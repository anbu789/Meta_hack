"""
inference.py — PharmaVigil OpenEnv Agent
Root-level inference script for Meta PyTorch OpenEnv Hackathon.

PyTorch usage: encodes ADE report narratives using sentence-transformers
(all-MiniLM-L6-v2) and uses cosine similarity to rank/filter reports before
sending context to the LLM.

Env vars:
    API_BASE_URL   — HuggingFace inference endpoint base URL
    MODEL_NAME     — Model name (default: Qwen/Qwen2.5-7B-Instruct)
    HF_TOKEN       — HuggingFace token
    ENV_URL        — PharmaVigil FastAPI server (default: http://localhost:7860)
"""

import os
import re
import json
import time
from typing import Optional, List

import torch
import torch.nn.functional as F
import requests
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "pharmavigil"

# Max steps per task — keeps total runtime well under 30 min validator limit
MAX_STEPS_PER_TASK = 5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Score clamping — validator requires strictly between 0 and 1 (exclusive)
# ---------------------------------------------------------------------------

def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))

# ---------------------------------------------------------------------------
# Mandatory stdout log functions
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={clamp_score(score):.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# PyTorch embedding model
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer: Optional[AutoTokenizer] = None
_embed_model: Optional[AutoModel] = None


def _load_embed_model():
    global _tokenizer, _embed_model
    if _embed_model is None:
        _tokenizer   = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _embed_model.eval()


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def embed_texts(texts: list) -> torch.Tensor:
    _load_embed_model()
    encoded = _tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = _embed_model(**encoded)
    embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def rank_reports_by_query(reports: list, query: str, top_k: int = 5) -> list:
    if not reports:
        return reports
    narratives  = [r.get("narrative", r.get("adverse_event", "")) for r in reports]
    query_emb   = embed_texts([query])
    report_embs = embed_texts(narratives)
    scores      = (report_embs @ query_emb.T).squeeze(1)
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [reports[i] for i in top_indices]

# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------

def api_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def api_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a pharmacovigilance analyst (FDA/EMA/WHO standards).
Analyze ICSRs and return a single valid JSON action object — no markdown, no prose, no comments, no trailing commas.

Action schema:
{
  "action_type": one of [classify, flag_signal, compute_ror, identify_interaction, submit],
  "target_report_ids": [list of report_id strings],
  "classification": "severity string or null",
  "signal_flag": { ...structured finding... } or null,
  "reasoning": "detailed explanation (required)"
}
"""

TASK_HINTS = {
    "task1": (
        "Classify severity (Serious/Life-threatening/Moderate/Mild). "
        "Check if the event is in the drug label (unexpected=true if absent). "
        "Flag expedited_report=true if Serious+Unexpected (15-day rule). "
        "Assign MedDRA SOC (e.g. 'Cardiac disorders'). "
        "Step 1: action_type=classify. Step 2: action_type=submit with signal_flag containing "
        "unexpected, expedited_report, meddra_soc keys."
    ),
    "task2": (
        "Build 2x2 contingency tables for each drug-event pair. "
        "ROR = (a*d)/(b*c). 95% CI = exp(ln(ROR) +/- 1.96*sqrt(1/a+1/b+1/c+1/d)). "
        "Signal threshold: ROR>2.0 AND CI_lower>1.0. "
        "Step 1: compute_ror. Step 2: flag_signal with top_signal, non_signals, ranked_signals, contingency_table. Step 3: submit."
    ),
    "task3": (
        "Step 1: compute standard ROR for Azithromycin-QT prolongation (full corpus). "
        "Step 2: SEGMENT by Amiodarone co-administration. Recompute ROR in Amiodarone-ABSENT subgroup. "
        "Step 3: flag_signal with masking_drug=Amiodarone, masked_drug=Azithromycin, standard_ROR, "
        "stratified_ROR, suppression_magnitude, segmented=true, masking_confirmed=true. "
        "Step 4: submit with 'stratified analysis' and 'regulatory recommendation' in reasoning."
    ),
}


def call_llm(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()


def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    return json.loads(text.strip())

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    obs         = api_reset(task_id)
    task_hint   = TASK_HINTS.get(task_id, "")
    done        = False
    rewards: List[float] = []
    history: list = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not done:
            step_num  = obs.get("step_number", 0)
            max_steps = obs.get("max_steps", 10)
            reports   = obs.get("reports", [])
            workspace = obs.get("workspace", {})
            task_desc = obs.get("task_description", "")

            if steps_taken >= MAX_STEPS_PER_TASK:
                action = {
                    "action_type": "submit",
                    "target_report_ids": [],
                    "classification": None,
                    "signal_flag": None,
                    "reasoning": "Force submit to stay within time limit.",
                }
            else:
                query            = task_desc or task_hint
                relevant_reports = rank_reports_by_query(reports, query, top_k=min(5, len(reports)))

                user_content = (
                    f"Task: {task_id.upper()} | Step {step_num}/{max_steps}\n"
                    f"Description: {task_desc}\n"
                    f"Hint: {task_hint}\n\n"
                    f"Reports ({len(relevant_reports)} selected):\n"
                    f"{json.dumps(relevant_reports, indent=2)}\n\n"
                    f"Workspace: {json.dumps(workspace, indent=2)}\n\n"
                    "Output your next action as valid JSON only. No comments. No trailing commas."
                )

                history.append({"role": "user", "content": user_content})
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:]

                error = None
                try:
                    action_text = call_llm(messages)
                    action      = parse_action(action_text)
                except Exception as e:
                    error = str(e)
                    action = {
                        "action_type": "submit",
                        "target_report_ids": [],
                        "classification": None,
                        "signal_flag": None,
                        "reasoning": "Fallback submit due to error.",
                    }

                history.append({"role": "assistant", "content": json.dumps(action)})

            error = None
            try:
                result      = api_step(action)
                reward_obj  = result.get("reward", {})
                step_reward = reward_obj.get("step_reward", 0.0)
                done        = reward_obj.get("done", False)
                score       = reward_obj.get("cumulative_reward", score)
                obs         = result.get("observation", obs)
            except Exception as e:
                error = str(e)
                done  = True
                step_reward = 0.0

            steps_taken += 1
            rewards.append(step_reward)

            log_step(
                step=steps_taken,
                action=action.get("action_type", "unknown"),
                reward=step_reward,
                done=done,
                error=error,
            )

            if step_num >= max_steps:
                break

            time.sleep(0.1)

        success = score >= 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    # Clamp final score — validator rejects exactly 0.0 and 1.0
    return clamp_score(score)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scores = {}
    for task in ["task1", "task2", "task3"]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"[DEBUG] Episode {task} failed: {e}", flush=True)
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])
            scores[task] = 0.01

    print("\n" + "=" * 40, flush=True)
    print("PharmaVigil Inference Scores", flush=True)
    print("=" * 40, flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.3f}", flush=True)
    print(f"  average: {sum(scores.values()) / len(scores):.3f}", flush=True)
    print("=" * 40, flush=True)
