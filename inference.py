"""
inference.py — PharmaVigil OpenEnv Agent
Root-level inference script for Meta PyTorch OpenEnv Hackathon.

PyTorch usage: encodes ADE report narratives using sentence-transformers
(all-MiniLM-L6-v2) and uses cosine similarity to rank/filter reports before
sending context to the LLM.

Env vars:
    API_BASE_URL   — HuggingFace inference endpoint base URL
    MODEL_NAME     — Model name (default: gpt-4o)
    HF_TOKEN       — HuggingFace token
    ENV_URL        — PharmaVigil FastAPI server (default: http://localhost:7860)
    OPENAI_API_KEY — OpenAI key fallback
"""

import os
import json
import time
import logging
from typing import Optional

import torch
import torch.nn.functional as F
import requests
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
ENV_URL      = os.getenv("ENV_URL") or "http://localhost:7860"

client_kwargs = {"api_key": API_KEY}
if API_BASE_URL:
    client_kwargs["base_url"] = API_BASE_URL
client = OpenAI(**client_kwargs)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer: Optional[AutoTokenizer] = None
_embed_model: Optional[AutoModel]   = None


def _load_embed_model():
    global _tokenizer, _embed_model
    if _embed_model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _tokenizer   = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _embed_model.eval()


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def embed_texts(texts: list[str]) -> torch.Tensor:
    _load_embed_model()
    encoded = _tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = _embed_model(**encoded)
    embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def rank_reports_by_query(reports: list[dict], query: str, top_k: int = 10) -> list[dict]:
    if not reports:
        return reports
    narratives  = [r.get("narrative", r.get("adverse_event", "")) for r in reports]
    query_emb   = embed_texts([query])
    report_embs = embed_texts(narratives)
    scores      = (report_embs @ query_emb.T).squeeze(1)
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    ranked = [reports[i] for i in top_indices]
    log.info("PyTorch ranking: selected %d/%d reports", len(ranked), len(reports))
    return ranked


SYSTEM_PROMPT = """You are a pharmacovigilance analyst (FDA/EMA/WHO standards).
Analyze ICSRs and return a single valid JSON action object — no markdown, no prose.

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
        "Step 1: action_type=classify. Step 2: action_type=submit."
    ),
    "task2": (
        "Build 2x2 contingency tables for each drug-event pair. "
        "ROR = (a*d)/(b*c). 95% CI = exp(ln(ROR) ± 1.96*sqrt(1/a+1/b+1/c+1/d)). "
        "Signal threshold: ROR>2.0 AND CI_lower>1.0. "
        "Step 1: compute_ror. Step 2: flag_signal. Step 3: submit."
    ),
    "task3": (
        "Step 1: compute standard ROR for Azithromycin-QT prolongation (full corpus). "
        "Step 2: SEGMENT by Amiodarone co-administration. Recompute ROR in Amiodarone-ABSENT subgroup. "
        "Step 3: flag_signal with masking_drug, masked_drug, standard_ROR, stratified_ROR, suppression_magnitude, segmented=True. "
        "Step 4: submit with regulatory recommendation in reasoning."
    ),
}


def api_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def api_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def call_llm(messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=1024, temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def run_episode(task_id: str) -> float:
    log.info("═══ Starting episode: %s ═══", task_id)
    obs          = api_reset(task_id)
    task_hint    = TASK_HINTS.get(task_id, "")
    done         = False
    total_reward = 0.0
    history: list[dict] = []

    while not done:
        step_num         = obs.get("step_number", 0)
        max_steps        = obs.get("max_steps", 10)
        reports          = obs.get("reports", [])
        workspace        = obs.get("workspace", {})
        task_desc        = obs.get("task_description", "")
        query            = task_desc or task_hint
        relevant_reports = rank_reports_by_query(reports, query, top_k=min(15, len(reports)))

        user_content = (
            f"Task: {task_id.upper()} | Step {step_num}/{max_steps}\n"
            f"Description: {task_desc}\n"
            f"Hint: {task_hint}\n\n"
            f"Reports ({len(relevant_reports)} of {len(reports)} selected):\n"
            f"{json.dumps(relevant_reports, indent=2)}\n\n"
            f"Workspace: {json.dumps(workspace, indent=2)}\n\n"
            "Output your next action as valid JSON only."
        )

        history.append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        try:
            action_text = call_llm(messages)
            action      = parse_action(action_text)
        except Exception as e:
            log.error("LLM/parse error at step %d: %s", step_num, e)
            action = {
                "action_type": "submit",
                "target_report_ids": [],
                "classification": None,
                "signal_flag": None,
                "reasoning": "Fallback submit due to error.",
            }

        log.info("  Step %d → action_type=%s", step_num, action.get("action_type"))
        history.append({"role": "assistant", "content": json.dumps(action)})

        try:
            result = api_step(action)
        except Exception as e:
            log.error("Step API error: %s", e)
            break

        reward_obj   = result.get("reward", {})
        total_reward = reward_obj.get("cumulative_reward", total_reward)
        done         = reward_obj.get("done", False)
        obs          = result.get("observation", obs)

        log.info("  → cumulative=%.3f done=%s", total_reward, done)

        if step_num >= max_steps:
            log.warning("Max steps reached — exiting")
            break
        time.sleep(0.3)

    log.info("Episode %s final score: %.3f", task_id, total_reward)
    return total_reward


if __name__ == "__main__":
    log.info("PharmaVigil inference.py | ENV_URL=%s | MODEL=%s", ENV_URL, MODEL_NAME)

    scores = {}
    for task in ["task1", "task2", "task3"]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            log.error("Episode %s failed: %s", task, e)
            scores[task] = 0.0

    print("\n" + "═" * 40)
    print("PharmaVigil Inference Scores")
    print("═" * 40)
    for task, score in scores.items():
        print(f"  {task}: {score:.3f}")
    print(f"  average: {sum(scores.values()) / len(scores):.3f}")
    print("═" * 40)
