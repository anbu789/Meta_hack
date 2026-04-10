"""
baseline/baseline_agent.py — PharmaVigil Zero-Shot Baseline Agent
Uses HuggingFace Inference API.

Env vars:
    HF_TOKEN         — required (HuggingFace token)
    ENV_URL          — PharmaVigil FastAPI server (default: http://localhost:7860)
"""

import os
import re
import json
import time
import logging

import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENV_URL      = os.getenv("ENV_URL") or "http://localhost:7860"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a pharmacovigilance analyst with expertise in
FDA/EMA/WHO adverse event reporting, MedDRA terminology, and disproportionality
statistics (ROR, signal detection, masking effects).

Always respond with a single valid JSON action object and nothing else.
No markdown, no explanation outside the JSON. No comments inside JSON. No trailing commas.

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
        "Classify severity as Serious/Life-threatening/Moderate/Mild using WHO criteria. "
        "Determine if event is unexpected (not in drug label). "
        "Set expedited_report=true if Serious AND unexpected (15-day rule). "
        "Assign the correct MedDRA System Organ Class. "
        "Step 1: action_type=classify. Step 2: action_type=submit."
    ),
    "task2": (
        "Build 2x2 contingency tables for every drug-event pair. "
        "ROR = (a*d)/(b*c). 95% CI = exp(ln(ROR) +/- 1.96*sqrt(1/a+1/b+1/c+1/d)). "
        "Haldane correction: add 0.5 to all cells if any cell is 0. "
        "Signal threshold: ROR > 2.0 AND CI_lower > 1.0. "
        "Step 1: compute_ror. Step 2: flag_signal with top_signal + non_signals + ranked_signals. "
        "Step 3: submit."
    ),
    "task3": (
        "Step 1: compute standard ROR for Azithromycin-QT prolongation across full corpus. "
        "Step 2: segment corpus by Amiodarone co-administration (check concomitant_drugs field). "
        "Recompute Azithromycin-QT ROR in the Amiodarone-ABSENT subgroup (stratified ROR). "
        "Step 3: flag_signal with keys: masking_drug, masked_drug, standard_ROR, "
        "stratified_ROR, suppression_magnitude, masking_confirmed=true, segmented=true. "
        "Step 4: submit — include 'stratified analysis' and 'regulatory recommendation' in reasoning."
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


def call_llm(user_content: str) -> str:
    """Fresh prompt each step — no history accumulation to stay within context window."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # Strip // comments
    text = re.sub(r'//[^\n]*', '', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Remove control characters that break JSON parsing
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    return json.loads(text.strip())


def force_submit() -> dict:
    return {
        "action_type": "submit",
        "target_report_ids": [],
        "classification": None,
        "signal_flag": None,
        "reasoning": "Force submit at max steps.",
    }


def run_episode(task_id: str) -> float:
    log.info("═══ Baseline episode: %s ═══", task_id)
    obs          = api_reset(task_id)
    task_hint    = TASK_HINTS.get(task_id, "")
    done         = False
    total_reward = 0.0

    while not done:
        step_num  = obs.get("step_number", 0)
        max_steps = obs.get("max_steps", 10)
        reports   = obs.get("reports", [])
        workspace = obs.get("workspace", {})
        task_desc = obs.get("task_description", "")

        # Force submit on last step to prevent looping past max_steps
        if step_num >= max_steps - 1:
            action = force_submit()
        else:
            # Limit reports to 5 to stay within 8B context window
            reports_truncated = reports[:5]

            user_content = (
                f"Task: {task_id.upper()} | Step {step_num}/{max_steps}\n"
                f"Description: {task_desc}\n"
                f"Hint: {task_hint}\n\n"
                f"Reports (showing {len(reports_truncated)} of {len(reports)}):\n"
                f"{json.dumps(reports_truncated, indent=2)}\n\n"
                f"Workspace: {json.dumps(workspace, indent=2)}\n\n"
                "Output your next action as valid JSON only. No comments. No trailing commas."
            )

            try:
                action_text = call_llm(user_content)
                action      = parse_action(action_text)
            except Exception as e:
                log.error("LLM/parse error at step %d: %s", step_num, e)
                action = force_submit()

        log.info("  Step %d → action_type=%s", step_num, action.get("action_type"))

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

        # 1.0s sleep to avoid HF router rate limiting on free tier
        time.sleep(1.0)

    log.info("Episode %s final score: %.3f", task_id, total_reward)
    return total_reward


if __name__ == "__main__":
    if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
        raise SystemExit("ERROR: HF_TOKEN env var not set.")

    log.info("PharmaVigil Baseline Agent | ENV_URL=%s", ENV_URL)

    scores = {}
    for task in ["task1", "task2", "task3"]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            log.error("Episode %s failed: %s", task, e)
            scores[task] = 0.0

    print("\n" + "═" * 40)
    print("PharmaVigil Baseline Scores")
    print("═" * 40)
    for task, score in scores.items():
        print(f"  {task}: {score:.3f}")
    print(f"  average: {sum(scores.values()) / len(scores):.3f}")
    print("═" * 40)
    print("\nExpected ranges:")
    print("  task1: 0.60 – 0.75")
    print("  task2: 0.40 – 0.55")
    print("  task3: 0.10 – 0.25")
