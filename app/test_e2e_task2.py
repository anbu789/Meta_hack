"""
E2E test — Task 2: compute_ror → flag_signal → submit
Run with server live: uvicorn app.main:app --reload --port 7860
"""
import requests, json

BASE = "http://localhost:7860"

def sep(title): print(f"\n{'='*50}\n{title}\n{'='*50}")

TOP_SIGNAL = {
    "drug": "DrugX",
    "event": "hepatotoxicity",
    "ROR": 4.80,
    "CI_lower": 1.074,
    "CI_upper": 21.45,
    "contingency_table": {"a": 10, "b": 5, "c": 5, "d": 12}
}

# ── RESET ──────────────────────────────────────────
sep("RESET task2")
r = requests.post(f"{BASE}/reset", json={"task_id": "task2"})
assert r.status_code == 200, f"reset failed: {r.text}"
obs = r.json()
print(f"task_id      : {obs['task_id']}")
print(f"reports count: {len(obs['reports'])}")
report_ids = [rp["report_id"] for rp in obs["reports"]]

# ── STEP 1: compute_ror ─────────────────────────────
sep("STEP 1: compute_ror")
r = requests.post(f"{BASE}/step", json={
    "action_type": "compute_ror",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {"drug": "DrugX", "event": "hepatotoxicity", "ROR": 4.80},
    "reasoning": (
        "Built 2x2 contingency table for DrugX-hepatotoxicity: a=10, b=5, c=5, d=12. "
        "ROR = (10*12)/(5*5) = 4.80. 95% CI lower=1.074 > 1.0. "
        "Exceeds signal threshold ROR>2.0 and CI_lower>1.0. "
        "DrugY-rash and DrugZ-nausea are below threshold — non-signals."
    )
})
assert r.status_code == 200
reward = r.json()["reward"]
print(f"step_reward: {reward['step_reward']} | cumulative: {reward['cumulative_reward']}")
assert not reward["done"]

# ── STEP 2: flag_signal ─────────────────────────────
sep("STEP 2: flag_signal")
r = requests.post(f"{BASE}/step", json={
    "action_type": "flag_signal",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {
        "drug": "DrugX", "event": "hepatotoxicity",
        "top_signal": TOP_SIGNAL,
        "non_signals": ["DrugY-rash", "DrugZ-nausea"]
    },
    "reasoning": (
        "DrugX-hepatotoxicity meets disproportionality signal threshold: "
        "ROR=4.80 > 2.0 and CI_lower=1.074 > 1.0. "
        "Contingency table: a=10, b=5, c=5, d=12. "
        "DrugY-rash (ROR=1.67) and DrugZ-nausea (ROR=1.80) are non-signals. "
        "Flagging DrugX-hepatotoxicity for regulatory review."
    )
})
assert r.status_code == 200
reward = r.json()["reward"]
print(f"step_reward: {reward['step_reward']} | cumulative: {reward['cumulative_reward']}")
assert not reward["done"]

# ── STEP 3: submit ──────────────────────────────────
sep("STEP 3: submit")
r = requests.post(f"{BASE}/step", json={
    "action_type": "submit",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {
        "drug": "DrugX", "event": "hepatotoxicity",
        "top_signal": TOP_SIGNAL,
        "non_signals": ["DrugY-rash", "DrugZ-nausea"],
        "contingency_table": {"a": 10, "b": 5, "c": 5, "d": 12},
        "ranked_signals": [TOP_SIGNAL]
    },
    "reasoning": (
        "Final submission: DrugX-hepatotoxicity is the top-ranked signal. "
        "ROR=4.80, CI=[1.074, 21.45], contingency table a=10,b=5,c=5,d=12. "
        "Signal threshold met: ROR>2.0 and CI_lower>1.0. "
        "Non-signals: DrugY-rash (ROR=1.67), DrugZ-nausea (ROR=1.80). "
        "Recommend expedited regulatory review for hepatotoxicity adverse event."
    )
})
assert r.status_code == 200
result = r.json()
reward = result["reward"]
print(f"step_reward      : {reward['step_reward']}")
print(f"cumulative_reward: {reward['cumulative_reward']}")
print(f"done             : {reward['done']}")
print(f"partial_credits  : {json.dumps(reward['partial_credits'], indent=2)}")
print(f"feedback         : {reward['feedback']}")
assert reward["done"]

sep("SUMMARY")
print(f"Final cumulative_reward: {reward['cumulative_reward']}")
print("\nTask 2 E2E ✅" if reward["cumulative_reward"] >= 0.7 else "\nTask 2 E2E ⚠️  low score — check grader")
