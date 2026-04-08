"""
E2E test — Task 1: classify → submit
Run with server live: uvicorn app.main:app --reload --port 7860
"""
import requests, json

BASE = "http://localhost:7860"

def sep(title): print(f"\n{'='*50}\n{title}\n{'='*50}")

# ── RESET ──────────────────────────────────────────
sep("RESET task1")
r = requests.post(f"{BASE}/reset", json={"task_id": "task1"})
assert r.status_code == 200, f"reset failed: {r.text}"
obs = r.json()
print(f"task_id      : {obs['task_id']}")
print(f"step_number  : {obs['step_number']}")
print(f"max_steps    : {obs['max_steps']}")
print(f"reports count: {len(obs['reports'])}")
report_ids = [rp["report_id"] for rp in obs["reports"]]
print(f"report_ids   : {report_ids}")

# ── STEP 1: classify ────────────────────────────────
sep("STEP 1: classify")
classify_action = {
    "action_type": "classify",
    "target_report_ids": report_ids,
    "classification": "Serious",
    "signal_flag": None,
    "reasoning": (
        "The patient experienced QT prolongation with syncope 3 days after starting "
        "Clarithromycin. QTc of 520ms constitutes a serious cardiac adverse event. "
        "This is unexpected per the drug label and meets WHO seriousness criteria "
        "(life-threatening condition). Expedited 15-day regulatory reporting is required. "
        "MedDRA SOC: Cardiac disorders."
    )
}
r = requests.post(f"{BASE}/step", json=classify_action)
assert r.status_code == 200, f"step failed: {r.text}"
result = r.json()
reward = result["reward"]
print(f"step_reward      : {reward['step_reward']}")
print(f"cumulative_reward: {reward['cumulative_reward']}")
print(f"done             : {reward['done']}")
print(f"partial_credits  : {json.dumps(reward['partial_credits'], indent=2)}")
print(f"feedback         : {reward['feedback']}")
assert not reward["done"], "should not be done after classify"
assert reward["step_reward"] >= 0.0

# ── STEP 2: submit ──────────────────────────────────
sep("STEP 2: submit")
submit_action = {
    "action_type": "submit",
    "target_report_ids": report_ids,
    "classification": "Serious",
    "signal_flag": {
        "unexpected": True,
        "expedited_reporting": True,
        "meddra_soc": "Cardiac disorders"
    },
    "reasoning": (
        "Final assessment: Severity=Serious (QTc 520ms, syncope, life-threatening). "
        "Event is unexpected — QT prolongation not listed in Clarithromycin label. "
        "Expedited 15-day report required per FDA/EMA regulations. "
        "MedDRA System Organ Class: Cardiac disorders."
    )
}
r = requests.post(f"{BASE}/step", json=submit_action)
assert r.status_code == 200, f"submit failed: {r.text}"
result = r.json()
reward = result["reward"]
print(f"step_reward      : {reward['step_reward']}")
print(f"cumulative_reward: {reward['cumulative_reward']}")
print(f"done             : {reward['done']}")
print(f"partial_credits  : {json.dumps(reward['partial_credits'], indent=2)}")
print(f"feedback         : {reward['feedback']}")
assert reward["done"], "should be done after submit"
assert reward["cumulative_reward"] > 0.0

# ── SUMMARY ────────────────────────────────────────
sep("SUMMARY")
print(f"Final cumulative_reward: {reward['cumulative_reward']}")
print(f"Expected range: 0.80 – 1.00 (step rewards + grader score)")
print("\nTask 1 E2E ✅" if reward["cumulative_reward"] >= 0.5 else "\nTask 1 E2E ⚠️  low score — check grader")
