"""
E2E test — Task 3: compute_ror → compute_ror (stratified) → flag_signal → submit
Run with server live: uvicorn app.main:app --reload --port 7860
"""
import requests, json

BASE = "http://localhost:7860"
def sep(title): print(f"\n{'='*50}\n{title}\n{'='*50}")

# ── RESET ──────────────────────────────────────────
sep("RESET task3")
r = requests.post(f"{BASE}/reset", json={"task_id": "task3"})
assert r.status_code == 200, f"reset failed: {r.text}"
obs = r.json()
print(f"task_id      : {obs['task_id']}")
print(f"reports count: {len(obs['reports'])}")
report_ids = [rp["report_id"] for rp in obs["reports"]]

# ── STEP 1: compute_ror (standard, full corpus) ─────
sep("STEP 1: compute_ror (standard)")
r = requests.post(f"{BASE}/step", json={
    "action_type": "compute_ror",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {"drug": "Azithromycin", "event": "QT prolongation", "standard_ROR": 1.4},
    "reasoning": (
        "Standard ROR for Azithromycin-QT prolongation across the full corpus: "
        "a=7, b=10, c=10, d=20 → ROR=1.40. Below signal threshold of 2.0. "
        "Amiodarone has a strong QT prolongation signal — potential masking effect. "
        "Will stratify corpus by Amiodarone co-administration to investigate."
    )
})
assert r.status_code == 200
reward = r.json()["reward"]
print(f"step_reward: {reward['step_reward']} | cumulative: {reward['cumulative_reward']}")
assert not reward["done"]

# ── STEP 2: compute_ror (stratified, Amiodarone-absent) ─
sep("STEP 2: compute_ror (stratified)")
r = requests.post(f"{BASE}/step", json={
    "action_type": "compute_ror",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {
        "drug": "Azithromycin", "event": "QT prolongation",
        "standard_ROR": 1.4,
        "stratified_ROR": 3.8,
        "segmented": True
    },
    "reasoning": (
        "Segmented corpus by Amiodarone co-administration. "
        "Azithromycin-QT prolongation in Amiodarone-absent subgroup: "
        "a=4, b=4, c=5, d=19 → stratified ROR=3.80, above threshold 2.0. "
        "Masking confirmed: Amiodarone suppresses Azithromycin signal in full corpus. "
        "Suppression magnitude = 3.8 - 1.4 = 2.4."
    )
})
assert r.status_code == 200
reward = r.json()["reward"]
print(f"step_reward: {reward['step_reward']} | cumulative: {reward['cumulative_reward']}")
assert not reward["done"]

# ── STEP 3: flag_signal ─────────────────────────────
sep("STEP 3: flag_signal")
r = requests.post(f"{BASE}/step", json={
    "action_type": "flag_signal",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {
        "drug": "Azithromycin", "event": "QT prolongation",
        "masking_drug": "Amiodarone",
        "masked_drug": "Azithromycin",
        "masking_confirmed": True,
        "standard_ROR": 1.4,
        "stratified_ROR": 3.8,
        "suppression_magnitude": 2.4,
        "segmented": True
    },
    "reasoning": (
        "Masking effect confirmed: Amiodarone masks Azithromycin QT prolongation signal. "
        "Standard ROR=1.40 (below threshold), stratified ROR=3.80 (above threshold). "
        "Suppression magnitude=2.4. Regulatory recommendation: stratified analysis "
        "required in signal detection workflow. Recommend label update investigation."
    )
})
assert r.status_code == 200
reward = r.json()["reward"]
print(f"step_reward: {reward['step_reward']} | cumulative: {reward['cumulative_reward']}")
assert not reward["done"]

# ── STEP 4: submit ──────────────────────────────────
sep("STEP 4: submit")
r = requests.post(f"{BASE}/step", json={
    "action_type": "submit",
    "target_report_ids": report_ids,
    "classification": None,
    "signal_flag": {
        "drug": "Azithromycin", "event": "QT prolongation",
        "masking_drug": "Amiodarone",
        "masked_drug": "Azithromycin",
        "masked_event": "QT prolongation",
        "masking_confirmed": True,
        "standard_ROR": 1.4,
        "stratified_ROR": 3.8,
        "suppression_magnitude": 2.4,
        "segmented": True,
        "corpus_segmented": True
    },
    "reasoning": (
        "Final submission: Amiodarone masks the Azithromycin QT prolongation signal. "
        "Standard ROR=1.40 (below threshold 2.0). After stratified analysis excluding "
        "Amiodarone co-administered reports: ROR=3.80 (above threshold). "
        "Suppression magnitude=2.4. Regulatory recommendation: mandatory stratified "
        "analysis for all QT-prolonging drugs. Signal should be flagged for label review."
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
print("\nTask 3 E2E ✅" if reward["cumulative_reward"] >= 0.7 else "\nTask 3 E2E ⚠️  low score — check grader")
