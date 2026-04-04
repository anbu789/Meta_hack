"""
Day 3 — Task 2 end-to-end test
Run from repo root: python test_task2.py
Server must be running: uvicorn app.main:app --reload --port 7860
"""

import requests
import json

BASE_URL = "http://localhost:7860"


def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ---------------------------------------------------------------
# 1. Reset
# ---------------------------------------------------------------
separator("1. POST /reset  task_id=task2")
r = requests.post(f"{BASE_URL}/reset", json={"task_id": "task2"})
assert r.status_code == 200, f"reset failed: {r.status_code} {r.text}"
obs = r.json()
print(f"task_id       : {obs['task_id']}")
print(f"reports count : {len(obs['reports'])}")
print(f"max_steps     : {obs['max_steps']}")
print(f"step_number   : {obs['step_number']}")
assert obs["task_id"] == "task2"
assert len(obs["reports"]) == 32, f"Expected 32 reports, got {len(obs['reports'])}"
print("✅ reset OK")


# ---------------------------------------------------------------
# Verify contingency table counts from corpus
# ---------------------------------------------------------------
separator("2. Verify contingency table from corpus")
reports = obs["reports"]
drugx_hepa = sum(1 for r in reports if r["drug_name"] == "DrugX" and r["adverse_event"] == "hepatotoxicity")
drugy_rash = sum(1 for r in reports if r["drug_name"] == "DrugY" and r["adverse_event"] == "rash")
drugz_nausea = sum(1 for r in reports if r["drug_name"] == "DrugZ" and r["adverse_event"] == "nausea")
print(f"DrugX-hepatotoxicity count (a): {drugx_hepa}  (expected 10)")
print(f"DrugY-rash count              : {drugy_rash}  (expected 2)")
print(f"DrugZ-nausea count            : {drugz_nausea}  (expected 3)")
assert drugx_hepa == 10
assert drugy_rash == 2
assert drugz_nausea == 3
print("✅ Contingency counts verified")


# ---------------------------------------------------------------
# 3. Step — compute_ror intermediate
# ---------------------------------------------------------------
separator("3. POST /step  action_type=compute_ror")
step1 = {
    "action_type": "compute_ror",
    "target_report_ids": [r["report_id"] for r in reports if r["drug_name"] == "DrugX"],
    "reasoning": (
        "Building 2x2 contingency table for DrugX-hepatotoxicity. "
        "a=10 (DrugX+hepa), b=5 (DrugX-hepa), c=5 (noDrugX+hepa), d=12 (noDrugX-hepa). "
        "ROR = (10*12)/(5*5) = 4.80. CI_lower = exp(ln(4.80) - 1.96*sqrt(1/10+1/5+1/5+1/12)) = 1.074."
    ),
    "signal_flag": None,
    "classification": None,
}
r = requests.post(f"{BASE_URL}/step", json=step1)
assert r.status_code == 200, f"step1 failed: {r.status_code} {r.text}"
res = r.json()
print(f"step_reward       : {res['reward']['step_reward']}")
print(f"cumulative_reward : {res['reward']['cumulative_reward']}")
print(f"feedback          : {res['reward']['feedback']}")
print("✅ Step 1 (compute_ror) OK")


# ---------------------------------------------------------------
# 4. Step — submit with CORRECT signal
# ---------------------------------------------------------------
separator("4. POST /step  action_type=submit (correct signal)")
submit_correct = {
    "action_type": "submit",
    "target_report_ids": [r["report_id"] for r in reports],
    "reasoning": (
        "After computing ROR for all drug-event pairs, DrugX-hepatotoxicity has "
        "ROR=4.80 (CI_lower=1.074, CI_upper=21.45), which exceeds signal threshold "
        "(ROR>2.0 AND CI_lower>1.0). DrugY-rash ROR=1.667 and DrugZ-nausea ROR=1.800 "
        "are both below threshold. DrugX-hepatotoxicity is the only confirmed signal."
    ),
    "signal_flag": {
        "top_signal": {
            "drug": "DrugX",
            "event": "hepatotoxicity",
            "ROR": 4.80,
            "CI_lower": 1.074,
            "CI_upper": 21.45,
            "contingency_table": {"a": 10, "b": 5, "c": 5, "d": 12},
        },
        "non_signals": ["DrugY-rash", "DrugZ-nausea"],
        "ranked_signals": [
            {"drug": "DrugX", "event": "hepatotoxicity", "ROR": 4.80}
        ],
    },
    "classification": None,
}
r = requests.post(f"{BASE_URL}/step", json=submit_correct)
assert r.status_code == 200, f"submit failed: {r.status_code} {r.text}"
res = r.json()
score = res["reward"]["cumulative_reward"]
print(f"cumulative_reward : {score}")
print(f"done              : {res['reward']['done']}")
print(f"feedback:\n{res['reward']['feedback']}")
assert res["reward"]["done"] is True
assert score >= 0.85, f"Expected score >= 0.85 for correct submit, got {score}"
print(f"✅ Submit (correct) scored {score:.3f}")


# ---------------------------------------------------------------
# 5. Reset + submit with WRONG signal — verify score is lower
# ---------------------------------------------------------------
separator("5. Reset + submit wrong signal (score variance check)")
requests.post(f"{BASE_URL}/reset", json={"task_id": "task2"})
submit_wrong = {
    "action_type": "submit",
    "target_report_ids": [],
    "reasoning": "DrugZ-nausea appears to be the main signal.",
    "signal_flag": {
        "top_signal": {
            "drug": "DrugZ",
            "event": "nausea",
            "ROR": 1.80,
            "CI_lower": 0.6,
            "CI_upper": 5.2,
            "contingency_table": {"a": 3, "b": 6, "c": 5, "d": 18},
        },
        "non_signals": [],
        "ranked_signals": [{"drug": "DrugZ", "event": "nausea", "ROR": 1.80}],
    },
    "classification": None,
}
r = requests.post(f"{BASE_URL}/step", json=submit_wrong)
res = r.json()
wrong_score = res["reward"]["cumulative_reward"]
print(f"Wrong signal score: {wrong_score:.3f}  (should be < {score:.3f})")
assert wrong_score < score, "Wrong signal should score lower than correct signal"
print(f"✅ Score variance confirmed: correct={score:.3f} > wrong={wrong_score:.3f}")


# ---------------------------------------------------------------
# 6. /grader endpoint
# ---------------------------------------------------------------
separator("6. POST /grader")
r = requests.post(f"{BASE_URL}/grader")
assert r.status_code == 200, f"grader failed: {r.status_code}"
print(json.dumps(r.json(), indent=2))
print("✅ /grader OK")


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
separator("ALL TESTS PASSED")
print(f"Task 2 correct-submit score : {score:.3f}")
print(f"Task 2 wrong-submit score   : {wrong_score:.3f}")
print("Day 3 complete ✅")
