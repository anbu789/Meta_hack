import json
from app.models import Action

# Ground truth — do not modify
GROUND_TRUTH = {
    "correct_severity": "Serious",
    "correct_unexpected": True,
    "correct_expedited_flag": True,
    "correct_meddra_soc": "Cardiac disorders"
}

# Accepted aliases for MedDRA SOC (case-insensitive)
MEDDRA_SOC_ALIASES = [
    "cardiac disorders",
    "cardiac disorder"
]


def _parse_classification(action: dict) -> dict | None:
    """
    Extract the classification dict from the action.
    Accepts either a dict or a JSON string in action['classification'].
    Returns None if unparseable.
    """
    raw = action.get("classification")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _check_severity(classification: dict) -> tuple[float, str]:
    agent_val = str(classification.get("severity", "")).strip()
    expected = GROUND_TRUTH["correct_severity"]
    if agent_val.lower() == expected.lower():
        return 0.25, f"Severity correct: '{agent_val}'"
    return 0.0, f"Severity incorrect: got '{agent_val}', expected '{expected}'"


def _check_unexpected(classification: dict) -> tuple[float, str]:
    raw = classification.get("unexpected")
    if isinstance(raw, bool):
        agent_val = raw
    elif isinstance(raw, str):
        agent_val = raw.strip().lower() in ("true", "yes", "1")
    else:
        return 0.0, f"Unexpected field missing or unparseable: got '{raw}'"

    expected = GROUND_TRUTH["correct_unexpected"]
    if agent_val == expected:
        return 0.25, f"Unexpected classification correct: {agent_val}"
    return 0.0, f"Unexpected classification incorrect: got '{agent_val}', expected '{expected}'"


def _check_expedited(classification: dict) -> tuple[float, str]:
    raw = classification.get("expedited_report")
    if isinstance(raw, bool):
        agent_val = raw
    elif isinstance(raw, str):
        agent_val = raw.strip().lower() in ("true", "yes", "1")
    else:
        return 0.0, f"Expedited report field missing or unparseable: got '{raw}'"

    expected = GROUND_TRUTH["correct_expedited_flag"]
    if agent_val == expected:
        return 0.25, f"Expedited report flag correct: {agent_val}"
    return 0.0, f"Expedited report flag incorrect: got '{agent_val}', expected '{expected}'"


def _check_meddra_soc(classification: dict) -> tuple[float, str]:
    agent_val = str(classification.get("meddra_soc", "")).strip().lower()
    if agent_val in MEDDRA_SOC_ALIASES:
        return 0.25, f"MedDRA SOC correct: '{classification.get('meddra_soc')}'"
    return 0.0, (
        f"MedDRA SOC incorrect: got '{classification.get('meddra_soc')}', "
        f"expected '{GROUND_TRUTH['correct_meddra_soc']}'"
    )


def grade_task1(action: dict) -> dict:
    """
    Grade a Task 1 classify action.

    Args:
        action: the agent's Action dict (action_type, classification, reasoning, ...)

    Returns:
        {
            "score": float,          # 0.0 – 1.0
            "partial_credits": dict, # per-criterion breakdown
            "feedback": str          # human-readable summary
        }
    """
    # Must be a classify action
    if action.get("action_type") != "classify":
        return {
            "score": 0.0,
            "partial_credits": {},
            "feedback": f"Expected action_type 'classify', got '{action.get('action_type')}'. No score awarded."
        }

    classification = _parse_classification(action)
    if classification is None:
        return {
            "score": 0.0,
            "partial_credits": {},
            "feedback": "Could not parse 'classification' field. Must be a dict or valid JSON string."
        }

    # Run all 4 checks
    sev_score,  sev_msg  = _check_severity(classification)
    unex_score, unex_msg = _check_unexpected(classification)
    exp_score,  exp_msg  = _check_expedited(classification)
    soc_score,  soc_msg  = _check_meddra_soc(classification)

    total = round(sev_score + unex_score + exp_score + soc_score, 2)

    partial_credits = {
        "severity_classification": {"score": sev_score,  "max": 0.25, "message": sev_msg},
        "unexpected_determination": {"score": unex_score, "max": 0.25, "message": unex_msg},
        "expedited_report_flag":    {"score": exp_score,  "max": 0.25, "message": exp_msg},
        "meddra_soc_assignment":    {"score": soc_score,  "max": 0.25, "message": soc_msg},
    }

    correct_count = sum(1 for v in partial_credits.values() if v["score"] > 0)
    feedback = (
        f"Task 1 grader: {correct_count}/4 criteria correct. "
        f"Total score: {total:.2f}. "
        + " | ".join(v["message"] for v in partial_credits.values())
    )

    return {
        "score": total,
        "partial_credits": partial_credits,
        "feedback": feedback
    }


def grade(episode, action: Action) -> tuple[float, dict, str]:
    """
    Adapter called by environment.py on submit.
    Signature: grade(episode, action) -> (score, partial_credits, feedback)
    """
    result = grade_task1(action.dict())
    return result["score"], result["partial_credits"], result["feedback"]
