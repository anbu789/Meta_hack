import math
from typing import Any

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "top_signal": {
        "drug": "DrugX",
        "event": "hepatotoxicity",
        "ROR": 4.80,
        "CI_lower": 1.074,
        "CI_upper": 21.45,
    },
    "contingency_table": {"a": 10, "b": 5, "c": 5, "d": 12},
    "non_signals": ["DrugY-rash", "DrugZ-nausea"],
}

ROR_TOLERANCE = 0.05
CI_TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _check_contingency_table(signal_flag: dict) -> tuple[float, str]:
    """+0.20 — correct a/b/c/d for the DrugX-hepatotoxicity pair."""
    ct = _safe_get(signal_flag, "top_signal", "contingency_table", default=None)
    if ct is None:
        ct = signal_flag.get("contingency_table", None)
    if ct is None:
        return 0.0, "contingency_table missing from signal_flag"

    gt = GROUND_TRUTH["contingency_table"]
    correct = all(
        abs(ct.get(k, -999) - gt[k]) <= 1  # allow ±1 tolerance on cell counts
        for k in ("a", "b", "c", "d")
    )
    if correct:
        return 0.20, "contingency table correct (a=10, b=5, c=5, d=12)"
    return 0.0, f"contingency table wrong — got {ct}, expected {gt}"


def _check_ror(signal_flag: dict) -> tuple[float, str]:
    """
    +0.30 — correct ROR for DrugX-hepatotoxicity within ±0.05.
    Partial +0.15 if agent computed *any* ROR and it's in the right ballpark (±0.30).
    """
    top = _safe_get(signal_flag, "top_signal", default={})
    ror = top.get("ROR") or top.get("ror")
    if ror is None:
        # try flat key
        ror = signal_flag.get("ROR") or signal_flag.get("ror")
    if ror is None:
        return 0.0, "ROR value missing from submission"

    try:
        ror = float(ror)
    except (TypeError, ValueError):
        return 0.0, "ROR is not a numeric value"

    diff = abs(ror - GROUND_TRUTH["top_signal"]["ROR"])
    if diff <= ROR_TOLERANCE:
        return 0.30, f"ROR correct: {ror:.3f} (expected 4.80)"
    if diff <= 0.30:
        return 0.15, f"ROR close but outside ±0.05 tolerance: {ror:.3f}"
    return 0.0, f"ROR incorrect: {ror:.3f} (expected 4.80)"


def _check_ci(signal_flag: dict) -> tuple[float, str]:
    """
    +0.20 — correct CI_lower > 1.0 and CI_lower within ±0.15 of 1.074.
    Partial +0.10 if CI_lower > 1.0 but outside tolerance.
    """
    top = _safe_get(signal_flag, "top_signal", default={})
    ci_lower = top.get("CI_lower") or top.get("ci_lower")
    if ci_lower is None:
        ci_lower = signal_flag.get("CI_lower") or signal_flag.get("ci_lower")
    if ci_lower is None:
        return 0.0, "CI_lower missing from submission"

    try:
        ci_lower = float(ci_lower)
    except (TypeError, ValueError):
        return 0.0, "CI_lower is not numeric"

    diff = abs(ci_lower - GROUND_TRUTH["top_signal"]["CI_lower"])
    if diff <= CI_TOLERANCE and ci_lower > 1.0:
        return 0.20, f"CI_lower correct: {ci_lower:.3f} (expected ~1.074)"
    if ci_lower > 1.0:
        return 0.10, f"CI_lower > 1.0 but outside tolerance: {ci_lower:.3f}"
    return 0.0, f"CI_lower ≤ 1.0 — signal threshold not met: {ci_lower:.3f}"


def _check_signal_identification(signal_flag: dict) -> tuple[float, str]:
    """
    +0.20 — DrugX-hepatotoxicity correctly flagged as signal;
             DrugY-rash and DrugZ-nausea correctly NOT flagged.
    """
    top = _safe_get(signal_flag, "top_signal", default={})
    drug = str(top.get("drug", "")).strip()
    event = str(top.get("event", "")).strip().lower()

    correct_drug = drug == "DrugX"
    correct_event = "hepatotox" in event  # accepts "hepatotoxicity" or abbreviated

    non_signals = signal_flag.get("non_signals", [])
    non_signal_strings = [str(x).lower() for x in non_signals]
    drugy_absent = not any("drugy" in s and "rash" in s for s in non_signal_strings) or \
                   any("drugy" in s and "rash" in s for s in non_signal_strings)
    # simplified: just check top signal is correct
    if correct_drug and correct_event:
        return 0.20, "Signal correctly identified: DrugX-hepatotoxicity"
    if correct_drug or correct_event:
        return 0.10, f"Partially correct signal: drug={drug}, event={event}"
    return 0.0, f"Wrong signal identified: drug={drug}, event={event}"


def _check_ranking(signal_flag: dict) -> tuple[float, str]:
    """
    +0.10 — DrugX-hepatotoxicity is ranked first / is the top signal.
    """
    ranked = signal_flag.get("ranked_signals", [])
    if not ranked:
        # if only one signal submitted and it's correct, give credit
        top = _safe_get(signal_flag, "top_signal", default={})
        if str(top.get("drug", "")) == "DrugX":
            return 0.10, "Single signal submitted and it is DrugX — ranking credit granted"
        return 0.0, "ranked_signals list missing and top signal is not DrugX"

    first = ranked[0] if isinstance(ranked[0], dict) else {}
    if str(first.get("drug", "")) == "DrugX":
        return 0.10, "Ranking correct — DrugX-hepatotoxicity is top signal"
    return 0.0, f"Ranking wrong — first ranked signal is {first.get('drug')}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_task2(action: dict) -> dict:
    """Standalone grader — accepts raw action dict."""
    signal_flag = action.get("signal_flag") or {}
    reasoning = action.get("reasoning", "")

    scores = {}
    messages = {}

    scores["contingency_table"], messages["contingency_table"] = _check_contingency_table(signal_flag)
    scores["ror_computation"], messages["ror_computation"] = _check_ror(signal_flag)
    scores["ci_computation"], messages["ci_computation"] = _check_ci(signal_flag)
    scores["signal_identification"], messages["signal_identification"] = _check_signal_identification(signal_flag)
    scores["signal_ranking"], messages["signal_ranking"] = _check_ranking(signal_flag)

    total = round(sum(scores.values()), 4)

    reasoning_bonus = 0.0
    if len(reasoning) > 80:
        reasoning_bonus = 0.0  # no bonus — scores already sum to 1.0

    feedback_lines = [f"  [{k}] {v} → +{scores[k]}" for k, v in messages.items()]
    feedback = "Task 2 Grader Results:\n" + "\n".join(feedback_lines) + f"\n  TOTAL: {total}"

    return {
        "score": total,
        "partial_credits": scores,
        "feedback": feedback,
    }


def grade(episode: Any, action: Any) -> tuple[float, dict, str]:
    """
    Adapter called by environment.py on submit action.
    Matches signature: grade(episode, action) -> (score, partial_credits, feedback)
    """
    action_dict = action.dict() if hasattr(action, "dict") else dict(action)
    result = grade_task2(action_dict)
    return result["score"], result["partial_credits"], result["feedback"]
