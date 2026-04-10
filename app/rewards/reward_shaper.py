# app/rewards/reward_shaper.py

DOMAIN_KEYWORDS = [
    "severity", "serious", "unexpected", "expedited", "meddra", "soc",
    "ror", "reporting odds ratio", "contingency", "signal", "ci", "confidence",
    "hepatotoxicity", "qt prolongation", "masking", "stratified", "amiodarone",
    "azithromycin", "concomitant", "adverse", "event", "drug", "patient",
    "classification", "cardiac", "threshold", "suppression", "disproportionality"
]

TASK_STAGE_MAP = {
    "task1": {1: "classify", 2: "submit"},
    "task2": {1: "compute_ror", 2: "flag_signal", 3: "submit"},
    "task3": {1: "compute_ror", 2: "compute_ror", 3: "flag_signal", 4: "submit"},
}

# Hard cap so cumulative reward never exceeds 1.0
MAX_CUMULATIVE_REWARD = 0.99


def _check_action_type(task_id: str, step_number: int, action_type: str) -> tuple:
    stage_map = TASK_STAGE_MAP.get(task_id, {})
    max_step = max(stage_map.keys()) if stage_map else 1
    expected = stage_map.get(min(step_number, max_step), "submit")
    ok = action_type == expected
    msg = f"action_type '{action_type}' {'matches' if ok else f'expected {expected}'} for {task_id} step {step_number}"
    return ok, msg


def _check_reasoning(reasoning: str) -> tuple:
    if not reasoning or len(reasoning) < 50:
        return False, f"reasoning too short ({len(reasoning) if reasoning else 0} chars, need >50)"
    hit = next((kw for kw in DOMAIN_KEYWORDS if kw in reasoning.lower()), None)
    if not hit:
        return False, "reasoning lacks domain-specific terminology"
    return True, f"reasoning is specific (len={len(reasoning)}, keyword='{hit}')"


def _check_report_ids(target_report_ids: list, valid_ids: set) -> tuple:
    if not target_report_ids:
        return False, "target_report_ids is empty"
    invalid = [rid for rid in target_report_ids if rid not in valid_ids]
    if invalid:
        return False, f"unknown report IDs: {invalid}"
    return True, f"{len(target_report_ids)} valid report ID(s) targeted"


def _check_intermediate_output(action) -> tuple:
    action_type = action.action_type if hasattr(action, "action_type") else action.get("action_type", "")

    if action_type == "submit":
        return True, "submit — output check skipped"

    if action_type == "classify":
        clf = action.classification if hasattr(action, "classification") else action.get("classification")
        if not clf:
            return False, "classify missing 'classification' field"
        if clf.lower() not in {"mild", "moderate", "serious", "life-threatening"}:
            return False, f"classification '{clf}' not a recognised severity"
        return True, f"classification '{clf}' is valid"

    if action_type in ("compute_ror", "flag_signal", "identify_interaction"):
        sf = action.signal_flag if hasattr(action, "signal_flag") else action.get("signal_flag")
        if not sf:
            return False, f"{action_type} missing 'signal_flag'"
        if not isinstance(sf, dict):
            return False, "signal_flag must be a dict"
        missing = {"drug", "event"} - sf.keys()
        if missing:
            return False, f"signal_flag missing keys: {missing}"
        return True, "signal_flag has required 'drug' and 'event' keys"

    return True, f"no check defined for '{action_type}'"


def shape_reward(task_id: str, step_number: int, action, valid_report_ids: set) -> tuple:
    action_type = action.action_type if hasattr(action, "action_type") else action.get("action_type", "")
    reasoning   = action.reasoning   if hasattr(action, "reasoning")   else action.get("reasoning", "")
    target_ids  = action.target_report_ids if hasattr(action, "target_report_ids") else action.get("target_report_ids", [])

    ok_type,   msg_type   = _check_action_type(task_id, step_number, action_type)
    ok_reason, msg_reason = _check_reasoning(reasoning)
    ok_ids,    msg_ids    = _check_report_ids(target_ids, valid_report_ids)
    ok_output, msg_output = _check_intermediate_output(action)

    credits = {
        "correct_action_type":    0.05 if ok_type   else 0.0,
        "reasoning_quality":      0.05 if ok_reason else 0.0,
        "valid_report_ids":       0.05 if ok_ids    else 0.0,
        "intermediate_structure": 0.05 if ok_output else 0.0,
    }
    step_reward = sum(credits.values())
    feedback = " | ".join([
        f"[action] {'✓' if ok_type   else '✗'} {msg_type}",
        f"[reason] {'✓' if ok_reason else '✗'} {msg_reason}",
        f"[ids] {'✓' if ok_ids    else '✗'} {msg_ids}",
        f"[output] {'✓' if ok_output else '✗'} {msg_output}",
        f"step={step_reward:.2f}",
    ])
    return round(step_reward, 4), credits, feedback


class RewardShaper:
    """No-arg constructor. Called by environment.py as RewardShaper()"""

    def __init__(self):
        self._cumulative = 0.0

    def reset(self):
        self._cumulative = 0.0

    def compute(self, action, episode) -> tuple:
        """
        Called by environment.py as:
            self._shaper.compute(action=action, episode=episode)
        Returns (step_reward, partial_credits, feedback) with cumulative capped at MAX_CUMULATIVE_REWARD.
        """
        valid_ids = {r["report_id"] if isinstance(r, dict) else r.report_id
                     for r in episode.reports}
        step_reward, credits, feedback = shape_reward(
            episode.task_id, episode.step_number, action, valid_ids
        )

        # Cap so cumulative never exceeds MAX_CUMULATIVE_REWARD
        self._cumulative += step_reward
        if self._cumulative > MAX_CUMULATIVE_REWARD:
            step_reward = max(0.0, step_reward - (self._cumulative - MAX_CUMULATIVE_REWARD))
            self._cumulative = MAX_CUMULATIVE_REWARD

        return round(step_reward, 4), credits, feedback
