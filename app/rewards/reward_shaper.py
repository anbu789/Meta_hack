from __future__ import annotations

from app.models import Action


# ---------------------------------------------------------------------------
# Step-level reward shaper
# Full implementation comes on Day 5.
# This stub gives +0.05 per correct signal so other files can import safely.
# ---------------------------------------------------------------------------

class RewardShaper:
    """
    Computes step-level partial rewards for every non-submit action.
    Each of the 4 signals below is worth +0.05 (max +0.20 per step).

    Full reward breakdown:
        +0.05  correct action_type for current task stage
        +0.05  reasoning field is non-empty and substantive
        +0.05  target_report_ids are non-empty
        +0.05  intermediate output (classification or signal_flag) is present
    """

    def compute(
        self,
        action: Action,
        episode,  # EpisodeState — not type-hinted to avoid circular import
    ) -> tuple[float, dict, str]:
        """
        Returns (step_reward, partial_credits_dict, feedback_str).
        """
        credits: dict[str, float] = {}
        feedback_parts: list[str] = []

        # Signal 1: action_type appropriate (non-submit actions all get credit here)
        credits["correct_action_type"] = 0.05
        feedback_parts.append("Action type recorded (+0.05).")

        # Signal 2: reasoning is substantive (more than 10 chars)
        if action.reasoning and len(action.reasoning.strip()) > 10:
            credits["substantive_reasoning"] = 0.05
            feedback_parts.append("Reasoning is substantive (+0.05).")
        else:
            credits["substantive_reasoning"] = 0.0
            feedback_parts.append("Reasoning too short or missing (0.00).")

        # Signal 3: target_report_ids non-empty
        if action.target_report_ids:
            credits["target_ids_provided"] = 0.05
            feedback_parts.append("Target report IDs provided (+0.05).")
        else:
            credits["target_ids_provided"] = 0.0
            feedback_parts.append("No target report IDs provided (0.00).")

        # Signal 4: intermediate output present
        if action.classification or action.signal_flag:
            credits["intermediate_output"] = 0.05
            feedback_parts.append("Intermediate output present (+0.05).")
        else:
            credits["intermediate_output"] = 0.0
            feedback_parts.append("No intermediate output provided (0.00).")

        total = sum(credits.values())
        return total, credits, " ".join(feedback_parts)
