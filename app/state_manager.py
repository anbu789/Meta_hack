from __future__ import annotations

from app.models import AdverseEventReport, Observation


# ---------------------------------------------------------------------------
# Task metadata — single source of truth for all task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict] = {
    "task1": {
        "description": (
            "You are a pharmacovigilance analyst. Classify the severity of the provided "
            "adverse event report using WHO/FDA criteria (Mild / Moderate / Serious / "
            "Life-threatening). Determine if the event is unexpected (not listed in the "
            "drug label). Flag if the case requires expedited 15-day regulatory reporting. "
            "Identify the correct MedDRA System Organ Class (SOC). Submit your findings."
        ),
        "max_steps": 5,
        "corpus_file": "corpus_task1.json",
        "hints": [
            "An event is Serious if it causes death, hospitalisation, disability, or is medically significant.",
            "Expedited reporting is required for Serious + Unexpected events within 15 days.",
            "QT prolongation maps to MedDRA SOC: Cardiac disorders.",
        ],
    },
    "task2": {
        "description": (
            "You are a pharmacovigilance analyst. A corpus of 20-30 ADE reports spanning "
            "3 drugs and 5 adverse events has been provided. Build a 2x2 contingency table "
            "for each drug-event pair. Compute the Reporting Odds Ratio (ROR) and 95% "
            "confidence interval. Flag pairs where ROR > 2.0 AND lower CI > 1.0. Rank "
            "flagged signals by ROR strength and submit the top signal with justification. "
            "Apply Haldane correction (add 0.5 to all cells) if any cell is zero."
        ),
        "max_steps": 10,
        "corpus_file": "corpus_task2.json",
        "hints": [
            "ROR = (a * d) / (b * c) where a=drug+event, b=drug-event, c=nodrug+event, d=nodrug-event.",
            "95% CI = exp(ln(ROR) ± 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)).",
            "Signal threshold: ROR > 2.0 AND lower bound of 95% CI > 1.0.",
            "If any cell in the contingency table is 0, add 0.5 to all four cells (Haldane correction).",
        ],
    },
    "task3": {
        "description": (
            "You are a pharmacovigilance analyst. A masking effect may be present in this "
            "corpus. First run standard ROR analysis across all reports. Then segment the "
            "corpus by co-administration status of the suspected masking drug. Recompute "
            "ROR for the suspected masked drug in the subgroup without the masking drug. "
            "If ROR exceeds the signal threshold only in the stratified analysis, confirm "
            "the masking effect and recommend a stratified analysis in the regulatory "
            "submission."
        ),
        "max_steps": 15,
        "corpus_file": "corpus_task3.json",
        "hints": [
            "Check if Azithromycin's standard ROR is below threshold but rises above 2.0 when Amiodarone-co-administered patients are excluded.",
            "Masking effect = masked_drug ROR below threshold in full corpus, above threshold in stratified subgroup.",
            "Magnitude of suppression = stratified_ROR - standard_ROR.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Episode state — one instance per active episode
# ---------------------------------------------------------------------------

class EpisodeState:
    """
    Holds all mutable state for a single episode.
    StateManager creates and stores one of these per reset().
    """

    def __init__(self, task_id: str, reports: list[AdverseEventReport]) -> None:
        cfg = TASK_CONFIGS[task_id]
        self.task_id: str = task_id
        self.task_description: str = cfg["description"]
        self.max_steps: int = cfg["max_steps"]
        self.hints: list[str] = cfg["hints"]
        self.reports: list[AdverseEventReport] = reports

        # Mutable across steps
        self.step_number: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.workspace: dict = {}
        self.action_history: list[dict] = []  # for circular reasoning detection

    def to_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            task_description=self.task_description,
            reports=self.reports,
            workspace=self.workspace,
            step_number=self.step_number,
            max_steps=self.max_steps,
            hints=self.hints,
        )

    def increment_step(self) -> None:
        self.step_number += 1

    def is_over_limit(self) -> bool:
        return self.step_number >= self.max_steps

    def record_action(self, action_type: str, reasoning: str) -> None:
        self.action_history.append({"action_type": action_type, "reasoning": reasoning})

    def is_circular(self, action_type: str, reasoning: str) -> bool:
        """
        Detects if the agent is repeating the exact same action type
        and reasoning back-to-back (circular reasoning penalty).
        """
        if len(self.action_history) < 1:
            return False
        last = self.action_history[-1]
        return (
            last["action_type"] == action_type
            and last["reasoning"].strip() == reasoning.strip()
        )


# ---------------------------------------------------------------------------
# State manager — singleton that holds the active episode
# ---------------------------------------------------------------------------

class StateManager:
    """
    Manages the lifecycle of episodes.
    One StateManager instance is created at app startup and shared across requests.
    """

    def __init__(self) -> None:
        self._episode: EpisodeState | None = None

    def new_episode(self, task_id: str, reports: list[AdverseEventReport]) -> EpisodeState:
        """Create a fresh episode, discarding any previous one."""
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Must be one of: {list(TASK_CONFIGS)}")
        self._episode = EpisodeState(task_id=task_id, reports=reports)
        return self._episode

    def get_episode(self) -> EpisodeState:
        """Return the current active episode. Raises if none exists."""
        if self._episode is None:
            raise RuntimeError("No active episode. Call POST /reset first.")
        return self._episode

    def has_episode(self) -> bool:
        return self._episode is not None
