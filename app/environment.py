from __future__ import annotations

import json
from pathlib import Path

from app.models import Action, Observation, Reward, AdverseEventReport
from app.state_manager import StateManager, TASK_CONFIGS
from app.rewards.reward_shaper import RewardShaper

# ---------------------------------------------------------------------------
# Data directory — all corpus JSON files live here
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

VALID_ACTION_TYPES = {
    "classify",
    "flag_signal",
    "compute_ror",
    "identify_interaction",
    "submit",
}


def _load_corpus(task_id: str) -> list[AdverseEventReport]:
    """Load the seeded report corpus for a given task from disk."""
    filename = TASK_CONFIGS[task_id]["corpus_file"]
    corpus_path = DATA_DIR / filename

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. "
            f"Make sure app/data/{filename} exists before running."
        )

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))

    # Support both a bare list and a dict wrapper {"reports": [...]}
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "reports" in raw:
        records = raw["reports"]
    else:
        raise ValueError(f"Unexpected corpus format in {filename}. Expected a list or {{\"reports\": [...]}}.")

    return [AdverseEventReport(**r) for r in records]


def _validate_report_ids(
    target_ids: list[str],
    corpus: list[AdverseEventReport],
) -> tuple[bool, str]:
    """
    Check that every report ID the agent referenced actually exists in the corpus.
    Returns (all_valid: bool, feedback: str).
    """
    valid_ids = {r.report_id for r in corpus}
    bad_ids = [rid for rid in target_ids if rid not in valid_ids]
    if bad_ids:
        return False, f"Referenced non-existent report IDs: {bad_ids}. Penalty applied."
    return True, ""


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class PharmaVigilEnv:
    """
    Core environment logic.
    Implements reset(), state(), and step() as required by OpenEnv spec.
    Graders are imported lazily per task to keep startup fast.
    """

    def __init__(self, state_manager: StateManager) -> None:
        self._sm = state_manager
        self._shaper = RewardShaper()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """
        Start a fresh episode for the given task.
        Loads the corpus from disk, creates a new EpisodeState,
        returns the initial Observation.
        """
        task_id = task_id.strip().lower()
        if task_id not in TASK_CONFIGS:
            task_id = "task1"  # safe fallback (validator sends empty body)

        reports = _load_corpus(task_id)
        episode = self._sm.new_episode(task_id=task_id, reports=reports)
        return episode.to_observation()

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> Observation:
        """Return a snapshot of the current episode state without advancing it."""
        episode = self._sm.get_episode()
        return episode.to_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> tuple[Observation, Reward]:
        """
        Advance the episode by one step.
        Applies the reward shaper for step-level rewards and penalties,
        then delegates final grading to the task-specific grader on submit.
        """
        episode = self._sm.get_episode()

        if episode.done:
            # Episode already finished — return terminal state with zero reward
            return episode.to_observation(), Reward(
                step_reward=0.0,
                cumulative_reward=episode.cumulative_reward,
                done=True,
                partial_credits={},
                feedback="Episode is already complete. No further actions accepted.",
            )

        # --- Increment step counter ------------------------------------
        episode.increment_step()

        # --- Base penalty: invalid action_type -------------------------
        if action.action_type not in VALID_ACTION_TYPES:
            penalty = -0.10
            episode.cumulative_reward = max(0.0, episode.cumulative_reward + penalty)
            return episode.to_observation(), Reward(
                step_reward=penalty,
                cumulative_reward=episode.cumulative_reward,
                done=False,
                partial_credits={"invalid_action_type": penalty},
                feedback=f"Unknown action_type '{action.action_type}'. Valid types: {sorted(VALID_ACTION_TYPES)}.",
            )

        # --- Penalty: circular reasoning -------------------------------
        circular_penalty = 0.0
        circular_feedback = ""
        if episode.is_circular(action.action_type, action.reasoning):
            circular_penalty = -0.15
            circular_feedback = " Circular reasoning detected (-0.15)."

        # --- Penalty: referencing non-existent report IDs --------------
        id_valid, id_feedback = _validate_report_ids(
            action.target_report_ids, episode.reports
        )
        id_penalty = 0.0 if id_valid else -0.10

        # --- Record this action in history (after circular check) ------
        episode.record_action(action.action_type, action.reasoning)

        # --- Handle submit --------------------------------------------
        if action.action_type == "submit":
            return self._handle_submit(episode, action, circular_penalty, id_penalty, circular_feedback, id_feedback)

        # --- Step-level reward from shaper ----------------------------
        step_reward, partial_credits, shaper_feedback = self._shaper.compute(
            action=action,
            episode=episode,
        )

        # --- Apply penalties on top of step reward --------------------
        total_step = step_reward + circular_penalty + id_penalty
        episode.cumulative_reward = max(0.0, episode.cumulative_reward + total_step)

        # Update workspace with any intermediate output in action
        if action.signal_flag:
            episode.workspace[f"step_{episode.step_number}_signal"] = action.signal_flag
        if action.classification:
            episode.workspace[f"step_{episode.step_number}_classification"] = action.classification

        # --- Penalty: exceeded max steps without submitting ------------
        if episode.is_over_limit():
            limit_penalty = -0.25
            episode.cumulative_reward = max(0.0, episode.cumulative_reward + limit_penalty)
            episode.done = True
            return episode.to_observation(), Reward(
                step_reward=total_step + limit_penalty,
                cumulative_reward=episode.cumulative_reward,
                done=True,
                partial_credits={**partial_credits, "exceeded_max_steps": limit_penalty},
                feedback=f"Max steps ({episode.max_steps}) exceeded without submitting. Episode ended. {shaper_feedback}{circular_feedback}{id_feedback}",
            )

        feedback = shaper_feedback + circular_feedback + id_feedback
        return episode.to_observation(), Reward(
            step_reward=total_step,
            cumulative_reward=episode.cumulative_reward,
            done=episode.done,
            partial_credits={**partial_credits, "circular_penalty": circular_penalty, "id_penalty": id_penalty},
            feedback=feedback.strip() or "Step recorded.",
        )

    # ------------------------------------------------------------------
    # submit handler — calls the appropriate grader
    # ------------------------------------------------------------------

    def _handle_submit(
        self,
        episode,
        action: Action,
        circular_penalty: float,
        id_penalty: float,
        circular_feedback: str,
        id_feedback: str,
    ) -> tuple[Observation, Reward]:
        """
        Delegate to the task-specific grader and finalise the episode.
        Graders are imported here (lazy import) to keep startup fast.
        """
        # Penalty for submitting without doing required steps
        early_submit_penalty = 0.0
        early_feedback = ""
        if episode.step_number <= 1:
            early_submit_penalty = -0.20
            early_feedback = " Submitted without completing required steps (-0.20)."

        # Import the correct grader
        if episode.task_id == "task1":
            from app.graders.grader_task1 import grade as grade_fn
        elif episode.task_id == "task2":
            from app.graders.grader_task2 import grade as grade_fn
        elif episode.task_id == "task3":
            from app.graders.grader_task3 import grade as grade_fn
        else:
            grade_fn = None

        if grade_fn is None:
            grader_score = 0.0
            grader_credits = {}
            grader_feedback = "No grader found for this task."
        else:
            grader_score, grader_credits, grader_feedback = grade_fn(
                episode=episode, action=action
            )

        total_step = grader_score + circular_penalty + id_penalty + early_submit_penalty
        episode.cumulative_reward = max(0.0, min(1.0, episode.cumulative_reward + total_step))
        episode.done = True

        full_feedback = (
            f"Episode complete. Grader score: {grader_score:.2f}. "
            f"{grader_feedback}{circular_feedback}{id_feedback}{early_feedback}"
        ).strip()

        return episode.to_observation(), Reward(
            step_reward=total_step,
            cumulative_reward=episode.cumulative_reward,
            done=True,
            partial_credits={
                **grader_credits,
                "circular_penalty": circular_penalty,
                "id_penalty": id_penalty,
                "early_submit_penalty": early_submit_penalty,
            },
            feedback=full_feedback,
        )
