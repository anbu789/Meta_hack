from __future__ import annotations

import json
from pathlib import Path

from app.models import Action, Observation, Reward, AdverseEventReport
from app.state_manager import StateManager, TASK_CONFIGS
from app.rewards.reward_shaper import RewardShaper

DATA_DIR = Path(__file__).parent / "data"

VALID_ACTION_TYPES = {
    "classify",
    "flag_signal",
    "compute_ror",
    "identify_interaction",
    "submit",
}


def _load_corpus(task_id: str) -> list[AdverseEventReport]:
    filename = TASK_CONFIGS[task_id]["corpus_file"]
    corpus_path = DATA_DIR / filename

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. "
            f"Make sure app/data/{filename} exists before running."
        )

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "reports" in raw:
        records = raw["reports"]
    else:
        raise ValueError(f"Unexpected corpus format in {filename}.")

    return [AdverseEventReport(**r) for r in records]


def _validate_report_ids(
    target_ids: list[str],
    corpus: list[AdverseEventReport],
) -> tuple[bool, str]:
    valid_ids = {r.report_id for r in corpus}
    bad_ids = [rid for rid in target_ids if rid not in valid_ids]
    if bad_ids:
        return False, f"Referenced non-existent report IDs: {bad_ids}. Penalty applied."
    return True, ""


def _sanitize_action(action: Action) -> Action:
    """
    Fill in missing optional fields so Pydantic validation never 422s
    on actions that the LLM forgot to include optional keys for.
    """
    if action.target_report_ids is None:
        action.target_report_ids = []
    if action.classification is None:
        action.classification = None
    if action.signal_flag is None:
        action.signal_flag = None
    if not action.reasoning:
        action.reasoning = "No reasoning provided."
    return action


class PharmaVigilEnv:
    def __init__(self, state_manager: StateManager) -> None:
        self._sm = state_manager
        self._shaper = RewardShaper()

    def reset(self, task_id: str = "task1") -> Observation:
        task_id = task_id.strip().lower()
        if task_id not in TASK_CONFIGS:
            task_id = "task1"

        reports = _load_corpus(task_id)
        episode = self._sm.new_episode(task_id=task_id, reports=reports)

        # Reset reward shaper for each new episode
        self._shaper.reset()

        return episode.to_observation()

    def state(self) -> Observation:
        episode = self._sm.get_episode()
        return episode.to_observation()

    def step(self, action: Action) -> tuple[Observation, Reward]:
        episode = self._sm.get_episode()

        # Sanitize action to prevent 422 from missing optional fields
        action = _sanitize_action(action)

        if episode.done:
            return episode.to_observation(), Reward(
                step_reward=0.0,
                cumulative_reward=episode.cumulative_reward,
                done=True,
                partial_credits={},
                feedback="Episode is already complete. No further actions accepted.",
            )

        episode.increment_step()

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

        circular_penalty = 0.0
        circular_feedback = ""
        if episode.is_circular(action.action_type, action.reasoning):
            circular_penalty = -0.15
            circular_feedback = " Circular reasoning detected (-0.15)."

        id_valid, id_feedback = _validate_report_ids(
            action.target_report_ids, episode.reports
        )
        id_penalty = 0.0 if id_valid else -0.10

        episode.record_action(action.action_type, action.reasoning)

        if action.action_type == "submit":
            return self._handle_submit(episode, action, circular_penalty, id_penalty, circular_feedback, id_feedback)

        step_reward, partial_credits, shaper_feedback = self._shaper.compute(
            action=action,
            episode=episode,
        )

        total_step = step_reward + circular_penalty + id_penalty
        episode.cumulative_reward = max(0.0, episode.cumulative_reward + total_step)

        if action.signal_flag:
            episode.workspace[f"step_{episode.step_number}_signal"] = action.signal_flag
        if action.classification:
            episode.workspace[f"step_{episode.step_number}_classification"] = action.classification

        if episode.is_over_limit():
            limit_penalty = -0.25
            episode.cumulative_reward = max(0.0, episode.cumulative_reward + limit_penalty)
            episode.done = True
            return episode.to_observation(), Reward(
                step_reward=total_step + limit_penalty,
                cumulative_reward=episode.cumulative_reward,
                done=True,
                partial_credits={**partial_credits, "exceeded_max_steps": limit_penalty},
                feedback=f"Max steps ({episode.max_steps}) exceeded without submitting. {shaper_feedback}{circular_feedback}{id_feedback}",
            )

        feedback = shaper_feedback + circular_feedback + id_feedback
        return episode.to_observation(), Reward(
            step_reward=total_step,
            cumulative_reward=episode.cumulative_reward,
            done=episode.done,
            partial_credits={**partial_credits, "circular_penalty": circular_penalty, "id_penalty": id_penalty},
            feedback=feedback.strip() or "Step recorded.",
        )

    def _handle_submit(
        self,
        episode,
        action: Action,
        circular_penalty: float,
        id_penalty: float,
        circular_feedback: str,
        id_feedback: str,
    ) -> tuple[Observation, Reward]:
        early_submit_penalty = 0.0
        early_feedback = ""
        if episode.step_number <= 1:
            early_submit_penalty = -0.20
            early_feedback = " Submitted without completing required steps (-0.20)."

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
