from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.models import (
    Action,
    Observation,
    Reward,
    ResetRequest,
    StepResponse,
)
from app.state_manager import StateManager, TASK_CONFIGS
from app.environment import PharmaVigilEnv

# ---------------------------------------------------------------------------
# App + shared instances
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PharmaVigil",
    description="OpenEnv pharmacovigilance signal detection environment.",
    version="1.0.0",
)

# One state manager and one env shared across all requests (in-process state)
_state_manager = StateManager()
_env = PharmaVigilEnv(state_manager=_state_manager)


# ---------------------------------------------------------------------------
# Health — required by HF Space ping and pre-validation script
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()) -> Observation:
    """
    Start a new episode.
    Accepts an optional task_id — defaults to 'task1' if omitted.
    The pre-validation script sends an empty body {}, which is handled
    safely by the ResetRequest default.
    """
    try:
        obs = _env.reset(task_id=body.task_id)
        return obs
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Advance the episode by one step."""
    try:
        obs, reward = _env.step(action)
        return StepResponse(observation=obs, reward=reward)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=Observation)
def state() -> Observation:
    """Return current episode state without advancing it."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Hackathon-required endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def tasks() -> JSONResponse:
    """
    List all 3 tasks with their descriptions, difficulty, max_steps,
    and the full action schema.
    """
    task_list = []
    for task_id, cfg in TASK_CONFIGS.items():
        task_list.append({
            "id": task_id,
            "name": {
                "task1": "Adverse Event Severity Classification",
                "task2": "Disproportionality Signal Detection (ROR)",
                "task3": "Masking Effect Detection",
            }[task_id],
            "difficulty": {"task1": "easy", "task2": "medium", "task3": "hard"}[task_id],
            "max_steps": cfg["max_steps"],
            "description": cfg["description"],
        })

    action_schema = {
        "action_type": "enum[classify, flag_signal, compute_ror, identify_interaction, submit]",
        "target_report_ids": "array[string] — report IDs this action applies to",
        "classification": "string | null — severity classification (for classify action)",
        "signal_flag": "object | null — structured signal finding (for flag_signal action)",
        "reasoning": "string — required on every action",
    }

    return JSONResponse({"tasks": task_list, "action_schema": action_schema})


@app.post("/grader")
def grader() -> JSONResponse:
    """
    Return the final grader score for the completed episode.
    Reads the cumulative reward from the current episode state.
    """
    if not _state_manager.has_episode():
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    episode = _state_manager.get_episode()
    return JSONResponse({
        "task_id": episode.task_id,
        "score": round(episode.cumulative_reward, 4),
        "done": episode.done,
        "step_number": episode.step_number,
        "max_steps": episode.max_steps,
    })


@app.get("/baseline")
def baseline() -> JSONResponse:
    """
    Trigger the baseline agent and return scores for all 3 tasks.
    Imports and runs baseline_agent inline.
    Note: this is a synchronous call and will take time.
    """
    try:
        from baseline.baseline_agent import run_episode
        scores = {}
        for task_id in ["task1", "task2", "task3"]:
            scores[task_id] = round(run_episode(task_id), 4)
        return JSONResponse({"baseline_scores": scores})
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline agent failed: {exc}. Make sure OPENAI_API_KEY is set.",
        )
