from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core domain object
# ---------------------------------------------------------------------------

class AdverseEventReport(BaseModel):
    """
    Represents a single Individual Case Safety Report (ICSR).
    This is the atomic unit of data the agent analyses.
    """
    report_id: str = Field(..., description="Unique report identifier, e.g. ADE-2024-00412")
    drug_name: str = Field(..., description="Primary suspect drug name")
    drug_dose_mg: float = Field(..., description="Dose in milligrams")
    patient_age: int = Field(..., description="Patient age in years")
    patient_sex: str = Field(..., description="M / F / Unknown")
    adverse_event: str = Field(..., description="MedDRA preferred term for the adverse event")
    onset_days: int = Field(..., description="Days from drug start to event onset")
    severity: str = Field(..., description="Mild / Moderate / Serious / Life-threatening")
    outcome: str = Field(..., description="Recovered / Ongoing / Fatal / Unknown")
    concomitant_drugs: list[str] = Field(default_factory=list, description="Other drugs taken simultaneously")
    narrative: str = Field(..., description="Free-text nurse or doctor note")
    country: str = Field(..., description="Country where report was filed")
    report_date: str = Field(..., description="ISO date string, e.g. 2024-03-15")


# ---------------------------------------------------------------------------
# OpenEnv required models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Returned by reset() and as part of each step() response.
    This is what the agent sees at every step.
    """
    task_id: str = Field(..., description="Which task is active: task1 / task2 / task3")
    task_description: str = Field(..., description="Human-readable objective for the agent")
    reports: list[AdverseEventReport] = Field(..., description="Corpus of ICSR reports for this episode")
    workspace: dict = Field(default_factory=dict, description="Agent working memory — stores intermediate outputs")
    step_number: int = Field(..., description="Current step index, starts at 1")
    max_steps: int = Field(..., description="Maximum steps allowed before penalty")
    hints: list[str] = Field(default_factory=list, description="Optional scaffolding hints for partial credit")


class Action(BaseModel):
    """
    Sent by the agent on every step() call.
    The reasoning field is always required — graders check it.
    """
    action_type: str = Field(
        ...,
        description="One of: classify | flag_signal | compute_ror | identify_interaction | submit"
    )
    target_report_ids: list[str] = Field(
        default_factory=list,
        description="Report IDs this action applies to. Must be valid IDs from the corpus."
    )
    classification: Optional[str] = Field(
        None,
        description="Severity classification when action_type is 'classify'"
    )
    signal_flag: Optional[dict] = Field(
        None,
        description="Structured signal finding when action_type is 'flag_signal'"
    )
    reasoning: str = Field(
        ...,
        description="Agent's explanation of why it took this action. Always required."
    )


class Reward(BaseModel):
    """
    Returned by step() after every action.
    Provides dense feedback throughout the episode, not just at the end.
    """
    step_reward: float = Field(..., description="Reward earned for this specific step (0.0 to 1.0)")
    cumulative_reward: float = Field(..., description="Total reward accumulated this episode")
    done: bool = Field(..., description="True when the episode is complete")
    partial_credits: dict = Field(
        default_factory=dict,
        description="Breakdown of what sub-components were rewarded this step"
    )
    feedback: str = Field(..., description="Human-readable explanation of the score given")


# ---------------------------------------------------------------------------
# Request models for API endpoints
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """
    Body for POST /reset.
    task_id is optional — the validator sends an empty body {},
    so we default to task1 to avoid a 422 Pydantic error.
    """
    task_id: str = Field(default="task1", description="task1 / task2 / task3")


class StepResponse(BaseModel):
    """
    Full response returned by POST /step.
    """
    observation: Observation
    reward: Reward
