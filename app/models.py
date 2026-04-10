from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AdverseEventReport(BaseModel):
    report_id: str = Field(..., description="Unique report identifier, e.g. ADE-2024-00412")
    drug_name: str = Field(..., description="Primary suspect drug name")
    drug_dose_mg: float = Field(..., description="Dose in milligrams")
    patient_age: int = Field(..., description="Patient age in years")
    patient_sex: str = Field(..., description="M / F / Unknown")
    adverse_event: str = Field(..., description="MedDRA preferred term for the adverse event")
    onset_days: int = Field(..., description="Days from drug start to event onset")
    severity: str = Field(..., description="Mild / Moderate / Serious / Life-threatening")
    outcome: str = Field(..., description="Recovered / Ongoing / Fatal / Unknown")
    concomitant_drugs: list[str] = Field(default_factory=list)
    narrative: str = Field(..., description="Free-text nurse or doctor note")
    country: str = Field(..., description="Country where report was filed")
    report_date: str = Field(..., description="ISO date string, e.g. 2024-03-15")


class Observation(BaseModel):
    task_id: str
    task_description: str
    reports: list[AdverseEventReport]
    workspace: dict = Field(default_factory=dict)
    step_number: int
    max_steps: int
    hints: list[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: str = Field(..., description="classify | flag_signal | compute_ror | identify_interaction | submit")
    target_report_ids: list[str] = Field(default_factory=list)
    classification: Optional[str] = Field(None)
    signal_flag: Optional[dict] = Field(None)
    # Made optional with default — prevents 422 when 8B model omits this field
    reasoning: str = Field(default="No reasoning provided.", description="Agent's explanation. Always required.")


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    done: bool
    partial_credits: dict = Field(default_factory=dict)
    feedback: str


class ResetRequest(BaseModel):
    task_id: str = Field(default="task1", description="task1 / task2 / task3")


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
