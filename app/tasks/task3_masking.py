import json
from pathlib import Path
from app.models import Observation, AdverseEventReport

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus_task3.json"

TASK_DESCRIPTION = """You are a pharmacovigilance analyst investigating a potential masking effect.

A masking effect occurs when Drug A's strong signal for an adverse event statistically suppresses Drug B's signal for the same event in co-administered patients.

Your corpus contains 47 Individual Case Safety Reports (ICSRs) involving Azithromycin, Amiodarone, and other drugs.

You must:
1. Compute the standard ROR for Azithromycin-QT prolongation across the full corpus
2. Segment the corpus by Amiodarone co-administration status (present / absent)
3. Recompute Azithromycin-QT ROR in the Amiodarone-absent subgroup
4. Identify whether a masking effect exists and the magnitude of suppression
5. Submit your finding with a regulatory recommendation

Action sequence: compute_ror → compute_ror (stratified) → flag_signal → submit

Signal threshold: ROR > 2.0 AND CI_lower > 1.0
Masking confirmed if: standard ROR < threshold AND stratified ROR > threshold
"""

def load_task3() -> Observation:
    with open(CORPUS_PATH) as f:
        raw = json.load(f)

    reports = [AdverseEventReport(**r) for r in raw]

    return Observation(
        task_id="task3",
        task_description=TASK_DESCRIPTION,
        reports=reports,
        workspace={
            "standard_ror": None,
            "stratified_ror": None,
            "masking_confirmed": None,
            "suppression_magnitude": None,
            "regulatory_recommendation": None,
        },
        step_number=0,
        max_steps=15,
        hints=[
            "Check concomitant_drugs field to identify Amiodarone co-administration",
            "A non-signal in the full corpus does not mean the drug is safe",
            "Stratify by removing reports where Amiodarone is present as drug_name or concomitant_drug",
        ],
    )
