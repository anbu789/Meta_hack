import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_task1() -> dict:
    """
    Load Task 1 corpus and return the initial observation payload.
    Called by environment.py on reset(task_id='task1').
    """
    corpus_path = DATA_DIR / "corpus_task1.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path) as f:
        reports = json.load(f)

    return {
        "task_id": "task1",
        "task_description": (
            "You are a pharmacovigilance analyst. You have received one Individual Case "
            "Safety Report (ICSR). You must:\n"
            "  1. Classify the severity of the adverse event (Mild / Moderate / Serious / Life-threatening)\n"
            "  2. Determine whether the event is unexpected (not listed in the drug label)\n"
            "  3. Flag whether this report requires expedited 15-day regulatory reporting\n"
            "  4. Identify the correct MedDRA System Organ Class (SOC) for the adverse event\n\n"
            "Submit your findings using action_type='classify'. "
            "Your 'classification' field must be a JSON string with keys: "
            "'severity', 'unexpected', 'expedited_report', 'meddra_soc'. "
            "Always include detailed reasoning."
        ),
        "reports": reports,
        "workspace": {},
        "step_number": 0,
        "max_steps": 5,
        "hints": [
            "Apply WHO seriousness criteria: death, life-threatening, hospitalisation, disability, congenital anomaly, or medically significant event.",
            "Check the drug label — if the adverse event is not listed, it is unexpected.",
            "Expedited 15-day reporting is required for serious AND unexpected events.",
            "Use MedDRA System Organ Class (SOC) level — the highest level of the MedDRA hierarchy."
        ]
    }


def get_task1_schema() -> dict:
    """Returns the action schema description shown at /tasks endpoint."""
    return {
        "task_id": "task1",
        "name": "Adverse Event Severity Classification",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "Classify a single ADE report by severity and determine regulatory "
            "reporting requirements using WHO/FDA criteria."
        ),
        "action_schema": {
            "action_type": "classify",
            "target_report_ids": ["list of report IDs this action applies to"],
            "classification": {
                "severity": "Mild | Moderate | Serious | Life-threatening",
                "unexpected": "true | false",
                "expedited_report": "true | false",
                "meddra_soc": "MedDRA System Organ Class string"
            },
            "reasoning": "required — explain your clinical reasoning"
        }
    }
