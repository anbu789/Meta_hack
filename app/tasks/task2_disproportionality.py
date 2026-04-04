import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "corpus_task2.json"

TASK_DESCRIPTION = (
    "You are given a corpus of 32 adverse drug event reports spanning 3 drugs "
    "(DrugX, DrugY, DrugZ) and 5 adverse events. "
    "Your objective: (1) Build a 2x2 contingency table for each drug-event pair. "
    "(2) Compute the Reporting Odds Ratio (ROR) and 95% confidence interval for each pair. "
    "(3) Identify pairs that meet the signal threshold: ROR > 2.0 AND CI_lower > 1.0. "
    "(4) Rank flagged signals by ROR strength. "
    "(5) Submit the top signal with full justification."
)

HINTS = [
    "ROR = (a * d) / (b * c) where a=drug+event, b=drug-event, c=no_drug+event, d=no_drug-event",
    "95% CI = exp(ln(ROR) ± 1.96 * sqrt(1/a + 1/b + 1/c + 1/d))",
    "Signal threshold: ROR > 2.0 AND CI_lower > 1.0",
    "Apply Haldane correction (add 0.5 to all cells) if any cell is zero",
    "Use action_type='compute_ror' for intermediate steps, 'submit' to finalize",
]


def load_task2(max_steps: int = 10) -> dict:
    with open(DATA_PATH, "r") as f:
        reports = json.load(f)

    return {
        "task_id": "task2",
        "task_description": TASK_DESCRIPTION,
        "reports": reports,
        "workspace": {
            "contingency_tables": {},
            "ror_results": {},
            "flagged_signals": [],
            "ranked_signals": [],
        },
        "step_number": 0,
        "max_steps": max_steps,
        "hints": HINTS,
    }
