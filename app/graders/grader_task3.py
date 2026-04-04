from app.models import Action

GROUND_TRUTH = {
    "masking_drug": "Amiodarone",
    "masked_drug": "Azithromycin",
    "masked_event": "QT prolongation",
    "standard_ROR": 1.4,
    "stratified_ROR": 3.8,
    "suppression_magnitude": 2.4,
    "masking_confirmed": True,
}

REGULATORY_KEYWORDS = [
    "stratified", "stratify", "segment", "subgroup",
    "regulatory", "recommendation", "label", "warning",
    "investigation", "signal", "masked", "masking",
]


def grade_task3(action: dict) -> dict:
    signal_flag = action.get("signal_flag") or {}
    reasoning = (action.get("reasoning") or "").lower()

    credits = {}
    score = 0.0

    # +0.15 correct standard ROR (within ±0.05 of 1.4)
    submitted_standard = signal_flag.get("standard_ROR") or signal_flag.get("standard_ror")
    if submitted_standard is not None:
        try:
            if abs(float(submitted_standard) - GROUND_TRUTH["standard_ROR"]) <= 0.05:
                credits["standard_ror"] = 0.15
                score += 0.15
            else:
                credits["standard_ror"] = 0.0
        except (TypeError, ValueError):
            credits["standard_ror"] = 0.0
    else:
        credits["standard_ror"] = 0.0

    # +0.20 correct corpus segmentation (flag present in signal_flag)
    segmented = signal_flag.get("segmented") or signal_flag.get("corpus_segmented")
    if segmented is True:
        credits["corpus_segmentation"] = 0.20
        score += 0.20
    else:
        credits["corpus_segmentation"] = 0.0

    # +0.20 correct stratified ROR (within ±0.05 of 3.8)
    submitted_stratified = signal_flag.get("stratified_ROR") or signal_flag.get("stratified_ror")
    if submitted_stratified is not None:
        try:
            if abs(float(submitted_stratified) - GROUND_TRUTH["stratified_ROR"]) <= 0.05:
                credits["stratified_ror"] = 0.20
                score += 0.20
            else:
                credits["stratified_ror"] = 0.0
        except (TypeError, ValueError):
            credits["stratified_ror"] = 0.0
    else:
        credits["stratified_ror"] = 0.0

    # +0.20 correct masking identification (masking_confirmed=True, correct drug pair)
    masking_confirmed = signal_flag.get("masking_confirmed")
    masking_drug = (signal_flag.get("masking_drug") or "").strip()
    masked_drug = (signal_flag.get("masked_drug") or "").strip()

    if (
        masking_confirmed is True
        and masking_drug.lower() == GROUND_TRUTH["masking_drug"].lower()
        and masked_drug.lower() == GROUND_TRUTH["masked_drug"].lower()
    ):
        credits["masking_identification"] = 0.20
        score += 0.20
    else:
        credits["masking_identification"] = 0.0

    # +0.15 correct suppression magnitude (within ±0.1 of 2.4)
    submitted_magnitude = signal_flag.get("suppression_magnitude")
    if submitted_magnitude is not None:
        try:
            if abs(float(submitted_magnitude) - GROUND_TRUTH["suppression_magnitude"]) <= 0.1:
                credits["suppression_magnitude"] = 0.15
                score += 0.15
            else:
                credits["suppression_magnitude"] = 0.0
        except (TypeError, ValueError):
            credits["suppression_magnitude"] = 0.0
    else:
        credits["suppression_magnitude"] = 0.0

    # +0.10 regulatory recommendation in reasoning
    if any(kw in reasoning for kw in REGULATORY_KEYWORDS):
        credits["regulatory_recommendation"] = 0.10
        score += 0.10
    else:
        credits["regulatory_recommendation"] = 0.0

    score = round(min(score, 1.0), 4)

    feedback_parts = []
    for key, val in credits.items():
        status = "✅" if val > 0 else "❌"
        feedback_parts.append(f"{status} {key}: {val}")
    feedback = f"Task 3 score: {score}\n" + "\n".join(feedback_parts)

    return {"score": score, "partial_credits": credits, "feedback": feedback}


def grade(episode, action: Action) -> tuple[float, dict, str]:
    result = grade_task3(action.model_dump())
    return result["score"], result["partial_credits"], result["feedback"]
