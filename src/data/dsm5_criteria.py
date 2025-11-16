"""
DSM-5 Depression Criteria Descriptions

Text descriptions of DSM-5 Major Depressive Disorder criteria for NLI-style matching.
Based on DSM-5 diagnostic criteria for Major Depressive Episode.
"""

# Map symptom labels to their DSM-5 criterion descriptions
DSM5_CRITERIA = {
    'DEPRESSED_MOOD': (
        "The patient exhibits a depressed mood most of the day, nearly every day, "
        "as indicated by subjective report or observation by others. This includes "
        "feeling sad, empty, or hopeless."
    ),
    'ANHEDONIA': (
        "The patient shows markedly diminished interest or pleasure in all, or almost all, "
        "activities most of the day, nearly every day. This is a loss of enjoyment in "
        "previously pleasurable activities."
    ),
    'APPETITE_CHANGE': (
        "The patient demonstrates significant weight loss when not dieting or weight gain, "
        "or decrease or increase in appetite nearly every day. This includes changes in "
        "eating patterns or significant weight fluctuations."
    ),
    'SLEEP_ISSUES': (
        "The patient experiences insomnia or hypersomnia nearly every day. This includes "
        "difficulty falling asleep, staying asleep, early morning awakening, or excessive sleeping."
    ),
    'PSYCHOMOTOR': (
        "The patient exhibits psychomotor agitation or retardation nearly every day, "
        "observable by others. This includes restlessness, inability to sit still, or "
        "slowed movements and speech."
    ),
    'FATIGUE': (
        "The patient reports fatigue or loss of energy nearly every day. This includes "
        "feeling tired, exhausted, or lacking energy to perform daily activities."
    ),
    'WORTHLESSNESS': (
        "The patient expresses feelings of worthlessness or excessive or inappropriate guilt "
        "nearly every day. This may include self-blame, negative self-evaluation, or "
        "feeling like a burden to others."
    ),
    'COGNITIVE_ISSUES': (
        "The patient demonstrates diminished ability to think or concentrate, or indecisiveness, "
        "nearly every day. This includes difficulty focusing, making decisions, or experiencing "
        "mental fog or confusion."
    ),
    'SUICIDAL_THOUGHTS': (
        "The patient has recurrent thoughts of death, recurrent suicidal ideation without a "
        "specific plan, or a suicide attempt or a specific plan for committing suicide."
    ),
    'SPECIAL_CASE': (
        "The patient's presentation requires expert clinical judgment to determine if it "
        "represents a depressive symptom. This includes ambiguous or complex cases."
    ),
}


# Shorter criterion versions for memory efficiency (optional)
DSM5_CRITERIA_SHORT = {
    'DEPRESSED_MOOD': "Depressed mood most of the day, nearly every day.",
    'ANHEDONIA': "Diminished interest or pleasure in activities.",
    'APPETITE_CHANGE': "Significant weight or appetite change.",
    'SLEEP_ISSUES': "Insomnia or hypersomnia nearly every day.",
    'PSYCHOMOTOR': "Psychomotor agitation or retardation.",
    'FATIGUE': "Fatigue or loss of energy nearly every day.",
    'WORTHLESSNESS': "Feelings of worthlessness or excessive guilt.",
    'COGNITIVE_ISSUES': "Diminished ability to think or concentrate.",
    'SUICIDAL_THOUGHTS': "Recurrent thoughts of death or suicidal ideation.",
    'SPECIAL_CASE': "Complex case requiring expert clinical judgment.",
}


def get_criterion_text(symptom_label: str, use_short: bool = False) -> str:
    """
    Get criterion description for a symptom label.

    Args:
        symptom_label: One of the SYMPTOM_LABELS (e.g., 'DEPRESSED_MOOD')
        use_short: Whether to use short version (default: False)

    Returns:
        Criterion description text
    """
    criteria_dict = DSM5_CRITERIA_SHORT if use_short else DSM5_CRITERIA

    if symptom_label not in criteria_dict:
        raise ValueError(f"Unknown symptom label: {symptom_label}")

    return criteria_dict[symptom_label]


def get_all_criterion_texts(use_short: bool = False) -> dict:
    """
    Get all criterion descriptions.

    Args:
        use_short: Whether to use short versions

    Returns:
        Dictionary mapping symptom labels to criterion texts
    """
    return DSM5_CRITERIA_SHORT.copy() if use_short else DSM5_CRITERIA.copy()
