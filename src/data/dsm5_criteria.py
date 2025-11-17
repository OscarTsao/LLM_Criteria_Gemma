"""
DSM-5 Depression Criteria Descriptions for NLI Matching Task.

Each criterion corresponds to a symptom of Major Depressive Disorder from DSM-5.
"""

# DSM-5 Criterion descriptions for the 9 core symptoms
DSM5_CRITERIA = {
    'DEPRESSED_MOOD': (
        "The person experiences depressed mood most of the day, nearly every day, "
        "as indicated by subjective report or observation. This may include feelings "
        "of sadness, emptiness, hopelessness, or tearfulness."
    ),
    'ANHEDONIA': (
        "The person shows markedly diminished interest or pleasure in all, or almost all, "
        "activities most of the day, nearly every day. This includes loss of interest in "
        "hobbies, social activities, sex, or things previously enjoyed."
    ),
    'APPETITE_CHANGE': (
        "The person experiences significant weight loss when not dieting, weight gain, "
        "or decrease or increase in appetite nearly every day. This includes changes "
        "in eating patterns or food intake."
    ),
    'SLEEP_ISSUES': (
        "The person experiences insomnia or hypersomnia nearly every day. This includes "
        "difficulty falling asleep, staying asleep, sleeping too much, or irregular "
        "sleep patterns."
    ),
    'PSYCHOMOTOR': (
        "The person exhibits psychomotor agitation or retardation nearly every day that "
        "is observable by others. This includes restlessness, pacing, slowed movements, "
        "or slowed speech."
    ),
    'FATIGUE': (
        "The person experiences fatigue or loss of energy nearly every day. This includes "
        "feeling tired, depleted, or lacking motivation to perform daily activities."
    ),
    'WORTHLESSNESS': (
        "The person experiences feelings of worthlessness or excessive or inappropriate "
        "guilt nearly every day. This includes negative self-evaluation, self-blame, "
        "or rumination about past failures."
    ),
    'COGNITIVE_ISSUES': (
        "The person experiences diminished ability to think or concentrate, or indecisiveness, "
        "nearly every day. This includes difficulty focusing, making decisions, or experiencing "
        "impaired reasoning."
    ),
    'SUICIDAL_THOUGHTS': (
        "The person has recurrent thoughts of death, recurrent suicidal ideation without a "
        "specific plan, or a suicide attempt or specific plan for committing suicide."
    ),
}

# Ordered list of symptom labels (excluding SPECIAL_CASE for criteria matching)
SYMPTOM_LABELS = [
    'DEPRESSED_MOOD',
    'ANHEDONIA',
    'APPETITE_CHANGE',
    'SLEEP_ISSUES',
    'PSYCHOMOTOR',
    'FATIGUE',
    'WORTHLESSNESS',
    'COGNITIVE_ISSUES',
    'SUICIDAL_THOUGHTS',
]

# Number of criteria for matching (9 DSM-5 symptoms)
NUM_CRITERIA = len(SYMPTOM_LABELS)


def get_criterion_text(symptom_label: str) -> str:
    """
    Get the criterion description for a given symptom label.

    Args:
        symptom_label: One of the 9 DSM-5 symptom labels

    Returns:
        The criterion description text

    Raises:
        KeyError: If symptom_label is not found
    """
    return DSM5_CRITERIA[symptom_label]


def get_all_criteria() -> dict:
    """
    Get all criterion descriptions.

    Returns:
        Dictionary mapping symptom labels to criterion descriptions
    """
    return DSM5_CRITERIA.copy()


def get_symptom_labels() -> list:
    """
    Get ordered list of symptom labels.

    Returns:
        List of 9 symptom label strings
    """
    return SYMPTOM_LABELS.copy()
