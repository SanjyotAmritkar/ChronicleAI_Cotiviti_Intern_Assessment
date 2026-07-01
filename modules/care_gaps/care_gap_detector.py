CARE_RULES = {
    "Diabetes": {
        "condition_keywords": ["diabetes", "glycemic", "hyperglycemia", "a1c"],
        "expected_actions": {
            "HbA1c monitoring": ["a1c", "hba1c"],
            "retinal screening": ["retinal", "eye exam", "ophthalmology", "retinopathy"],
            "diabetes education": ["diabetes education", "nutrition counseling"],
            "foot assessment": ["foot exam", "foot care", "foot checks"],
        },
    },
    "Heart Failure": {
        "condition_keywords": ["heart failure", "reduced ejection fraction", "volume overload"],
        "expected_actions": {
            "echocardiogram": ["echocardiogram", "ejection fraction"],
            "cardiology follow-up": ["cardiology", "cardiologist"],
            "daily weights": ["daily weight", "daily weights", "weight log"],
            "sodium counseling": ["low sodium", "sodium restriction", "sodium"],
        },
    },
    "Coronary Artery Disease": {
        "condition_keywords": [
            "coronary artery disease",
            "stable angina",
            "angina",
            "ischemia",
            "stent",
        ],
        "expected_actions": {
            "lipid panel": ["lipid panel", "ldl"],
            "stress test": ["stress test"],
            "cardiology consultation": ["cardiology", "cardiologist"],
            "cardiac rehabilitation": ["cardiac rehabilitation"],
        },
    },
}


def _journey_text(visits: list[dict]) -> str:
    return "\n".join(visit.get("content", "") for visit in visits).lower()


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text

    return keyword in text.split()


def identify_conditions(visits: list[dict]) -> list[str]:
    """Identify chronic condition categories documented in the journey."""
    text = _journey_text(visits)
    conditions = []

    for condition, rule in CARE_RULES.items():
        if any(_contains_keyword(text, keyword) for keyword in rule["condition_keywords"]):
            conditions.append(condition)

    return conditions


def identify_completed_actions(visits: list[dict], conditions: list[str] | None = None) -> dict:
    """Identify expected follow-up activities documented anywhere in the journey."""
    text = _journey_text(visits)
    active_conditions = conditions or identify_conditions(visits)
    completed_actions = {}

    for condition in active_conditions:
        completed_actions[condition] = []
        for action, keywords in CARE_RULES[condition]["expected_actions"].items():
            if any(_contains_keyword(text, keyword) for keyword in keywords):
                completed_actions[condition].append(action)

    return completed_actions


def identify_missing_followups(
    visits: list[dict],
    conditions: list[str] | None = None,
    completed_actions: dict | None = None,
) -> list[dict]:
    """Return observational gaps for expected activities not found in the notes."""
    active_conditions = conditions or identify_conditions(visits)
    documented_actions = completed_actions or identify_completed_actions(visits, active_conditions)
    missing_followups = []

    for condition in active_conditions:
        completed_for_condition = set(documented_actions.get(condition, []))
        for action in CARE_RULES[condition]["expected_actions"]:
            if action not in completed_for_condition:
                missing_followups.append(
                    {
                        "condition": condition,
                        "category": "Potential Care Gap",
                        "observation": f"No {action} identified",
                    }
                )

    return missing_followups


def evaluate_care_gaps(visits: list[dict]) -> list[dict]:
    """Evaluate potential informational or follow-up gaps using CARE_RULES."""
    conditions = identify_conditions(visits)

    if not conditions:
        return [
            {
                "condition": "Unknown",
                "category": "Documentation Gap",
                "observation": "No supported chronic condition category identified",
            }
        ]

    completed_actions = identify_completed_actions(visits, conditions)
    missing_followups = identify_missing_followups(visits, conditions, completed_actions)

    if missing_followups:
        return missing_followups

    return [
        {
            "condition": condition,
            "category": "Potential Observation",
            "observation": "Expected follow-up activities were documented in the available notes",
        }
        for condition in conditions
    ]


def generate_gap_summary(visits: list[dict]) -> list[dict]:
    """Generate observational care gap summary records."""
    return evaluate_care_gaps(visits)
