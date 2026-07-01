import re


SECTION_HEADERS = [
    "Encounter Date",
    "Chief Complaint",
    "History of Present Illness",
    "Symptoms",
    "Assessment",
    "Plan",
    "Medications",
]

SYMPTOM_KEYWORDS = {
    "ankle edema": "Edema",
    "ankle swelling": "Edema",
    "blurred vision": "Blurred vision",
    "bruising": "Bruising",
    "chest pain": "Chest pain",
    "chest pressure": "Chest pressure",
    "diaphoresis": "Diaphoresis",
    "dizziness": "Dizziness",
    "dyspnea": "Dyspnea",
    "edema": "Edema",
    "fatigue": "Fatigue",
    "foot tingling": "Foot tingling",
    "hypoglycemia": "Hypoglycemia",
    "nausea": "Nausea",
    "orthopnea": "Orthopnea",
    "palpitations": "Palpitations",
    "polydipsia": "Polydipsia",
    "polyuria": "Polyuria",
    "shortness of breath": "Shortness of breath",
    "syncope": "Syncope",
    "visual": "Visual symptoms",
    "weight gain": "Weight gain",
}

DIAGNOSTIC_KEYWORDS = {
    "a1c": "HbA1c ordered",
    "basic metabolic panel": "Basic metabolic panel ordered",
    "bmp": "Basic metabolic panel ordered",
    "bnp": "BNP ordered",
    "chest x-ray": "Chest X-ray ordered",
    "chest xray": "Chest X-ray ordered",
    "coronary angiography": "Coronary angiography ordered",
    "echocardiogram": "Echocardiogram ordered",
    "eye exam": "Diabetic eye exam ordered",
    "lipid panel": "Lipid panel ordered",
    "ophthalmology": "Diabetic eye exam ordered",
    "stress test": "Stress test ordered",
    "urine albumin": "Urine albumin ordered",
}


def _extract_section(note_text: str, section_name: str) -> str:
    other_headers = [header for header in SECTION_HEADERS if header != section_name]
    pattern = (
        rf"{re.escape(section_name)}\s*\n"
        rf"(.*?)"
        rf"(?=\n(?:{'|'.join(re.escape(header) for header in other_headers)})\s*\n|\Z)"
    )
    match = re.search(pattern, note_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""

    return match.group(1).strip()


def _bullet_lines(section_text: str) -> list[str]:
    lines = []
    for line in section_text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            lines.append(cleaned)

    return lines


def _normalize_medication_name(medication_text: str) -> str:
    text = medication_text.strip()
    text = re.sub(r"^\-\s*", "", text)
    match = re.match(r"([A-Za-z][A-Za-z\- ]*?)(?:\s+\d|\s+if\b|\s+daily\b|\s+twice\b|\s+weekly\b|$)", text)
    if not match:
        return text

    return " ".join(match.group(1).split()).title()


def _medication_map(note_text: str) -> dict[str, str]:
    medications = {}
    for medication in extract_medications(note_text):
        medications[_normalize_medication_name(medication)] = medication

    return medications


def _sorted_list(values: set[str]) -> list[str]:
    return sorted(values, key=str.lower)


def extract_symptoms(note_text: str) -> set[str]:
    """Extract documented symptoms as canonical terms."""
    symptoms_text = _extract_section(note_text, "Symptoms")
    symptoms = set()

    for line in _bullet_lines(symptoms_text):
        lowered = line.lower()
        if lowered.startswith("no "):
            continue
        for keyword, canonical_name in SYMPTOM_KEYWORDS.items():
            if keyword in lowered:
                symptoms.add(canonical_name)

    return symptoms


def extract_medications(note_text: str) -> set[str]:
    """Extract medication lines from the Medications section."""
    medications_text = _extract_section(note_text, "Medications")
    return set(_bullet_lines(medications_text))


def extract_diagnostics(note_text: str) -> set[str]:
    """Extract ordered diagnostics from the Plan section."""
    plan_text = _extract_section(note_text, "Plan").lower()
    diagnostics = set()

    for keyword, diagnostic_label in DIAGNOSTIC_KEYWORDS.items():
        if keyword in plan_text:
            diagnostics.add(diagnostic_label)

    return diagnostics


def detect_new_findings(previous_visit: str | None, current_visit: str) -> list[str]:
    previous_symptoms = extract_symptoms(previous_visit or "")
    current_symptoms = extract_symptoms(current_visit)
    return _sorted_list(current_symptoms - previous_symptoms)


def detect_resolved_findings(previous_visit: str | None, current_visit: str) -> list[str]:
    previous_symptoms = extract_symptoms(previous_visit or "")
    current_symptoms = extract_symptoms(current_visit)
    return _sorted_list(previous_symptoms - current_symptoms)


def detect_medication_changes(previous_visit: str | None, current_visit: str) -> dict[str, list[str]]:
    previous_medications = _medication_map(previous_visit or "")
    current_medications = _medication_map(current_visit)

    previous_names = set(previous_medications)
    current_names = set(current_medications)
    shared_names = previous_names & current_names

    dose_modifications = []
    for medication_name in sorted(shared_names, key=str.lower):
        previous_text = previous_medications[medication_name]
        current_text = current_medications[medication_name]
        if previous_text != current_text:
            dose_modifications.append(f"{medication_name}: {previous_text} -> {current_text}")

    return {
        "medications_started": _sorted_list(current_names - previous_names),
        "medications_discontinued": _sorted_list(previous_names - current_names),
        "dose_modifications": dose_modifications,
    }


def detect_diagnostic_changes(previous_visit: str | None, current_visit: str) -> list[str]:
    previous_diagnostics = extract_diagnostics(previous_visit or "")
    current_diagnostics = extract_diagnostics(current_visit)
    return _sorted_list(current_diagnostics - previous_diagnostics)


def compare_visits(visits: list[dict]) -> dict:
    """Compare encounters sequentially and return aggregate clinical changes."""
    summary = {
        "new_findings": [],
        "resolved_findings": [],
        "medications_started": [],
        "medications_discontinued": [],
        "dose_modifications": [],
        "diagnostics": [],
        "visit_changes": [],
    }

    seen_new_findings = set()
    seen_resolved_findings = set()
    seen_started = set()
    seen_discontinued = set()
    seen_dose_changes = set()
    seen_diagnostics = set()

    previous_note = None
    for visit in visits:
        current_note = visit.get("content", "")
        medication_changes = detect_medication_changes(previous_note, current_note)
        visit_change = {
            "encounter": visit.get("visit_number"),
            "new_findings": detect_new_findings(previous_note, current_note),
            "resolved_findings": detect_resolved_findings(previous_note, current_note),
            "medications_started": medication_changes["medications_started"],
            "medications_discontinued": medication_changes["medications_discontinued"],
            "dose_modifications": medication_changes["dose_modifications"],
            "diagnostics": detect_diagnostic_changes(previous_note, current_note),
        }

        summary["visit_changes"].append(visit_change)
        seen_new_findings.update(visit_change["new_findings"])
        seen_resolved_findings.update(visit_change["resolved_findings"])
        seen_started.update(visit_change["medications_started"])
        seen_discontinued.update(visit_change["medications_discontinued"])
        seen_dose_changes.update(visit_change["dose_modifications"])
        seen_diagnostics.update(visit_change["diagnostics"])

        previous_note = current_note

    summary["new_findings"] = _sorted_list(seen_new_findings)
    summary["resolved_findings"] = _sorted_list(seen_resolved_findings)
    summary["medications_started"] = _sorted_list(seen_started)
    summary["medications_discontinued"] = _sorted_list(seen_discontinued)
    summary["dose_modifications"] = _sorted_list(seen_dose_changes)
    summary["diagnostics"] = _sorted_list(seen_diagnostics)

    return summary


def generate_change_summary(visits: list[dict]) -> dict:
    """Generate the clinical change summary for a loaded patient journey."""
    return compare_visits(visits)
