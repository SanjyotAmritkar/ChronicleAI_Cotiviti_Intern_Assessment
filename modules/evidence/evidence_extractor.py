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
    "Blurred vision": ["blurred vision"],
    "Bruising": ["bruising"],
    "Chest pain": ["chest pain"],
    "Chest pressure": ["chest pressure"],
    "Diaphoresis": ["diaphoresis"],
    "Dizziness": ["dizziness"],
    "Dyspnea": ["dyspnea", "shortness of breath"],
    "Edema": ["edema", "ankle swelling", "ankle edema"],
    "Fatigue": ["fatigue"],
    "Foot tingling": ["foot tingling", "tingling"],
    "Hypoglycemia": ["hypoglycemia"],
    "Nausea": ["nausea"],
    "Orthopnea": ["orthopnea", "pillows"],
    "Palpitations": ["palpitations"],
    "Polydipsia": ["polydipsia", "thirst"],
    "Polyuria": ["polyuria", "urination"],
    "Shortness of breath": ["shortness of breath", "dyspnea"],
    "Syncope": ["syncope"],
    "Visual symptoms": ["visual", "vision"],
    "Weight gain": ["weight gain", "gained"],
}

DIAGNOSTIC_KEYWORDS = {
    "Basic metabolic panel ordered": ["basic metabolic panel", "bmp"],
    "BNP ordered": ["bnp"],
    "Chest X-ray ordered": ["chest x-ray", "chest xray"],
    "Coronary angiography ordered": ["coronary angiography"],
    "Diabetic eye exam ordered": ["eye exam", "ophthalmology"],
    "Echocardiogram ordered": ["echocardiogram"],
    "HbA1c ordered": ["a1c"],
    "Lipid panel ordered": ["lipid panel"],
    "Stress test ordered": ["stress test"],
    "Urine albumin ordered": ["urine albumin"],
}

TRAJECTORY_KEYWORDS = {
    "Increasing symptom burden identified": [
        "worsening",
        "progressive",
        "increased",
        "persistent",
        "weight gain",
        "orthopnea",
        "dyspnea",
    ],
    "Medication intensification observed": [
        "increase",
        "add",
        "start",
        "escalation",
    ],
    "Diagnostic workup expanded": [
        "order",
        "refer",
        "echocardiogram",
        "stress test",
        "bnp",
        "a1c",
    ],
    "Worsening or persistent clinical language detected": [
        "worsening",
        "progressive",
        "persistent",
        "volume overload",
        "abnormal",
    ],
    "Symptoms resolving across encounters": [
        "improved",
        "improvement",
        "resolved",
        "no recurrent",
        "no angina",
    ],
    "Improvement or stability language documented": [
        "improved",
        "improvement",
        "stable",
        "controlled",
    ],
    "No significant treatment escalation detected": [
        "continue",
        "current regimen",
        "current medications",
    ],
    "Treatment changes coincide with improving clinical status": [
        "improved",
        "improvement",
        "continue",
    ],
    "Minimal clinical changes detected": [
        "stable",
        "routine",
        "continue",
    ],
    "Therapy remains unchanged": [
        "continue",
        "current",
    ],
    "No expanded diagnostic workup detected": [
        "follow up",
        "continue",
    ],
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


def _section_lines(section_text: str) -> list[str]:
    lines = []
    for line in section_text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            lines.append(cleaned)

    return lines


def _visit_by_number(visits: list[dict], encounter_number: int | None) -> dict | None:
    for visit in visits:
        if visit.get("visit_number") == encounter_number:
            return visit

    return None


def _encounter_label(visit: dict) -> str:
    return f"Encounter {visit.get('visit_number')}"


def _evidence_record(finding: str, visit: dict, section: str, evidence: str) -> dict:
    return {
        "finding": finding,
        "encounter": _encounter_label(visit),
        "section": section,
        "evidence": evidence,
    }


def extract_supporting_text(note_text: str, section_name: str, keywords: list[str]) -> str:
    """Return the first section line matching any keyword."""
    section_text = _extract_section(note_text, section_name)
    lines = _section_lines(section_text)

    for line in lines:
        lowered = line.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return line

    return ""


def find_symptom_evidence(visits: list[dict], changes: dict) -> list[dict]:
    evidence = []

    for visit_change in changes.get("visit_changes", []):
        visit = _visit_by_number(visits, visit_change.get("encounter"))
        if not visit:
            continue

        for finding in visit_change.get("new_findings", []):
            supporting_text = extract_supporting_text(
                visit.get("content", ""),
                "Symptoms",
                SYMPTOM_KEYWORDS.get(finding, [finding]),
            )
            if supporting_text:
                evidence.append(_evidence_record(finding, visit, "Symptoms", supporting_text))

        for finding in visit_change.get("resolved_findings", []):
            lowered_finding = finding.lower()
            supporting_text = extract_supporting_text(
                visit.get("content", ""),
                "Symptoms",
                [
                    "no " + lowered_finding,
                    "no recurrent " + lowered_finding,
                    "denies " + lowered_finding,
                ],
            )
            if not supporting_text:
                supporting_text = f"{finding} no longer documented in Symptoms section"
            evidence.append(_evidence_record(f"{finding} resolved", visit, "Symptoms", supporting_text))

    return evidence


def find_medication_evidence(visits: list[dict], changes: dict) -> list[dict]:
    evidence = []

    for visit_change in changes.get("visit_changes", []):
        visit = _visit_by_number(visits, visit_change.get("encounter"))
        if not visit:
            continue

        for medication in visit_change.get("medications_started", []):
            supporting_text = extract_supporting_text(
                visit.get("content", ""),
                "Medications",
                [medication],
            )
            if supporting_text:
                evidence.append(
                    _evidence_record(f"{medication} started", visit, "Medications", supporting_text)
                )

        for medication in visit_change.get("medications_discontinued", []):
            evidence.append(
                _evidence_record(
                    f"{medication} discontinued",
                    visit,
                    "Medications",
                    f"{medication} no longer listed in Medications section",
                )
            )

        for dose_change in visit_change.get("dose_modifications", []):
            medication_name = dose_change.split(":", 1)[0]
            supporting_text = extract_supporting_text(
                visit.get("content", ""),
                "Medications",
                [medication_name],
            )
            if supporting_text:
                evidence.append(
                    _evidence_record(
                        f"{medication_name} escalation",
                        visit,
                        "Medications",
                        supporting_text,
                    )
                )

    return evidence


def find_diagnostic_evidence(visits: list[dict], changes: dict) -> list[dict]:
    evidence = []

    for visit_change in changes.get("visit_changes", []):
        visit = _visit_by_number(visits, visit_change.get("encounter"))
        if not visit:
            continue

        for diagnostic in visit_change.get("diagnostics", []):
            supporting_text = extract_supporting_text(
                visit.get("content", ""),
                "Plan",
                DIAGNOSTIC_KEYWORDS.get(diagnostic, [diagnostic.replace(" ordered", "")]),
            )
            if supporting_text:
                evidence.append(_evidence_record(diagnostic, visit, "Plan", supporting_text))

    return evidence


def find_trajectory_evidence(visits: list[dict], trajectory: dict) -> list[dict]:
    evidence = []
    recent_visits = visits[-2:] if len(visits) >= 2 else visits

    for reasoning in trajectory.get("reasoning", []):
        keywords = TRAJECTORY_KEYWORDS.get(reasoning, [reasoning])
        matched_reasoning = False
        for visit in recent_visits:
            for section in ["History of Present Illness", "Assessment", "Plan", "Symptoms"]:
                supporting_text = extract_supporting_text(visit.get("content", ""), section, keywords)
                if supporting_text:
                    evidence.append(
                        _evidence_record(
                            trajectory.get("status", "Trajectory"),
                            visit,
                            section,
                            supporting_text,
                        )
                    )
                    matched_reasoning = True
                    break
            if matched_reasoning:
                break

    return evidence


def build_evidence_dictionary(visits: list[dict], changes: dict, trajectory: dict) -> dict:
    """Build deterministic evidence attribution grouped by evidence type."""
    symptom_evidence = find_symptom_evidence(visits, changes)
    medication_evidence = find_medication_evidence(visits, changes)
    diagnostic_evidence = find_diagnostic_evidence(visits, changes)
    trajectory_evidence = find_trajectory_evidence(visits, trajectory)

    return {
        "symptoms": symptom_evidence,
        "medications": medication_evidence,
        "diagnostics": diagnostic_evidence,
        "trajectory": trajectory_evidence,
        "all": symptom_evidence + medication_evidence + diagnostic_evidence + trajectory_evidence,
    }
