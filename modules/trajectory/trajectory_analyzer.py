WORSENING_KEYWORDS = [
    "worsening",
    "progressive",
    "persistent",
    "increasing",
    "volume overload",
    "hospitalization",
    "abnormal",
    "escalation",
]

IMPROVING_KEYWORDS = [
    "improved",
    "improvement",
    "resolved",
    "better",
    "stable",
    "recovering",
    "controlled",
    "responding",
]


def _note_text(visits: list[dict], recent_only: bool = False) -> str:
    selected_visits = visits[-2:] if recent_only else visits
    return "\n".join(visit.get("content", "") for visit in selected_visits).lower()


def _recent_changes(changes: dict) -> list[dict]:
    return changes.get("visit_changes", [])[-2:]


def count_new_findings(changes: dict, recent_only: bool = False) -> int:
    """Count new findings across the journey or in recent encounters."""
    if recent_only:
        return sum(len(visit_change.get("new_findings", [])) for visit_change in _recent_changes(changes))

    return len(changes.get("new_findings", []))


def count_resolved_findings(changes: dict, recent_only: bool = False) -> int:
    """Count resolved findings across the journey or in recent encounters."""
    if recent_only:
        return sum(
            len(visit_change.get("resolved_findings", []))
            for visit_change in _recent_changes(changes)
        )

    return len(changes.get("resolved_findings", []))


def count_medication_escalations(changes: dict, recent_only: bool = False) -> int:
    """Count medication starts and dose modifications after baseline encounter."""
    visit_changes = _recent_changes(changes) if recent_only else changes.get("visit_changes", [])[1:]
    return sum(
        len(visit_change.get("medications_started", []))
        + len(visit_change.get("dose_modifications", []))
        for visit_change in visit_changes
    )


def count_diagnostic_orders(changes: dict, recent_only: bool = False) -> int:
    """Count diagnostic orders after baseline encounter."""
    visit_changes = _recent_changes(changes) if recent_only else changes.get("visit_changes", [])[1:]
    return sum(len(visit_change.get("diagnostics", [])) for visit_change in visit_changes)


def _keyword_count(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def compute_confidence(
    status: str,
    new_findings: int,
    resolved_findings: int,
    medication_escalations: int,
    diagnostic_orders: int,
    worsening_language_count: int,
    improving_language_count: int,
) -> int:
    """Compute a transparent heuristic confidence score."""
    confidence = 64

    if status == "Potentially Worsening":
        confidence += min(new_findings * 5, 15)
        confidence += min(medication_escalations * 4, 12)
        confidence += min(diagnostic_orders * 3, 9)
        confidence += min(worsening_language_count * 5, 15)
        confidence -= min(resolved_findings * 3, 12)
    elif status == "Improving":
        confidence += min(resolved_findings * 5, 18)
        confidence += min(improving_language_count * 4, 16)
        confidence -= min(new_findings * 3, 12)
        confidence -= min(medication_escalations * 2, 8)
    else:
        confidence += 8
        confidence -= min((new_findings + resolved_findings) * 2, 10)
        confidence -= min(medication_escalations * 3, 9)
        confidence -= min(diagnostic_orders * 2, 8)

    return max(50, min(confidence, 95))


def generate_reasoning(
    status: str,
    changes: dict,
    new_findings: int,
    resolved_findings: int,
    medication_escalations: int,
    diagnostic_orders: int,
    worsening_language_count: int,
    improving_language_count: int,
) -> list[str]:
    """Generate concise explainable reasoning bullets."""
    reasoning = []

    if status == "Potentially Worsening":
        if new_findings > resolved_findings:
            reasoning.append("Increasing symptom burden identified")
        if medication_escalations:
            reasoning.append("Medication intensification observed")
        if diagnostic_orders:
            reasoning.append("Diagnostic workup expanded")
        if worsening_language_count:
            reasoning.append("Worsening or persistent clinical language detected")
    elif status == "Improving":
        if resolved_findings:
            reasoning.append("Symptoms resolving across encounters")
        if improving_language_count:
            reasoning.append("Improvement or stability language documented")
        if medication_escalations == 0:
            reasoning.append("No significant treatment escalation detected")
        else:
            reasoning.append("Treatment changes coincide with improving clinical status")
    else:
        reasoning.append("Minimal clinical changes detected")
        if medication_escalations == 0:
            reasoning.append("Therapy remains unchanged")
        if diagnostic_orders == 0:
            reasoning.append("No expanded diagnostic workup detected")

    if not reasoning:
        reasoning.append("Trajectory based on balanced rule-based indicators")

    return reasoning


def analyze_trajectory(visits: list[dict], changes: dict) -> dict:
    """Assess overall clinical progression using transparent deterministic rules."""
    recent_new = count_new_findings(changes, recent_only=True)
    recent_resolved = count_resolved_findings(changes, recent_only=True)
    recent_medication_escalations = count_medication_escalations(changes, recent_only=True)
    recent_diagnostic_orders = count_diagnostic_orders(changes, recent_only=True)

    total_new = count_new_findings(changes)
    total_resolved = count_resolved_findings(changes)
    total_medication_escalations = count_medication_escalations(changes)
    total_diagnostic_orders = count_diagnostic_orders(changes)

    recent_text = _note_text(visits, recent_only=True)
    all_text = _note_text(visits)
    worsening_language_count = _keyword_count(recent_text, WORSENING_KEYWORDS)
    improving_language_count = _keyword_count(recent_text, IMPROVING_KEYWORDS)
    total_worsening_language_count = _keyword_count(all_text, WORSENING_KEYWORDS)

    worsening_score = (
        recent_new * 3
        + max(recent_new - recent_resolved, 0) * 3
        + recent_medication_escalations * 2
        + recent_diagnostic_orders
        + worsening_language_count * 3
        + total_worsening_language_count
    )
    improving_score = (
        recent_resolved * 3
        + max(recent_resolved - recent_new, 0) * 3
        + improving_language_count * 3
    )
    stable_score = 5
    if recent_new <= 1 and recent_resolved <= 1:
        stable_score += 3
    if recent_medication_escalations == 0:
        stable_score += 2
    if recent_diagnostic_orders == 0:
        stable_score += 2

    if worsening_score >= improving_score + 4 and worsening_score >= stable_score:
        status = "Potentially Worsening"
    elif improving_score >= worsening_score and improving_score >= stable_score:
        status = "Improving"
    else:
        status = "Stable"

    confidence = compute_confidence(
        status=status,
        new_findings=total_new,
        resolved_findings=total_resolved,
        medication_escalations=total_medication_escalations,
        diagnostic_orders=total_diagnostic_orders,
        worsening_language_count=worsening_language_count,
        improving_language_count=improving_language_count,
    )
    reasoning = generate_reasoning(
        status=status,
        changes=changes,
        new_findings=recent_new,
        resolved_findings=recent_resolved,
        medication_escalations=recent_medication_escalations,
        diagnostic_orders=recent_diagnostic_orders,
        worsening_language_count=worsening_language_count,
        improving_language_count=improving_language_count,
    )

    return {
        "status": status,
        "confidence": confidence,
        "reasoning": reasoning,
    }
