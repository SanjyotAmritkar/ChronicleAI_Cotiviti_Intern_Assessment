import time
from datetime import datetime

import streamlit as st

from modules.care_gaps.care_gap_detector import generate_gap_summary
from modules.change_detection.change_detector import extract_symptoms, generate_change_summary
from modules.evidence.evidence_extractor import build_evidence_dictionary
from modules.llm.clinical_summarizer import (
    DEFAULT_MODEL,
    generate_clinical_summary,
    generate_trajectory_explanation,
    get_last_llm_error,
    initialize_client,
    parse_structured_summary,
)
from modules.timeline.timeline_generator import generate_timeline
from modules.trajectory.trajectory_analyzer import analyze_trajectory
from modules.utils.data_loader import list_patients, load_patient


LLM_SUMMARY_CACHE_VERSION = "compact-summary-v1"

st.set_page_config(
    page_title="ChronicleAI",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_cached_llm_insights(
    patient_condition: str,
    timeline: list[dict],
    trajectory: dict,
    clinical_changes: dict,
    supporting_evidence: dict,
    care_gaps: list[dict],
    model: str,
    cache_version: str,
) -> dict:
    summary = generate_clinical_summary(
        patient_condition=patient_condition,
        timeline=timeline,
        trajectory=trajectory,
        clinical_changes=clinical_changes,
        supporting_evidence=supporting_evidence,
        care_gaps=care_gaps,
        model=model,
    )
    trajectory_explanation = generate_trajectory_explanation(
        patient_condition=patient_condition,
        trajectory=trajectory,
        clinical_changes=clinical_changes,
        supporting_evidence=supporting_evidence,
        model=model,
    )

    return {
        "summary": summary,
        "trajectory_explanation": trajectory_explanation,
        "error": get_last_llm_error(),
    }


def status_icon(status: str) -> str:
    if status == "Improving":
        return "Improving"
    if status == "Stable":
        return "Stable"
    return "Potentially Worsening"


def compact_status(status: str) -> str:
    if status == "Improving":
        return "Improving"
    if status == "Stable":
        return "Stable"
    return "Worsening"


def display_status(status: str) -> str:
    if status == "Potentially Worsening":
        return "Potentially Worsening"
    return status


def status_caption(status: str) -> str:
    if status == "Improving":
        return "Overall clinical progression"
    if status == "Stable":
        return "Limited directional change"
    return "Worsening signals detected"


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def format_month(date_text: str) -> str:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%B %Y")
    except ValueError:
        return date_text


def format_short_date(date_text: str) -> str:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        return date_text


def timeline_label(event: str) -> str:
    event_lower = event.lower()
    if "evaluation" in event_lower:
        return "Evaluation started"
    if "diagnostic" in event_lower:
        return "Diagnostics ordered"
    if "intervention" in event_lower:
        return "Intervention noted"
    if "recovery" in event_lower:
        return "Improving course"
    if "improving" in event_lower:
        return "Improving course"
    if "management" in event_lower:
        return "Management updated"
    if "resolved" in event_lower:
        return "Symptoms resolved"
    return event


def render_timeline(timeline_events: list[dict]) -> None:
    if not timeline_events:
        st.info("No timeline events generated.")
        return

    st.caption("Chronological overview of key clinical events")
    timeline_cols = st.columns(len(timeline_events))
    for index, (column, event) in enumerate(zip(timeline_cols, timeline_events), start=1):
        with column:
            with st.container(border=True):
                st.markdown(f"**{timeline_label(event['event'])}**")
                st.caption(format_short_date(event["date"]))
                st.caption(f"Encounter {index}")


def split_sentences(text: str, limit: int = 3) -> list[str]:
    if not text:
        return []
    normalized = text.replace("\n", " ").strip()
    parts = []
    for chunk in normalized.replace("?", ".").replace("!", ".").split("."):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts[:limit]


def max_items(items: list[str], limit: int = 3) -> list[str]:
    return [item for item in items if item][:limit]


def compact_text(text: str, max_words: int = 14) -> str:
    cleaned_text = text.strip().lstrip("+-").strip()
    words = cleaned_text.replace(";", ",").split()
    if len(words) <= max_words:
        return cleaned_text
    return " ".join(words[:max_words]).rstrip(",") + "..."


def render_bullets(items: list[str], empty_text: str, max_words: int = 14, limit: int = 3) -> None:
    visible_items = max_items(items, limit=limit)
    if not visible_items:
        st.caption(empty_text)
        return
    for item in visible_items:
        st.markdown(f"- {compact_text(item, max_words=max_words)}")


def render_badge_list(items: list[str], empty_text: str, max_words: int = 14) -> None:
    render_bullets(items, empty_text, max_words=max_words, limit=4)


def render_kpi_card(label: str, value: str, caption: str) -> None:
    with st.container(border=True):
        st.caption(label.upper())
        st.markdown(f"### {value}")
        st.caption(caption)


def render_ai_summary(summary: str | None, trajectory_explanation: str | None, trajectory: dict) -> None:
    with st.container(border=True):
        st.markdown("### AI Clinical Summary")
        st.caption("Concise longitudinal synthesis. Expand each section for details.")
        if not summary:
            st.info("Enable LLM Insights and run Analyze Journey to generate the AI summary.")
            return

        structured = parse_structured_summary(summary, trajectory_explanation)
        longitudinal_items = max_items(structured["clinical_course"])
        if not longitudinal_items and structured["overview"]:
            longitudinal_items = [structured["overview"]]

        rationale_items = split_sentences(structured["trajectory_rationale"], limit=4)
        if not rationale_items:
            rationale_items = max_items(trajectory.get("reasoning", []))

        if structured["overview"]:
            st.caption(compact_text(structured["overview"], max_words=18))

        with st.expander("Longitudinal Summary", expanded=True):
            render_badge_list(longitudinal_items, "No longitudinal summary generated.", max_words=10)

        with st.expander("Clinical Signals", expanded=False):
            render_badge_list(structured["key_signals"], "No clinical signals generated.", max_words=9)

        with st.expander("Trajectory Rationale", expanded=False):
            render_badge_list(rationale_items, "No trajectory rationale generated.", max_words=11)


def temporal_finding_categories(visits: list[dict]) -> dict[str, list[str]]:
    symptom_sets = [extract_symptoms(visit["content"]) for visit in visits]
    if not symptom_sets:
        return {"new": [], "resolved": [], "persistent": []}

    first = symptom_sets[0]
    last = symptom_sets[-1]
    all_findings = set().union(*symptom_sets)
    persistent = set.intersection(*symptom_sets) if symptom_sets else set()
    resolved = (first - last) - persistent
    new = (set().union(*symptom_sets[1:]) - first) - persistent - resolved

    return {
        "new": sorted(new, key=str.lower),
        "resolved": sorted(resolved, key=str.lower),
        "persistent": sorted(persistent, key=str.lower),
    }


def medication_change_items(change_summary: dict) -> list[str]:
    items = []
    items.extend(f"{medication} started" for medication in change_summary["medications_started"])
    items.extend(
        f"{medication} discontinued"
        for medication in change_summary["medications_discontinued"]
    )
    items.extend(change_summary["dose_modifications"])
    return items


def confidence_drivers(change_summary: dict, gap_summary: list[dict], trajectory: dict) -> dict[str, list[str]]:
    new_count = len(change_summary["new_findings"])
    resolved_count = len(change_summary["resolved_findings"])
    medication_count = len(change_summary["medications_started"]) + len(change_summary["dose_modifications"])
    diagnostic_count = len(change_summary["diagnostics"])
    gap_count = sum(
        1
        for gap in gap_summary
        if gap["category"] in {"Potential Care Gap", "Documentation Gap"}
    )

    positive = []
    negative = []

    if resolved_count >= new_count:
        positive.append("symptom consistency")
    if medication_count:
        positive.append("medication optimization")
    if diagnostic_count:
        positive.append("diagnostics completed")
    if trajectory["status"] == "Improving":
        positive.append("recovery documented")
    if not gap_count:
        positive.append("follow-up activities documented")

    if new_count > resolved_count:
        negative.append("emerging symptoms")
    if gap_count:
        negative.append("incomplete follow-up documentation")
    if not diagnostic_count:
        negative.append("limited diagnostic signal")
    if trajectory["status"] == "Potentially Worsening":
        negative.append("worsening language detected")

    return {
        "positive": positive[:4],
        "negative": negative[:4],
    }


def render_confidence_drivers(drivers: dict[str, list[str]]) -> None:
    with st.container(border=True):
        st.markdown("**Confidence Drivers**")
        pos_col, neg_col = st.columns(2)
        with pos_col:
            st.caption("Supportive")
            render_badge_list(drivers["positive"], "No positive drivers identified.")
        with neg_col:
            st.caption("Limiting")
            render_bullets(drivers["negative"], "No limiting drivers identified.")


def render_analysis_progress() -> None:
    steps = [
        ("Temporal Reconstruction", 10),
        ("Entity Extraction", 25),
        ("Clinical Change Detection", 40),
        ("Trajectory Classification", 55),
        ("Evidence Attribution", 70),
        ("LLM Synthesis", 85),
        ("Care Gap Detection", 100),
    ]
    progress = st.progress(0)
    status = st.empty()
    with st.spinner("Analyzing patient journey..."):
        for label, percent in steps:
            status.markdown(f"**{label}** - {percent}%")
            progress.progress(percent)
            time.sleep(0.1)
    status.success("Analysis complete.")


def run_analysis(patient_id: str, enable_llm: bool) -> dict:
    patient = load_patient(patient_id)
    change_summary = generate_change_summary(patient["visits"])
    timeline_events = generate_timeline(patient["visits"])
    trajectory = analyze_trajectory(patient["visits"], change_summary)
    evidence_dictionary = build_evidence_dictionary(
        patient["visits"],
        change_summary,
        trajectory,
    )
    gap_summary = generate_gap_summary(patient["visits"])

    llm_summary = None
    llm_trajectory_explanation = None
    llm_error = None
    if enable_llm:
        if initialize_client() is None:
            llm_error = get_last_llm_error()
        else:
            llm_insights = get_cached_llm_insights(
                patient_condition=patient["metadata"]["condition"],
                timeline=timeline_events,
                trajectory=trajectory,
                clinical_changes=change_summary,
                supporting_evidence=evidence_dictionary,
                care_gaps=gap_summary,
                model=DEFAULT_MODEL,
                cache_version=LLM_SUMMARY_CACHE_VERSION,
            )
            llm_summary = llm_insights["summary"]
            llm_trajectory_explanation = llm_insights["trajectory_explanation"]
            llm_error = llm_insights["error"]

    return {
        "patient": patient,
        "change_summary": change_summary,
        "timeline_events": timeline_events,
        "trajectory": trajectory,
        "evidence_dictionary": evidence_dictionary,
        "gap_summary": gap_summary,
        "llm_summary": llm_summary,
        "llm_trajectory_explanation": llm_trajectory_explanation,
        "llm_error": llm_error,
    }


def render_sidebar() -> tuple[str, bool, bool]:
    with st.sidebar:
        st.markdown("## ChronicleAI")
        st.caption("Clinical Narrative Intelligence Engine")
        st.divider()

        st.caption("PROJECT")
        st.markdown("**Overview**")
        st.caption("Longitudinal notes into explainable patient journeys.")
        st.divider()

        st.caption("PATIENT")
        patients = list_patients()
        if not patients:
            st.warning("No sample patient journeys found.")
            st.stop()

        selected_patient = st.selectbox(
            "Patient",
            patients,
            format_func=lambda patient_id: patient_id.replace("_", " ").title(),
            label_visibility="collapsed",
        )
        st.write("")
        enable_llm = st.toggle("Enable LLM Insights", value=False)
        analyze = st.button("Analyze Journey", type="primary", use_container_width=True)

        st.divider()
        st.caption("MODEL INFORMATION")
        with st.expander("Models Used", expanded=True):
            st.caption("Deterministic NLP")
            st.caption("Anthropic Claude LLM")
            st.caption("Trajectory Engine")
            st.caption("Evidence Attribution Engine")
            st.caption("Care Gap Detector")

        st.divider()
        st.caption("About ChronicleAI")
        st.caption("Transforms longitudinal clinical notes into explainable patient journeys.")

    return selected_patient, enable_llm, analyze


def maybe_reset_analysis(selected_patient: str, enable_llm: bool) -> None:
    current_key = (selected_patient, enable_llm)
    if st.session_state.get("analysis_key") != current_key:
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_payload"] = None
        st.session_state["analysis_key"] = current_key


selected_patient, enable_llm_insights, analyze_clicked = render_sidebar()
maybe_reset_analysis(selected_patient, enable_llm_insights)

header_col, status_col = st.columns([0.72, 0.28])
with header_col:
    st.markdown("# ChronicleAI")
    st.markdown("#### Clinical Narrative Intelligence Engine")
    st.caption("Transforming longitudinal clinical notes into explainable patient journeys.")
with status_col:
    if st.session_state.get("analysis_ready"):
        st.success("Analysis completed")
        st.caption("Updated just now")
    else:
        st.info("Ready for analysis")
        st.caption("Select a patient to begin")
st.divider()

if analyze_clicked:
    render_analysis_progress()
    st.session_state["analysis_payload"] = run_analysis(selected_patient, enable_llm_insights)
    st.session_state["analysis_ready"] = True

if not st.session_state.get("analysis_ready") or not st.session_state.get("analysis_payload"):
    with st.container(border=True):
        st.markdown("### Ready to analyze")
        st.caption("Select a patient and click Analyze Journey to reconstruct the clinical timeline.")
    st.stop()

payload = st.session_state["analysis_payload"]
patient = payload["patient"]
change_summary = payload["change_summary"]
timeline_events = payload["timeline_events"]
trajectory = payload["trajectory"]
evidence_dictionary = payload["evidence_dictionary"]
gap_summary = payload["gap_summary"]

gap_count = sum(
    1
    for gap in gap_summary
    if gap["category"] in {"Potential Care Gap", "Documentation Gap"}
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_card(
        "Trajectory",
        compact_status(trajectory["status"]),
        status_caption(trajectory["status"]),
    )
with kpi_cols[1]:
    render_kpi_card(
        "Confidence",
        f"{trajectory['confidence']}%",
        "Heuristic confidence score",
    )
with kpi_cols[2]:
    render_kpi_card(
        "Encounters",
        str(len(patient["visits"])),
        "Clinical notes reviewed",
    )
with kpi_cols[3]:
    render_kpi_card(
        "Care Gaps",
        str(gap_count),
        "Potential gaps detected",
    )

st.write("")
section_header("Patient Intelligence", "Timeline and AI-generated longitudinal synthesis")
journey_col, summary_col = st.columns([0.6, 0.4])
with journey_col:
    with st.container(border=True):
        st.markdown("### Patient Journey")
        render_timeline(timeline_events)
with summary_col:
    if enable_llm_insights and payload["llm_error"] and not payload["llm_summary"]:
        st.warning(f"LLM Insights unavailable: {payload['llm_error']}")
    render_ai_summary(
        payload["llm_summary"],
        payload["llm_trajectory_explanation"],
        trajectory,
    )

section_header("Trajectory Analysis", "Rule-based trajectory classification and confidence drivers")
trajectory_col, drivers_col = st.columns([0.55, 0.45])
with trajectory_col:
    with st.container(border=True):
        st.markdown(f"### {display_status(trajectory['status'])}")
        metric_col, factor_col = st.columns([0.35, 0.65])
        with metric_col:
            st.metric("Confidence", f"{trajectory['confidence']}%")
        with factor_col:
            st.caption("Contributing Factors")
            render_badge_list(trajectory["reasoning"], "No trajectory factors generated.", max_words=12)
with drivers_col:
    render_confidence_drivers(confidence_drivers(change_summary, gap_summary, trajectory))

section_header("Clinical Evolution", "Temporal finding categories across encounters")
finding_categories = temporal_finding_categories(patient["visits"])
with st.container(border=True):
    evolution_cols = st.columns(4)
    with evolution_cols[0]:
        st.markdown("**New Findings**")
        render_bullets(finding_categories["new"], "No later-emerging findings detected.")
    with evolution_cols[1]:
        st.markdown("**Resolved Findings**")
        render_bullets(finding_categories["resolved"], "No initially documented findings resolved.")
    with evolution_cols[2]:
        st.markdown("**Persistent Findings**")
        render_bullets(finding_categories["persistent"], "No findings persisted across every encounter.")
    with evolution_cols[3]:
        st.markdown("**Medication Changes**")
        render_bullets(medication_change_items(change_summary), "No medication changes detected.")

section_header("Supporting Evidence", "Traceable source snippets behind extracted findings")
with st.container(border=True):
    evidence_tabs = st.tabs(
        [
            f"Symptoms ({len(evidence_dictionary['symptoms'])})",
            f"Medications ({len(evidence_dictionary['medications'])})",
            f"Diagnostics ({len(evidence_dictionary['diagnostics'])})",
            f"Trajectory ({len(evidence_dictionary['trajectory'])})",
        ]
    )
    evidence_groups = [
        evidence_dictionary["symptoms"],
        evidence_dictionary["medications"],
        evidence_dictionary["diagnostics"],
        evidence_dictionary["trajectory"],
    ]

    for tab, evidence_items in zip(evidence_tabs, evidence_groups):
        with tab:
            if not evidence_items:
                st.caption("No supporting evidence found for this category.")
                continue
            for item in evidence_items[:6]:
                with st.expander(item["finding"], expanded=False):
                    st.caption(f"{item['encounter']} - {item['section']}")
                    st.write(item["evidence"])
            if len(evidence_items) > 6:
                st.caption(f"{len(evidence_items) - 6} additional evidence items hidden for readability.")

section_header("Potential Care Gaps", "Observational documentation and follow-up gaps")
with st.container(border=True):
    if not gap_summary:
        st.success("No potential care gaps identified in the available notes.")
    else:
        gap_cols = st.columns(min(3, len(gap_summary)))
        for index, gap in enumerate(gap_summary):
            with gap_cols[index % len(gap_cols)]:
                st.caption(gap["category"])
                st.markdown(f"**{gap['condition']}**")
                st.caption(gap["observation"])

section_header("Raw Encounters", "Original source notes, collapsed by default")
for index, visit in enumerate(patient["visits"]):
    visit_date = timeline_events[index]["date"] if index < len(timeline_events) else ""
    label = f"Encounter {visit['visit_number']}"
    if visit_date:
        label = f"{label} - {format_short_date(visit_date)}"
    with st.expander(label, expanded=False):
        st.text(visit["content"])
