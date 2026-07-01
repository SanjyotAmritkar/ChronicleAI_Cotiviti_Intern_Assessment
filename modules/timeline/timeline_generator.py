import re

import pandas as pd


def _extract_section(note_text: str, section_name: str) -> str:
    section_headers = [
        "Encounter Date",
        "Chief Complaint",
        "History of Present Illness",
        "Symptoms",
        "Assessment",
        "Plan",
        "Medications",
    ]
    other_headers = [header for header in section_headers if header != section_name]
    pattern = (
        rf"{re.escape(section_name)}\s*\n"
        rf"(.*?)"
        rf"(?=\n(?:{'|'.join(re.escape(header) for header in other_headers)})\s*\n|\Z)"
    )
    match = re.search(pattern, note_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""

    return match.group(1).strip()


def extract_date(note_text: str) -> str:
    """Extract the encounter date from a clinical note."""
    date_text = _extract_section(note_text, "Encounter Date")
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
    if not match:
        return ""

    return match.group(0)


def extract_event(note_text: str) -> str:
    """Create a deterministic timeline event from the assessment section."""
    assessment = _extract_section(note_text, "Assessment").lower()

    if "heart failure exacerbation" in assessment or "volume overload" in assessment:
        return "Heart failure progression documented"
    if "heart failure" in assessment:
        return "Heart failure status documented"
    if "coronary artery disease" in assessment or "stable angina" in assessment:
        if "suspected" in assessment or "abnormal stress test" in assessment:
            return "Coronary disease evaluation initiated"
        if "stent" in assessment or "pci" in assessment:
            return "Coronary intervention recovery documented"
        return "Coronary disease management documented"
    if "diabetes" in assessment or "glycemic" in assessment:
        if "improving" in assessment or "improvement" in assessment or "improved" in assessment:
            return "Diabetes management improving"
        if "new diagnosis" in assessment or "suspected" in assessment:
            return "Diabetes evaluation initiated"
        if "persistent hyperglycemia" in assessment or "uncontrolled" in assessment:
            return "Diabetes management intensified"
        return "Diabetes management documented"

    return "Clinical status documented"


def generate_timeline(visits: list[dict]) -> list[dict]:
    """Generate chronological timeline events from loaded visit dictionaries."""
    timeline = []

    for visit in visits:
        note_text = visit.get("content", "")
        event_date = extract_date(note_text)
        event = extract_event(note_text)

        if event_date:
            timeline.append(
                {
                    "date": event_date,
                    "event": event,
                }
            )

    return sorted(timeline, key=lambda item: item["date"])


def timeline_dataframe(timeline_events: list[dict]) -> pd.DataFrame:
    """Convert timeline events into a chronological dataframe."""
    dataframe = pd.DataFrame(timeline_events, columns=["date", "event"])
    if dataframe.empty:
        return dataframe

    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    dataframe["date"] = dataframe["date"].dt.strftime("%Y-%m-%d")
    return dataframe


def plot_timeline(dataframe: pd.DataFrame):
    """Build a Plotly timeline chart."""
    import plotly.express as px

    if dataframe.empty:
        return None

    plot_data = dataframe.copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"])
    plot_data["encounter"] = [f"Encounter {index + 1}" for index in range(len(plot_data))]

    figure = px.scatter(
        plot_data,
        x="date",
        y="encounter",
        text="event",
        hover_data={"date": True, "event": True, "encounter": False},
        title="Patient Journey Timeline",
    )
    figure.update_traces(mode="markers+text", textposition="top center")
    figure.update_layout(
        xaxis_title="Encounter Date",
        yaxis_title="",
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure
