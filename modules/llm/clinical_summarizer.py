import json
import os
import re
from typing import Any


DEFAULT_MODEL = "claude-3-5-haiku-20241022"
FALLBACK_MODEL = "claude-3-haiku-20240307"
LAST_LLM_ERROR = None

SYSTEM_PROMPT = (
    "You are a healthcare NLP assistant specializing in longitudinal patient "
    "narrative analysis.\n\n"
    "Generate concise summaries of patient journeys.\n"
    "Focus on progression over time.\n"
    "Avoid diagnosis.\n"
    "Avoid treatment recommendations.\n"
    "Describe clinical evolution only.\n"
    "Professional tone.\n"
    "Use concise bullet-friendly language.\n"
    "Each generated bullet should be 12 words or fewer.\n"
    "Return only valid JSON. Do not include Markdown, HTML, XML, code fences, or extra text."
)


def initialize_client():
    """Initialize an Anthropic client from ANTHROPIC_API_KEY, returning None if unavailable."""
    global LAST_LLM_ERROR

    try:
        from dotenv import load_dotenv
        from anthropic import Anthropic
    except ImportError as error:
        LAST_LLM_ERROR = f"Missing dependency: {error.name}"
        return None

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    legacy_api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and legacy_api_key and legacy_api_key.startswith("sk-ant-"):
        api_key = legacy_api_key

    if not api_key:
        LAST_LLM_ERROR = "ANTHROPIC_API_KEY was not found in the environment"
        return None

    LAST_LLM_ERROR = None
    return Anthropic(api_key=api_key)


def get_last_llm_error() -> str | None:
    """Return the last non-secret LLM error message."""
    return LAST_LLM_ERROR


def list_available_models() -> list[str]:
    """Return Anthropic model IDs visible to the configured API key."""
    global LAST_LLM_ERROR

    client = initialize_client()
    if client is None:
        return []

    try:
        models = client.models.list()
    except Exception as error:
        LAST_LLM_ERROR = f"{error.__class__.__name__}: {error}"
        return []

    model_ids = [model.id for model in models.data if getattr(model, "id", None)]
    LAST_LLM_ERROR = None
    return model_ids


def _model_candidates(model: str | None) -> list[str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    requested_model = os.getenv("ANTHROPIC_MODEL") or model or DEFAULT_MODEL
    candidates = [requested_model]
    for fallback_model in list_available_models() + [DEFAULT_MODEL, FALLBACK_MODEL]:
        if fallback_model not in candidates:
            candidates.append(fallback_model)

    return candidates


def _format_list(items: list[Any], empty_label: str = "None documented") -> str:
    if not items:
        return f"- {empty_label}"

    return "\n".join(f"- {item}" for item in items)


def _format_timeline(timeline: list[dict]) -> str:
    if not timeline:
        return "- No timeline events available"

    return "\n".join(
        f"- {event.get('date', 'Unknown date')}: {event.get('event', 'Clinical event documented')}"
        for event in timeline
    )


def _format_evidence(evidence: dict) -> str:
    evidence_items = evidence.get("all", [])[:12] if evidence else []
    if not evidence_items:
        return "- No supporting evidence available"

    return "\n".join(
        "- "
        f"{item.get('finding', 'Finding')} | "
        f"{item.get('encounter', 'Encounter')} | "
        f"{item.get('section', 'Section')}: "
        f"{item.get('evidence', '')}"
        for item in evidence_items
    )


def _format_care_gaps(care_gaps: list[dict]) -> str:
    if not care_gaps:
        return "- No potential care gaps identified"

    return "\n".join(
        "- "
        f"{gap.get('category', 'Potential Observation')} | "
        f"{gap.get('condition', 'Condition')}: "
        f"{gap.get('observation', '')}"
        for gap in care_gaps
    )


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"```(?:json|html|markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?span[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(Clinical Narrative Summary|AI Clinical Summary)\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return text
    if end == -1 or end < start:
        return text[start:]
    return text[start : end + 1]


def _extract_json_string(text: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""

    value = match.group(1)
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def _extract_json_array(text: str, key: str) -> list[str]:
    pattern = rf'"{re.escape(key)}"\s*:\s*\[(.*?)(?=\]\s*,\s*"|\]\s*}}|,\s*"[a-zA-Z_]+":|\Z)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return []

    array_text = match.group(1)
    values = re.findall(r'"((?:\\.|[^"\\])*)"', array_text)
    parsed_values = []
    for value in values:
        try:
            parsed_values.append(json.loads(f'"{value}"'))
        except json.JSONDecodeError:
            parsed_values.append(value)

    return parsed_values


def _salvage_structured_summary(text: str) -> dict:
    return {
        "overview": _extract_json_string(text, "overview"),
        "clinical_course": _extract_json_array(text, "clinical_course"),
        "key_signals": _extract_json_array(text, "key_signals"),
        "trajectory_rationale": _extract_json_string(text, "trajectory_rationale"),
        "care_gap_note": _extract_json_string(text, "care_gap_note"),
    }


def parse_structured_summary(summary_text: str | None, trajectory_text: str | None = None) -> dict:
    """Parse LLM summary output into a stable structure for Streamlit rendering."""
    default_summary = {
        "overview": "",
        "clinical_course": [],
        "key_signals": [],
        "trajectory_rationale": "",
        "care_gap_note": "",
    }

    if not summary_text:
        return default_summary

    cleaned_summary = _clean_generated_text(summary_text)
    json_candidate = _extract_json_object(cleaned_summary)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        default_summary["overview"] = _clean_generated_text(str(parsed.get("overview", "")))
        default_summary["clinical_course"] = [
            _clean_generated_text(str(item))
            for item in parsed.get("clinical_course", [])
            if str(item).strip()
        ]
        default_summary["key_signals"] = [
            _clean_generated_text(str(item))
            for item in parsed.get("key_signals", [])
            if str(item).strip()
        ]
        default_summary["trajectory_rationale"] = _clean_generated_text(
            str(parsed.get("trajectory_rationale", ""))
        )
        default_summary["care_gap_note"] = _clean_generated_text(str(parsed.get("care_gap_note", "")))
    elif json_candidate.lstrip().startswith("{"):
        salvaged = _salvage_structured_summary(json_candidate)
        default_summary["overview"] = _clean_generated_text(salvaged["overview"])
        default_summary["clinical_course"] = [
            _clean_generated_text(item)
            for item in salvaged["clinical_course"]
            if item.strip()
        ]
        default_summary["key_signals"] = [
            _clean_generated_text(item)
            for item in salvaged["key_signals"]
            if item.strip()
        ]
        default_summary["trajectory_rationale"] = _clean_generated_text(
            salvaged["trajectory_rationale"]
        )
        default_summary["care_gap_note"] = _clean_generated_text(salvaged["care_gap_note"])
    else:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned_summary)
            if sentence.strip()
        ]
        default_summary["overview"] = sentences[0] if sentences else cleaned_summary
        default_summary["clinical_course"] = sentences[1:4]
        default_summary["care_gap_note"] = sentences[-1] if len(sentences) > 4 else ""

    if trajectory_text:
        default_summary["trajectory_rationale"] = _clean_generated_text(trajectory_text)

    return default_summary


def build_summary_prompt(
    patient_condition: str,
    timeline: list[dict],
    trajectory: dict,
    clinical_changes: dict,
    supporting_evidence: dict,
    care_gaps: list[dict],
) -> dict:
    """Build the system and user prompts for clinical narrative summarization."""
    medication_changes = (
        clinical_changes.get("medications_started", [])
        + clinical_changes.get("medications_discontinued", [])
        + clinical_changes.get("dose_modifications", [])
    )

    user_prompt = f"""
Patient Condition
{patient_condition}

Timeline
{_format_timeline(timeline)}

Trajectory
- Status: {trajectory.get("status", "Unknown")}
- Confidence: {trajectory.get("confidence", "Unknown")}%
- Reasoning:
{_format_list(trajectory.get("reasoning", []))}

New Findings
{_format_list(clinical_changes.get("new_findings", []))}

Resolved Findings
{_format_list(clinical_changes.get("resolved_findings", []))}

Medication Changes
{_format_list(medication_changes)}

Diagnostics
{_format_list(clinical_changes.get("diagnostics", []))}

Supporting Evidence
{_format_evidence(supporting_evidence)}

Care Gaps
{_format_care_gaps(care_gaps)}

Generate:
Clinical Narrative Summary

Return JSON using exactly this schema:
{{
  "overview": "One concise sentence, 16 words or fewer.",
  "clinical_course": [
    "Short bullet 1, 12 words or fewer.",
    "Short bullet 2, 12 words or fewer.",
    "Short bullet 3, 12 words or fewer."
  ],
  "key_signals": [
    "Short signal 1, 12 words or fewer.",
    "Short signal 2, 12 words or fewer.",
    "Short signal 3, 12 words or fewer."
  ],
  "trajectory_rationale": "One concise sentence, 18 words or fewer.",
  "care_gap_note": "One concise sentence, 14 words or fewer."
}}
""".strip()

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }


def _generate_response(prompt: dict, model: str = DEFAULT_MODEL) -> str | None:
    global LAST_LLM_ERROR

    client = initialize_client()
    if client is None:
        return None

    response = None
    errors = []
    for model_name in _model_candidates(model):
        try:
            response = client.messages.create(
                model=model_name,
                system=prompt["system"],
                messages=[
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            break
        except Exception as error:
            errors.append(f"{model_name}: {error.__class__.__name__}: {error}")

    if response is None:
        LAST_LLM_ERROR = " | ".join(errors)
        return None

    message_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    message = "\n".join(message_parts)
    LAST_LLM_ERROR = None
    return message.strip() if message else None


def generate_clinical_summary(
    patient_condition: str,
    timeline: list[dict],
    trajectory: dict,
    clinical_changes: dict,
    supporting_evidence: dict,
    care_gaps: list[dict],
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Generate an optional AI clinical summary from deterministic ChronicleAI outputs."""
    prompt = build_summary_prompt(
        patient_condition=patient_condition,
        timeline=timeline,
        trajectory=trajectory,
        clinical_changes=clinical_changes,
        supporting_evidence=supporting_evidence,
        care_gaps=care_gaps,
    )
    return _generate_response(prompt, model=model)


def generate_trajectory_explanation(
    patient_condition: str,
    trajectory: dict,
    clinical_changes: dict,
    supporting_evidence: dict,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Generate an optional explanation for the assigned trajectory classification."""
    prompt = {
        "system": (
            SYSTEM_PROMPT
            + "\n\nExplain why the provided trajectory status was assigned. "
            "Do not recommend treatments or diagnoses. Return one concise plain-text paragraph only. "
            "Do not include Markdown, HTML, XML, or code fences."
        ),
        "user": f"""
Patient Condition
{patient_condition}

Trajectory Status
{trajectory.get("status", "Unknown")}

Trajectory Confidence
{trajectory.get("confidence", "Unknown")}%

Trajectory Reasoning
{_format_list(trajectory.get("reasoning", []))}

New Findings
{_format_list(clinical_changes.get("new_findings", []))}

Resolved Findings
{_format_list(clinical_changes.get("resolved_findings", []))}

Medication Changes
{_format_list(clinical_changes.get("dose_modifications", []))}

Supporting Evidence
{_format_evidence(supporting_evidence)}

Question
Why was this trajectory assigned?
""".strip(),
    }
    return _generate_response(prompt, model=model)
