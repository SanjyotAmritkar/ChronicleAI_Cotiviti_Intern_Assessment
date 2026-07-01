# ChronicleAI

Clinical Narrative Intelligence Engine

Transforming longitudinal clinical notes into explainable patient journeys.

## Overview

ChronicleAI is a proof-of-concept healthcare NLP platform developed for the Cotiviti Intern Assessment.

The system converts longitudinal clinical documentation into explainable patient journeys using deterministic NLP techniques and Large Language Models. It is designed as a hackathon-style prototype that emphasizes transparency, traceability, and clinical narrative understanding.

## Features

- Timeline reconstruction
- Clinical trajectory analysis
- Medication evolution tracking
- Symptom progression detection
- Evidence attribution
- Care gap identification
- Longitudinal patient summarization
- Explainable AI outputs
- LLM-powered narrative synthesis

## Architecture

```text
Patient Notes
    |
    v
NLP Extraction
    |
    v
Temporal Reconstruction
    |
    v
Clinical Change Detection
    |
    v
Trajectory Analysis
    |
    v
Evidence Attribution
    |
    v
LLM Summary Generation
    |
    v
ChronicleAI Dashboard
```

## Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- Deterministic NLP
- Large Language Models
- Anthropic API for the current prototype LLM integration
- OpenAI-compatible model configuration planned for future provider support

## Repository Structure

```text
ChronicleAI_Cotiviti_Intern_Assessment/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── LICENSE
├── assets/
│   └── screenshots/
├── data/
│   └── synthetic_patients/
│       ├── cardiac_case/
│       ├── diabetes_case/
│       └── heart_failure_case/
├── modules/
│   ├── analytics/
│   ├── care_gaps/
│   ├── change_detection/
│   ├── evidence/
│   ├── llm/
│   ├── timeline/
│   ├── trajectory/
│   └── utils/
├── docs/
├── notebooks/
├── presentation/
├── report/
└── video/
```

## Setup Instructions

### Clone Repository

```bash
git clone <repository-url>
cd ChronicleAI_Cotiviti_Intern_Assessment
```

### Create Virtual Environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

```bash
cp .env.example .env
```

Add your local API key to `.env`. Do not commit `.env`.

### Run Application

```bash
streamlit run app.py
```

## Screenshots

Screenshots should be added to `assets/screenshots/` before final submission.

- Dashboard Overview
- Trajectory Analysis
- Clinical Evolution
- LLM Summary
- Care Gap Detection

## Deliverables

- Final report: `report/ChronicleAI_Report.docx`
- Presentation: `presentation/ChronicleAI_Presentation.pptx`
- Demo video: `video/ChronicleAI_Demo.mp4`
- Architecture diagrams: `docs/architecture.png`, `docs/project_structure.png`

Placeholder notes are included in these folders until final exported files are added.

## Future Work

- FHIR integration
- Real-world EHR support
- Predictive trajectory modeling
- Multimodal clinical intelligence

## Security Notes

- Synthetic patient notes are used for demonstration only.
- No real patient data should be committed to this repository.
- API keys must remain in `.env`, which is ignored by Git.
