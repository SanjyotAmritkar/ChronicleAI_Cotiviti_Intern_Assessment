# ChronicleAI Documentation

ChronicleAI is a proof-of-concept clinical narrative intelligence platform designed to transform longitudinal clinical documentation into explainable patient journeys.

## Included Diagrams

### architecture.png

End-to-end system architecture illustrating data ingestion, trajectory analysis, evidence attribution, LLM synthesis, and visualization components.

### workflow.png

High-level clinical workflow showing how sequential encounters are transformed into longitudinal intelligence.

### llm_pipeline.png

Hybrid NLP and Large Language Model pipeline used for summarization, reasoning, and evidence-backed clinical insights.

## Core Components

### Data Layer

Synthetic patient encounter datasets representing cardiac disease, diabetes management, and heart failure progression.

### Change Detection Engine

Identifies:

- emerging findings
- resolved symptoms
- medication changes
- diagnostics

### Trajectory Engine

Classifies patients as:

- Improving
- Stable
- Worsening

### Care Gap Detector

Evaluates expected follow-up activities against documented care pathways.

### LLM Layer

Generates:

- longitudinal summaries
- trajectory rationale
- clinical signals
- explainable narratives

### Evidence Attribution

Links generated conclusions back to supporting encounter documentation.

### User Interface

Streamlit dashboard providing:

- patient timeline
- summary view
- supporting evidence
- care gaps
- longitudinal insights