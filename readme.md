# ClinicalTrialEligibilityChecker

ClinicalTrialEligibilityChecker is an end-to-end clinical trial eligibility
assessment system designed to support both automated relevance scoring and
clinician-facing interpretability.

The system integrates large language models, neural ranking models,
and a web-based user interface to enable transparent, reproducible,
and scalable patient–trial matching.

---

## System Overview

ClinicalTrialEligibilityChecker consists of two main components:

1. **ClinicalTrialEligibilityChecker_Server**  
   A FastAPI-based inference server that exposes REST APIs for
   eligibility scoring, explanation generation, and batch evaluation.

2. **ClinicalTrialEligibilityChecker_Client**  
   A Streamlit-based client application that allows clinicians
   and researchers to interact with the system through a
   user-friendly interface.

---

## Architecture

The system follows a clean client–server architecture:

- **Client (Streamlit UI)**  
  Handles user interaction, visualization, and clinician-facing workflows.

- **Server (FastAPI)**  
  Exposes REST APIs and orchestrates model inference.

- **Model Layer**  
  - NEUREQ neural eligibility model  
  - TCH_CLF teacher Longformer reranker  
  - Large Language Model for eligibility reasoning and justification

- **Audit Layer**  
  Structured JSON logs persisted for traceability and inspection.

The client and server are intentionally decoupled to allow:
- Independent deployment
- Easier auditing and debugging
- Future replacement or extension of models or UI components

---

## Models

### NEUREQ
- Question-driven eligibility modeling framework.
- Uses a fixed 10-question schema capturing core eligibility dimensions.
- Each question produces a ternary response (YES / NO / NA) with justification.
- Responses are scored using a trained neural model.

### TCH_CLF
- Teacher Longformer-based clinical trial reranker.
- Extracts structured trial attributes (age, gender, conditions, criteria).
- Produces a relevance score with optional deterministic reasoning.

### Large Language Model
- **Model:** deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- Used for:
  - Generating 10-question eligibility answers with justifications (NEUREQ)
  - Optional deterministic reasoning for TCH_CLF
- Integrated using a model-specific chat template and decoding strategy

**Note:**  
The server implementation is tightly coupled to this LLM’s chat structure and
generation behavior. Replacing the LLM will require non-trivial code changes
to prompt formatting, chat templating, and output parsing.

---

## Model Checkpoints

Trained model checkpoint files are not included in this repository due to
size constraints.

The inference server expects the required `.pt` files to be present under:

`ClinicalTrialEligibilityChecker_Server/models/`

Instructions and download links for the official model checkpoints are
provided in:

`ClinicalTrialEligibilityChecker_Server/models/readme.md`

---

## Batch Evaluation

The system supports batch evaluation where:
- Multiple patient cases are evaluated against multiple trials
- Execution is sequential and GPU-safe
- Live progress is exposed to the client
- Partial results are available before batch completion
- All explanations are served from cached audit logs

---

## Auditability and Reproducibility

ClinicalTrialEligibilityChecker emphasizes clinical transparency by:
- Logging all intermediate representations
- Storing deterministic seeds for reproducibility
- Persisting per-trial explanations
- Separating inference from presentation

All logs are stored as structured JSON files and can be inspected
independently of the client application.

---

## System Requirements

This system is designed for research and prototyping use and can be run
in both CPU-only and GPU-enabled environments, with reduced performance
in CPU-only mode.

### Operating System
- **Server:** Linux (required)
- **Client:** Windows, Linux, or macOS with a graphical environment

### Python
- Python 3.9 or newer

### Server-side (Inference)
- NVIDIA GPU strongly recommended
- Large Language Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- GPU memory requirements:
  - Reasoning and justification generation may approach or exceed **24 GB**
  - GPUs in the 24–48 GB range (e.g., NVIDIA L40) are recommended
- CPU-only execution is supported with significantly reduced performance
- Minimum **120 GB free disk space** recommended for model downloads,
  checkpoints, and audit logs

### Client-side (UI)
- Runs entirely on CPU
- Requires a modern web browser
- Streamlit-based graphical interface

Detailed dependency installation and environment setup instructions are
documented in the component-specific README files.

---

## Directory Structure

```text
ClinicalTrialEligibilityChecker/
├── ClinicalTrialEligibilityChecker_Server/
│   ├── server.py
│   ├── models/
│   ├── audit_logs/
│   └── readme.md
│
├── ClinicalTrialEligibilityChecker_Client/
│   ├── app.py
│   ├── assets/
│   ├── config.json
│   └── readme.md
│
└── readme.md
```

---

## Setup and Execution

Refer to the README files inside each subdirectory for
component-specific setup and execution instructions:

```text
ClinicalTrialEligibilityChecker_Server/readme.md
ClinicalTrialEligibilityChecker_Client/readme.md
```
---

## Intended Use

This system is intended for:
- Clinical trial eligibility research
- Model evaluation and benchmarking
- Decision-support prototyping
- Human-in-the-loop eligibility inspection

It is not intended to replace clinical judgment.

## License

Internal research use only.
Contact the authors for redistribution, clinical deployment,
or commercial use permissions.

