Clinical Trial Eligibility Checker — NEUREQ + TCH_CLF Inference Server
=====================================================

This directory contains the server-side inference component of the
Clinical Trial Eligibility Checker system, implemented as a unified 
FastAPI service for clinical trial eligibility assessment.

The server exposes REST APIs for two complementary eligibility models:

1. NEUREQ
   A large language model–driven, question-based eligibility modeling
   pipeline followed by a neural eligibility re-ranking model.

2. TCH_CLF
   A teacher Longformer-based clinical trial reranker with optional
   deterministic large language model reasoning.

The server supports both single patient–trial evaluation and batch
evaluation across multiple patients and trials, with full audit logging
to support reproducibility, traceability, and clinical inspection.


FEATURES
--------
- Unified REST API for NEUREQ and TCH_CLF inference
- Deterministic inference using stable random seeds
- GPU-safe sequential batch execution
- Live batch progress tracking for client applications
- Cached audit logs for post-hoc inspection
- Clinician-facing explanations and justifications
- Explicit separation between inference and UI layers


API ENDPOINTS
-------------

Single Evaluation
-----------------
POST /predict/neureq
    Runs the NEUREQ pipeline:
    LLM prompt -> 10-question eligibility schema -> NEUREQ neural model.

POST /predict/tch_clf
    Runs the teacher Longformer reranker with optional
    deterministic LLM-based reasoning.


Batch Evaluation
----------------
POST /predict/batch
    Evaluates multiple patients against multiple trials sequentially.

GET /predict/batch/status/{batch_id}
    Returns live progress updates and partial results for an
    ongoing batch evaluation.

GET /predict/batch/details/{batch_id}/{patient_id}/{trial_id}
    Returns cached per-trial explanations (NEUREQ or TCH_CLF)
    without recomputation.


MODEL OVERVIEW
--------------

NEUREQ
------
- Uses a fixed 10-question eligibility schema.
- Each question has a ternary response: YES, NO, or NA.
- Each response includes a textual justification.
- Responses are mapped to a numerical vector.
- Final relevance score is produced by a trained LSTM-based model.

TCH_CLF
-------
- Uses a Clinical Longformer encoder.
- Extracts structured trial attributes such as age, gender,
  conditions, and eligibility criteria.
- Produces a relevance score and optional deterministic reasoning.


AUDIT LOGGING
-------------
All evaluations generate structured JSON audit logs containing:
- Input patient and trial text
- Extracted and normalized trial fields
- Intermediate representations
- Final relevance scores
- Model explanations and reasoning
- Deterministic seeds used during inference

Logs are organized under:

```text
audit_logs/
  neureq/
  tch_clf/
  batch/
```

Batch evaluation logs reference the corresponding single-evaluation
audit files to ensure full traceability.


CONFIGURATION
-------------
Model paths and runtime settings are defined at the top of server.py,
including:

- Large language model name
- NEUREQ model checkpoint path
- Teacher model checkpoint path
- Prompt template file path


RUNNING THE SERVER
------------------
Start the server using:

uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
or run "bash run_server.sh"

A single worker is recommended to preserve GPU safety and
deterministic execution.


NOTES
-----
- Batch execution is strictly sequential to avoid GPU contention.
- All batch explanations are served from cached audit logs.
- The server is designed to be consumed by an external
  clinician-facing client application.


LICENSE
-------
Internal research use only.
Contact the authors for redistribution or deployment permissions.
