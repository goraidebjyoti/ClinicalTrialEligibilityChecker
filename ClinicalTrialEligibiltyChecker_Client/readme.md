Clinical Trial Eligibility Checker — Client Application
=======================================================

This directory contains the clinician-facing client application for the
Clinical Trial Eligibility Checker system.

The client is implemented using Streamlit and communicates with the
ClinicalTrialEligibilityChecker_Server via REST APIs to perform eligibility evaluation
for clinical trial matching.


SYSTEM OVERVIEW
---------------
The client supports two evaluation modes:

1. Single Trial Check
   - Evaluate one patient case against one clinical trial.
   - Display model score and explanations immediately.

2. Batch Evaluation
   - Evaluate multiple patient cases against multiple trials.
   - Progress is shown in real time.
   - Results appear incrementally as trials are processed.
   - Clinicians can inspect detailed explanations per trial
     without recomputation.


SUPPORTED METHODS
-----------------
The client supports two backend methods:

NEUREQ
- Question-driven eligibility modeling.
- Displays a 10-question eligibility table.
- Each question includes a response and justification.

TCH_CLF
- Teacher Longformer-based reranking model.
- Displays relevance score and model-generated reasoning.


CLIENT FEATURES
---------------
- Live server connectivity status indicator
- Single patient–trial evaluation
- Batch evaluation with GPU-safe sequential execution
- Dynamic progress bar with per-patient progress
- Live updating results table
- Clickable trial IDs for detailed explanations
- Popup-style detail view using cached server results
- No recomputation during explanation viewing


INPUT FORMATS
-------------

Patient TSV File
----------------
- Tab-separated file with no header.
- Each row must contain:
    patient_id<TAB>patient_case_description
- Maximum: 5 patient cases per batch.

Trial JSON Files
----------------
- Each file must contain:
   ```text
  {
      "trial_id": "<ID>",
      "trial_text": "<full trial description>"
   }
   ```
- Maximum: 50 trials per batch.


BATCH EXECUTION BEHAVIOR
-----------------------
- Trials are evaluated sequentially for each patient.
- All trials for Patient 1 are completed before moving to Patient 2.
- This design ensures GPU safety and deterministic execution.
- Partial results are shown immediately as trials complete.


DETAILED RESULTS
----------------
Clicking a trial ID opens a detailed explanation view:

For NEUREQ:
- Full 10-question eligibility table
- Responses and justifications
- Final eligibility score

For TCH_CLF:
- Final relevance score
- Model-generated reasoning text

All details are loaded from cached audit logs on the server.


CONFIGURATION
-------------
The client reads server configuration from:

config.json

Required field:
- server_url: Base URL of the ClinicalTrialEligibilityChecker_Server


RUNNING THE CLIENT
------------------
Start the client using:

streamlit run app.py

Ensure the server is running and reachable before launching
the client application.


DIRECTORY STRUCTURE
-------------------

```text
assets/
  logo.png

app.py
config.json
readme.md (this file)
```

NOTES
-----
- The client does not perform any model inference.
- All heavy computation is handled by the server.
- The client is suitable for clinician-facing evaluation workflows.


LICENSE
-------
Internal research use only.
Contact the authors for redistribution or deployment permissions.

