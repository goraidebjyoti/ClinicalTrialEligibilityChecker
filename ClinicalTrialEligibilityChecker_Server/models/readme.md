Models Directory
================

This directory is expected to contain all trained model checkpoints used by
the ClinicalTrialEligibilityChecker inference server.

Due to file size constraints, model checkpoint files are not stored directly
in this repository. They must be downloaded separately and placed in this
directory before running the server.


REQUIRED MODEL FILES
--------------------

model_epoch_12.pt
-----------------
- Trained neural eligibility re-ranking model used in NEUREQ Phase 2.
- Implements a BiLSTM-based sequence model with additive attention.
- Operates over a sequence of 10 eligibility questions, where each question is
  represented using:
    - Question position embeddings
    - Categorical answer embeddings (YES / NO / NA)
    - Justification embeddings encoded using a frozen ClinicalBERT model
- The BiLSTM captures interactions across eligibility dimensions, while the
  attention mechanism focuses on the most decisive criteria.
- Outputs a scalar relevance score (logit) used for re-ranking clinical trials.

best_teacher_alpha0.2.pt
------------------------
- Trained teacher reranker model used by the TCH_CLF pipeline.
- Built on top of a Clinical Longformer encoder.
- Produces a relevance score between a patient case and a clinical trial.
- Designed to align with the offline teacher model used during training.


MODEL DOWNLOAD
--------------
The official trained model checkpoints are hosted externally due to size
limitations.

Download links:
- Google Drive: <[LINK HERE](https://drive.google.com/drive/folders/1ILcOmIajQxNttBvBqRX1CFtTA4V9KVug?usp=sharing)>

After downloading, place the files in this directory:

ClinicalTrialEligibilityChecker_Server/models/

Ensure that filenames exactly match those expected by the server.


MODEL USAGE
-----------
- Models are loaded by the server at runtime.
- The NEUREQ model is loaded at server startup (CPU-safe and lightweight).
- The TCH_CLF model is lazy-loaded on first use to conserve memory and GPU
  resources.
- Model paths are defined in server.py and should not be changed unless
  filenames are updated accordingly.


COMPATIBILITY
-------------
- The NEUREQ model expects a 10 × 1 input tensor corresponding to ternary
  eligibility responses.
- The TCH_CLF model expects tokenized input compatible with the Clinical
  Longformer tokenizer.
- Model checkpoints are architecture-specific and not interchangeable.


REPRODUCIBILITY
---------------
- These checkpoints correspond to fixed training runs.
- Inference is deterministic when used with fixed random seeds.
- Any change to these files may alter scoring behavior and should be
  documented.


NOTES
-----
- Do not rename model files without updating paths in server.py.
- Do not modify model files unless retraining is intended.
- For archival or versioned experiments, copy this directory rather than
  overwriting existing checkpoints.


LICENSE
-------
Internal research artifacts.
Not for redistribution without permission.
