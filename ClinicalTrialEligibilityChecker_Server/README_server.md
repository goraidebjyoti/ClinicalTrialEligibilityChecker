# server.py

FastAPI inference server exposing two independent patient-trial scoring
pipelines — **NEUREQ** and **TCH_CLF** — plus a batch-evaluation layer built
on top of them. This process owns every model in the system (an LLM, a
frozen sentence encoder, and two task-specific classifier heads); the
companion Streamlit client (`app.py`) only talks to this server over HTTP
and performs no inference itself.

Run with:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```
`--workers 1` is required. All models are loaded once into process memory
as module-level globals and batch requests are processed strictly
sequentially to avoid concurrent GPU access; running multiple workers would
load duplicate model copies and break the sequential-batch guarantee.

## What each pipeline does

**NEUREQ** (`POST /predict/neureq`)
1. Formats a prompt (from `prompt.txt`) asking the LLM to answer 10 fixed
   yes/no/NA eligibility questions about a (patient, trial) pair, each with
   a short justification.
2. Runs the LLM with greedy, deterministic decoding (seeded from a SHA-256
   hash of the prompt text, so identical inputs always produce identical
   output).
3. Parses the LLM's JSON output into a normalized 10-question answer
   structure, tolerating malformed/missing entries.
4. Encodes each justification string with a frozen Bio_ClinicalBERT model,
   embeds the question id and ternary answer, and feeds the resulting
   10-step sequence through a BiLSTM + additive-attention classifier
   (`EligibilityBiLSTM`) to produce a single relevance score in `[0, 1]`.

**TCH_CLF** (`POST /predict/tch_clf`)
1. Rule-parses the raw trial text (regex, header-anchored) into structured
   fields: title, brief summary, conditions, gender, min/max age,
   eligibility criteria.
2. Concatenates the extracted fields into one normalized trial string.
3. Tokenizes `(patient_query, concatenated_trial_text)` as a sentence pair
   and scores it with `TeacherReranker`, a classifier head on top of a
   Clinical-Longformer encoder, producing a relevance score in `[0, 1]`.
4. Optionally (`generate_reasoning=True`) also prompts the LLM for a
   free-text explanation paragraph. This text is returned for display only
   and does **not** affect the score computed in step 3.

**Batch mode** (`POST /predict/batch`) runs either pipeline over the full
cross-product of a list of patients and a list of trials, sequentially,
writing progress to an in-memory dict pollable via
`GET /predict/batch/status/{batch_id}` and per-pair audit records readable
via `GET /predict/batch/details/{batch_id}/{patient_id}/{trial_id}`.

## Configuration

Edit the constants at the top of `server.py` before running:

| Constant | Purpose |
|---|---|
| `LLM_MODEL_NAME` | HuggingFace model id for the LLM used by both pipelines (`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`) |
| `NEUREQ_STATE` | Path to the trained `EligibilityBiLSTM` weights (`.pt` state dict) |
| `CLINICAL_BERT_MODEL` | Frozen encoder (`emilyalsentzer/Bio_ClinicalBERT`) used to embed NEUREQ justification text |
| `PROMPT_FILE` | Path to the NEUREQ prompt template (plain text with two `{}`-style format slots: patient text, trial text) |
| `TEACHER_MODEL_PATH` | Path to the trained `TeacherReranker` weights (`.pt` state dict) |
| `TEACHER_MODEL_NAME` | Base checkpoint (`yikuan8/Clinical-Longformer`) used for the TCH_CLF encoder and tokenizer |
| `LOG_DIR` / `NEUREQ_LOG_DIR` / `TCH_LOG_DIR` / `BATCH_LOG_DIR` | Root and per-pipeline subdirectories for audit-log output |
| `MAX_NEW_TOKENS` | Generation cap for NEUREQ's structured JSON answer |
| `REASONING_MAX_TOKENS` | Generation cap for TCH_CLF's free-text reasoning paragraph |
| `LLM_LOADED_AT_STARTUP` | If `True`, the LLM loads eagerly at process startup; if `False`, it is lazy-loaded on first use |

`NEUREQ_STATE` and `TEACHER_MODEL_PATH` must point at valid local
checkpoint files, and `PROMPT_FILE` must exist (a minimal fallback template
is used if it's missing, but with degraded prompt quality).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/neureq` | Score one (patient, trial) pair with NEUREQ |
| `POST` | `/predict/tch_clf` | Score one (patient, trial) pair with TCH_CLF |
| `POST` | `/predict/batch` | Score a cross-product of patients × trials with either method |
| `GET` | `/predict/batch/status/{batch_id}` | Poll live progress of a batch job |
| `GET` | `/predict/batch/details/{batch_id}/{patient_id}/{trial_id}` | Fetch the full breakdown for one already-evaluated pair |
| `GET` | `/docs` | Auto-generated FastAPI/OpenAPI docs (also used by the client as a lightweight liveness probe) |

### `POST /predict/neureq`

**Input (JSON body)**
```json
{
  "query": "<patient case description, free text>",
  "trial": "<clinical trial text, free text>"
}
```

**Output (JSON)**
```json
{
  "score": 0.8421,
  "seed": 3924871045,
  "raw_llm_output": "<unparsed LLM generation>",
  "cleaned_answers": {
    "1": {"response": "YES", "justification": "..."},
    "2": {"response": "NO",  "justification": "..."},
    "...": "... up to key \"10\""
  },
  "answer_ids": [0, 1, 2, 0, 0, 1, 2, 0, 1, 0],
  "final_score": 0.8421,
  "questions": ["Age Eligibility - ...", "... 10 items total"],
  "audit_log": "audit_logs/neureq/case_20260716_101530_123456.json"
}
```
`answer_ids` are the integer indices used for the model's answer
embedding lookup: `YES=0, NO=1, NA=2`.

### `POST /predict/tch_clf`

**Input (JSON body)**
```json
{
  "query": "<patient case description, free text>",
  "trial": "<clinical trial text, free text>",
  "generate_reasoning": true
}
```
`generate_reasoning` is optional (defaults to `true`); set to `false` to
skip the LLM reasoning step and only compute the classifier score.

**Output (JSON)**
```json
{
  "score": 0.7213,
  "reasoning": "<free-text explanation, empty string if generate_reasoning=false>",
  "seed": 1029384756,
  "extracted": {
    "id": "NCT01234567",
    "study_title": "...",
    "brief_summary": "...",
    "conditions": "...",
    "gender": "male and female",
    "min_age": 18.0,
    "max_age": 65.0,
    "eligibility": {"criteria": "..."}
  },
  "concatenated_text": "study_title: ... || brief_summary: ... || ...",
  "found_headers": ["Study Title", "Conditions", "..."],
  "audit_log": "audit_logs/tch_clf/case_20260716_101530_654321.json"
}
```

### `POST /predict/batch`

**Input (JSON body)**
```json
{
  "method": "NEUREQ",
  "threshold": 0.5,
  "generate_reasoning": true,
  "patients": [
    {"patient_id": "P001", "patient_text": "..."},
    {"patient_id": "P002", "patient_text": "..."}
  ],
  "trials": [
    {"trial_id": "T001", "trial_text": "..."},
    {"trial_id": "T002", "trial_text": "..."}
  ]
}
```
`method` must be `"NEUREQ"` or `"TCH_CLF"`. Every patient is evaluated
against every trial (a full cross-product).

**Output (JSON)** — returned only after the entire batch has finished
running:
```json
{
  "batch_id": "batch_20260716_101530_111111",
  "status": "completed",
  "method": "NEUREQ",
  "threshold": 0.5,
  "results": {
    "P001": {"eligible_trials": ["T001"], "non_eligible_trials": ["T002"]},
    "P002": {"eligible_trials": [],       "non_eligible_trials": ["T001", "T002"]}
  }
}
```

### `GET /predict/batch/status/{batch_id}`

Returns the live in-memory progress record, e.g. while a batch is still
running:
```json
{
  "status": "running",
  "current_patient": "P002",
  "current_trial_index": {"P001": 2, "P002": 1},
  "total_trials": 2,
  "results": {
    "P001": {"eligible_trials": ["T001"], "non_eligible_trials": ["T002"]},
    "P002": {"eligible_trials": [], "non_eligible_trials": []}
  }
}
```
`404` if `batch_id` is unknown (never submitted, or the server process
restarted since — this state is in-memory only).

### `GET /predict/batch/details/{batch_id}/{patient_id}/{trial_id}`

Returns the full per-pair record for one already-evaluated combination,
read back from disk (no re-inference). Shape depends on `method`:
```json
{
  "method": "NEUREQ",
  "patient_id": "P001",
  "trial_id": "T001",
  "neureq": {
    "score": 0.8421,
    "questions": ["...", "... 10 items"],
    "cleaned_answers": {"1": {"response": "YES", "justification": "..."}, "...": "..."}
  }
}
```
or, for `TCH_CLF`:
```json
{
  "method": "TCH_CLF",
  "patient_id": "P001",
  "trial_id": "T001",
  "tch_clf": {
    "score": 0.7213,
    "reasoning": "...",
    "extracted": {"study_title": "...", "...": "..."}
  }
}
```
`404` if the batch/patient/trial combination has no matching log file on
disk.

## Output files (audit logs)

Every scoring call writes a structured JSON record to disk, in addition to
whatever it returns over HTTP:

- `audit_logs/neureq/case_<timestamp>.json` — one file per `/predict/neureq`
  call: patient/trial text, raw LLM output, seed, parsed answers, question
  bank, final score.
- `audit_logs/tch_clf/case_<timestamp>.json` — one file per `/predict/tch_clf`
  call: patient/trial text (raw and normalized), extracted trial fields,
  concatenated model input, teacher score, reasoning text, seed.
- `audit_logs/batch/<batch_id>/<patient_id>/<trial_id>.json` — one file per
  (patient, trial) pair inside a batch run, containing the score, threshold,
  timestamp, and a pointer (`audit_log`) to the corresponding richer
  per-pair record in `neureq/` or `tch_clf/` above.

`batch_id` and `case_id` values are timestamp-derived strings
(`batch_YYYYMMDD_HHMMSS_ffffff` / `case_YYYYMMDD_HHMMSS_ffffff`), unique
per process run down to microsecond resolution.

## Dependencies

`fastapi`, `pydantic`, `torch`, `transformers` (and a BitsAndBytes-compatible
environment for `BitsAndBytesConfig`, even though 4-bit/8-bit quantization
is disabled by default in this file). Model weights for `NEUREQ_STATE` and
`TEACHER_MODEL_PATH` must be present locally; `LLM_MODEL_NAME`,
`CLINICAL_BERT_MODEL`, and `TEACHER_MODEL_NAME` are pulled from the
HuggingFace Hub on first load unless cached locally.
