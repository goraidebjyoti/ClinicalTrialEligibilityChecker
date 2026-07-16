# app.py

Streamlit front-end for the patient–trial eligibility service. This file
contains no model logic — it builds HTTP requests against the FastAPI
backend (`server.py`) and renders whatever JSON comes back. It supports two
workflows: a synchronous single-pair check, and a batch job (multiple
patients × multiple trial files) with live progress polling and a
per-result detail popup.

Run with:
```bash
streamlit run app.py
```

## Requirements to run

- A running instance of the companion `server.py` FastAPI server, reachable
  over HTTP from wherever this Streamlit app runs.
- A `config.json` file in the same working directory as `app.py`.
- An `assets/logo.png` image file in the same working directory (used as
  both the browser tab icon and the header logo).

## Input files

### `config.json` (required, read once at startup)

```json
{ "server_url": "http://<server-host>:8000/predict" }
```
Only `server_url` is read. A bare API root (`http://host:8000`) is also
accepted — the app strips a trailing `/predict` if present so every
endpoint can be built consistently by joining a relative path onto the same
base URL. If this file is missing or malformed, the app shows an error and
halts (`st.stop()`).

### `assets/logo.png` (required)

Any standard image file; used as the header logo and page favicon.

### Single Trial Check — inputs (typed directly into the UI, no file)

- **Patient Case Description** — free text, patient demographic/clinical
  summary.
- **Trial Text** — free text, the clinical trial protocol/eligibility
  section. For best results with the TCH_CLF method, this text should
  contain labeled fields recognized by the server's parser, e.g.:
  ```
  Study Title: ...
  Brief Summary: ...
  Conditions: ...
  Gender: ...
  Minimum Age: ...
  Maximum Age: ...
  Eligibility Criteria: ...
  ```

### Batch Evaluation — uploaded files

**Patient TSV** (`.tsv`, max 5 rows) — headerless, tab-separated, exactly
two columns per row:
```
P001<TAB>Patient description text for P001...
P002<TAB>Patient description text for P002...
```
Columns are read positionally as `patient_id`, `patient_text`. Any row with
an empty field, or more than 5 rows total, is rejected client-side before
submission.

**Trial JSON files** (`.json`, max 50 files, multiple files uploaded at
once) — each file must be a single JSON object with these required keys:
```json
{
  "trial_id": "T001",
  "trial_text": "Study Title: ...\nEligibility Criteria: ..."
}
```
Any file missing either key, or more than 50 files total, is rejected
client-side before submission.

**Eligibility Threshold** — a slider (0.0–1.0, step 0.05, default 0.5)
controlling the score cutoff used to bucket each (patient, trial) result
into eligible vs. non-eligible.

## What the UI does

1. **Header** — logo, title, and a connectivity badge (🟢/🔴) based on a
   lightweight `GET /docs` probe against the server, with a manual
   **Refresh** button.
2. **Method selector** — `NEUREQ` or `TCH_CLF`, shared by both workflows
   below; determines which backend endpoint is called and how results are
   rendered.
3. **Single Trial Check** — two text areas (patient / trial) and a
   **Check Eligibility** button. Sends one request to
   `/predict/neureq` or `/predict/tch_clf` and renders:
   - `TCH_CLF`: the score and the free-text model reasoning.
   - `NEUREQ`: the score and a 10-row table (question, response, justification)
     with cell background colors (green=YES, red=NO, grey=NA).
4. **Batch Evaluation** — file uploaders for the patient TSV and trial JSON
   files, a threshold slider, and a **Run Batch Evaluation** button. On
   submit: validates and converts the uploads, `POST`s to `/predict/batch`,
   then polls `GET /predict/batch/status/{batch_id}` once per second,
   updating a progress bar and a live per-patient eligible/non-eligible
   table until the job reports `"completed"`.
5. **Results** — once a batch finishes, a summary table (one row per
   patient) plus an expander per patient listing every trial as a clickable
   button, split into Eligible / Non-Eligible groups.
6. **Trial Details popup** — clicking a trial button fetches
   `GET /predict/batch/details/{batch_id}/{patient_id}/{trial_id}` and
   displays either the NEUREQ per-question breakdown or the TCH_CLF score +
   reasoning for that specific pair, with a **Close** button to dismiss it.

## Output

This app does not write any files to disk. All output is rendered directly
in the browser: score values, the NEUREQ per-question table, TCH_CLF
reasoning text, the batch progress bar/table, and the final results
summary/detail popup. Any persistent record of a session's results lives
only in the backend server's audit logs (see `server.py`'s README), not in
this client.

## Session state

The following values are held in Streamlit's `session_state` for the
duration of a browser session (reset on full page reload):
`connected`, `last_checked`, `batch_id`, `batch_results`, `popup`.

## Dependencies

`streamlit`, `requests`, `pandas`. No model libraries (`torch`,
`transformers`, etc.) are required by this file — all inference happens on
the server side.
