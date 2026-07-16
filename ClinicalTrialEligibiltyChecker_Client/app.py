# app.py — Clinical Trial Eligibility Checker (client)
#
# Streamlit front-end for the eligibility-checking service. This module owns
# zero model logic: every prediction request is delegated to the FastAPI
# backend (server.py) over HTTP, and this file only builds requests, renders
# JSON responses, and manages UI/session state. Two workflows are exposed:
#   1. Single Check   -> one patient description vs one trial text, synchronous.
#   2. Batch Evaluation -> N patients x M trials, submitted as an async job on
#      the server, polled here until completion.

import streamlit as st
import requests
import json
import time
import pandas as pd
from urllib.parse import urljoin
from io import StringIO
from datetime import datetime


def highlight_response(val):
    """
    Pandas Styler callback (used with df.style.applymap) that maps a
    NEUREQ ternary answer string to an inline CSS style string so the
    rendered eligibility table is color-coded:
        YES -> green highlight, NO -> red highlight, NA -> grey highlight.
    Any other/unexpected value falls through with no styling applied.
    """
    if val == "YES":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif val == "NO":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
    elif val == "NA":
        return "background-color: #e2e3e5; color: #383d41; font-weight: bold;"
    return ""


# ----------------- Load config -----------------
# Reads deployment-specific settings (currently just the backend base URL)
# from a local config.json sitting next to this script. Failure here is
# fatal for the whole app (no server URL == nothing else can work), so we
# surface the error via st.error and halt execution with st.stop().
try:
    config = json.load(open("config.json"))
    SERVER_URL = config["server_url"]
except Exception as e:
    st.error(f"Failed to load config.json: {e}")
    st.stop()

# Normalize the configured server URL into a bare API root, so every
# endpoint below can be built by joining a fixed relative path onto `base`
# regardless of whether config.json pointed at ".../predict" or the root.
base = SERVER_URL.rstrip("/")
if base.endswith("/predict"):
    base = base[: -len("/predict")]

# ----------------- Constants -----------------
# Hard upper bounds enforced client-side before a batch job is submitted,
# mirroring/protecting the batch endpoint from oversized payloads.
MAX_PATIENTS = 5
MAX_TRIALS = 50

# ----------------- Page setup -----------------
# Streamlit page-level configuration: browser tab title/icon and a wide
# (full-width) layout, since the UI uses multi-column layouts throughout.
st.set_page_config(
    page_title="Clinical Trial Eligibility Checker",
    page_icon="assets/logo.png",
    layout="wide"
)

# ----------------- Connection utils -----------------
def check_conn():
    """
    Lightweight liveness probe: hits the FastAPI auto-generated /docs page
    on the backend with a short timeout. Returns True only if the request
    completes without raising (status code is not checked, so a reachable
    server that returns any HTTP response, including an error page, still
    counts as "connected" — this is purely a network/socket reachability
    check, not a health check of the loaded models).
    """
    try:
        requests.get(urljoin(base + "/", "docs"), timeout=5)
        return True
    except Exception:
        return False

# Session-scoped connection state, initialized once per browser session.
if "connected" not in st.session_state:
    st.session_state.connected = False
if "last_checked" not in st.session_state:
    st.session_state.last_checked = None

def refresh_connection():
    """Re-run the connectivity probe and stamp the wall-clock time it ran."""
    st.session_state.connected = check_conn()
    st.session_state.last_checked = datetime.now().strftime("%H:%M:%S")

# Run one connectivity check automatically on first render of a fresh session.
if st.session_state.last_checked is None:
    refresh_connection()

# ----------------- Header -----------------
# Three-column header: logo | title | connection status + refresh control.
# Column width ratios are tuned so the title dominates and the status badge
# stays compact on the right edge.
header_col1, header_col2, header_col3 = st.columns(
    [0.6, 6.8, 1.6], vertical_alignment="center"
)

with header_col1:
    st.image("assets/logo.png", width=100)

with header_col2:
    st.markdown(
        "<h1 style='margin-bottom:0; margin-left:-16px;'>Clinical Trial Eligibility Checker</h1>",
        unsafe_allow_html=True
    )

with header_col3:
    # Render a green "Connected" or red "Disconnected" pill based on the
    # last connectivity probe result stored in session state.
    if st.session_state.connected:
        st.markdown(
            "<div style='padding:6px 10px; background:#d4edda; color:#155724; "
            "border-radius:6px; font-weight:600; text-align:center;'>"
            "🟢 Connected</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding:6px 10px; background:#f8d7da; color:#721c24; "
            "border-radius:6px; font-weight:600; text-align:center;'>"
            "🔴 Disconnected</div>",
            unsafe_allow_html=True
        )

    st.caption(
        f"Last checked: {st.session_state.last_checked}"
        if st.session_state.last_checked else ""
    )

    # Manual refresh button: re-probes connectivity, then forces a full
    # Streamlit script rerun so the badge/caption reflect the new state.
    if st.button("🔄 Refresh", width="stretch"):
        refresh_connection()
        st.rerun()

# ----------------- Method -----------------
# Model selector shared by both the single-check and batch workflows below.
# Value drives which backend endpoint is called and how the response is
# rendered (NEUREQ -> per-question table, TCH_CLF -> free-text reasoning).
METHODS = ["NEUREQ", "TCH_CLF"]
method = st.selectbox("Method", METHODS)

if not st.session_state.connected:
    st.error("Cannot reach hospital server. Please refresh or contact IT.")

# =====================================================================
# 🔹 SINGLE CHECK
# =====================================================================
# Ad-hoc, synchronous evaluation of exactly one (patient, trial) pair.
# Useful for interactive exploration/debugging of a single case without
# going through the batch upload flow below.

st.markdown("## Single Trial Check")

left, right = st.columns(2)
with left:
    patient_text = st.text_area("Patient Case Description", height=250)
with right:
    trial_text = st.text_area("Trial Text", height=250)

if st.button("Check Eligibility", type="primary", width="stretch"):
    # Guard: don't fire a request if the backend is known to be unreachable.
    if not st.session_state.connected:
        st.error("Server not connected")
        st.stop()

    # Guard: both free-text inputs are required for a meaningful prediction.
    if not patient_text.strip() or not trial_text.strip():
        st.error("Both fields required")
    else:
        # Route to the model-specific endpoint selected above.
        endpoint = "/predict/neureq" if method == "NEUREQ" else "/predict/tch_clf"
        payload = {
            "query": patient_text,
            "trial": trial_text,
            "generate_reasoning": True   # only consumed by /predict/tch_clf
        }

        # Blocking POST — the spinner covers LLM generation + model scoring
        # time on the server, which can be non-trivial (large-model inference).
        with st.spinner("Analyzing..."):
            res = requests.post(urljoin(base + "/", endpoint), json=payload).json()

        st.markdown(f"### Score: **{res['score']:.4f}**")

        # TCH_CLF response: render the free-form LLM-generated reasoning text.
        if method == "TCH_CLF":
            st.markdown("#### Model Reasoning")
            st.write(res.get("reasoning", ""))

        # NEUREQ response: reconstruct a per-question table from the
        # parallel `questions` list and `cleaned_answers` dict (keyed by
        # 1-based question index as a string) returned by the server, then
        # apply the color-coded styling defined in highlight_response().
        if method == "NEUREQ":
            rows = []
            for i, q in enumerate(res["questions"], start=1):
                a = res["cleaned_answers"][str(i)]
                rows.append({
                    "Q#": i,
                    "Question": q,
                    "Response": a["response"],
                    "Justification": a["justification"]
                })
            df = pd.DataFrame(rows)
            styled_df = df.style.applymap(highlight_response, subset=["Response"])
            st.dataframe(styled_df, width="stretch", hide_index=True)
            st.caption("NA = Information not available or cannot be determined from the trial or patient description.")

# =====================================================================
# 🔹 BATCH MODE
# =====================================================================
# Bulk evaluation: upload a TSV of patients and a set of trial JSON files,
# submit them as one batch job to the server, and poll for progress until
# the job completes. Designed for evaluating many patient/trial
# combinations without re-submitting single requests one at a time.

st.markdown("---")
st.markdown("## Batch Evaluation")

# File uploaders — types are restricted at the widget level (.tsv / .json)
# as a first line of defense; row/file count limits are enforced below.
patient_file = st.file_uploader("Upload Patient TSV (max 5 patients)", type=["tsv"])
trial_files = st.file_uploader(
    "Upload Trial JSON files (max 50 trials)",
    type=["json"],
    accept_multiple_files=True
)

# Score cutoff used to bucket each (patient, trial) result into
# eligible / non-eligible in the results table below.
threshold = st.slider("Eligibility Threshold", 0.0, 1.0, 0.5, 0.05)

# Session-scoped state for batch results and the trial-detail popup,
# so they survive Streamlit's rerun-per-interaction execution model.
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "popup" not in st.session_state:
    st.session_state.popup = None

if st.button("Run Batch Evaluation", type="primary", width="stretch"):

    if not st.session_state.connected:
        st.error("Server not connected")
        st.stop()

    if not patient_file or not trial_files:
        st.error("Both patient TSV and trial JSON files are required")
        st.stop()

    # ---- Validate patient TSV ----
    # Expected format: headerless TSV, exactly two columns
    # (patient_id, patient_text), one row per patient.
    patients_df = pd.read_csv(
    StringIO(patient_file.getvalue().decode()),
    sep="\t",
    header=None,
    names=["patient_id", "patient_text"]
    )

    # Validate row count
    if len(patients_df) > MAX_PATIENTS:
        st.error(f"Maximum {MAX_PATIENTS} patient cases allowed per batch")
        st.stop()

    # Validate content
    if patients_df.isnull().any().any():
        st.error("Patient TSV contains empty fields")
        st.stop()

    # Convert the DataFrame into the list-of-dicts shape expected by the
    # /predict/batch request schema (PatientCase objects server-side).
    patients = [
        {
            "patient_id": str(row["patient_id"]),
            "patient_text": row["patient_text"]
        }
        for _, row in patients_df.iterrows()
    ]

    # ---- Validate trial JSON files ----
    if len(trial_files) > MAX_TRIALS:
        st.error(f"Maximum {MAX_TRIALS} trial files allowed per batch")
        st.stop()

    # Each uploaded file must be a JSON object with "trial_id" and
    # "trial_text" keys. Any malformed file aborts the whole batch submit.
    trials = []
    for f in trial_files:
        data = json.load(f)
        if "trial_id" not in data or "trial_text" not in data:
            st.error("Each trial JSON must contain 'trial_id' and 'trial_text'")
            st.stop()
        trials.append({
            "trial_id": data["trial_id"],
            "trial_text": data["trial_text"]
        })

    # Request body for the batch endpoint: method + threshold apply to
    # every (patient, trial) pair evaluated in this job.
    payload = {
        "method": method,
        "threshold": threshold,
        "patients": patients,
        "trials": trials,
        "generate_reasoning": True
    }

    # ---- Start batch ----
    # Kicks off the batch job server-side. Note: predict_batch() on the
    # server currently runs synchronously and returns the full result set
    # in this same response (see server.py); the batch_id returned here is
    # still used below to poll /predict/batch/status for a progress view.
    start_res = requests.post(
        urljoin(base + "/", "/predict/batch"),
        json=payload
    ).json()

    batch_id = start_res["batch_id"]
    st.session_state.batch_id = batch_id

    # Placeholders that get overwritten in-place on each poll iteration
    # instead of re-rendering the whole page (avoids flicker/duplication).
    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    table_placeholder = st.empty()

    # ---- Poll batch progress ----
    # Simple 1-second polling loop against the status endpoint. Loop exits
    # once the server reports status == "completed".
    while True:
        status = requests.get(
            urljoin(base + "/", f"/predict/batch/status/{batch_id}")
        ).json()

        if status["status"] == "completed":
            break

        pid = status["current_patient"]
        total = status["total_trials"]

        if pid:
            done = status["current_trial_index"].get(pid, 0)

            progress_text.markdown(
                f"**Patient {pid}: {done} / {total} trials processed**"
            )

            if total > 0:
                progress_bar.progress(min(done / total, 1.0))


        # live table update — re-render the eligible/non-eligible trial
        # lists per patient using whatever partial results are available
        # so far in this poll response.
        rows = []
        for p, data in status["results"].items():
            rows.append({
                "Patient Case": p,
                "Eligible Trials": ", ".join(data["eligible_trials"]),
                "Non-Eligible Trials": ", ".join(data["non_eligible_trials"])
            })

        if rows:
            table_placeholder.dataframe(
                pd.DataFrame(rows),
                width="stretch"
            )

        time.sleep(1)

    # ---- Final results ----
    # Persist the completed batch results into session state so the
    # results section below renders on every subsequent script rerun
    # (e.g. when a user clicks a trial button), not just this one pass.
    st.session_state.batch_results = status["results"]
    progress_bar.progress(1.0)
    progress_text.markdown("✅ **Batch evaluation completed**")

# =====================================================================
# 🔹 RESULTS TABLE
# =====================================================================
# Renders whenever batch_results is populated in session state — either
# right after a batch just finished, or on a later rerun triggered by
# interacting with the expander/buttons below.

if st.session_state.batch_results:
    st.markdown("## Results")

    # Summary table: one row per patient, eligible/non-eligible trial IDs
    # joined into comma-separated strings for compact display.
    rows = []
    for pid, data in st.session_state.batch_results.items():
        rows.append({
            "Patient Case": pid,
            "Eligible Trials": ", ".join(data["eligible_trials"]),
            "Non-Eligible Trials": ", ".join(data["non_eligible_trials"])
        })

    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.markdown("### Click a trial ID to view details")

    # Per-patient expander containing a clickable button for every trial
    # (both eligible and non-eligible). Clicking a trial button stores the
    # (patient_id, trial_id, method) selection in session_state.popup,
    # which triggers the detail panel rendered in the next section.
    for pid, data in st.session_state.batch_results.items():
        with st.expander(f"Patient {pid}"):
            for group, trials in [
                ("Eligible", data["eligible_trials"]),
                ("Non-Eligible", data["non_eligible_trials"])
            ]:
                st.markdown(f"**{group} Trials**")
                for tid in trials:
                    if st.button(tid, key=f"{pid}_{tid}"):
                        st.session_state.popup = {
                            "patient_id": pid,
                            "trial_id": tid,
                            "method": method
                        }

# =====================================================================
# 🔹 POPUP
# =====================================================================
# Detail view for a single (patient, trial) pair selected from the batch
# results above. Fetches the cached per-pair audit data from the server
# (no re-inference — this hits the batch/details endpoint which reads
# from the audit logs written during the original batch run).

if st.session_state.popup:
    st.markdown("---")
    st.markdown("## Trial Details")

    info = st.session_state.popup
    st.markdown(f"**Patient:** {info['patient_id']}")
    st.markdown(f"**Trial:** {info['trial_id']}")

    # Pull the full eligibility breakdown / reasoning for this specific
    # (batch_id, patient_id, trial_id) triple from the server's audit log.
    details = requests.get(
    urljoin(
        base + "/",
        f"/predict/batch/details/"
        f"{st.session_state.batch_id}/"
        f"{info['patient_id']}/"
        f"{info['trial_id']}"
        )
    ).json()

    # NEUREQ detail: same per-question table rendering as the single-check
    # flow above, sourced from the batch audit log instead of a live call.
    if details["method"] == "NEUREQ":
        st.markdown("### NEUREQ Eligibility Breakdown")

        rows = []
        for i, q in enumerate(details["neureq"]["questions"], start=1):
            a = details["neureq"]["cleaned_answers"][str(i)]
            rows.append({
                "Q#": i,
                "Question": q,
                "Response": a["response"],
                "Justification": a["justification"]
            })

        df = pd.DataFrame(rows)
        styled_df = df.style.applymap(highlight_response, subset=["Response"])
        st.dataframe(styled_df, width="stretch", hide_index=True)
        st.caption("NA = Information not available or cannot be determined from the trial or patient description.")
        st.success(f"Final Score: {details['neureq']['score']:.4f}")

    # TCH_CLF detail: score + stored free-text reasoning paragraph.
    elif details["method"] == "TCH_CLF":
        st.markdown("### TCH_CLF Reasoning")
        st.success(f"Final Score: {details['tch_clf']['score']:.4f}")
        st.write(details["tch_clf"]["reasoning"])


    if st.button("Close"):
        st.session_state.popup = None

st.caption("Clinical Trial Eligibility Checker v1.0 | Proprietary Software")