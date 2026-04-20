# app.py — Clinical Trial Eligibility Checker (client)

import streamlit as st
import requests
import json
import time
import pandas as pd
from urllib.parse import urljoin
from io import StringIO
from datetime import datetime

def highlight_response(val):
    if val == "YES":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif val == "NO":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
    elif val == "NA":
        return "background-color: #e2e3e5; color: #383d41; font-weight: bold;"
    return ""


# ----------------- Load config -----------------
try:
    config = json.load(open("config.json"))
    SERVER_URL = config["server_url"]
except Exception as e:
    st.error(f"Failed to load config.json: {e}")
    st.stop()

base = SERVER_URL.rstrip("/")
if base.endswith("/predict"):
    base = base[: -len("/predict")]

# ----------------- Constants -----------------
MAX_PATIENTS = 5
MAX_TRIALS = 50

# ----------------- Page setup -----------------
st.set_page_config(
    page_title="Clinical Trial Eligibility Checker",
    page_icon="assets/logo.png",
    layout="wide"
)

# ----------------- Connection utils -----------------
def check_conn():
    try:
        requests.get(urljoin(base + "/", "docs"), timeout=5)
        return True
    except Exception:
        return False

if "connected" not in st.session_state:
    st.session_state.connected = False
if "last_checked" not in st.session_state:
    st.session_state.last_checked = None

def refresh_connection():
    st.session_state.connected = check_conn()
    st.session_state.last_checked = datetime.now().strftime("%H:%M:%S")

# initial check
if st.session_state.last_checked is None:
    refresh_connection()

# ----------------- Header -----------------
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

    if st.button("🔄 Refresh", width="stretch"):
        refresh_connection()
        st.rerun()

# ----------------- Method -----------------
METHODS = ["NEUREQ", "TCH_CLF"]
method = st.selectbox("Method", METHODS)

if not st.session_state.connected:
    st.error("Cannot reach hospital server. Please refresh or contact IT.")

# =====================================================================
# 🔹 SINGLE CHECK
# =====================================================================

st.markdown("## Single Trial Check")

left, right = st.columns(2)
with left:
    patient_text = st.text_area("Patient Case Description", height=250)
with right:
    trial_text = st.text_area("Trial Text", height=250)

if st.button("Check Eligibility", type="primary", width="stretch"):
    if not st.session_state.connected:
        st.error("Server not connected")
        st.stop()

    if not patient_text.strip() or not trial_text.strip():
        st.error("Both fields required")
    else:
        endpoint = "/predict/neureq" if method == "NEUREQ" else "/predict/tch_clf"
        payload = {
            "query": patient_text,
            "trial": trial_text,
            "generate_reasoning": True
        }

        with st.spinner("Analyzing..."):
            res = requests.post(urljoin(base + "/", endpoint), json=payload).json()

        st.markdown(f"### Score: **{res['score']:.4f}**")

        if method == "TCH_CLF":
            st.markdown("#### Model Reasoning")
            st.write(res.get("reasoning", ""))

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

st.markdown("---")
st.markdown("## Batch Evaluation")

patient_file = st.file_uploader("Upload Patient TSV (max 5 patients)", type=["tsv"])
trial_files = st.file_uploader(
    "Upload Trial JSON files (max 50 trials)",
    type=["json"],
    accept_multiple_files=True
)

threshold = st.slider("Eligibility Threshold", 0.0, 1.0, 0.5, 0.05)

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

    payload = {
        "method": method,
        "threshold": threshold,
        "patients": patients,
        "trials": trials,
        "generate_reasoning": True
    }

    # ---- Start batch ----
    start_res = requests.post(
        urljoin(base + "/", "/predict/batch"),
        json=payload
    ).json()

    batch_id = start_res["batch_id"]
    st.session_state.batch_id = batch_id

    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    table_placeholder = st.empty()

    # ---- Poll batch progress ----
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


        # live table update
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
    st.session_state.batch_results = status["results"]
    progress_bar.progress(1.0)
    progress_text.markdown("✅ **Batch evaluation completed**")

# =====================================================================
# 🔹 RESULTS TABLE
# =====================================================================

if st.session_state.batch_results:
    st.markdown("## Results")

    rows = []
    for pid, data in st.session_state.batch_results.items():
        rows.append({
            "Patient Case": pid,
            "Eligible Trials": ", ".join(data["eligible_trials"]),
            "Non-Eligible Trials": ", ".join(data["non_eligible_trials"])
        })

    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.markdown("### Click a trial ID to view details")

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

if st.session_state.popup:
    st.markdown("---")
    st.markdown("## Trial Details")

    info = st.session_state.popup
    st.markdown(f"**Patient:** {info['patient_id']}")
    st.markdown(f"**Trial:** {info['trial_id']}")

    details = requests.get(
    urljoin(
        base + "/",
        f"/predict/batch/details/"
        f"{st.session_state.batch_id}/"
        f"{info['patient_id']}/"
        f"{info['trial_id']}"
        )
    ).json()

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

    elif details["method"] == "TCH_CLF":
        st.markdown("### TCH_CLF Reasoning")
        st.success(f"Final Score: {details['tch_clf']['score']:.4f}")
        st.write(details["tch_clf"]["reasoning"])


    if st.button("Close"):
        st.session_state.popup = None

st.caption("Clinical Trial Eligibility Checker v1.0 | Proprietary Software")
