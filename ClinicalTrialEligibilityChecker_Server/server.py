# server.py — Unified service with separate endpoints:
#   POST /predict/neureq  -> NEUREQ pipeline (LLM -> 10-question -> NEUREQ BiLSTM+AddAttn model)
#   POST /predict/tch_clf -> Teacher Longformer reranker + optional live DeepSeek reasoning
#
# Notes:
# - Edit MODEL/STATE/PATH constants below to match your environment.
# - Start with: uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

import os
import json
import re
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
)

# ------------------ CONFIG (edit as needed) ------------------
LLM_MODEL_NAME        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # DeepSeek LLM
NEUREQ_STATE          = "models/model_epoch_12.pt"                   # BiLSTM+AddAttn state dict
CLINICAL_BERT_MODEL   = "emilyalsentzer/Bio_ClinicalBERT"             # frozen encoder for justifications
PROMPT_FILE           = "prompt.txt"                                   # NEUREQ prompt template
TEACHER_MODEL_PATH    = "models/best_teacher_alpha0.2.pt"
TEACHER_MODEL_NAME    = "yikuan8/Clinical-Longformer"                  # tokenizer for teacher

LOG_DIR        = "audit_logs"
NEUREQ_LOG_DIR = os.path.join(LOG_DIR, "neureq")
TCH_LOG_DIR    = os.path.join(LOG_DIR, "tch_clf")
BATCH_LOG_DIR  = os.path.join(LOG_DIR, "batch")

os.makedirs(NEUREQ_LOG_DIR, exist_ok=True)
os.makedirs(TCH_LOG_DIR,    exist_ok=True)
os.makedirs(BATCH_LOG_DIR,  exist_ok=True)

MAX_NEW_TOKENS      = 4096
REASONING_MAX_TOKENS = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedServer")

# ------------------ QUESTIONS (displayed to clinician) ------------------
QUESTIONS = [
    "Age Eligibility - Does the patient's age fall within the trial's specified range?",
    "Gender Eligibility - Is the trial open to the patient's gender?",
    "Condition Relevance - Do the patient's symptoms, diagnosis, or condition match the trial's focus?",
    "Diagnostic Findings Match - Do lab tests, imaging, or biomarkers align with the trial's criteria?",
    "Prior Treatment Consideration - Has the patient undergone treatments relevant to the trial's eligibility criteria?",
    "Inclusion/Exclusion Criteria - Does the patient meet specific trial conditions (e.g., comorbidities, concurrent medications)?",
    "Pathophysiologic Mechanism - Does the patient's condition suggest an underlying disease mechanism relevant to the trial?",
    "Functional Status - Does the patient's sensory, motor, or cognitive function align with trial requirements?",
    "Interest in Experimental Therapy - Has the patient shown willingness for investigational treatments?",
    "Treatment Target Alignment - Does the trial's treatment directly address the patient's condition or symptoms?"
]

# ------------------ Globals ------------------
tokenizer = llm = PROMPT_TEMPLATE = None

# NEUREQ — new BiLSTM model + frozen ClinicalBERT encoder
neureq_model      = None
bert_tokenizer    = None   # ClinicalBERT tokenizer (NEUREQ justification encoder)
bert_model_enc    = None   # ClinicalBERT model    (NEUREQ justification encoder)

# teacher (lazy)
_teacher_model     = None
_teacher_tokenizer = None

# Whether to lazy-load the LLM used for live reasoning.
LLM_LOADED_AT_STARTUP = True

# Answer mapping for BiLSTM embedding lookup (must match training)
ANSWER_MAP = {"YES": 0, "NO": 1, "NA": 2}

# Legacy response map kept for any downstream caller that still reads ternary floats
RESPONSE_MAP = {"YES": 1.0, "NO": 0.0, "NA": 0.5}

app = FastAPI(title="NEUREQ + TCH_CLF Server")

# ------------------ Batch Progress (in-memory) ------------------
BATCH_PROGRESS = {}

# ------------------ Utility functions ------------------

def _stable_seed_from_prompt(prompt: str) -> int:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _strip_think_and_fences(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```', '', text)
    return text

def _find_balanced_json_substrings(text: str) -> List[Tuple[int, int, str]]:
    starts  = []
    results = []
    for i, ch in enumerate(text):
        if ch == '{':
            starts.append(i)
        elif ch == '}' and starts:
            start = starts.pop()
            end   = i + 1
            results.append((start, end, text[start:end]))
    return results

def _try_parse_candidates(text: str) -> Optional[Dict[str, Any]]:
    candidates = _find_balanced_json_substrings(text)
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1] - t[0], reverse=True)
    required = set(str(i) for i in range(1, 11))
    for start, end, substr in candidates:
        s       = substr.strip()
        s_clean = re.sub(r',\s*([}\]])', r'\1', s)
        try:
            parsed = json.loads(s_clean)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if required.issubset(set(parsed.keys())):
            return parsed
        if "cleaned_answers" in parsed and isinstance(parsed["cleaned_answers"], dict):
            ca = parsed["cleaned_answers"]
            if required.issubset(set(ca.keys())):
                return ca
        if "answers" in parsed and isinstance(parsed["answers"], dict):
            ans = parsed["answers"]
            if required.issubset(set(ans.keys())):
                return ans
    return None

def extract_valid_json(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_think_and_fences(raw_text)
    return _try_parse_candidates(cleaned)

def extract_valid_json_preserve(raw_text: str) -> Tuple[Dict[str, Dict[str, str]], list]:
    """
    Returns:
        cleaned_answers : dict  {str(1..10): {"response": str, "justification": str}}
        answer_ids      : list  of int  (ANSWER_MAP indices, length 10)
    """
    parsed = extract_valid_json(raw_text)
    if not parsed:
        answers    = {str(i): {"response": "NA", "justification": "Parsing failed"} for i in range(1, 11)}
        answer_ids = [ANSWER_MAP["NA"]] * 10
        return answers, answer_ids

    answers, answer_ids = {}, []
    for i in range(1, 11):
        key       = str(i)
        raw_entry = parsed.get(key, None)
        resp_raw  = None
        just      = ""

        if isinstance(raw_entry, dict):
            resp_raw = (
                raw_entry.get("response")
                or raw_entry.get("answer")
                or raw_entry.get("value")
            )
            just = (
                raw_entry.get("justification")
                or raw_entry.get("reason")
                or raw_entry.get("explain")
                or ""
            )
        elif isinstance(raw_entry, str):
            resp_raw = raw_entry
        elif raw_entry is None:
            resp_raw = None
        else:
            try:
                resp_raw = str(raw_entry)
            except Exception:
                resp_raw = None

        resp_preserved = resp_raw if resp_raw is not None else "NA"
        resp_upper     = str(resp_preserved).strip().upper()

        answers[key] = {"response": resp_preserved, "justification": (just or "").strip()}
        answer_ids.append(ANSWER_MAP.get(resp_upper, ANSWER_MAP["NA"]))

    return answers, answer_ids


# ------------------ NEUREQ model (BiLSTM + Additive Attention) ------------------

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        scores  = self.v(torch.tanh(self.attn(H))).squeeze(-1)
        alpha   = torch.softmax(scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * H, dim=1)
        return context, alpha


class EligibilityBiLSTM(nn.Module):
    """
    Input per sample: 10 questions, each with
        - question id  (0-9)   → Embedding(10, 8)
        - answer id    (0-2)   → Embedding(3, 3)  [YES=0, NO=1, NA=2]
        - justification CLS    → Bio_ClinicalBERT [768-d]
    BiLSTM hidden = 64 (bidirectional → 128), then additive attention → classifier.
    """
    def __init__(self):
        super().__init__()
        self.question_embed = nn.Embedding(10, 8)
        self.answer_embed   = nn.Embedding(3, 3)

        self.bilstm = nn.LSTM(
            input_size  = 8 + 3 + 768,   # q_emb + a_emb + bert_cls
            hidden_size = 64,
            num_layers  = 1,
            bidirectional=True,
            batch_first = True
        )

        self.attention  = AdditiveAttention(128)   # 64*2 = 128

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, q_ids, a_ids, j_embs):
        q_emb = self.question_embed(q_ids)          # (B, 10, 8)
        a_emb = self.answer_embed(a_ids)             # (B, 10, 3)

        x = torch.cat([q_emb, a_emb, j_embs], dim=-1)  # (B, 10, 779)
        H, _ = self.bilstm(x)                           # (B, 10, 128)

        context, _ = self.attention(H)                  # (B, 128)
        logit = self.classifier(context).squeeze(-1)    # (B,)
        return logit


# ------------------ Teacher Reranker (TCH_CLF — unchanged) ------------------

class TeacherReranker(nn.Module):
    def __init__(self, base_model_name=TEACHER_MODEL_NAME):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(base_model_name)
        hidden = self.longformer.config.hidden_size
        if hasattr(self.longformer, "pooler"):
            self.longformer.pooler = None
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 384), nn.GELU(), nn.Dropout(0.1), nn.Linear(384, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        cls     = outputs.last_hidden_state[:, 0, :]
        logits  = self.classifier(cls)
        return logits.squeeze(1)


# ------------------ NEUREQ justification encoder ------------------

@torch.no_grad()
def encode_justification(text: str) -> torch.Tensor:
    """Encode a single justification string → CLS vector [768]."""
    inputs = bert_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)
    outputs = bert_model_enc(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)   # (768,)


# ------------------ NEUREQ scoring function ------------------

@torch.no_grad()
def score_neureq(cleaned_answers: Optional[Dict[str, Dict[str, str]]]) -> float:
    """
    Score a single patient-trial pair using the new EligibilityBiLSTM.
    Returns a probability in [0, 1] via sigmoid of the raw logit.
    cleaned_answers == None → returns 0.0 (bottom of ranking).
    """
    if cleaned_answers is None:
        return 0.0

    question_ids       = []
    answer_ids_list    = []
    justification_embs = []

    for qid in range(1, 11):
        q         = cleaned_answers.get(str(qid), None)
        response  = q.get("response", "NA")  if q else "NA"
        just_text = q.get("justification", "") if q else ""

        question_ids.append(qid - 1)
        answer_ids_list.append(ANSWER_MAP.get(str(response).strip().upper(), ANSWER_MAP["NA"]))
        justification_embs.append(encode_justification(just_text))

    q_ids  = torch.tensor(question_ids,    dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1,10)
    a_ids  = torch.tensor(answer_ids_list, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1,10)
    j_embs = torch.stack(justification_embs).unsqueeze(0).to(DEVICE)                  # (1,10,768)

    logit = neureq_model(q_ids, a_ids, j_embs)                # (1,)
    prob  = float(torch.sigmoid(logit).item())
    return prob


# ------------------ Startup: load essentials ------------------

@app.on_event("startup")
def startup():
    global tokenizer, llm, PROMPT_TEMPLATE
    global neureq_model, bert_tokenizer, bert_model_enc

    # load prompt template
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            PROMPT_TEMPLATE = f.read()
    except Exception:
        PROMPT_TEMPLATE = "{0}\n\n{1}"

    # load LLM + tokenizer at startup (NEUREQ pipeline needs it)
    if LLM_LOADED_AT_STARTUP:
        logger.info("Loading LLM and tokenizer at startup...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=False)
        )
        logger.info("LLM loaded.")
    else:
        tokenizer = None
        llm       = None

    # load frozen ClinicalBERT (justification encoder for NEUREQ)
    logger.info("Loading Bio_ClinicalBERT (frozen justification encoder)...")
    bert_tokenizer  = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
    bert_model_enc  = AutoModel.from_pretrained(CLINICAL_BERT_MODEL).to(DEVICE)
    bert_model_enc.eval()
    for p in bert_model_enc.parameters():
        p.requires_grad = False
    logger.info("Bio_ClinicalBERT loaded and frozen.")

    # load new NEUREQ BiLSTM model
    logger.info("Loading NEUREQ EligibilityBiLSTM...")
    _neureq = EligibilityBiLSTM().to(DEVICE)
    _neureq.load_state_dict(torch.load(NEUREQ_STATE, map_location=DEVICE))
    _neureq.eval()
    neureq_model = _neureq
    logger.info("NEUREQ model loaded. Server ready.")


# ------------------ Lazy loaders for teacher and LLM ------------------

def load_teacher_if_needed():
    global _teacher_model, _teacher_tokenizer
    if _teacher_model is None:
        logger.info("Lazy-loading Teacher reranker and tokenizer...")
        _teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
        m = TeacherReranker()
        m.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location="cpu"))
        m.to(DEVICE)
        m.eval()
        _teacher_model = m
        logger.info("Teacher loaded.")
    return _teacher_model, _teacher_tokenizer

def ensure_llm_loaded():
    global tokenizer, llm
    if tokenizer is None or llm is None:
        logger.info("Lazy-loading LLM and tokenizer for reasoning...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=False)
        )
        logger.info("LLM loaded (lazy).")
    return tokenizer, llm


# ------------------ Request models ------------------

class NeureqRequest(BaseModel):
    query: str
    trial: str

class TchClfRequest(BaseModel):
    query: str
    trial: str
    generate_reasoning: Optional[bool] = True


# ------------------ Batch Request Models ------------------

class PatientCase(BaseModel):
    patient_id:   str
    patient_text: str

class TrialFile(BaseModel):
    trial_id:   str
    trial_text: str

class BatchRequest(BaseModel):
    method:             str                     # "NEUREQ" or "TCH_CLF"
    threshold:          float = 0.5
    patients:           List[PatientCase]
    trials:             List[TrialFile]
    generate_reasoning: Optional[bool] = True


# ------------------ NEUREQ handler (/predict/neureq) ------------------

@app.post("/predict/neureq")
def predict_neureq(req: NeureqRequest):
    if tokenizer is None or llm is None:
        ensure_llm_loaded()

    # Build prompt
    prompt_text = PROMPT_TEMPLATE.format(req.query, req.trial)
    seed = _stable_seed_from_prompt(prompt_text)
    try:
        torch.manual_seed(seed)
    except Exception:
        logger.warning("Could not set torch.manual_seed for determinism")

    # Prepare LLM inputs
    messages = [{"role": "user", "content": prompt_text}]
    inputs   = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    device   = next(llm.parameters()).device
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs.to(device)}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate deterministically (greedy)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_start = inputs["input_ids"].shape[-1]
    raw_text  = tokenizer.decode(output_ids[0][gen_start:], skip_special_tokens=True)

    # Save initial raw log
    case_id  = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_path = os.path.join(NEUREQ_LOG_DIR, f"{case_id}.json")
    initial_log = {
        "case_id":       case_id,
        "timestamp":     datetime.now().isoformat(),
        "patient_text":  req.query,
        "trial_text":    req.trial,
        "raw_llm_output": raw_text,
        "seed_used":     int(seed)
    }
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(initial_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write initial NEUREQ log: {e}")

    # Extract structured answers from LLM JSON
    # extract_valid_json_preserve now returns (cleaned_answers, answer_ids)
    cleaned_answers, answer_ids = extract_valid_json_preserve(raw_text)

    # Score with new BiLSTM + Additive Attention model
    score = score_neureq(cleaned_answers)

    full_log = {
        **initial_log,
        "cleaned_answers": cleaned_answers,
        "answer_ids":      answer_ids,          # [0-2] per question (YES=0, NO=1, NA=2)
        "final_score":     round(score, 4),
        "questions":       QUESTIONS
    }
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)
        logger.info(f"NEUREQ log saved: {log_path} | score={score:.4f} | seed={seed}")
    except Exception as e:
        logger.error(f"Failed to save full NEUREQ log: {e}")

    return {
        "score":           round(score, 4),
        "seed":            int(seed),
        "raw_llm_output":  raw_text,
        "cleaned_answers": cleaned_answers,
        "answer_ids":      answer_ids,
        "final_score":     round(score, 4),
        "questions":       QUESTIONS,
        "audit_log":       log_path
    }


# ------------------ TCH_CLF handler (/predict/tch_clf) ------------------

# Robust STOP pattern: list common ClinicalTrials-style headers
STOP_HEADERS = [
    "Study Title", "Official Title", "Brief Summary", "Detailed Description", "Conditions",
    "Condition", "Interventions", "Intervention", "Eligibility Criteria", "Inclusion Criteria",
    "Exclusion Criteria", "Gender", "Sex", "Minimum Age", "Maximum Age", "Ages Eligible",
    "Status", "Phase", "Start Date", "Primary Completion Date", "Last Update Posted"
]
STOP_RE = r"(?=\n(?:{})(?:\s*:)|$)".format("|".join([re.escape(h) for h in STOP_HEADERS]))

def extract_field(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None

def normalize_age(age_str):
    if not age_str or str(age_str).strip().lower() in ["n/a", "na", "not applicable", "no limit"]:
        return None
    age_str = str(age_str).strip()
    match = re.match(r"(\d+(?:\.\d+)?)\s*(year|yr|years|yrs|month|months|mo|day|days|d)", age_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit  = match.group(2).lower()
        if "year" in unit:
            return value
        elif "month" in unit:
            return value / 12.0
        elif "day" in unit:
            return value / 365.25
        else:
            return None
    else:
        num_match = re.match(r"(\d+(?:\.\d+)?)", age_str, re.IGNORECASE)
        if num_match:
            return float(num_match.group(1))
        return None

def normalize_gender(gender_str, eligibility_text):
    text_to_check = ""
    if gender_str:
        text_to_check = str(gender_str).lower()
    elif eligibility_text:
        text_to_check = str(eligibility_text).lower()

    if not text_to_check:
        return "male and female"
    if 'all' in text_to_check:
        return "male and female"

    male_terms   = ['male', 'man', 'boy', 'men', 'boys']
    female_terms = ['female', 'woman', 'girl', 'women', 'girls']

    has_male   = any(term in text_to_check for term in male_terms)
    has_female = any(term in text_to_check for term in female_terms)

    if has_male and has_female:
        return "male and female"
    elif has_male:
        return "male"
    elif has_female:
        return "female"
    else:
        return "male and female"

def _normalize_incoming_trial_text(raw: str) -> str:
    if raw is None:
        return ""
    t = raw.replace("\\n", "\n")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


@app.post("/predict/tch_clf")
def predict_tch_clf(req: TchClfRequest):
    original_trial_text   = req.trial
    normalized_trial_text = _normalize_incoming_trial_text(original_trial_text)

    trial_id = None
    m_nct = re.search(r"\b(NCT\d{6,8})\b", normalized_trial_text, re.IGNORECASE)
    if m_nct:
        trial_id = m_nct.group(1).upper()

    study_title    = extract_field(r"Study Title:\s*(.+?)"     + STOP_RE, normalized_trial_text) or None
    official_title = extract_field(r"Official Title:\s*(.+?)" + STOP_RE, normalized_trial_text) or None
    if not study_title and official_title:
        study_title = official_title

    brief_summary    = extract_field(r"Brief Summary:\s*(.+?)"                    + STOP_RE, normalized_trial_text) or None
    conditions       = extract_field(r"Conditions?:\s*(.+?)"                      + STOP_RE, normalized_trial_text) or None
    gender_raw       = extract_field(r"(?:Gender|Sex(?:es)? Eligible?)\s*:\s*(.+?)" + STOP_RE, normalized_trial_text)
    min_age_raw      = extract_field(r"Minimum Age:\s*(.+?)"                      + STOP_RE, normalized_trial_text)
    max_age_raw      = extract_field(r"Maximum Age:\s*(.+?)"                      + STOP_RE, normalized_trial_text)
    eligibility_text = extract_field(r"Eligibility Criteria:\s*(.+?)"             + STOP_RE, normalized_trial_text) or None

    if not min_age_raw and not max_age_raw:
        combined_age = extract_field(r"Ages Eligible.*?:\s*(.+?)" + STOP_RE, normalized_trial_text)
        if combined_age:
            range_match = re.match(r"(\d+.*?)(?:\s+to\s+(\d+.*))?$", combined_age, re.IGNORECASE)
            if range_match:
                min_age_raw = range_match.group(1).strip() if range_match.group(1) else None
                max_age_raw = range_match.group(2).strip() if range_match.group(2) else None
            else:
                min_age_raw = combined_age.strip()

    min_age_norm = normalize_age(min_age_raw) if min_age_raw else None
    max_age_norm = normalize_age(max_age_raw) if max_age_raw else None

    if min_age_norm is None:
        min_age_norm = 0.0
    if max_age_norm is None:
        max_age_norm = 150.0

    gender_norm = normalize_gender(gender_raw, eligibility_text)

    extracted = {
        "id":            trial_id,
        "study_title":   study_title,
        "brief_summary": brief_summary,
        "conditions":    conditions,
        "gender":        gender_norm,
        "min_age":       min_age_norm,
        "max_age":       max_age_norm,
        "eligibility":   {"criteria": eligibility_text}
    }

    parts = []
    field_order = [
        ("study_title",  extracted["study_title"]),
        ("brief_summary",extracted["brief_summary"]),
        ("conditions",   extracted["conditions"]),
        ("gender",       extracted["gender"]),
        ("min_age",      str(extracted["min_age"])  if extracted["min_age"]  is not None else None),
        ("max_age",      str(extracted["max_age"])  if extracted["max_age"]  is not None else None),
        ("eligibility",  extracted["eligibility"].get("criteria"))
    ]
    for nm, val in field_order:
        if val is not None and val != "":
            parts.append(f"{nm}: {val}")
    concatenated_text = " || ".join(parts)

    found_headers = []
    for h in STOP_HEADERS:
        if re.search(rf"^\s*{re.escape(h)}\s*:", normalized_trial_text, flags=re.IGNORECASE | re.MULTILINE):
            found_headers.append(h)

    teacher_model, teacher_tokenizer = load_teacher_if_needed()
    enc = teacher_tokenizer(
        req.query,
        concatenated_text,
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logit = teacher_model(enc["input_ids"], enc["attention_mask"]).item()
        score = float(torch.sigmoid(torch.tensor(logit)).item())

    reasoning_text = ""
    reason_seed    = 0
    if req.generate_reasoning:
        ensure_llm_loaded()
        reason_prompt = (
            "### Role: You are an expert in biomedical AI ...\n"
            f"### Patient Description: {req.query}\n"
            f"### Clinical Trial: {req.trial}\n"
            "### Output: Provide a JSON object with keys 'reasoning' (>=3 sentences) and 'relevance' ('Relevant'|'Non-Relevant').\n"
        )
        reason_seed = _stable_seed_from_prompt(reason_prompt)
        try:
            torch.manual_seed(reason_seed)
        except Exception:
            pass
        messages_r = [{"role": "user", "content": reason_prompt}]
        inputs_r   = tokenizer.apply_chat_template(messages_r, add_generation_prompt=True, return_tensors="pt")
        device_llm = next(llm.parameters()).device
        if isinstance(inputs_r, torch.Tensor):
            inputs_r = {"input_ids": inputs_r.to(device_llm)}
        else:
            inputs_r = {k: v.to(device_llm) for k, v in inputs_r.items()}
        with torch.no_grad():
            out_ids = llm.generate(
                **inputs_r,
                max_new_tokens=REASONING_MAX_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_start  = inputs_r["input_ids"].shape[-1]
        raw_reason = tokenizer.decode(out_ids[0][gen_start:], skip_special_tokens=True)
        parsed     = None
        try:
            s = raw_reason.rfind("{")
            e = raw_reason.rfind("}") + 1
            if s != -1 and e > s:
                parsed_cand = json.loads(raw_reason[s:e])
                if isinstance(parsed_cand, dict) and "reasoning" in parsed_cand:
                    parsed = parsed_cand
        except Exception:
            parsed = None
        if parsed:
            reasoning_text = parsed.get("reasoning", "").strip()
        else:
            reasoning_text = raw_reason.strip()

    case_id  = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    tch_log  = {
        "case_id":                case_id,
        "timestamp":              datetime.now().isoformat(),
        "patient_text":           req.query,
        "trial_text":             original_trial_text,
        "trial_text_normalized":  normalized_trial_text,
        "found_headers":          found_headers,
        "extracted":              extracted,
        "concatenated_text":      concatenated_text,
        "teacher_score":          round(score, 4),
        "reasoning":              reasoning_text,
        "seed_used":              int(reason_seed)
    }
    log_path = os.path.join(TCH_LOG_DIR, f"{case_id}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(tch_log, f, indent=2, ensure_ascii=False)
        logger.info(f"TCH_CLF log saved: {log_path} | score={score:.4f}")
    except Exception as e:
        logger.error(f"Failed to save TCH_CLF log: {e}")

    return {
        "score":              round(score, 4),
        "reasoning":          reasoning_text,
        "seed":               int(reason_seed),
        "extracted":          extracted,
        "concatenated_text":  concatenated_text,
        "found_headers":      found_headers,
        "audit_log":          log_path
    }


# ------------------ Batch handler (/predict/batch) ------------------

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):

    if req.method not in {"NEUREQ", "TCH_CLF"}:
        raise HTTPException(status_code=400, detail="method must be NEUREQ or TCH_CLF")

    batch_id  = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    batch_dir = os.path.join(BATCH_LOG_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    BATCH_PROGRESS[batch_id] = {
        "status":               "running",
        "current_patient":      None,
        "current_trial_index":  {},
        "total_trials":         len(req.trials),
        "results":              {}
    }

    for p in req.patients:
        BATCH_PROGRESS[batch_id]["results"][p.patient_id] = {
            "eligible_trials":     [],
            "non_eligible_trials": []
        }

    for p in req.patients:
        BATCH_PROGRESS[batch_id]["current_trial_index"][p.patient_id] = 0

    results: Dict[str, Dict[str, List[str]]] = {}

    logger.info(
        f"Starting batch {batch_id} | "
        f"method={req.method} | "
        f"patients={len(req.patients)} | "
        f"trials={len(req.trials)}"
    )

    for patient in req.patients:

        eligible_trials     = []
        non_eligible_trials = []

        patient_dir = os.path.join(batch_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        for trial in req.trials:

            logger.info(
                f"[{batch_id}] Evaluating "
                f"patient={patient.patient_id} "
                f"trial={trial.trial_id}"
            )

            if req.method == "NEUREQ":
                response = predict_neureq(
                    NeureqRequest(
                        query=patient.patient_text,
                        trial=trial.trial_text
                    )
                )
            else:
                response = predict_tch_clf(
                    TchClfRequest(
                        query=patient.patient_text,
                        trial=trial.trial_text,
                        generate_reasoning=req.generate_reasoning
                    )
                )

            score = response.get("score", 0.0)

            if score >= req.threshold:
                eligible_trials.append(trial.trial_id)
            else:
                non_eligible_trials.append(trial.trial_id)

            BATCH_PROGRESS[batch_id]["current_patient"] = patient.patient_id
            BATCH_PROGRESS[batch_id]["current_trial_index"][patient.patient_id] += 1
            BATCH_PROGRESS[batch_id]["results"][patient.patient_id] = {
                "eligible_trials":     eligible_trials.copy(),
                "non_eligible_trials": non_eligible_trials.copy()
            }

            eval_log = {
                "patient_id": patient.patient_id,
                "trial_id":   trial.trial_id,
                "method":     req.method,
                "score":      round(score, 4),
                "threshold":  req.threshold,
                "timestamp":  datetime.now().isoformat(),
                "audit_log":  response.get("audit_log")
            }

            with open(
                os.path.join(patient_dir, f"{trial.trial_id}.json"),
                "w", encoding="utf-8"
            ) as f:
                json.dump(eval_log, f, indent=2, ensure_ascii=False)

        results[patient.patient_id] = {
            "eligible_trials":     eligible_trials,
            "non_eligible_trials": non_eligible_trials
        }

    logger.info(f"Batch {batch_id} completed")
    BATCH_PROGRESS[batch_id]["status"] = "completed"

    return {
        "batch_id":  batch_id,
        "status":    "completed",
        "method":    req.method,
        "threshold": req.threshold,
        "results":   results
    }


@app.get("/predict/batch/status/{batch_id}")
def get_batch_status(batch_id: str):
    if batch_id not in BATCH_PROGRESS:
        raise HTTPException(status_code=404, detail="Unknown batch_id")
    return BATCH_PROGRESS[batch_id]


@app.get("/predict/batch/details/{batch_id}/{patient_id}/{trial_id}")
def get_batch_details(batch_id: str, patient_id: str, trial_id: str):

    batch_dir = os.path.join(BATCH_LOG_DIR, batch_id, patient_id)
    if not os.path.isdir(batch_dir):
        raise HTTPException(status_code=404, detail="Batch or patient not found")

    trial_log_path = os.path.join(batch_dir, f"{trial_id}.json")
    if not os.path.isfile(trial_log_path):
        raise HTTPException(status_code=404, detail="Trial not found")

    with open(trial_log_path, "r", encoding="utf-8") as f:
        eval_log = json.load(f)

    response = {
        "method":     eval_log["method"],
        "patient_id": patient_id,
        "trial_id":   trial_id
    }

    if eval_log["method"] == "NEUREQ":
        audit_path = eval_log.get("audit_log")
        if audit_path and os.path.isfile(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            response["neureq"] = {
                "score":           d["final_score"],
                "questions":       d["questions"],
                "cleaned_answers": d["cleaned_answers"]
            }

    if eval_log["method"] == "TCH_CLF":
        audit_path = eval_log.get("audit_log")
        if audit_path and os.path.isfile(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            response["tch_clf"] = {
                "score":     d["teacher_score"],
                "reasoning": d.get("reasoning", ""),
                "extracted": d.get("extracted", {})
            }

    return response


# ------------------ End of server.py ------------------
