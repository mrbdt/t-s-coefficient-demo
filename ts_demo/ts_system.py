from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# -----------------------------
# Environment / Config
# -----------------------------
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DB_PATH = os.getenv("TS_DB_PATH", "ts_kb.sqlite3")

TS_MAX_CHUNKS = int(os.getenv("TS_MAX_CHUNKS", "30"))
LAMBDA_DECAY = float(os.getenv("TS_LAMBDA_DECAY", "0.002"))  # awareness decay for old exposures
TAU_RELIABILITY_DAYS = float(os.getenv("TS_TAU_RELIABILITY_DAYS", "365"))  # recency weighting

HORIZON_BUCKETS = ["1D", "1W", "1M", "3M", "1Y", "3Y"]
HORIZON_TRADING_DAYS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252, "3Y": 756}

# Heuristic weights (for "impact now" / UI ranking)
NEAR_WEIGHTS = {"1D": 1.00, "1W": 0.75, "1M": 0.55, "3M": 0.40, "1Y": 0.25, "3Y": 0.15}
HALF_LIFE_DAYS = {"1D": 2, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "3Y": 900}

# Reach defaults (v1)
SOURCE_REACH = {
    "SEC_FILING": 0.70,
    "EARNINGS_CALL": 0.65,
    "PRESS_RELEASE": 0.75,
    "NEWS_PAYWALLED": 0.55,
    "NEWS_FREE": 0.80,
    "X_CSUITE": 0.60,
    "X_ANALYST": 0.45,
    "RUMOUR": 0.25,
}

# Dedup thresholds
EMBED_SAME_STRICT = 0.93
EMBED_SAME_LOOSE = 0.85
FUZZ_SAME_STRICT = 0.92

# Status labels for demo readability
def status_from_similarity(sim: float) -> str:
    if sim >= 0.95:
        return "RECONFIRMED"
    if sim >= 0.65:
        return "KNOWN"
    return "NEW"


# -----------------------------
# Structured extraction schema
# -----------------------------
Polarity = Literal["BULLISH", "BEARISH", "MIXED", "NEUTRAL"]
Modality = Literal["ASSERTION", "FORECAST", "INTENTION", "CONDITIONAL", "RISK", "OPINION"]
Evidential = Literal["REPORTED_NUMBER", "OPERATIONAL_OBSERVATION", "INTERNAL_METRIC", "UNSPECIFIED"]
SpeakerRole = Literal["COMPANY_OFFICIAL", "MANAGEMENT", "ANALYST", "OTHER"]

class ExtractedClaim(BaseModel):
    # Core claim
    claim: str = Field(..., description="Atomic, standalone claim. Include key numbers and the relevant period if present.")
    polarity: Polarity = Field(..., description="BULLISH/BEARISH/MIXED/NEUTRAL")
    materiality_0_1: float = Field(..., ge=0.0, le=1.0, description="If true and broadly known, how valuation-relevant is it?")
    credibility_0_1: float = Field(..., ge=0.0, le=1.0, description="How likely true given the source and wording?")
    surprise_0_1: float = Field(..., ge=0.0, le=1.0, description="How unexpected vs common priors for this company/sector?")
    horizon_profile: Dict[str, float] = Field(..., description="Distribution over 1D/1W/1M/3M/1Y/3Y; should sum ~1.0.")
    rationale: str = Field(..., description="One paragraph on why this matters for valuation.")
    quote: str = Field(..., description="Exact supporting quote/snippet from the chunk.")

    # Pragmatics / attribution
    is_forward_looking: bool = Field(..., description="True if the claim is about future expectations, guidance, intentions, or projected outcomes.")
    modality: Modality = Field(..., description="ASSERTION/FORECAST/INTENTION/CONDITIONAL/RISK/OPINION")
    commitment_0_1: float = Field(..., ge=0.0, le=1.0, description="Higher for 'will'/'we will'; lower for 'may'/'could'.")
    conditionality_0_1: float = Field(..., ge=0.0, le=1.0, description="Higher when heavily conditional ('if', 'subject to', 'depending on').")
    evidential_basis: Evidential = Field(..., description="Is it a reported number, observation, internal metric, or unspecified?")
    speaker_role: SpeakerRole = Field(..., description="Who is effectively the source of the quoted snippet?")

class ExtractedClaims(BaseModel):
    claims: List[ExtractedClaim]


# -----------------------------
# Text normalisation
# -----------------------------
def table_to_markdown_from_bs4(table_tag) -> str:
    """
    Convert a BeautifulSoup <table> tag to markdown.
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for td in tr.find_all(["th", "td"]):
            text = td.get_text(" ", strip=True)
            cells.append(text.replace("\n", " ").strip())
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    header = rows[0]
    if all((h == "" or any(ch.isalpha() for ch in h)) for h in header):
        body = rows[1:]
    else:
        header = [f"col{i+1}" for i in range(max_cols)]
        body = rows

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def table_to_markdown_from_list(table: List[List[str]]) -> str:
    """
    Convert a table represented as a list-of-lists (pdfplumber) into markdown.
    """
    if not table:
        return ""
    max_cols = max(len(r) for r in table)
    rows = [list(map(lambda x: (x or "").strip().replace("\n", " "), r)) for r in table]
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    header = rows[0] if any(cell.isalpha() for cell in " ".join(rows[0])) else [f"col{i+1}" for i in range(max_cols)]
    body = rows[1:] if header == rows[0] else rows

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def normalise_to_text(path: Path | str) -> str:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        parts: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []

                for t in tables:
                    md = table_to_markdown_from_list(t)
                    if md:
                        parts.append(f"[START_TABLE page={page_no}]\n{md}\n[END_TABLE]\n")

                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
        return "\n\n".join(parts)

    if ext in {".html", ".htm"}:
        html = path.read_text(errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        for table in soup.find_all("table"):
            try:
                md = table_to_markdown_from_bs4(table)
                md_block = soup.new_tag("p")
                md_block.string = f"\n\n[START_TABLE]\n{md}\n[END_TABLE]\n\n"
                table.replace_with(md_block)
            except Exception:
                continue

        text = soup.get_text("\n")
        return "\n".join([line.rstrip() for line in text.splitlines() if line.strip()])

    if ext == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)

    return path.read_text(errors="ignore")


def chunk_text(text: str, max_chars: int = 7000, overlap: int = 700) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def sample_chunks(chunks: List[str], k: int) -> List[str]:
    if len(chunks) <= k:
        return chunks
    # start / middle / end sampling
    a = chunks[: max(1, k // 3)]
    mid_start = max(0, len(chunks) // 2 - max(1, k // 6))
    b = chunks[mid_start: mid_start + max(1, k // 3)]
    c = chunks[-max(1, k // 3):]
    out = a + b + c
    return out[:k]


# -----------------------------
# DB schema
# -----------------------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        canonical_text TEXT NOT NULL,
        embedding_json TEXT NOT NULL,

        sign INTEGER NOT NULL,
        materiality REAL NOT NULL,
        novelty REAL NOT NULL,
        surprise REAL NOT NULL,
        ts_coef REAL NOT NULL,
        horizon_json TEXT NOT NULL,

        source_id TEXT NOT NULL,
        speaker_role TEXT NOT NULL,

        is_forward_looking INTEGER NOT NULL,
        modality TEXT NOT NULL,
        commitment REAL NOT NULL,
        conditionality REAL NOT NULL,
        evidential_basis TEXT NOT NULL,

        p0_cred REAL NOT NULL,
        p_prag REAL NOT NULL,
        p_true_latest REAL NOT NULL,
        p_true_at_issue REAL NOT NULL,

        issued_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS exposures (
        exposure_id TEXT PRIMARY KEY,
        fact_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        reach REAL NOT NULL,
        authority REAL NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        doc_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        source_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        url TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS doc_claims (
        doc_claim_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        fact_id TEXT NOT NULL,

        extracted_claim TEXT NOT NULL,
        quote TEXT NOT NULL,
        rationale TEXT NOT NULL,
        polarity TEXT NOT NULL,

        best_match_similarity REAL NOT NULL,
        status TEXT NOT NULL,

        delta_awareness REAL NOT NULL,
        p_true_used REAL NOT NULL,
        pred_horizon_json TEXT NOT NULL,
        pred_total REAL NOT NULL,

        FOREIGN KEY(doc_id) REFERENCES docs(doc_id),
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS doc_scores (
        doc_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,

        pred_horizon_json TEXT NOT NULL,
        pred_near_term REAL NOT NULL,

        n_claims INTEGER NOT NULL,
        n_new INTEGER NOT NULL,
        n_known INTEGER NOT NULL,
        n_reconfirmed INTEGER NOT NULL,

        FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        reliability REAL NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS resolutions (
        resolution_id TEXT PRIMARY KEY,
        fact_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        resolved_at TEXT NOT NULL,
        outcome INTEGER NOT NULL,     -- 1 true, 0 false
        confidence REAL NOT NULL,
        evidence TEXT NOT NULL,
        method TEXT NOT NULL,
        p_pred_at_issue REAL NOT NULL,
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_ticker ON facts(ticker);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exposures_fact ON exposures(fact_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ticker_time ON docs(ticker, timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docclaims_doc ON doc_claims(doc_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resolutions_source ON resolutions(source_id);")

    conn.commit()


# -----------------------------
# Utilities
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def polarity_to_sign(p: Polarity) -> int:
    if p == "BULLISH":
        return +1
    if p == "BEARISH":
        return -1
    return 0


def normalise_horizon(h: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(h.get(k, 0.0)) for k in HORIZON_BUCKETS}
    s = float(sum(out.values()))
    if s <= 0:
        out = {"1D": 0.25, "1W": 0.25, "1M": 0.20, "3M": 0.15, "1Y": 0.10, "3Y": 0.05}
        s = float(sum(out.values()))
    return {k: out[k] / s for k in HORIZON_BUCKETS}


def embed(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text, encoding_format="float")
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(vec) + 1e-12)
    return vec / n

# ---------- Structured parse probe and robust parsing helpers ----------
# Global toggle — will be set at runtime
STRUCTURED_PARSE_ENABLED = True

def _supports_structured_parse(client: OpenAI) -> bool:
    """
    Quick probe to check if responses.parse(...) with our Pydantic schema is supported.
    """
    try:
        resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "Return an empty claims list."},
                {"role": "user", "content": "TEXT: test"},
            ],
            text_format=ExtractedClaims,
        )
        _ = resp.output_parsed
        return True
    except Exception as e:
        print(f"[LLM] Structured parse probe failed ({type(e).__name__}): {e}")
        return False


def _safe_json_load(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"claims": []}
        return {"claims": []}


def _normalize_evidential(v: str) -> str:
    if not v or not isinstance(v, str):
        return "UNSPECIFIED"
    vv = v.strip().upper()
    if "REPORT" in vv or "NUMBER" in vv or "REPORTED" in vv:
        return "REPORTED_NUMBER"
    if "OPERATION" in vv or "OBSERV" in vv or "OPERATIONAL" in vv:
        return "OPERATIONAL_OBSERVATION"
    if "INTERNAL" in vv or "METRIC" in vv:
        return "INTERNAL_METRIC"
    return "UNSPECIFIED"


def _normalize_modality(v: Optional[str]) -> str:
    if not v:
        return "ASSERTION"
    vv = v.strip().upper()
    for opt in ["FORECAST","INTENTION","CONDITIONAL","RISK","OPINION","ASSERTION"]:
        if opt in vv:
            return opt
    return "ASSERTION"


def _normalize_speaker_role(v: Optional[str]) -> str:
    if not v:
        return "OTHER"
    vv = v.strip().upper()
    if "COMPANY" in vv or "SEC" in vv or "FORM" in vv:
        return "COMPANY_OFFICIAL"
    if "MANAGEMENT" in vv or "CEO" in vv or "CFO" in vv:
        return "MANAGEMENT"
    if "ANALYST" in vv or "SELL" in vv or "BUY" in vv:
        return "ANALYST"
    return "OTHER"


def parse_raw_claims(data: dict, chunk: str) -> List[ExtractedClaim]:
    """
    Defensive mapping from arbitrary JSON returned by the model into our ExtractedClaim Pydantic objects.
    """
    out: List[ExtractedClaim] = []
    if not isinstance(data, dict):
        return out
    raw_claims = data.get("claims") or data.get("items") or []
    if not isinstance(raw_claims, list):
        return out

    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = rc.get("claim") or rc.get("text") or (chunk[:400] + ("..." if len(chunk) > 400 else ""))

        polarity = (rc.get("polarity") or "NEUTRAL").upper()
        if polarity not in {"BULLISH","BEARISH","MIXED","NEUTRAL"}:
            pt = claim_text.lower()
            if any(w in pt for w in ["increase","upside","beat","outperform","grow","gain","benefit"]):
                polarity = "BULLISH"
            elif any(w in pt for w in ["decline","drop","miss","slowdown","risk","loss","downside","concern"]):
                polarity = "BEARISH"
            else:
                polarity = "NEUTRAL"

        def _safe_float(k, default):
            try:
                v = rc.get(k, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        materiality = max(0.0, min(1.0, _safe_float("materiality_0_1", 0.15)))
        credibility = max(0.0, min(1.0, _safe_float("credibility_0_1", 0.6)))
        surprise = max(0.0, min(1.0, _safe_float("surprise_0_1", 0.25)))

        hp = rc.get("horizon_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        if sum([float(v) for v in hp.values()]) <= 0:
            hp = {"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05}
        s = float(sum(float(hp.get(k,0.0)) for k in HORIZON_BUCKETS))
        if s <= 0:
            s = 1.0
        horizon_profile = {k: float(hp.get(k,0.0))/s for k in HORIZON_BUCKETS}

        rationale = rc.get("rationale") or rc.get("why") or ""
        quote = rc.get("quote") or rc.get("snippet") or (claim_text if len(claim_text) < 600 else claim_text[:600] + "...")

        is_forward = bool(rc.get("is_forward_looking", False))
        modality = _normalize_modality(rc.get("modality"))
        commitment = max(0.0, min(1.0, _safe_float("commitment_0_1", 0.5)))
        conditionality = max(0.0, min(1.0, _safe_float("conditionality_0_1", 0.0)))
        evidential = _normalize_evidential(rc.get("evidential_basis") or rc.get("evidence") or "")
        speaker_role = _normalize_speaker_role(rc.get("speaker_role"))

        try:
            ec = ExtractedClaim(
                claim=str(claim_text),
                polarity=polarity,
                materiality_0_1=float(materiality),
                credibility_0_1=float(credibility),
                surprise_0_1=float(surprise),
                horizon_profile=horizon_profile,
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=bool(is_forward),
                modality=modality,
                commitment_0_1=float(commitment),
                conditionality_0_1=float(conditionality),
                evidential_basis=evidential,
                speaker_role=speaker_role,
            )
            out.append(ec)
        except Exception:
            out.append(ExtractedClaim(
                claim=str(claim_text),
                polarity="NEUTRAL",
                materiality_0_1=0.15,
                credibility_0_1=0.6,
                surprise_0_1=0.25,
                horizon_profile={"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05},
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=False,
                modality="ASSERTION",
                commitment_0_1=0.5,
                conditionality_0_1=0.0,
                evidential_basis="UNSPECIFIED",
                speaker_role="OTHER",
            ))

    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def get_source_reliability(conn: sqlite3.Connection, source_id: str) -> float:
    cur = conn.execute("SELECT reliability FROM sources WHERE source_id = ?", (source_id,))
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO sources(source_id, reliability, updated_at) VALUES (?,?,?)",
            (source_id, 1.0, utc_now().isoformat()),
        )
        conn.commit()
        return 1.0
    return float(row[0])


def pragmatics_adjust(p0: float, commitment: float, conditionality: float, is_forward_looking: bool) -> float:
    """
    Pull probability towards 0.5 when language is weak or heavily conditional.
    For non-forward-looking assertions, just return p0.
    """
    p0 = float(np.clip(p0, 0.0, 1.0))
    if not is_forward_looking:
        return p0

    commitment = float(np.clip(commitment, 0.0, 1.0))
    conditionality = float(np.clip(conditionality, 0.0, 1.0))

    strength = (0.4 + 0.6 * commitment) * (1.0 - 0.5 * conditionality)
    p = 0.5 + (p0 - 0.5) * strength
    return float(np.clip(p, 0.0, 1.0))


def apply_reliability(p_prag: float, reliability: float) -> float:
    """
    Shrink towards 0.5 based on source reliability.
    """
    p_prag = float(np.clip(p_prag, 0.0, 1.0))
    reliability = float(np.clip(reliability, 0.0, 1.0))
    p = 0.5 + (p_prag - 0.5) * reliability
    return float(np.clip(p, 0.0, 1.0))


def combine_independent_evidence(p_old: float, p_new: float, authority: float) -> float:
    """
    Simple monotonic combiner: new high-authority evidence raises belief.
    (We do not handle contradictions in v1.)
    """
    p_old = float(np.clip(p_old, 0.0, 1.0))
    p_new = float(np.clip(p_new, 0.0, 1.0))
    authority = float(np.clip(authority, 0.0, 1.0))
    p_eff = p_new * authority
    return float(np.clip(1.0 - (1.0 - p_old) * (1.0 - p_eff), 0.0, 1.0))


def compute_ts_coef(sign: int, materiality: float, novelty: float, surprise: float) -> float:
    """
    TS_coef is fixed: "impact if true and widely believed".
    """
    return float(sign) * float(materiality) * float(novelty) * float(surprise)


def awareness(conn: sqlite3.Connection, fact_id: str, as_of: dt.datetime, lam: float = LAMBDA_DECAY) -> float:
    cur = conn.execute("SELECT reach, timestamp FROM exposures WHERE fact_id = ?", (fact_id,))
    rows = cur.fetchall()
    if not rows:
        return 0.0

    prod = 1.0
    for reach, ts in rows:
        t_e = dt.datetime.fromisoformat(ts)
        age_days = max(0.0, (as_of - t_e).total_seconds() / 86400.0)
        effective = float(reach) * float(np.exp(-lam * age_days))
        prod *= (1.0 - np.clip(effective, 0.0, 1.0))
    return float(1.0 - prod)


def impact_now(ts_coef: float, p_true: float, horizon: Dict[str, float], age_days: float, awareness_now: float) -> float:
    """
    UI ranking: includes time decay and near-term weighting.
    """
    total = 0.0
    for b in HORIZON_BUCKETS:
        w = NEAR_WEIGHTS[b]
        hl = HALF_LIFE_DAYS[b]
        decay = float(np.exp(-age_days * np.log(2.0) / hl))
        total += horizon[b] * w * decay
    return float(ts_coef * p_true * awareness_now * total)


def fetch_facts_for_ticker(conn: sqlite3.Connection, ticker: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT fact_id, canonical_text, embedding_json, ts_coef FROM facts WHERE ticker = ?",
        (ticker,),
    )
    rows = []
    for fact_id, text, emb_json, ts_coef in cur.fetchall():
        rows.append({
            "fact_id": fact_id,
            "canonical_text": text,
            "embedding": np.array(json.loads(emb_json), dtype=np.float32),
            "ts_coef": float(ts_coef),
        })
    return rows


def novelty_against_kb(claim_text: str, claim_vec: np.ndarray, existing: List[Dict[str, Any]]) -> Tuple[float, Optional[str], float]:
    best_sim = 0.0
    best_id: Optional[str] = None
    best_text = None

    for f in existing:
        sim = cosine(claim_vec, f["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_id = f["fact_id"]
            best_text = f["canonical_text"]

    if best_id is not None and best_text is not None:
        fuzz_ratio = fuzz.token_set_ratio(claim_text, best_text) / 100.0
        best_sim = max(best_sim, fuzz_ratio)

    novelty = float(np.clip(1.0 - best_sim, 0.0, 1.0))
    return novelty, best_id, float(best_sim)


def treat_as_same_fact(embed_sim: float, fuzz_sim: float) -> bool:
    return (embed_sim >= EMBED_SAME_STRICT) or (embed_sim >= EMBED_SAME_LOOSE and fuzz_sim >= FUZZ_SAME_STRICT)


# -----------------------------
# OpenAI extraction
# -----------------------------
# ---------- REPLACE the existing extract_claims_from_chunk(...) entirely with this ----------

def _normalize_evidential(v: str) -> str:
    if not v or not isinstance(v, str):
        return "UNSPECIFIED"
    vv = v.strip().upper()
    if "REPORT" in vv or "NUMBER" in vv or "REPORTED" in vv:
        return "REPORTED_NUMBER"
    if "OPERATION" in vv or "OBSERV" in vv:
        return "OPERATIONAL_OBSERVATION"
    if "INTERNAL" in vv or "METRIC" in vv:
        return "INTERNAL_METRIC"
    return "UNSPECIFIED"

def _normalize_modality(v: Optional[str]) -> str:
    if not v:
        return "ASSERTION"
    vv = v.strip().upper()
    for opt in ["FORECAST","INTENTION","CONDITIONAL","RISK","OPINION","ASSERTION"]:
        if opt in vv:
            return opt
    return "ASSERTION"

def _normalize_speaker_role(v: Optional[str]) -> str:
    if not v:
        return "OTHER"
    vv = v.strip().upper()
    if "COMPANY" in vv or "SEC" in vv or "FORM" in vv:
        return "COMPANY_OFFICIAL"
    if "MANAGEMENT" in vv or "CEO" in vv or "CFO" in vv:
        return "MANAGEMENT"
    if "ANALYST" in vv or "SELL" in vv or "BUY" in vv:
        return "ANALYST"
    return "OTHER"

def parse_raw_claims(data: dict, chunk: str) -> List[ExtractedClaim]:
    """
    Defensive mapping from arbitrary JSON returned by the model into our ExtractedClaim objects.
    Uses sensible defaults for missing fields and normalises enums.
    """
    out: List[ExtractedClaim] = []
    if not isinstance(data, dict):
        return out
    raw_claims = data.get("claims") or data.get("items") or []
    if not isinstance(raw_claims, list):
        return out

    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = rc.get("claim") or rc.get("text") or (chunk[:400] + ("..." if len(chunk) > 400 else ""))
        # Basic defaults
        polarity = (rc.get("polarity") or "NEUTRAL").upper()
        if polarity not in {"BULLISH","BEARISH","MIXED","NEUTRAL"}:
            # try to infer from words
            pt = claim_text.lower()
            if any(w in pt for w in ["increase","upside","beat","outperform","grow","gain","benefit"]):
                polarity = "BULLISH"
            elif any(w in pt for w in ["decline","drop","miss","slowdown","risk","loss","downside","concern"]):
                polarity = "BEARISH"
            else:
                polarity = "NEUTRAL"

        # numeric fields with safe clamping
        def _safe_float(k, default):
            try:
                v = rc.get(k, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        materiality = max(0.0, min(1.0, _safe_float("materiality_0_1", 0.15)))
        credibility = max(0.0, min(1.0, _safe_float("credibility_0_1", 0.6)))
        surprise = max(0.0, min(1.0, _safe_float("surprise_0_1", 0.25)))

        # horizon_profile default / normalise
        hp = rc.get("horizon_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        # fallback distribution
        if sum([float(v) for v in hp.values()]) <= 0:
            hp = {"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05}
        # normalise
        s = float(sum(float(hp.get(k,0.0)) for k in HORIZON_BUCKETS))
        if s <= 0:
            s = 1.0
        horizon_profile = {k: float(hp.get(k,0.0))/s for k in HORIZON_BUCKETS}

        rationale = rc.get("rationale") or rc.get("why") or ""
        quote = rc.get("quote") or rc.get("snippet") or (claim_text if len(claim_text) < 600 else claim_text[:600] + "...")

        # pragmatics defaults
        is_forward = bool(rc.get("is_forward_looking", False))
        modality = _normalize_modality(rc.get("modality"))
        commitment = max(0.0, min(1.0, _safe_float("commitment_0_1", 0.5)))
        conditionality = max(0.0, min(1.0, _safe_float("conditionality_0_1", 0.0)))
        evidential = _normalize_evidential(rc.get("evidential_basis") or rc.get("evidence") or "")
        speaker_role = _normalize_speaker_role(rc.get("speaker_role"))

        # Build ExtractedClaim object with fallback values where possible
        try:
            ec = ExtractedClaim(
                claim=str(claim_text),
                polarity=polarity,  # pydantic will coerce or validate
                materiality_0_1=float(materiality),
                credibility_0_1=float(credibility),
                surprise_0_1=float(surprise),
                horizon_profile=horizon_profile,
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=bool(is_forward),
                modality=modality,
                commitment_0_1=float(commitment),
                conditionality_0_1=float(conditionality),
                evidential_basis=evidential,
                speaker_role=speaker_role,
            )
            out.append(ec)
        except Exception as e:
            # Last-resort fallback: use dict and leave pydantic to convert later if needed
            out.append(ExtractedClaim(
                claim=str(claim_text),
                polarity="NEUTRAL",
                materiality_0_1=0.15,
                credibility_0_1=0.6,
                surprise_0_1=0.25,
                horizon_profile={"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05},
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=False,
                modality="ASSERTION",
                commitment_0_1=0.5,
                conditionality_0_1=0.0,
                evidential_basis="UNSPECIFIED",
                speaker_role="OTHER",
            ))

    return out


def extract_claims_from_chunk(
    client: OpenAI,
    chunk: str,
    ticker: str,
    doc_type: str,
    source_type: str,
    max_split_depth: int = 2
) -> List[ExtractedClaim]:
    """
    Extract claims from a chunk. Non-destructive approach:
    - If structured parse is available, try it (only if probe enabled).
    - If structured parse fails or is disabled, use responses.create JSON output.
    - If responses.create fails due to token/context size, split chunk recursively and combine results.
    """
    instructions = (
        "You are an expert equity research analyst.\n"
        "Extract ONLY price-relevant, atomic claims.\n"
        "Ignore boilerplate and generic risk language unless it is clearly new/changed.\n"
        "Each claim must be standalone and specific; include key numbers AND period when present.\n"
        "Provide pragmatics fields:\n"
        "- is_forward_looking (future expectations/intentions/guidance)\n"
        "- modality (ASSERTION/FORECAST/INTENTION/CONDITIONAL/RISK/OPINION)\n"
        "- commitment_0_1 (\"will\" > \"expect\" > \"could\")\n"
        "- conditionality_0_1 (\"subject to\"/\"if\" increases)\n"
        "- evidential_basis\n"
        "- speaker_role (COMPANY_OFFICIAL for SEC; otherwise MANAGEMENT/ANALYST/OTHER based on quote)\n"
        "Return between 0 and 8 claims. If nothing material, return an empty list."
    )

    user = (
        f"TICKER: {ticker}\nDOC_TYPE: {doc_type}\nSOURCE_TYPE: {source_type}\n\n"
        f"TEXT:\n{chunk}"
    )

    global STRUCTURED_PARSE_ENABLED
    if STRUCTURED_PARSE_ENABLED:
        try:
            print("[LLM] structured parse request...")
            resp = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user},
                ],
                text_format=ExtractedClaims,
            )
            parsed: ExtractedClaims = resp.output_parsed
            return parsed.claims
        except Exception as e:
            print(f"[LLM] structured parse failed ({type(e).__name__}): {e}")
            STRUCTURED_PARSE_ENABLED = False
            print("[LLM] disabling structured parse for remainder of run; will use JSON fallback.")

    try:
        print("[LLM] calling responses.create with JSON-only instruction...")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": instructions + "\nReturn ONLY valid JSON with top-level key 'claims'."},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_object"}},
        )
        raw = (resp.output_text or "").strip()
        data = _safe_json_load(raw)
        parsed_claims = parse_raw_claims(data, chunk)
        return parsed_claims

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[LLM] responses.create failed: {err_msg}")

        msg = str(e).lower()
        token_error_indicators = ["context length", "max_tokens", "input length", "exceeds", "token limit", "context_length"]
        if any(k in msg for k in token_error_indicators) and max_split_depth > 0:
            print("[LLM] Detected token/context error — splitting chunk and retrying (preserving content).")
            mid = len(chunk) // 2
            left = chunk[:mid]
            right = chunk[mid:]
            left_claims = extract_claims_from_chunk(client, left, ticker, doc_type, source_type, max_split_depth - 1)
            right_claims = extract_claims_from_chunk(client, right, ticker, doc_type, source_type, max_split_depth - 1)
            return left_claims + right_claims

        return []


# -----------------------------
# Reliability update (recency-weighted Brier)
# -----------------------------
def recalc_source_reliability(conn: sqlite3.Connection, source_id: str) -> float:
    """
    Reliability = 1 - min(1, Brier/0.25), with exponential recency weighting.
    Uses p_pred_at_issue stored at resolution time.
    """
    cur = conn.execute(
        "SELECT resolved_at, outcome, p_pred_at_issue FROM resolutions WHERE source_id = ?",
        (source_id,),
    )
    rows = cur.fetchall()
    if not rows:
        rel = 1.0
    else:
        now = utc_now()
        w_sum = 0.0
        b_sum = 0.0
        for resolved_at, outcome, p_pred in rows:
            t = dt.datetime.fromisoformat(resolved_at)
            age_days = max(0.0, (now - t).total_seconds() / 86400.0)
            w = float(np.exp(-age_days / TAU_RELIABILITY_DAYS))
            y = float(outcome)
            p = float(np.clip(p_pred, 0.0, 1.0))
            b = (p - y) ** 2
            w_sum += w
            b_sum += w * b
        B = b_sum / max(1e-12, w_sum)
        rel = 1.0 - min(1.0, B / 0.25)

    conn.execute(
        "INSERT INTO sources(source_id, reliability, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(source_id) DO UPDATE SET reliability=excluded.reliability, updated_at=excluded.updated_at",
        (source_id, float(rel), utc_now().isoformat()),
    )
    conn.commit()
    return float(rel)


def resolve_fact(
    fact_id: str,
    outcome: bool,
    confidence: float,
    evidence: str,
    method: str = "MANUAL",
) -> None:
    """
    Record a resolution and update the relevant source reliability.
    """
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    cur = conn.execute("SELECT source_id, p_true_at_issue FROM facts WHERE fact_id = ?", (fact_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Unknown fact_id: {fact_id}")

    source_id, p_pred_at_issue = row
    resolution_id = f"res_{fact_id}_{hashlib.sha256((evidence+method).encode('utf-8')).hexdigest()[:10]}"

    conn.execute(
        "INSERT OR IGNORE INTO resolutions(resolution_id,fact_id,source_id,resolved_at,outcome,confidence,evidence,method,p_pred_at_issue) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            resolution_id,
            fact_id,
            source_id,
            utc_now().isoformat(),
            1 if outcome else 0,
            float(np.clip(confidence, 0.0, 1.0)),
            evidence.strip(),
            method,
            float(p_pred_at_issue),
        ),
    )
    conn.commit()
    recalc_source_reliability(conn, source_id)
    conn.close()


# -----------------------------
# Ingest pipeline
# -----------------------------
@dataclass
class IngestResult:
    doc_id: str
    ticker: str
    doc_type: str
    source_type: str
    timestamp: str
    n_claims: int
    pred_by_horizon: Dict[str, float]
    pred_near_term: float
    top_claims: List[Dict[str, Any]]


def ingest_document(
    path: Path,
    ticker: str,
    doc_type: str,
    source_type: str,
    timestamp: dt.datetime,
    authority: float,
    url: Optional[str] = None,
    as_of: Optional[dt.datetime] = None,
) -> IngestResult:
    """
    Ingest one artefact into the KB, compute:
    - claim-level delta-awareness predictions per horizon
    - doc-level predictions per horizon

    Use as_of=timestamp for walk-forward demos.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (set it in .env or your shell).")

    as_of = as_of or timestamp

    client = OpenAI()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    doc_hash = sha256_file(path)
    doc_id = f"doc_{ticker}_{doc_hash[:12]}"

    conn.execute(
        "INSERT OR IGNORE INTO docs(doc_id,ticker,doc_type,source_type,timestamp,sha256,url) VALUES (?,?,?,?,?,?,?)",
        (doc_id, ticker, doc_type, source_type, timestamp.isoformat(), doc_hash, url),
    )
    conn.commit()

    text = normalise_to_text(path)
    chunks = sample_chunks(chunk_text(text), TS_MAX_CHUNKS)

    existing_facts = fetch_facts_for_ticker(conn, ticker)

    # Extract
    extracted: List[ExtractedClaim] = []
    for i, ch in enumerate(chunks):
        print(f"[ingest] Extracting claims from chunk {i+1}/{len(chunks)} ...")
        extracted.extend(extract_claims_from_chunk(client, ch, ticker, doc_type, source_type))
        # small pause to avoid tiny bursts to the LLM and keep console readable
        import time
        time.sleep(0.01)

    # Within-doc dedup (embedding)
    merged: List[ExtractedClaim] = []
    seen_vecs: List[np.ndarray] = []
    for c in extracted:
        v = embed(client, c.claim)
        if any(cosine(v, sv) > 0.92 for sv in seen_vecs):
            continue
        seen_vecs.append(v)
        merged.append(c)

    reach = float(SOURCE_REACH.get(source_type, 0.50))
    authority = float(np.clip(authority, 0.0, 1.0))

    pred_by_horizon = {b: 0.0 for b in HORIZON_BUCKETS}
    near_term_total = 0.0

    # For UI ranking
    claim_rows: List[Dict[str, Any]] = []

    n_new = n_known = n_reconfirmed = 0

    for c in merged:
        claim_vec = embed(client, c.claim)
        novelty, matched_id, best_sim = novelty_against_kb(c.claim, claim_vec, existing_facts)

        # Fuzzy similarity for same-fact decision
        fuzz_sim = 0.0
        if matched_id is not None:
            best_text = next(x["canonical_text"] for x in existing_facts if x["fact_id"] == matched_id)
            fuzz_sim = fuzz.token_set_ratio(c.claim, best_text) / 100.0

        same_fact = matched_id is not None and treat_as_same_fact(embed_sim=best_sim, fuzz_sim=fuzz_sim)

        sign = polarity_to_sign(c.polarity)
        horizon = normalise_horizon(c.horizon_profile)

        # Source reliability and pragmatics
        source_id = f"{ticker}:{c.speaker_role}"
        rel = get_source_reliability(conn, source_id)
        p0 = float(np.clip(c.credibility_0_1, 0.0, 1.0))
        p_prag = pragmatics_adjust(p0, c.commitment_0_1, c.conditionality_0_1, c.is_forward_looking)
        p_true_new = apply_reliability(p_prag, rel)

        if not same_fact:
            fact_id = f"fact_{ticker}_{hashlib.sha256(c.claim.encode('utf-8')).hexdigest()[:16]}"
            ts_coef = compute_ts_coef(sign, c.materiality_0_1, novelty, c.surprise_0_1)

            now = utc_now().isoformat()
            conn.execute(
                """
                INSERT OR IGNORE INTO facts(
                    fact_id,ticker,canonical_text,embedding_json,
                    sign,materiality,novelty,surprise,ts_coef,horizon_json,
                    source_id,speaker_role,
                    is_forward_looking,modality,commitment,conditionality,evidential_basis,
                    p0_cred,p_prag,p_true_latest,p_true_at_issue,
                    issued_at,created_at,updated_at
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fact_id, ticker, c.claim, json.dumps(claim_vec.tolist()),
                    int(sign), float(c.materiality_0_1), float(novelty), float(c.surprise_0_1), float(ts_coef), json.dumps(horizon),
                    source_id, c.speaker_role,
                    1 if c.is_forward_looking else 0, c.modality, float(c.commitment_0_1), float(c.conditionality_0_1), c.evidential_basis,
                    float(p0), float(p_prag), float(p_true_new), float(p_true_new),
                    timestamp.isoformat(), now, now
                ),
            )
            conn.commit()

            existing_facts.append({
                "fact_id": fact_id,
                "canonical_text": c.claim,
                "embedding": claim_vec,
                "ts_coef": float(ts_coef),
            })

            status = "NEW"
        else:
            fact_id = matched_id  # type: ignore[assignment]
            cur = conn.execute("SELECT ts_coef, horizon_json, p_true_latest, issued_at FROM facts WHERE fact_id = ?", (fact_id,))
            ts_coef_db, horizon_json_db, p_true_latest_db, issued_at_db = cur.fetchone()
            ts_coef = float(ts_coef_db)
            horizon = json.loads(horizon_json_db)

            # incorporate new evidence monotonically
            p_true_latest = combine_independent_evidence(float(p_true_latest_db), p_true_new, authority)
            conn.execute(
                "UPDATE facts SET p_true_latest=?, updated_at=? WHERE fact_id=?",
                (float(p_true_latest), utc_now().isoformat(), fact_id),
            )
            conn.commit()

            status = status_from_similarity(best_sim)

        if status == "NEW":
            n_new += 1
        elif status == "KNOWN":
            n_known += 1
        else:
            n_reconfirmed += 1

        # Delta-awareness at event time
        aw_before = awareness(conn, fact_id, as_of)
        exposure_id = f"exp_{fact_id}_{doc_id}"
        conn.execute(
            "INSERT OR IGNORE INTO exposures(exposure_id,fact_id,doc_id,source_type,reach,authority,timestamp) VALUES (?,?,?,?,?,?,?)",
            (exposure_id, fact_id, doc_id, source_type, reach, authority, timestamp.isoformat()),
        )
        conn.commit()
        aw_after = awareness(conn, fact_id, as_of)
        delta_aw = float(max(0.0, aw_after - aw_before))

        # Probability used for this event
        cur = conn.execute("SELECT p_true_latest, issued_at FROM facts WHERE fact_id = ?", (fact_id,))
        p_true_used, issued_at = cur.fetchone()
        p_true_used = float(p_true_used)

        # Per-horizon prediction contribution
        pred_h = {b: float(ts_coef * p_true_used * delta_aw * float(horizon[b])) for b in HORIZON_BUCKETS}
        pred_total = float(sum(pred_h.values()))

        for b in HORIZON_BUCKETS:
            pred_by_horizon[b] += pred_h[b]
        near_term_total += float(sum(pred_h[b] * NEAR_WEIGHTS[b] for b in HORIZON_BUCKETS))

        # For ranking in UI
        age_days = max(0.0, (as_of - dt.datetime.fromisoformat(issued_at)).total_seconds() / 86400.0)
        aw_now = aw_after
        impact_rank = impact_now(ts_coef, p_true_used, horizon, age_days, aw_now)

        doc_claim_id = f"dc_{doc_id}_{fact_id}_{hashlib.sha256(c.claim.encode('utf-8')).hexdigest()[:8]}"
        conn.execute(
            """
            INSERT OR IGNORE INTO doc_claims(
                doc_claim_id,doc_id,fact_id,
                extracted_claim,quote,rationale,polarity,
                best_match_similarity,status,
                delta_awareness,p_true_used,pred_horizon_json,pred_total
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                doc_claim_id, doc_id, fact_id,
                c.claim, c.quote, c.rationale, c.polarity,
                float(best_sim), status,
                float(delta_aw), float(p_true_used), json.dumps(pred_h), float(pred_total)
            ),
        )
        conn.commit()

        claim_rows.append({
            "fact_id": fact_id,
            "status": status,
            "best_match_similarity": float(best_sim),
            "delta_awareness": float(delta_aw),
            "ts_coef": float(ts_coef),
            "p_true": float(p_true_used),
            "impact_rank": float(impact_rank),
            "pred_total": float(pred_total),
            "pred_horizon": pred_h,
            "claim": c.claim,
            "quote": c.quote,
            "rationale": c.rationale,
            "pragmatics": {
                "is_forward_looking": bool(c.is_forward_looking),
                "modality": c.modality,
                "commitment_0_1": float(c.commitment_0_1),
                "conditionality_0_1": float(c.conditionality_0_1),
                "evidential_basis": c.evidential_basis,
                "speaker_role": c.speaker_role,
            }
        })

    # Document-level score row
    conn.execute(
        """
        INSERT OR REPLACE INTO doc_scores(
            doc_id,ticker,timestamp,
            pred_horizon_json,pred_near_term,
            n_claims,n_new,n_known,n_reconfirmed
        )
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            doc_id, ticker, timestamp.isoformat(),
            json.dumps(pred_by_horizon),
            float(near_term_total),
            int(len(claim_rows)), int(n_new), int(n_known), int(n_reconfirmed)
        ),
    )
    conn.commit()
    conn.close()

    # Return top claims by impact_rank for showmanship
    claim_rows.sort(key=lambda x: abs(x["impact_rank"]), reverse=True)
    return IngestResult(
        doc_id=doc_id,
        ticker=ticker,
        doc_type=doc_type,
        source_type=source_type,
        timestamp=timestamp.isoformat(),
        n_claims=len(claim_rows),
        pred_by_horizon=pred_by_horizon,
        pred_near_term=float(near_term_total),
        top_claims=claim_rows[:15],
    )


# -----------------------------
# Convenience queries for UI / scripts
# -----------------------------
def list_docs(ticker: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute(
        "SELECT d.doc_id, d.doc_type, d.source_type, d.timestamp, s.pred_near_term, s.pred_horizon_json, s.n_new, s.n_known, s.n_reconfirmed "
        "FROM docs d LEFT JOIN doc_scores s ON d.doc_id = s.doc_id "
        "WHERE d.ticker = ? ORDER BY d.timestamp ASC",
        (ticker,),
    )
    out = []
    for row in cur.fetchall():
        doc_id, doc_type, source_type, ts, pred_near, pred_h_json, n_new, n_known, n_reconf = row
        out.append({
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_type": source_type,
            "timestamp": ts,
            "pred_near_term": float(pred_near or 0.0),
            "pred_horizon": json.loads(pred_h_json) if pred_h_json else {b: 0.0 for b in HORIZON_BUCKETS},
            "n_new": int(n_new or 0),
            "n_known": int(n_known or 0),
            "n_reconfirmed": int(n_reconf or 0),
        })
    conn.close()
    return out


def list_doc_claims(doc_id: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute(
        "SELECT fact_id, extracted_claim, status, best_match_similarity, delta_awareness, p_true_used, pred_total, pred_horizon_json, quote, rationale "
        "FROM doc_claims WHERE doc_id = ?",
        (doc_id,),
    )
    out = []
    for r in cur.fetchall():
        fact_id, claim, status, sim, da, ptrue, ptot, phj, quote, rationale = r
        out.append({
            "fact_id": fact_id,
            "claim": claim,
            "status": status,
            "best_match_similarity": float(sim),
            "delta_awareness": float(da),
            "p_true_used": float(ptrue),
            "pred_total": float(ptot),
            "pred_horizon": json.loads(phj),
            "quote": quote,
            "rationale": rationale,
        })
    conn.close()
    out.sort(key=lambda x: abs(x["pred_total"]), reverse=True)
    return out


def list_unresolved_forward_looking(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute(
        """
        SELECT f.fact_id, f.canonical_text, f.source_id, f.speaker_role, f.modality, f.commitment, f.conditionality, f.p_true_at_issue, f.issued_at
        FROM facts f
        LEFT JOIN resolutions r ON f.fact_id = r.fact_id
        WHERE f.ticker = ? AND f.is_forward_looking = 1 AND r.fact_id IS NULL
        ORDER BY f.issued_at DESC
        LIMIT ?
        """,
        (ticker, int(limit)),
    )
    out = []
    for r in cur.fetchall():
        fact_id, text, source_id, speaker_role, modality, commitment, conditionality, p_issue, issued_at = r
        out.append({
            "fact_id": fact_id,
            "claim": text,
            "source_id": source_id,
            "speaker_role": speaker_role,
            "modality": modality,
            "commitment_0_1": float(commitment),
            "conditionality_0_1": float(conditionality),
            "p_true_at_issue": float(p_issue),
            "issued_at": issued_at,
        })
    conn.close()
    return out


def list_sources() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute("SELECT source_id, reliability, updated_at FROM sources ORDER BY source_id ASC")
    out = [{"source_id": s, "reliability": float(r), "updated_at": t} for (s, r, t) in cur.fetchall()]
    conn.close()
    return out
