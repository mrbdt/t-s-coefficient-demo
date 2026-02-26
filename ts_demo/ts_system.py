"""Compatibility API surface for the T-S demo system.

Core building blocks are split into smaller modules under `core/`:
- `core.config` for environment/config constants
- `core.models` for extraction schemas
- `core.text_processing` for normalization/chunking helpers

This file keeps legacy function names for scripts/UI while hosting DB, scoring,
and orchestration utilities.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from rapidfuzz import fuzz

from core.config import (
    DB_PATH, EMBED_SAME_LOOSE, EMBED_SAME_STRICT, FUZZ_SAME_STRICT, HALF_LIFE_DAYS,
    HORIZON_BUCKETS, HORIZON_TRADING_DAYS, LAMBDA_DECAY, NEAR_WEIGHTS, OPENAI_EMBED_MODEL,
    OPENAI_MODEL, SOURCE_REACH, TAU_RELIABILITY_DAYS, TS_MAX_CHUNKS, status_from_similarity,
)
from core.models import ExtractedClaim, ExtractedClaims, Polarity
from core.text_processing import (
    chunk_text,
    normalise_to_text,
    sample_chunks,
    table_to_markdown_from_bs4,
    table_to_markdown_from_list,
)

def init_db(conn: sqlite3.Connection) -> None:
    """Create all required tables and indexes for the demo knowledge base."""
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
    """Compute a stable SHA-256 fingerprint for file identity and deduping."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> dt.datetime:
    """Return timezone-aware UTC now (avoids naive datetime bugs)."""
    return dt.datetime.now(dt.timezone.utc)


def polarity_to_sign(p: Polarity) -> int:
    """Convert categorical polarity into numeric sign used by scoring math."""
    if p == "BULLISH":
        return +1
    if p == "BEARISH":
        return -1
    return 0


def normalise_horizon(h: Dict[str, float]) -> Dict[str, float]:
    """Normalize horizon weights to a proper probability distribution.

    Guarantees:
    - All required buckets exist.
    - Values sum to ~1.0.
    - Falls back to a reasonable default profile if input is empty/invalid.
    """
    out = {k: float(h.get(k, 0.0)) for k in HORIZON_BUCKETS}
    s = float(sum(out.values()))
    if s <= 0:
        out = {"1D": 0.25, "1W": 0.25, "1M": 0.20, "3M": 0.15, "1Y": 0.10, "3Y": 0.05}
        s = float(sum(out.values()))
    return {k: out[k] / s for k in HORIZON_BUCKETS}


def embed(client: OpenAI, text: str) -> np.ndarray:
    """Create a unit-normalized embedding vector for similarity comparisons."""
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
    """Best-effort JSON parser for imperfect model outputs.

    We first try strict JSON, then attempt to recover the first JSON object-shaped
    substring. If both fail, we return an empty claims payload rather than crashing.
    """
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
    """Defensively coerce loosely-structured model JSON into `ExtractedClaim` objects.

    This parser exists because model outputs can drift. Instead of failing hard, we
    normalize fields, clamp ranges, and provide sensible defaults so ingestion can
    proceed while still producing auditable records.
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
    """Cosine similarity for pre-normalized vectors (dot product shortcut)."""
    return float(np.dot(a, b))


def get_source_reliability(conn: sqlite3.Connection, source_id: str) -> float:
    """Fetch source reliability prior; initialize to 1.0 if unseen."""
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
    """Baseline T-S coefficient (legacy/simple formula).

    Interpreted as directional impact magnitude under the counterfactual where the
    claim is true and broadly diffused in the market.
    """
    return float(sign) * float(materiality) * float(novelty) * float(surprise)

def compute_ts_coef_enhanced(
    sign: int,
    materiality: float,
    novelty: float,
    surprise: float,
    credibility: float,
    p_true: float,
    horizon: Dict[str, float],
    issued_at_iso: Optional[str],
    source_type: Optional[str],
) -> float:
    """
    Enhanced, pragmatic computation of ts_coef (signed).

    Formula (v1):
      ts_coef = sign * p_true * materiality * novelty * credibility * surprise * reach * near_term_weight * time_decay

    - sign: +1 / -1 / 0
    - p_true: probability used at issue (already adjusted by pragmatics & source reliability)
    - materiality, novelty, credibility, surprise : in [0,1]
    - reach: estimated audience reach for source_type (uses SOURCE_REACH fallback)
    - near_term_weight: weighted importance across horizons using NEAR_WEIGHTS
    - time_decay: exponential decay by age (TAU_RELIABILITY_DAYS)
    """
    try:
        sign = int(sign)
    except Exception:
        sign = 0

    # clamp inputs
    materiality = float(max(0.0, min(1.0, materiality or 0.0)))
    novelty = float(max(0.0, min(1.0, novelty or 0.0)))
    surprise = float(max(0.0, min(1.0, surprise or 0.0)))
    credibility = float(max(0.0, min(1.0, credibility or 0.0)))
    p_true = float(max(0.0, min(1.0, p_true or 0.5)))

    # reach estimate
    reach = float(SOURCE_REACH.get((source_type or "").upper(), 0.5))

    # near-term weight from horizon distribution
    near_term_weight = 0.0
    try:
        for b in HORIZON_BUCKETS:
            near_term_weight += float(horizon.get(b, 0.0)) * float(NEAR_WEIGHTS.get(b, 0.0))
    except Exception:
        near_term_weight = 0.5
    if near_term_weight <= 0.0:
        near_term_weight = 0.5

    # age / time decay
    age_days = 0.0
    if issued_at_iso:
        try:
            # issued_at_iso is an ISO timestamp string like timestamp.isoformat()
            issued_dt = dt.datetime.fromisoformat(issued_at_iso)
            age_days = max(0.0, (utc_now().replace(tzinfo=None) - issued_dt).days)
        except Exception:
            age_days = 0.0
    time_decay = float(math.exp(-age_days / max(1.0, TAU_RELIABILITY_DAYS)))

    magnitude = p_true * materiality * novelty * credibility * surprise * reach * near_term_weight * time_decay

    ts_coef = float(sign) * float(magnitude)

    # suppress sub-floating noise
    if abs(ts_coef) < 1e-12:
        return 0.0
    return float(ts_coef)

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
    """Load existing canonical facts for dedupe against new extracted claims."""
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
    """Compute novelty vs. knowledge base and return best candidate match.

    Returns `(novelty, best_fact_id, best_similarity)` where novelty is `1-sim`.
    Similarity blends embedding proximity with fuzzy token overlap for robustness.
    """
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
    """Decision rule for whether a new claim should merge into an existing fact."""
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
    """Extract valuation-relevant atomic claims from a text chunk.

    Strategy:
    1) Prefer structured parsing into `ExtractedClaims` for schema safety.
    2) If unavailable/failing, fallback to JSON mode and coerce via `parse_raw_claims`.
    3) If chunk is still too noisy/long, caller may recursively split and retry.

    This function intentionally never mutates KB state; it only returns parsed claims.
    """
    instructions = (
        "You are an expert equity research analyst.\n"
        "Extract ONLY price-relevant, atomic claims.\n"
        "Ignore boilerplate and generic risk language unless it is clearly new/changed.\n"
        "Each claim must be standalone and specific; include key numbers AND period when present.\n"
        "\n"
        "For EACH claim, output the following fields:\n"
        "- claim (string)\n"
        "- polarity (one of: BULLISH, BEARISH, MIXED, NEUTRAL)\n"
        "- materiality_0_1 (0..1)\n"
        "- credibility_0_1 (0..1)\n"
        "- surprise_0_1 (0..1)\n"
        "- horizon_profile (object with keys: 1D, 1W, 1M, 3M, 1Y, 3Y; values sum to ~1)\n"
        "- rationale (string)\n"
        "- quote (string; exact supporting snippet)\n"
        "- is_forward_looking (boolean)\n"
        "- modality (one of: ASSERTION, FORECAST, INTENTION, CONDITIONAL, RISK, OPINION)\n"
        "- commitment_0_1 (0..1; \"will\" > \"expect\" > \"could\")\n"
        "- conditionality_0_1 (0..1; \"subject to\"/\"if\" increases)\n"
        "- evidential_basis (one of: REPORTED_NUMBER, OPERATIONAL_OBSERVATION, INTERNAL_METRIC, UNSPECIFIED)\n"
        "- speaker_role (one of: COMPANY_OFFICIAL, MANAGEMENT, ANALYST, OTHER; COMPANY_OFFICIAL for SEC)\n"
        "\n"
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

    # Fallback to JSON-only responses.create
    try:
        print("[LLM] calling responses.create with JSON-only instruction...")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": instructions + "\nReturn ONLY valid JSON matching: {\"claims\": [ ... ]}. Do not include any extra keys outside 'claims'."},
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
    """Recompute a source reliability score from historical resolutions.

    Formula details:
    - Compute recency-weighted Brier error over `(p_pred_at_issue, outcome)` pairs.
    - Convert Brier -> reliability via `1 - min(1, Brier/0.25)` so range is [0, 1].
    - Persist the updated reliability in `sources` for future ingest runs.
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
    """Record a fact resolution and trigger reliability recalibration.

    Intended usage:
    - Called by analysts (manual adjudication) or automated evaluators.
    - Writes an immutable resolution event, then updates source reliability priors.
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
    """Ingest a single source document via the staged pipeline orchestrator."""
    from pipeline.pipeline_main import ingest_document_pipeline

    result = ingest_document_pipeline(
        path=path,
        ticker=ticker,
        doc_type=doc_type,
        source_type=source_type,
        timestamp=timestamp,
        authority=authority,
        url=url,
        as_of=as_of,
    )
    return IngestResult(
        doc_id=result.doc_id,
        ticker=result.ticker,
        doc_type=result.doc_type,
        source_type=result.source_type,
        timestamp=result.timestamp,
        n_claims=result.n_claims,
        pred_by_horizon=result.pred_by_horizon,
        pred_near_term=result.pred_near_term,
        top_claims=result.top_claims,
    )


# -----------------------------
# Convenience queries for UI / scripts
# -----------------------------
def list_docs(ticker: str) -> List[Dict[str, Any]]:
    """List ingested documents for a ticker (newest first)."""
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
    """List all claim rows associated with a given document id."""
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
    """Return forward-looking facts that have not yet been resolved/adjudicated."""
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
    """List sources and their current reliability estimates."""
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute("SELECT source_id, reliability, updated_at FROM sources ORDER BY source_id ASC")
    out = [{"source_id": s, "reliability": float(r), "updated_at": t} for (s, r, t) in cur.fetchall()]
    conn.close()
    return out
