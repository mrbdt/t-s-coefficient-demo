from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from core.config import (
    EMBED_SAME_LOOSE, EMBED_SAME_STRICT, FUZZ_SAME_STRICT, HALF_LIFE_DAYS,
    HORIZON_BUCKETS, LAMBDA_DECAY, NEAR_WEIGHTS, SOURCE_REACH, TAU_RELIABILITY_DAYS,
)
from core.models import Polarity
from core.database import utc_now


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
