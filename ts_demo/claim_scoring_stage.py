from __future__ import annotations

import numpy as np
import sqlite3
from typing import Any, Dict

from rapidfuzz import fuzz

from ts_system import (
    ExtractedClaim,
    apply_reliability,
    compute_ts_coef,
    compute_ts_coef_enhanced,
    get_source_reliability,
    normalise_horizon,
    novelty_against_kb,
    polarity_to_sign,
    pragmatics_adjust,
    treat_as_same_fact,
)


def score_claim_against_kb(
    conn: sqlite3.Connection,
    claim: ExtractedClaim,
    claim_vec,
    existing_facts,
    ticker: str,
    source_type: str,
    timestamp_iso: str,
) -> Dict[str, Any]:
    """Compute novelty, probability and ts-coefficient details for one claim."""
    novelty, matched_id, best_sim = novelty_against_kb(claim.claim, claim_vec, existing_facts)

    fuzz_sim = 0.0
    if matched_id is not None:
        best_text = next(x["canonical_text"] for x in existing_facts if x["fact_id"] == matched_id)
        fuzz_sim = fuzz.token_set_ratio(claim.claim, best_text) / 100.0
    same_fact = matched_id is not None and treat_as_same_fact(embed_sim=best_sim, fuzz_sim=fuzz_sim)

    sign = polarity_to_sign(claim.polarity)
    horizon = normalise_horizon(claim.horizon_profile)
    source_id = f"{ticker}:{claim.speaker_role}"
    rel = get_source_reliability(conn, source_id)

    p0 = float(np.clip(claim.credibility_0_1, 0.0, 1.0))
    p_prag = pragmatics_adjust(p0, claim.commitment_0_1, claim.conditionality_0_1, claim.is_forward_looking)
    p_true_new = apply_reliability(p_prag, rel)

    try:
        ts_coef = compute_ts_coef_enhanced(
            sign=int(sign),
            materiality=float(claim.materiality_0_1),
            novelty=float(novelty),
            surprise=float(claim.surprise_0_1),
            credibility=float(claim.credibility_0_1),
            p_true=float(p_true_new),
            horizon=horizon,
            issued_at_iso=timestamp_iso,
            source_type=source_type,
        )
    except Exception:
        ts_coef = compute_ts_coef(sign, claim.materiality_0_1, novelty, claim.surprise_0_1)

    return {
        "novelty": float(novelty),
        "matched_id": matched_id,
        "best_sim": float(best_sim),
        "same_fact": bool(same_fact),
        "sign": int(sign),
        "horizon": horizon,
        "source_id": source_id,
        "p0": float(p0),
        "p_prag": float(p_prag),
        "p_true_new": float(p_true_new),
        "ts_coef": float(ts_coef),
    }
