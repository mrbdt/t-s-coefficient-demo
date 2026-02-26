from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict

from ts_system import combine_independent_evidence, status_from_similarity, utc_now


def upsert_fact(
    conn: sqlite3.Connection,
    claim,
    claim_vec,
    scoring: Dict[str, Any],
    ticker: str,
    timestamp_iso: str,
    authority: float,
    existing_facts,
) -> Dict[str, Any]:
    """Create a new fact or update an existing one with new evidence."""
    if not scoring["same_fact"]:
        fact_id = f"fact_{ticker}_{hashlib.sha256(claim.claim.encode('utf-8')).hexdigest()[:16]}"
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
                fact_id, ticker, claim.claim, json.dumps(claim_vec.tolist()),
                scoring["sign"], float(claim.materiality_0_1), scoring["novelty"], float(claim.surprise_0_1), scoring["ts_coef"], json.dumps(scoring["horizon"]),
                scoring["source_id"], claim.speaker_role,
                1 if claim.is_forward_looking else 0, claim.modality, float(claim.commitment_0_1), float(claim.conditionality_0_1), claim.evidential_basis,
                scoring["p0"], scoring["p_prag"], scoring["p_true_new"], scoring["p_true_new"],
                timestamp_iso, now, now
            ),
        )
        conn.commit()
        existing_facts.append({"fact_id": fact_id, "canonical_text": claim.claim, "embedding": claim_vec, "ts_coef": scoring["ts_coef"]})
        return {"fact_id": fact_id, "status": "NEW", "ts_coef": scoring["ts_coef"], "horizon": scoring["horizon"], "best_sim": scoring["best_sim"]}

    fact_id = scoring["matched_id"]
    cur = conn.execute("SELECT ts_coef, horizon_json, p_true_latest FROM facts WHERE fact_id = ?", (fact_id,))
    ts_coef_db, horizon_json_db, p_true_latest_db = cur.fetchone()
    p_true_latest = combine_independent_evidence(float(p_true_latest_db), scoring["p_true_new"], authority)
    conn.execute("UPDATE facts SET p_true_latest=?, updated_at=? WHERE fact_id=?", (float(p_true_latest), utc_now().isoformat(), fact_id))
    conn.commit()
    return {
        "fact_id": fact_id,
        "status": status_from_similarity(scoring["best_sim"]),
        "ts_coef": float(ts_coef_db),
        "horizon": json.loads(horizon_json_db),
        "best_sim": scoring["best_sim"],
    }
