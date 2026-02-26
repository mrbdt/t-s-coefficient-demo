from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

from core.config import DB_PATH, HORIZON_BUCKETS
from core.database import init_db


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
