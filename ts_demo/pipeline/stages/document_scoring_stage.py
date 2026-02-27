from __future__ import annotations

import json
import sqlite3


def persist_document_score(
    conn: sqlite3.Connection,
    doc_id: str,
    ticker: str,
    timestamp_iso: str,
    pred_by_horizon,
    pred_near_term: float,
    n_claims: int,
    n_new: int,
    n_known: int,
    n_reconfirmed: int,
) -> None:
    """Store document-level aggregate scoring output."""
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
            doc_id, ticker, timestamp_iso,
            json.dumps(pred_by_horizon),
            float(pred_near_term),
            int(n_claims), int(n_new), int(n_known), int(n_reconfirmed),
        ),
    )
    conn.commit()
