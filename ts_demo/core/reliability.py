from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, List

import numpy as np

from core.config import DB_PATH
from core.database import init_db, utc_now


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

def list_sources() -> List[Dict[str, Any]]:
    """List sources and their current reliability estimates."""
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute("SELECT source_id, reliability, updated_at FROM sources ORDER BY source_id ASC")
    out = [{"source_id": s, "reliability": float(r), "updated_at": t} for (s, r, t) in cur.fetchall()]
    conn.close()
    return out
