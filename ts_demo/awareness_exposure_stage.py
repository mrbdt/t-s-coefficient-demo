from __future__ import annotations

import sqlite3
from typing import Dict

from ts_system import HORIZON_BUCKETS, awareness


def apply_exposure_and_measure(
    conn: sqlite3.Connection,
    fact_id: str,
    doc_id: str,
    source_type: str,
    reach: float,
    authority: float,
    timestamp_iso: str,
    as_of,
) -> Dict[str, float]:
    """Store an exposure row and return awareness before/after and delta."""
    aw_before = awareness(conn, fact_id, as_of)
    exposure_id = f"exp_{fact_id}_{doc_id}"
    conn.execute(
        "INSERT OR IGNORE INTO exposures(exposure_id,fact_id,doc_id,source_type,reach,authority,timestamp) VALUES (?,?,?,?,?,?,?)",
        (exposure_id, fact_id, doc_id, source_type, reach, authority, timestamp_iso),
    )
    conn.commit()
    aw_after = awareness(conn, fact_id, as_of)
    delta_aw = float(max(0.0, aw_after - aw_before))
    return {"aw_before": float(aw_before), "aw_after": float(aw_after), "delta_aw": delta_aw}


def horizon_prediction(ts_coef: float, p_true_used: float, delta_aw: float, horizon: Dict[str, float]) -> Dict[str, float]:
    """Compute per-horizon prediction contribution for one claim observation."""
    return {b: float(ts_coef * p_true_used * delta_aw * float(horizon[b])) for b in HORIZON_BUCKETS}
