from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .awareness_exposure_stage import apply_exposure_and_measure, horizon_prediction
from .claim_extraction_stage import extract_and_deduplicate_claims
from .claim_scoring_stage import score_claim_against_kb
from .chunking_stage import build_chunks
from .document_scoring_stage import persist_document_score
from .fact_storage_stage import upsert_fact
from .ingestion_stage import register_document
from .normalization_stage import normalize_document
from ts_system import (
    DB_PATH,
    HORIZON_BUCKETS,
    NEAR_WEIGHTS,
    SOURCE_REACH,
    _supports_structured_parse,
    embed,
    fetch_facts_for_ticker,
    impact_now,
    init_db,
)


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


def ingest_document_pipeline(
    path: Path,
    ticker: str,
    doc_type: str,
    source_type: str,
    timestamp: dt.datetime,
    authority: float,
    url: Optional[str] = None,
    as_of: Optional[dt.datetime] = None,
) -> IngestResult:
    """Pipeline orchestrator: ingestion → normalization → ... → document scoring."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (set it in .env or your shell).")

    as_of = as_of or timestamp
    client = OpenAI()

    import ts_system as ts
    if ts.STRUCTURED_PARSE_ENABLED:
        ts.STRUCTURED_PARSE_ENABLED = _supports_structured_parse(client)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    doc_id, _ = register_document(conn, path, ticker, doc_type, source_type, timestamp.isoformat(), url)
    text = normalize_document(path)
    chunks = build_chunks(text)
    existing_facts = fetch_facts_for_ticker(conn, ticker)
    claims = extract_and_deduplicate_claims(client, chunks, ticker, doc_type, source_type)

    reach = float(SOURCE_REACH.get(source_type, 0.50))
    authority = float(np.clip(authority, 0.0, 1.0))
    pred_by_horizon = {b: 0.0 for b in HORIZON_BUCKETS}
    near_term_total = 0.0
    claim_rows: List[Dict[str, Any]] = []
    n_new = n_known = n_reconfirmed = 0

    for claim in claims:
        claim_vec = embed(client, claim.claim)
        scoring = score_claim_against_kb(conn, claim, claim_vec, existing_facts, ticker, source_type, timestamp.isoformat())
        fact_state = upsert_fact(conn, claim, claim_vec, scoring, ticker, timestamp.isoformat(), authority, existing_facts)

        status = fact_state["status"]
        if status == "NEW":
            n_new += 1
        elif status == "KNOWN":
            n_known += 1
        else:
            n_reconfirmed += 1

        awareness_state = apply_exposure_and_measure(
            conn, fact_state["fact_id"], doc_id, source_type, reach, authority, timestamp.isoformat(), as_of
        )

        cur = conn.execute("SELECT p_true_latest, issued_at FROM facts WHERE fact_id = ?", (fact_state["fact_id"],))
        p_true_used, issued_at = cur.fetchone()
        p_true_used = float(p_true_used)

        pred_h = horizon_prediction(fact_state["ts_coef"], p_true_used, awareness_state["delta_aw"], fact_state["horizon"])
        pred_total = float(sum(pred_h.values()))

        for b in HORIZON_BUCKETS:
            pred_by_horizon[b] += pred_h[b]
        near_term_total += float(sum(pred_h[b] * NEAR_WEIGHTS[b] for b in HORIZON_BUCKETS))

        age_days = max(0.0, (as_of - dt.datetime.fromisoformat(issued_at)).total_seconds() / 86400.0)
        impact_rank = impact_now(fact_state["ts_coef"], p_true_used, fact_state["horizon"], age_days, awareness_state["aw_after"])

        doc_claim_id = f"dc_{doc_id}_{fact_state['fact_id']}_{hashlib.sha256(claim.claim.encode('utf-8')).hexdigest()[:8]}"
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
                doc_claim_id, doc_id, fact_state["fact_id"],
                claim.claim, claim.quote, claim.rationale, claim.polarity,
                float(fact_state["best_sim"]), status,
                float(awareness_state["delta_aw"]), float(p_true_used), json.dumps(pred_h), float(pred_total),
            ),
        )
        conn.commit()

        claim_rows.append({
            "fact_id": fact_state["fact_id"],
            "status": status,
            "best_match_similarity": float(fact_state["best_sim"]),
            "delta_awareness": float(awareness_state["delta_aw"]),
            "ts_coef": float(fact_state["ts_coef"]),
            "p_true": float(p_true_used),
            "impact_rank": float(impact_rank),
            "pred_total": float(pred_total),
            "pred_horizon": pred_h,
            "claim": claim.claim,
            "quote": claim.quote,
            "rationale": claim.rationale,
            "pragmatics": {
                "is_forward_looking": bool(claim.is_forward_looking),
                "modality": claim.modality,
                "commitment_0_1": float(claim.commitment_0_1),
                "conditionality_0_1": float(claim.conditionality_0_1),
                "evidential_basis": claim.evidential_basis,
                "speaker_role": claim.speaker_role,
            },
        })

    persist_document_score(
        conn, doc_id, ticker, timestamp.isoformat(), pred_by_horizon, near_term_total,
        len(claim_rows), n_new, n_known, n_reconfirmed,
    )
    conn.close()

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
