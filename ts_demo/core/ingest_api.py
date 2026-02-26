from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
