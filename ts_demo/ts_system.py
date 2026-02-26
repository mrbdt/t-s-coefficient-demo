"""Public compatibility API for the T-S demo.

This module now re-exports functionality from smaller modules so callers can keep
using `ts_system` while the implementation stays modular.
"""

from __future__ import annotations

from core.config import (
    DB_PATH,
    EMBED_SAME_LOOSE,
    EMBED_SAME_STRICT,
    FUZZ_SAME_STRICT,
    HALF_LIFE_DAYS,
    HORIZON_BUCKETS,
    HORIZON_TRADING_DAYS,
    LAMBDA_DECAY,
    NEAR_WEIGHTS,
    OPENAI_EMBED_MODEL,
    OPENAI_MODEL,
    SOURCE_REACH,
    TAU_RELIABILITY_DAYS,
    TS_MAX_CHUNKS,
    status_from_similarity,
)
from core.database import init_db, sha256_file, utc_now
from core.extraction import (
    STRUCTURED_PARSE_ENABLED,
    _safe_json_load,
    _supports_structured_parse,
    embed,
    extract_claims_from_chunk,
    parse_raw_claims,
)
from core.ingest_api import IngestResult, ingest_document
from core.models import (
    Evidential,
    ExtractedClaim,
    ExtractedClaims,
    Modality,
    Polarity,
    SpeakerRole,
)
from core.query import list_doc_claims, list_docs, list_unresolved_forward_looking
from core.reliability import list_sources, recalc_source_reliability, resolve_fact
from core.scoring import (
    apply_reliability,
    awareness,
    combine_independent_evidence,
    compute_ts_coef,
    compute_ts_coef_enhanced,
    cosine,
    fetch_facts_for_ticker,
    get_source_reliability,
    impact_now,
    normalise_horizon,
    novelty_against_kb,
    polarity_to_sign,
    pragmatics_adjust,
    treat_as_same_fact,
)
from core.text_processing import (
    chunk_text,
    normalise_to_text,
    sample_chunks,
    table_to_markdown_from_bs4,
    table_to_markdown_from_list,
)
