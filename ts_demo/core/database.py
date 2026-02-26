from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from pathlib import Path


def init_db(conn: sqlite3.Connection) -> None:
    """Create all required tables and indexes for the demo knowledge base."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        canonical_text TEXT NOT NULL,
        embedding_json TEXT NOT NULL,

        sign INTEGER NOT NULL,
        materiality REAL NOT NULL,
        novelty REAL NOT NULL,
        surprise REAL NOT NULL,
        ts_coef REAL NOT NULL,
        horizon_json TEXT NOT NULL,

        source_id TEXT NOT NULL,
        speaker_role TEXT NOT NULL,

        is_forward_looking INTEGER NOT NULL,
        modality TEXT NOT NULL,
        commitment REAL NOT NULL,
        conditionality REAL NOT NULL,
        evidential_basis TEXT NOT NULL,

        p0_cred REAL NOT NULL,
        p_prag REAL NOT NULL,
        p_true_latest REAL NOT NULL,
        p_true_at_issue REAL NOT NULL,

        issued_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS exposures (
        exposure_id TEXT PRIMARY KEY,
        fact_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        reach REAL NOT NULL,
        authority REAL NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        doc_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        source_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        url TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS doc_claims (
        doc_claim_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        fact_id TEXT NOT NULL,

        extracted_claim TEXT NOT NULL,
        quote TEXT NOT NULL,
        rationale TEXT NOT NULL,
        polarity TEXT NOT NULL,

        best_match_similarity REAL NOT NULL,
        status TEXT NOT NULL,

        delta_awareness REAL NOT NULL,
        p_true_used REAL NOT NULL,
        pred_horizon_json TEXT NOT NULL,
        pred_total REAL NOT NULL,

        FOREIGN KEY(doc_id) REFERENCES docs(doc_id),
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS doc_scores (
        doc_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,

        pred_horizon_json TEXT NOT NULL,
        pred_near_term REAL NOT NULL,

        n_claims INTEGER NOT NULL,
        n_new INTEGER NOT NULL,
        n_known INTEGER NOT NULL,
        n_reconfirmed INTEGER NOT NULL,

        FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        reliability REAL NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS resolutions (
        resolution_id TEXT PRIMARY KEY,
        fact_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        resolved_at TEXT NOT NULL,
        outcome INTEGER NOT NULL,     -- 1 true, 0 false
        confidence REAL NOT NULL,
        evidence TEXT NOT NULL,
        method TEXT NOT NULL,
        p_pred_at_issue REAL NOT NULL,
        FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
    );
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_ticker ON facts(ticker);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exposures_fact ON exposures(fact_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ticker_time ON docs(ticker, timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docclaims_doc ON doc_claims(doc_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resolutions_source ON resolutions(source_id);")

    conn.commit()

def sha256_file(path: Path) -> str:
    """Compute a stable SHA-256 fingerprint for file identity and deduping."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def utc_now() -> dt.datetime:
    """Return timezone-aware UTC now (avoids naive datetime bugs)."""
    return dt.datetime.now(dt.timezone.utc)
