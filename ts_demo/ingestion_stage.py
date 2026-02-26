from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from ts_system import sha256_file


def register_document(
    conn: sqlite3.Connection,
    path: Path,
    ticker: str,
    doc_type: str,
    source_type: str,
    timestamp_iso: str,
    url: Optional[str],
) -> Tuple[str, str]:
    """Create or reuse a document row and return (doc_id, doc_hash)."""
    doc_hash = sha256_file(path)
    doc_id = f"doc_{ticker}_{doc_hash[:12]}"
    conn.execute(
        "INSERT OR IGNORE INTO docs(doc_id,ticker,doc_type,source_type,timestamp,sha256,url) VALUES (?,?,?,?,?,?,?)",
        (doc_id, ticker, doc_type, source_type, timestamp_iso, doc_hash, url),
    )
    conn.commit()
    return doc_id, doc_hash
