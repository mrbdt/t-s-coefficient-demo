"""
Core pipeline for the T-S coefficient demo.

High-level flow implemented in this module:
1) Normalize raw documents (HTML/PDF/DOCX) into plain analysis text.
2) Ask an LLM to extract atomic valuation-relevant claims.
3) Deduplicate claims against existing facts using embeddings + fuzzy matching.
4) Update fact-level probabilities/reliability and compute impact scores.
5) Persist provenance so analysts can audit every score back to source text.

Most functions below are designed to be individually readable and testable:
- pure scoring/probability helpers (math-only)
- storage helpers (SQL writes/reads)
- orchestration helpers (chunking, extraction, ingestion)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import re
import math

import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# -----------------------------
# Environment / Config
# -----------------------------
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DB_PATH = os.getenv("TS_DB_PATH", "system_db.sqlite3")

PIPELINE_DIR = Path(os.getenv("TS_PIPELINE_DIR", "pipeline"))
EVENT_EVAL_DIR = Path(os.getenv("TS_EVENT_EVAL_DIR", "event_evaluation_results"))
PIPELINE_STAGE_DIRS = {
    "inputs": PIPELINE_DIR / "1_ingested_inputs",
    "normalised": PIPELINE_DIR / "2_normalised_documents",
    "all_chunks": PIPELINE_DIR / "3_all_chunks",
    "sampled_chunks": PIPELINE_DIR / "4_sampled_chunks",
    "chunk_claims": PIPELINE_DIR / "5_chunk_claims",
    "merged_claims": PIPELINE_DIR / "6_merged_claims",
    "fact_matching": PIPELINE_DIR / "7_fact_matching",
    "doc_outputs": PIPELINE_DIR / "8_doc_outputs",
}

TS_MAX_CHUNKS = int(os.getenv("TS_MAX_CHUNKS", "30"))
LAMBDA_DECAY = float(os.getenv("TS_LAMBDA_DECAY", "0.002"))  # awareness decay for old exposures
TAU_RELIABILITY_DAYS = float(os.getenv("TS_TAU_RELIABILITY_DAYS", "365"))  # recency weighting

HORIZON_BUCKETS = ["1D", "1W", "1M", "3M", "1Y", "3Y"]
HORIZON_TRADING_DAYS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252, "3Y": 756}

# Heuristic weights (for "impact now" / UI ranking)
NEAR_WEIGHTS = {"1D": 1.00, "1W": 0.75, "1M": 0.55, "3M": 0.40, "1Y": 0.25, "3Y": 0.15}
HALF_LIFE_DAYS = {"1D": 2, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "3Y": 900}

# Reach defaults (v1)
SOURCE_REACH = {
    "SEC_FILING": 0.70,
    "EARNINGS_CALL": 0.65,
    "PRESS_RELEASE": 0.75,
    "NEWS_PAYWALLED": 0.55,
    "NEWS_FREE": 0.80,
    "X_CSUITE": 0.60,
    "X_ANALYST": 0.45,
    "RUMOUR": 0.25,
}

# Dedup thresholds
EMBED_SAME_STRICT = 0.93
EMBED_SAME_LOOSE = 0.85
FUZZ_SAME_STRICT = 0.92

# Status labels for demo readability
def status_from_similarity(sim: float) -> str:
    """Map a numeric similarity score to a human-readable lifecycle label.

    Why this helper exists:
    - Analysts prefer categorical tags (NEW/KNOWN/RECONFIRMED) over raw cosine values.
    - The thresholds are intentionally simple, deterministic, and easy to tune.

    Threshold semantics:
    - >= 0.95  -> RECONFIRMED: nearly identical claim seen again.
    - >= 0.65  -> KNOWN: substantially overlaps an existing fact.
    - <  0.65  -> NEW: treated as a new fact candidate.
    """
    if sim >= 0.95:
        return "RECONFIRMED"
    if sim >= 0.65:
        return "KNOWN"
    return "NEW"


# -----------------------------
# Structured extraction schema
# -----------------------------
Polarity = Literal["BULLISH", "BEARISH", "MIXED", "NEUTRAL"]
Modality = Literal["ASSERTION", "FORECAST", "INTENTION", "CONDITIONAL", "RISK", "OPINION"]
Evidential = Literal["REPORTED_NUMBER", "OPERATIONAL_OBSERVATION", "INTERNAL_METRIC", "UNSPECIFIED"]
SpeakerRole = Literal["COMPANY_OFFICIAL", "MANAGEMENT", "ANALYST", "OTHER"]

class ExtractedClaim(BaseModel):
    # Core claim
    claim: str = Field(..., description="Atomic, standalone claim. Include key numbers and the relevant period if present.")
    polarity: Polarity = Field(..., description="BULLISH/BEARISH/MIXED/NEUTRAL")
    materiality_0_1: float = Field(..., ge=0.0, le=1.0, description="If true and broadly known, how valuation-relevant is it?")
    credibility_0_1: float = Field(..., ge=0.0, le=1.0, description="How likely true given the source and wording?")
    surprise_0_1: float = Field(..., ge=0.0, le=1.0, description="How unexpected vs common priors for this company/sector?")
    horizon_profile: Dict[str, float] = Field(..., description="Distribution over 1D/1W/1M/3M/1Y/3Y; should sum ~1.0.")
    rationale: str = Field(..., description="One paragraph on why this matters for valuation.")
    quote: str = Field(..., description="Exact supporting quote/snippet from the chunk.")

    # Pragmatics / attribution
    is_forward_looking: bool = Field(..., description="True if the claim is about future expectations, guidance, intentions, or projected outcomes.")
    modality: Modality = Field(..., description="ASSERTION/FORECAST/INTENTION/CONDITIONAL/RISK/OPINION")
    commitment_0_1: float = Field(..., ge=0.0, le=1.0, description="Higher for 'will'/'we will'; lower for 'may'/'could'.")
    conditionality_0_1: float = Field(..., ge=0.0, le=1.0, description="Higher when heavily conditional ('if', 'subject to', 'depending on').")
    evidential_basis: Evidential = Field(..., description="Is it a reported number, observation, internal metric, or unspecified?")
    speaker_role: SpeakerRole = Field(..., description="Who is effectively the source of the quoted snippet?")

class ExtractedClaims(BaseModel):
    claims: List[ExtractedClaim]


# -----------------------------
# Text normalisation
# -----------------------------
def table_to_markdown_from_bs4(table_tag) -> str:
    """Convert an HTML table (`bs4` tag) into markdown text.

    This makes tabular financial data visible to LLM extraction in plain text form
    without requiring HTML understanding at inference time.
    """
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for td in tr.find_all(["th", "td"]):
            text = td.get_text(" ", strip=True)
            cells.append(text.replace("\n", " ").strip())
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    header = rows[0]
    if all((h == "" or any(ch.isalpha() for ch in h)) for h in header):
        body = rows[1:]
    else:
        header = [f"col{i+1}" for i in range(max_cols)]
        body = rows

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def table_to_markdown_from_list(table: List[List[str]]) -> str:
    """Convert a `pdfplumber` list-of-lists table into markdown text.

    PDF extraction often yields ragged rows (different column counts by row).
    We pad rows to a consistent width so downstream consumers get stable structure.
    """
    if not table:
        return ""
    max_cols = max(len(r) for r in table)
    rows = [list(map(lambda x: (x or "").strip().replace("\n", " "), r)) for r in table]
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    header = rows[0] if any(cell.isalpha() for cell in " ".join(rows[0])) else [f"col{i+1}" for i in range(max_cols)]
    body = rows[1:] if header == rows[0] else rows

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def normalise_to_text(path: Path | str) -> str:
    """Convert heterogeneous source files into analysis-friendly text.

    Design goals:
    - Preserve economically relevant content (including tables and inline XBRL facts).
    - Remove markup noise that hurts extraction quality.
    - Return a single plain-text representation regardless of input format.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        parts: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []

                for t in tables:
                    md = table_to_markdown_from_list(t)
                    if md:
                        parts.append(f"[START_TABLE page={page_no}]\n{md}\n[END_TABLE]\n")

                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
        return "\n\n".join(parts)

    if ext in {".html", ".htm"}:
        html = path.read_text(errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        # Drop scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Remove taxonomy / XBRL href anchors, replace other anchors with their text
        for a in list(soup.find_all("a")):
            href = (a.get("href") or "").strip()
            txt = a.get_text(" ", strip=True)
            # Remove noisy taxonomy/XBRL links
            if href.startswith("http://fasb.org") or href.startswith("https://fasb.org") \
               or href.startswith("http://www.xbrl.org") or href.startswith("https://www.xbrl.org") \
               or txt.startswith("http://fasb.org") or txt.startswith("http://www.xbrl.org"):
                a.decompose()
            else:
                a.replace_with(txt)

        # Convert inline XBRL (ix:*, ixbrl:*, etc.) into readable markers.
        # This function is conservative: it does not drop facts, it replaces tags with a readable text line.
        def _extract_inline_xbrl_as_text(soup) -> None:
            """
            Replace inline XBRL tags with readable markers.
            This function:
            - treats tags whose tag-name contains a colon (e.g. us-gaap:Revenue) as concepts,
            - also looks for attributes like 'name' or 'concept',
            - sanitises the value by removing taxonomy URLs and excessive whitespace,
            - replaces the tag in-place with a single-line [XBRL_FACT] marker.
            """
            def _clean_text(s: str) -> str:
                if not s:
                    return ""
                # remove common taxonomy / xbrl urls entirely
                s = re.sub(r'https?://(?:www\.)?fasb\.org[^\s]*', '', s)
                s = re.sub(r'https?://(?:www\.)?xbrl\.org[^\s]*', '', s)
                s = re.sub(r'https?://[^\s#]*#[^\s]*', '', s)  # catch other fragment URLs
                # normalise whitespace
                s = re.sub(r'[\s\n\r]+', ' ', s).strip()
                return s

            # enumerate tags and replace XBRL-like items
            for tag in list(soup.find_all()):
                name = getattr(tag, "name", "") or ""
                try:
                    lname = name.lower()
                except Exception:
                    lname = ""
                is_xbrl_like = (":" in name) or lname.startswith("ix:") or lname.startswith("ixbrl:") or lname.startswith("xbrli:")
                if not is_xbrl_like:
                    continue

                # concept: prefer the tag name (e.g. us-gaap:Revenue), else attributes 'name'/'concept'/'id'
                concept = str(name)
                if not concept or concept.strip() == "":
                    concept = tag.get("name") or tag.get("concept") or tag.get("id") or ""
                concept = _clean_text(str(concept)).strip()

                # value: text inside tag, cleaned
                val = _clean_text(tag.get_text(" ", strip=True))

                # context/unit/decimals if present
                context = _clean_text(tag.get("contextref") or tag.get("contextRef") or tag.get("context") or "")
                unit = _clean_text(tag.get("unitref") or tag.get("unitRef") or tag.get("unit") or "")
                decimals = _clean_text(tag.get("decimals") or "")

                # build marker
                md = f"[XBRL_FACT] concept={concept} value={val}"
                if context:
                    md += f" context={context}"
                if unit:
                    md += f" unit={unit}"
                if decimals:
                    md += f" decimals={decimals}"

                # replace the tag node with the marker
                try:
                    tag.replace_with(md)
                except Exception:
                    # if replace fails, try to insert text and decompose tag
                    tag.insert_after(md)
                    tag.decompose()


        # Convert inline XBRL facts to readable text (conservative replacement)
        try:
            _extract_inline_xbrl_as_text(soup)
        except Exception:
            # if parsing fails, continue — we still have text fallback
            pass

        # Convert explicit tables to markdown blocks (existing behavior)
        for table in list(soup.find_all("table")):
            try:
                md = table_to_markdown_from_bs4(table)
                if md:
                    md_block = soup.new_tag("p")
                    md_block.string = f"\n\n[START_TABLE]\n{md}\n[END_TABLE]\n\n"
                    table.replace_with(md_block)
            except Exception:
                continue

        text = soup.get_text("\n")

        # FINAL CLEANUP: remove remaining taxonomy URLs / fragments and normalise whitespace
        # remove fasb/xbrl urls
        text = re.sub(r'https?://(?:www\.)?fasb\.org[^\s\n]*', '', text)
        text = re.sub(r'https?://(?:www\.)?xbrl\.org[^\s\n]*', '', text)
        # remove any remaining fragment URLs that include '#'
        text = re.sub(r'https?://[^\s\n]*#[^\s\n]*', '', text)
        # collapse whitespace
        text = re.sub(r'[\s\n\r]+', ' ', text).strip()

        # Put back sensible newlines for markers and tables
        text = text.replace('[START_TABLE]', '\n[START_TABLE]').replace('[END_TABLE]', '[END_TABLE]\n')
        text = text.replace('[XBRL_FACT]', '\n[XBRL_FACT]')

        # Trim and return
        return text

    if ext == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)

    return path.read_text(errors="ignore")


def chunk_text(text: str, max_chars: int = 7000, overlap: int = 700) -> List[str]:
    """Split long text into overlapping windows for LLM extraction.

    Overlap is intentional: claims near boundaries should still appear fully in at
    least one chunk so we do not lose context-sensitive statements.
    """
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def sample_chunks(chunks: List[str], k: int) -> List[str]:
    """Subsample chunks when documents are huge to cap extraction cost.

    Heuristic keeps early, middle, and late sections to preserve narrative coverage.
    """
    if len(chunks) <= k:
        return chunks
    # start / middle / end sampling
    a = chunks[: max(1, k // 3)]
    mid_start = max(0, len(chunks) // 2 - max(1, k // 6))
    b = chunks[mid_start: mid_start + max(1, k // 3)]
    c = chunks[-max(1, k // 3):]
    out = a + b + c
    return out[:k]


def sample_chunks_with_indices(chunks: List[str], k: int) -> List[Tuple[int, str]]:
    """Return sampled chunks together with their original 1-based chunk numbers.

    The plain `sample_chunks` helper is still useful when we only need the text.
    This companion helper exists so we can write clear pipeline artifacts showing
    which original chunks were kept for the LLM stage.
    """
    if len(chunks) <= k:
        return [(i + 1, chunk) for i, chunk in enumerate(chunks)]

    candidate_indices = []
    candidate_indices.extend(range(0, max(1, k // 3)))

    mid_start = max(0, len(chunks) // 2 - max(1, k // 6))
    candidate_indices.extend(range(mid_start, min(len(chunks), mid_start + max(1, k // 3))))

    tail_count = max(1, k // 3)
    candidate_indices.extend(range(max(0, len(chunks) - tail_count), len(chunks)))

    out: List[Tuple[int, str]] = []
    seen = set()
    for idx in candidate_indices:
        if idx in seen or idx >= len(chunks):
            continue
        seen.add(idx)
        out.append((idx + 1, chunks[idx]))
        if len(out) >= k:
            break
    return out


# -----------------------------
# DB schema
# -----------------------------
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

    conn.execute("""
    CREATE TABLE IF NOT EXISTS event_evaluation_runs (
        eval_run_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        market_ticker TEXT NOT NULL,
        created_at TEXT NOT NULL,
        out_csv_path TEXT NOT NULL,
        estimation_days INTEGER NOT NULL,
        buffer_days INTEGER NOT NULL,
        row_count INTEGER NOT NULL,
        summary_json TEXT NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS event_evaluation_rows (
        eval_row_id TEXT PRIMARY KEY,
        eval_run_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        row_json TEXT NOT NULL,
        FOREIGN KEY(eval_run_id) REFERENCES event_evaluation_runs(eval_run_id),
        FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
    );
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_ticker ON facts(ticker);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exposures_fact ON exposures(fact_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_ticker_time ON docs(ticker, timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docclaims_doc ON doc_claims(doc_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_resolutions_source ON resolutions(source_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_eval_rows_run ON event_evaluation_rows(eval_run_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_eval_rows_doc ON event_evaluation_rows(doc_id);")

    conn.commit()


# -----------------------------
# Utilities
# -----------------------------
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


def safe_file_stem(name: str) -> str:
    """Convert a free-form name into a file-safe stem.

    We keep the rule intentionally simple so pipeline artifact names stay readable.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned or "document"


def model_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model-like object into a plain dictionary."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot convert object of type {type(obj)} to dict")


def _json_default(value: Any) -> Any:
    """Best-effort JSON serializer for numpy/path/datetime helper types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def ensure_pipeline_dirs() -> None:
    """Create all pipeline output directories if they do not already exist."""
    for path in list(PIPELINE_STAGE_DIRS.values()) + [EVENT_EVAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clear_directory_contents(path: Path) -> None:
    """Delete everything inside a directory, but keep the directory itself.

    We preserve `.gitkeep` so the repo can still track otherwise-empty folders.
    """
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_pipeline_outputs() -> None:
    """Reset every pipeline stage directory to an empty state."""
    ensure_pipeline_dirs()
    for stage_dir in PIPELINE_STAGE_DIRS.values():
        clear_directory_contents(stage_dir)


def write_text_file(path: Path, text: str) -> None:
    """Write a UTF-8 text artifact, creating parent directories on demand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_file(path: Path, payload: Any) -> None:
    """Write a pretty JSON artifact for human inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def utc_now() -> dt.datetime:
    """Return timezone-aware UTC now (avoids naive datetime bugs)."""
    return dt.datetime.now(dt.timezone.utc)


def polarity_to_sign(p: Polarity) -> int:
    """Convert categorical polarity into numeric sign used by scoring math."""
    if p == "BULLISH":
        return +1
    if p == "BEARISH":
        return -1
    return 0


def normalise_horizon(h: Dict[str, float]) -> Dict[str, float]:
    """Normalize horizon weights to a proper probability distribution.

    Guarantees:
    - All required buckets exist.
    - Values sum to ~1.0.
    - Falls back to a reasonable default profile if input is empty/invalid.
    """
    out = {k: float(h.get(k, 0.0)) for k in HORIZON_BUCKETS}
    s = float(sum(out.values()))
    if s <= 0:
        out = {"1D": 0.25, "1W": 0.25, "1M": 0.20, "3M": 0.15, "1Y": 0.10, "3Y": 0.05}
        s = float(sum(out.values()))
    return {k: out[k] / s for k in HORIZON_BUCKETS}


def embed(client: OpenAI, text: str) -> np.ndarray:
    """Create a unit-normalized embedding vector for similarity comparisons."""
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text, encoding_format="float")
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(vec) + 1e-12)
    return vec / n

# ---------- Structured parse probe and robust parsing helpers ----------
# Global toggle — will be set at runtime
STRUCTURED_PARSE_ENABLED = True

def _supports_structured_parse(client: OpenAI) -> bool:
    """
    Quick probe to check if responses.parse(...) with our Pydantic schema is supported.
    """
    try:
        resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "Return an empty claims list."},
                {"role": "user", "content": "TEXT: test"},
            ],
            text_format=ExtractedClaims,
        )
        _ = resp.output_parsed
        return True
    except Exception as e:
        print(f"[LLM] Structured parse probe failed ({type(e).__name__}): {e}")
        return False


def _safe_json_load(raw: str) -> dict:
    """Best-effort JSON parser for imperfect model outputs.

    We first try strict JSON, then attempt to recover the first JSON object-shaped
    substring. If both fail, we return an empty claims payload rather than crashing.
    """
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"claims": []}
        return {"claims": []}


def _normalize_evidential(v: str) -> str:
    if not v or not isinstance(v, str):
        return "UNSPECIFIED"
    vv = v.strip().upper()
    if "REPORT" in vv or "NUMBER" in vv or "REPORTED" in vv:
        return "REPORTED_NUMBER"
    if "OPERATION" in vv or "OBSERV" in vv or "OPERATIONAL" in vv:
        return "OPERATIONAL_OBSERVATION"
    if "INTERNAL" in vv or "METRIC" in vv:
        return "INTERNAL_METRIC"
    return "UNSPECIFIED"


def _normalize_modality(v: Optional[str]) -> str:
    if not v:
        return "ASSERTION"
    vv = v.strip().upper()
    for opt in ["FORECAST","INTENTION","CONDITIONAL","RISK","OPINION","ASSERTION"]:
        if opt in vv:
            return opt
    return "ASSERTION"


def _normalize_speaker_role(v: Optional[str]) -> str:
    if not v:
        return "OTHER"
    vv = v.strip().upper()
    if "COMPANY" in vv or "SEC" in vv or "FORM" in vv:
        return "COMPANY_OFFICIAL"
    if "MANAGEMENT" in vv or "CEO" in vv or "CFO" in vv:
        return "MANAGEMENT"
    if "ANALYST" in vv or "SELL" in vv or "BUY" in vv:
        return "ANALYST"
    return "OTHER"


def parse_raw_claims(data: dict, chunk: str) -> List[ExtractedClaim]:
    """Defensively coerce loosely-structured model JSON into `ExtractedClaim` objects.

    This parser exists because model outputs can drift. Instead of failing hard, we
    normalize fields, clamp ranges, and provide sensible defaults so ingestion can
    proceed while still producing auditable records.
    """
    out: List[ExtractedClaim] = []
    if not isinstance(data, dict):
        return out
    raw_claims = data.get("claims") or data.get("items") or []
    if not isinstance(raw_claims, list):
        return out

    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = rc.get("claim") or rc.get("text") or (chunk[:400] + ("..." if len(chunk) > 400 else ""))

        polarity = (rc.get("polarity") or "NEUTRAL").upper()
        if polarity not in {"BULLISH","BEARISH","MIXED","NEUTRAL"}:
            pt = claim_text.lower()
            if any(w in pt for w in ["increase","upside","beat","outperform","grow","gain","benefit"]):
                polarity = "BULLISH"
            elif any(w in pt for w in ["decline","drop","miss","slowdown","risk","loss","downside","concern"]):
                polarity = "BEARISH"
            else:
                polarity = "NEUTRAL"

        def _safe_float(k, default):
            try:
                v = rc.get(k, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        materiality = max(0.0, min(1.0, _safe_float("materiality_0_1", 0.15)))
        credibility = max(0.0, min(1.0, _safe_float("credibility_0_1", 0.6)))
        surprise = max(0.0, min(1.0, _safe_float("surprise_0_1", 0.25)))

        hp = rc.get("horizon_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        if sum([float(v) for v in hp.values()]) <= 0:
            hp = {"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05}
        s = float(sum(float(hp.get(k,0.0)) for k in HORIZON_BUCKETS))
        if s <= 0:
            s = 1.0
        horizon_profile = {k: float(hp.get(k,0.0))/s for k in HORIZON_BUCKETS}

        rationale = rc.get("rationale") or rc.get("why") or ""
        quote = rc.get("quote") or rc.get("snippet") or (claim_text if len(claim_text) < 600 else claim_text[:600] + "...")

        is_forward = bool(rc.get("is_forward_looking", False))
        modality = _normalize_modality(rc.get("modality"))
        commitment = max(0.0, min(1.0, _safe_float("commitment_0_1", 0.5)))
        conditionality = max(0.0, min(1.0, _safe_float("conditionality_0_1", 0.0)))
        evidential = _normalize_evidential(rc.get("evidential_basis") or rc.get("evidence") or "")
        speaker_role = _normalize_speaker_role(rc.get("speaker_role"))

        try:
            ec = ExtractedClaim(
                claim=str(claim_text),
                polarity=polarity,
                materiality_0_1=float(materiality),
                credibility_0_1=float(credibility),
                surprise_0_1=float(surprise),
                horizon_profile=horizon_profile,
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=bool(is_forward),
                modality=modality,
                commitment_0_1=float(commitment),
                conditionality_0_1=float(conditionality),
                evidential_basis=evidential,
                speaker_role=speaker_role,
            )
            out.append(ec)
        except Exception:
            out.append(ExtractedClaim(
                claim=str(claim_text),
                polarity="NEUTRAL",
                materiality_0_1=0.15,
                credibility_0_1=0.6,
                surprise_0_1=0.25,
                horizon_profile={"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05},
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=False,
                modality="ASSERTION",
                commitment_0_1=0.5,
                conditionality_0_1=0.0,
                evidential_basis="UNSPECIFIED",
                speaker_role="OTHER",
            ))

    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for pre-normalized vectors (dot product shortcut)."""
    return float(np.dot(a, b))


def get_source_reliability(conn: sqlite3.Connection, source_id: str) -> float:
    """Fetch source reliability prior; initialize to 1.0 if unseen."""
    cur = conn.execute("SELECT reliability FROM sources WHERE source_id = ?", (source_id,))
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO sources(source_id, reliability, updated_at) VALUES (?,?,?)",
            (source_id, 1.0, utc_now().isoformat()),
        )
        conn.commit()
        return 1.0
    return float(row[0])


def pragmatics_adjust(p0: float, commitment: float, conditionality: float, is_forward_looking: bool) -> float:
    """
    Pull probability towards 0.5 when language is weak or heavily conditional.
    For non-forward-looking assertions, just return p0.
    """
    p0 = float(np.clip(p0, 0.0, 1.0))
    if not is_forward_looking:
        return p0

    commitment = float(np.clip(commitment, 0.0, 1.0))
    conditionality = float(np.clip(conditionality, 0.0, 1.0))

    strength = (0.4 + 0.6 * commitment) * (1.0 - 0.5 * conditionality)
    p = 0.5 + (p0 - 0.5) * strength
    return float(np.clip(p, 0.0, 1.0))


def apply_reliability(p_prag: float, reliability: float) -> float:
    """
    Shrink towards 0.5 based on source reliability.
    """
    p_prag = float(np.clip(p_prag, 0.0, 1.0))
    reliability = float(np.clip(reliability, 0.0, 1.0))
    p = 0.5 + (p_prag - 0.5) * reliability
    return float(np.clip(p, 0.0, 1.0))


def combine_independent_evidence(p_old: float, p_new: float, authority: float) -> float:
    """
    Simple monotonic combiner: new high-authority evidence raises belief.
    (We do not handle contradictions in v1.)
    """
    p_old = float(np.clip(p_old, 0.0, 1.0))
    p_new = float(np.clip(p_new, 0.0, 1.0))
    authority = float(np.clip(authority, 0.0, 1.0))
    p_eff = p_new * authority
    return float(np.clip(1.0 - (1.0 - p_old) * (1.0 - p_eff), 0.0, 1.0))


def compute_ts_coef(sign: int, materiality: float, novelty: float, surprise: float) -> float:
    """Baseline T-S coefficient (legacy/simple formula).

    Interpreted as directional impact magnitude under the counterfactual where the
    claim is true and broadly diffused in the market.
    """
    return float(sign) * float(materiality) * float(novelty) * float(surprise)

def compute_ts_coef_enhanced(
    sign: int,
    materiality: float,
    novelty: float,
    surprise: float,
    credibility: float,
    p_true: float,
    horizon: Dict[str, float],
    issued_at_iso: Optional[str],
    source_type: Optional[str],
) -> float:
    """
    Enhanced, pragmatic computation of ts_coef (signed).

    Formula (v1):
      ts_coef = sign * p_true * materiality * novelty * credibility * surprise * reach * near_term_weight * time_decay

    - sign: +1 / -1 / 0
    - p_true: probability used at issue (already adjusted by pragmatics & source reliability)
    - materiality, novelty, credibility, surprise : in [0,1]
    - reach: estimated audience reach for source_type (uses SOURCE_REACH fallback)
    - near_term_weight: weighted importance across horizons using NEAR_WEIGHTS
    - time_decay: exponential decay by age (TAU_RELIABILITY_DAYS)
    """
    try:
        sign = int(sign)
    except Exception:
        sign = 0

    # clamp inputs
    materiality = float(max(0.0, min(1.0, materiality or 0.0)))
    novelty = float(max(0.0, min(1.0, novelty or 0.0)))
    surprise = float(max(0.0, min(1.0, surprise or 0.0)))
    credibility = float(max(0.0, min(1.0, credibility or 0.0)))
    p_true = float(max(0.0, min(1.0, p_true or 0.5)))

    # reach estimate
    reach = float(SOURCE_REACH.get((source_type or "").upper(), 0.5))

    # near-term weight from horizon distribution
    near_term_weight = 0.0
    try:
        for b in HORIZON_BUCKETS:
            near_term_weight += float(horizon.get(b, 0.0)) * float(NEAR_WEIGHTS.get(b, 0.0))
    except Exception:
        near_term_weight = 0.5
    if near_term_weight <= 0.0:
        near_term_weight = 0.5

    # age / time decay
    age_days = 0.0
    if issued_at_iso:
        try:
            # issued_at_iso is an ISO timestamp string like timestamp.isoformat()
            issued_dt = dt.datetime.fromisoformat(issued_at_iso)
            age_days = max(0.0, (utc_now().replace(tzinfo=None) - issued_dt).days)
        except Exception:
            age_days = 0.0
    time_decay = float(math.exp(-age_days / max(1.0, TAU_RELIABILITY_DAYS)))

    magnitude = p_true * materiality * novelty * credibility * surprise * reach * near_term_weight * time_decay

    ts_coef = float(sign) * float(magnitude)

    # suppress sub-floating noise
    if abs(ts_coef) < 1e-12:
        return 0.0
    return float(ts_coef)

def awareness(conn: sqlite3.Connection, fact_id: str, as_of: dt.datetime, lam: float = LAMBDA_DECAY) -> float:
    cur = conn.execute("SELECT reach, timestamp FROM exposures WHERE fact_id = ?", (fact_id,))
    rows = cur.fetchall()
    if not rows:
        return 0.0

    prod = 1.0
    for reach, ts in rows:
        t_e = dt.datetime.fromisoformat(ts)
        age_days = max(0.0, (as_of - t_e).total_seconds() / 86400.0)
        effective = float(reach) * float(np.exp(-lam * age_days))
        prod *= (1.0 - np.clip(effective, 0.0, 1.0))
    return float(1.0 - prod)


def impact_now(ts_coef: float, p_true: float, horizon: Dict[str, float], age_days: float, awareness_now: float) -> float:
    """
    UI ranking: includes time decay and near-term weighting.
    """
    total = 0.0
    for b in HORIZON_BUCKETS:
        w = NEAR_WEIGHTS[b]
        hl = HALF_LIFE_DAYS[b]
        decay = float(np.exp(-age_days * np.log(2.0) / hl))
        total += horizon[b] * w * decay
    return float(ts_coef * p_true * awareness_now * total)


# -----------------------------
# Lightweight pragmatics / measurement helpers
# -----------------------------
# These helpers intentionally use simple regex rules. The goal is not to solve
# full semantic parsing, but to catch the most obvious measurable statements
# without adding new dependencies or a large amount of code.

_NUMBER_RE = re.compile(
    r'(?P<currency>[$£€])?\s*(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<suffix>bn|billion|m|million|k|thousand|%|percent|bps|basis points)?',
    flags=re.IGNORECASE,
)

_DATE_WORDS = {
    'jan', 'january', 'feb', 'february', 'mar', 'march', 'apr', 'april',
    'jun', 'june', 'jul', 'july', 'aug', 'august', 'sep', 'sept',
    'september', 'oct', 'october', 'nov', 'november', 'dec', 'december',
    'q1', 'q2', 'q3', 'q4', 'fy', 'fiscal', 'year', 'quarter',
}

_HEDGE_WORDS = [
    'may', 'might', 'could', 'can', 'possibly', 'potentially',
    'subject to', 'depending on', 'if', 'expects to', 'aim to',
]

_COMMIT_WORDS = [
    'we will', 'will', 'we commit', 'commit', 'we intend', 'intend',
    'we plan', 'plan to', 'target', 'promise',
]


def _count_text_hits(text: str, patterns: List[str]) -> int:
    """Count simple phrase hits using whole-word matching where possible."""
    total = 0
    for p in patterns:
        if ' ' in p:
            total += len(re.findall(re.escape(p), text))
        else:
            total += len(re.findall(rf'\b{re.escape(p)}\b', text))
    return total


def strip_numeric_surface(text: str) -> str:
    """Replace numeric expressions with a marker.

    This gives us a simple "metric skeleton" for fuzzy matching. Example:
    - "Revenue was $9 billion in 2025" -> "Revenue was <NUM> in <NUM>"
    """
    out = _NUMBER_RE.sub(' <NUM> ', text.lower())
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def parse_simple_measurements(text: str) -> List[float]:
    """Parse a few numeric expressions into plain floats.

    Heuristics are intentionally light-weight:
    - 9bn / 9 billion -> 9_000_000_000
    - 3m / 3 million -> 3_000_000
    - 12% -> 12
    - 150 bps -> 150

    We keep only the first few measurements because they are mainly used as a
    coarse signal for how measurable / changing a claim is.
    """
    out: List[float] = []
    for m_num in _NUMBER_RE.finditer(text):
        raw_val = (m_num.group('value') or '').replace(',', '')
        suffix = (m_num.group('suffix') or '').lower().strip()
        try:
            val = float(raw_val)
        except Exception:
            continue

        if suffix in {'bn', 'billion'}:
            val *= 1_000_000_000.0
        elif suffix in {'m', 'million'}:
            val *= 1_000_000.0
        elif suffix in {'k', 'thousand'}:
            val *= 1_000.0

        out.append(val)
        if len(out) >= 5:
            break
    return out


def measure_features(text: str) -> Dict[str, Any]:
    """Return simple measurability / concreteness features from raw text.

    Why this helper exists:
    - Your research question cares about measurable vs vague statements.
    - A deterministic score is easier to understand and audit than another LLM
      pass.
    """
    lower = text.lower()
    values = parse_simple_measurements(text)

    has_currency = bool(re.search(r'[$£€]|\b(usd|eur|gbp|dollars?|euros?|pounds?)\b', lower))
    has_percent = bool(re.search(r'\b\d+(?:\.\d+)?\s*(%|percent|bps|basis points)\b', lower))
    has_date = bool(re.search(r'\b20\d{2}\b', lower)) or any(re.search(rf'\b{re.escape(w)}\b', lower) for w in _DATE_WORDS)
    has_unit = bool(re.search(r'\b(days?|weeks?|months?|years?|users?|customers?|stores?|employees?|units?)\b', lower))
    has_unit = has_unit or has_currency or has_percent

    hedge_count = _count_text_hits(lower, _HEDGE_WORDS)
    commit_count = _count_text_hits(lower, _COMMIT_WORDS)

    # Simple 0..1 score: numbers matter most, then dates / units.
    measurability = 0.0
    if values:
        measurability += 0.45
    if has_date:
        measurability += 0.20
    if has_unit:
        measurability += 0.20
    if len(values) >= 2:
        measurability += 0.05
    if has_currency and has_percent:
        measurability += 0.10

    return {
        'values': values,
        'has_currency': has_currency,
        'has_percent': has_percent,
        'has_date': has_date,
        'has_unit': has_unit,
        'hedge_count': hedge_count,
        'commit_count': commit_count,
        'measurability_0_1': float(min(1.0, measurability)),
        # In this small demo we treat concreteness and measurability as the same
        # family of feature. You can split them later if you want finer theory.
        'concreteness_0_1': float(min(1.0, measurability)),
    }


def compare_quantitative_claims(new_text: str, old_text: str) -> Dict[str, float]:
    """Estimate whether two similar claims differ materially in quantity.

    This is the smallest useful fix for the issue you pointed out:
    - "we made 9bn" should not look the same as "we made 3bn"
    - both should also differ from "we made 1bn"

    The logic is simple:
    1) compare the claim skeleton with numbers removed,
    2) if the skeletons still look similar, compare the first measurement,
    3) turn the relative change into a 0..1 strength.
    """
    new_skel = strip_numeric_surface(new_text)
    old_skel = strip_numeric_surface(old_text)
    skeleton_sim = fuzz.token_set_ratio(new_skel, old_skel) / 100.0

    new_feat = measure_features(new_text)
    old_feat = measure_features(old_text)

    if skeleton_sim < 0.82:
        return {
            'skeleton_sim': float(skeleton_sim),
            'delta_strength': 0.0,
            'mismatch_penalty': 0.0,
        }

    if not new_feat['values'] or not old_feat['values']:
        return {
            'skeleton_sim': float(skeleton_sim),
            'delta_strength': 0.0,
            'mismatch_penalty': 0.0,
        }

    new_val = float(new_feat['values'][0])
    old_val = float(old_feat['values'][0])
    denom = max(abs(old_val), 1.0)
    rel_gap = abs(new_val - old_val) / denom

    # log1p makes the feature smoother than a raw ratio.
    delta_strength = min(1.0, math.log1p(rel_gap) / math.log(10.0))
    mismatch_penalty = 0.35 * delta_strength

    # If one claim looks like currency and the other looks like a percent,
    # apply a small extra penalty.
    if bool(new_feat['has_currency']) != bool(old_feat['has_currency']):
        mismatch_penalty += 0.10
    if bool(new_feat['has_percent']) != bool(old_feat['has_percent']):
        mismatch_penalty += 0.10

    return {
        'skeleton_sim': float(skeleton_sim),
        'delta_strength': float(min(1.0, delta_strength)),
        'mismatch_penalty': float(min(0.60, mismatch_penalty)),
    }


def infer_speech_act(text: str, modality: str) -> str:
    """Map simple corporate wording into a coarse speech-act label.

    We keep this deliberately small and readable:
    - INTENTION / promise language -> COMMISSIVE
    - most factual / forecast language -> ASSERTIVE
    - a few obvious imperative / formal cases -> DIRECTIVE / DECLARATIVE
    """
    lower = text.lower()

    if modality == 'INTENTION' or any(w in lower for w in ['we will', 'we commit', 'we intend', 'we plan']):
        return 'COMMISSIVE'
    if any(w in lower for w in ['must ', 'should ', 'please ', 'we ask']):
        return 'DIRECTIVE'
    if any(w in lower for w in ['hereby', 'appoint', 'declare', 'authorize']):
        return 'DECLARATIVE'
    if modality in {'ASSERTION', 'FORECAST', 'CONDITIONAL', 'RISK', 'OPINION'}:
        return 'ASSERTIVE'
    return 'OTHER'


def fetch_facts_for_ticker(conn: sqlite3.Connection, ticker: str) -> List[Dict[str, Any]]:
    """Load existing canonical facts for dedupe against new extracted claims."""
    cur = conn.execute(
        "SELECT fact_id, canonical_text, embedding_json, ts_coef FROM facts WHERE ticker = ?",
        (ticker,),
    )
    rows = []
    for fact_id, text, emb_json, ts_coef in cur.fetchall():
        rows.append({
            "fact_id": fact_id,
            "canonical_text": text,
            "embedding": np.array(json.loads(emb_json), dtype=np.float32),
            "ts_coef": float(ts_coef),
        })
    return rows


def novelty_against_kb(
    claim_text: str,
    claim_vec: np.ndarray,
    existing: List[Dict[str, Any]],
) -> Tuple[float, Optional[str], float, float]:
    """Compute novelty vs. knowledge base and return best candidate match.

    Returns:
    - novelty
    - best_fact_id
    - best_similarity (after light numeric penalty)
    - quantitative_delta (0..1)

    The new `quantitative_delta` is the small but important addition: it lets the
    system distinguish between semantically similar claims that carry very
    different numbers.
    """
    best_sim = 0.0
    best_id: Optional[str] = None
    best_text = None

    for f in existing:
        sim = cosine(claim_vec, f['embedding'])
        if sim > best_sim:
            best_sim = sim
            best_id = f['fact_id']
            best_text = f['canonical_text']

    quantitative_delta = 0.0
    if best_id is not None and best_text is not None:
        fuzz_ratio = fuzz.token_set_ratio(claim_text, best_text) / 100.0
        quant_cmp = compare_quantitative_claims(claim_text, best_text)
        quantitative_delta = float(quant_cmp['delta_strength'])

        best_sim = max(best_sim, fuzz_ratio)
        best_sim = max(0.0, best_sim - float(quant_cmp['mismatch_penalty']))

    novelty = float(np.clip(1.0 - best_sim, 0.0, 1.0))
    return novelty, best_id, float(best_sim), float(quantitative_delta)


def treat_as_same_fact(embed_sim: float, fuzz_sim: float, quantitative_delta: float = 0.0) -> bool:
    """Decision rule for whether a new claim should merge into an existing fact.

    Large quantitative deltas usually mean we are looking at a fresh observation
    rather than a mere reconfirmation of the previous fact.
    """
    if quantitative_delta >= 0.25:
        return False
    return (embed_sim >= EMBED_SAME_STRICT) or (embed_sim >= EMBED_SAME_LOOSE and fuzz_sim >= FUZZ_SAME_STRICT)


# -----------------------------
# OpenAI extraction
# -----------------------------
# ---------- REPLACE the existing extract_claims_from_chunk(...) entirely with this ----------

def _normalize_evidential(v: str) -> str:
    if not v or not isinstance(v, str):
        return "UNSPECIFIED"
    vv = v.strip().upper()
    if "REPORT" in vv or "NUMBER" in vv or "REPORTED" in vv:
        return "REPORTED_NUMBER"
    if "OPERATION" in vv or "OBSERV" in vv:
        return "OPERATIONAL_OBSERVATION"
    if "INTERNAL" in vv or "METRIC" in vv:
        return "INTERNAL_METRIC"
    return "UNSPECIFIED"

def _normalize_modality(v: Optional[str]) -> str:
    if not v:
        return "ASSERTION"
    vv = v.strip().upper()
    for opt in ["FORECAST","INTENTION","CONDITIONAL","RISK","OPINION","ASSERTION"]:
        if opt in vv:
            return opt
    return "ASSERTION"

def _normalize_speaker_role(v: Optional[str]) -> str:
    if not v:
        return "OTHER"
    vv = v.strip().upper()
    if "COMPANY" in vv or "SEC" in vv or "FORM" in vv:
        return "COMPANY_OFFICIAL"
    if "MANAGEMENT" in vv or "CEO" in vv or "CFO" in vv:
        return "MANAGEMENT"
    if "ANALYST" in vv or "SELL" in vv or "BUY" in vv:
        return "ANALYST"
    return "OTHER"

def parse_raw_claims(data: dict, chunk: str) -> List[ExtractedClaim]:
    """
    Defensive mapping from arbitrary JSON returned by the model into our ExtractedClaim objects.
    Uses sensible defaults for missing fields and normalises enums.
    """
    out: List[ExtractedClaim] = []
    if not isinstance(data, dict):
        return out
    raw_claims = data.get("claims") or data.get("items") or []
    if not isinstance(raw_claims, list):
        return out

    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = rc.get("claim") or rc.get("text") or (chunk[:400] + ("..." if len(chunk) > 400 else ""))
        # Basic defaults
        polarity = (rc.get("polarity") or "NEUTRAL").upper()
        if polarity not in {"BULLISH","BEARISH","MIXED","NEUTRAL"}:
            # try to infer from words
            pt = claim_text.lower()
            if any(w in pt for w in ["increase","upside","beat","outperform","grow","gain","benefit"]):
                polarity = "BULLISH"
            elif any(w in pt for w in ["decline","drop","miss","slowdown","risk","loss","downside","concern"]):
                polarity = "BEARISH"
            else:
                polarity = "NEUTRAL"

        # numeric fields with safe clamping
        def _safe_float(k, default):
            try:
                v = rc.get(k, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        materiality = max(0.0, min(1.0, _safe_float("materiality_0_1", 0.15)))
        credibility = max(0.0, min(1.0, _safe_float("credibility_0_1", 0.6)))
        surprise = max(0.0, min(1.0, _safe_float("surprise_0_1", 0.25)))

        # horizon_profile default / normalise
        hp = rc.get("horizon_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        # fallback distribution
        if sum([float(v) for v in hp.values()]) <= 0:
            hp = {"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05}
        # normalise
        s = float(sum(float(hp.get(k,0.0)) for k in HORIZON_BUCKETS))
        if s <= 0:
            s = 1.0
        horizon_profile = {k: float(hp.get(k,0.0))/s for k in HORIZON_BUCKETS}

        rationale = rc.get("rationale") or rc.get("why") or ""
        quote = rc.get("quote") or rc.get("snippet") or (claim_text if len(claim_text) < 600 else claim_text[:600] + "...")

        # pragmatics defaults
        is_forward = bool(rc.get("is_forward_looking", False))
        modality = _normalize_modality(rc.get("modality"))
        commitment = max(0.0, min(1.0, _safe_float("commitment_0_1", 0.5)))
        conditionality = max(0.0, min(1.0, _safe_float("conditionality_0_1", 0.0)))
        evidential = _normalize_evidential(rc.get("evidential_basis") or rc.get("evidence") or "")
        speaker_role = _normalize_speaker_role(rc.get("speaker_role"))

        # Build ExtractedClaim object with fallback values where possible
        try:
            ec = ExtractedClaim(
                claim=str(claim_text),
                polarity=polarity,  # pydantic will coerce or validate
                materiality_0_1=float(materiality),
                credibility_0_1=float(credibility),
                surprise_0_1=float(surprise),
                horizon_profile=horizon_profile,
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=bool(is_forward),
                modality=modality,
                commitment_0_1=float(commitment),
                conditionality_0_1=float(conditionality),
                evidential_basis=evidential,
                speaker_role=speaker_role,
            )
            out.append(ec)
        except Exception as e:
            # Last-resort fallback: use dict and leave pydantic to convert later if needed
            out.append(ExtractedClaim(
                claim=str(claim_text),
                polarity="NEUTRAL",
                materiality_0_1=0.15,
                credibility_0_1=0.6,
                surprise_0_1=0.25,
                horizon_profile={"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05},
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=False,
                modality="ASSERTION",
                commitment_0_1=0.5,
                conditionality_0_1=0.0,
                evidential_basis="UNSPECIFIED",
                speaker_role="OTHER",
            ))

    return out


def extract_claims_from_chunk(
    client: OpenAI,
    chunk: str,
    ticker: str,
    doc_type: str,
    source_type: str,
    max_split_depth: int = 2
) -> List[ExtractedClaim]:
    """Extract valuation-relevant atomic claims from a text chunk.

    Strategy:
    1) Prefer structured parsing into `ExtractedClaims` for schema safety.
    2) If unavailable/failing, fallback to JSON mode and coerce via `parse_raw_claims`.
    3) If chunk is still too noisy/long, caller may recursively split and retry.

    This function intentionally never mutates KB state; it only returns parsed claims.
    """
    instructions = (
        "You are an expert equity research analyst.\n"
        "Extract ONLY price-relevant, atomic claims.\n"
        "Ignore boilerplate and generic risk language unless it is clearly new/changed.\n"
        "Each claim must be standalone and specific; include key numbers AND period when present.\n"
        "\n"
        "For EACH claim, output the following fields:\n"
        "- claim (string)\n"
        "- polarity (one of: BULLISH, BEARISH, MIXED, NEUTRAL)\n"
        "- materiality_0_1 (0..1)\n"
        "- credibility_0_1 (0..1)\n"
        "- surprise_0_1 (0..1)\n"
        "- horizon_profile (object with keys: 1D, 1W, 1M, 3M, 1Y, 3Y; values sum to ~1)\n"
        "- rationale (string)\n"
        "- quote (string; exact supporting snippet)\n"
        "- is_forward_looking (boolean)\n"
        "- modality (one of: ASSERTION, FORECAST, INTENTION, CONDITIONAL, RISK, OPINION)\n"
        "- commitment_0_1 (0..1; \"will\" > \"expect\" > \"could\")\n"
        "- conditionality_0_1 (0..1; \"subject to\"/\"if\" increases)\n"
        "- evidential_basis (one of: REPORTED_NUMBER, OPERATIONAL_OBSERVATION, INTERNAL_METRIC, UNSPECIFIED)\n"
        "- speaker_role (one of: COMPANY_OFFICIAL, MANAGEMENT, ANALYST, OTHER; COMPANY_OFFICIAL for SEC)\n"
        "\n"
        "Return between 0 and 8 claims. If nothing material, return an empty list."
    )

    user = (
        f"TICKER: {ticker}\nDOC_TYPE: {doc_type}\nSOURCE_TYPE: {source_type}\n\n"
        f"TEXT:\n{chunk}"
    )

    global STRUCTURED_PARSE_ENABLED
    if STRUCTURED_PARSE_ENABLED:
        try:
            print("[LLM] structured parse request...")
            resp = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user},
                ],
                text_format=ExtractedClaims,
            )
            parsed: ExtractedClaims = resp.output_parsed
            return parsed.claims
        except Exception as e:
            print(f"[LLM] structured parse failed ({type(e).__name__}): {e}")
            STRUCTURED_PARSE_ENABLED = False
            print("[LLM] disabling structured parse for remainder of run; will use JSON fallback.")

    # Fallback to JSON-only responses.create
    try:
        print("[LLM] calling responses.create with JSON-only instruction...")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": instructions + "\nReturn ONLY valid JSON matching: {\"claims\": [ ... ]}. Do not include any extra keys outside 'claims'."},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_object"}},
        )
        raw = (resp.output_text or "").strip()
        data = _safe_json_load(raw)
        parsed_claims = parse_raw_claims(data, chunk)
        return parsed_claims

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[LLM] responses.create failed: {err_msg}")

        msg = str(e).lower()
        token_error_indicators = ["context length", "max_tokens", "input length", "exceeds", "token limit", "context_length"]
        if any(k in msg for k in token_error_indicators) and max_split_depth > 0:
            print("[LLM] Detected token/context error — splitting chunk and retrying (preserving content).")
            mid = len(chunk) // 2
            left = chunk[:mid]
            right = chunk[mid:]
            left_claims = extract_claims_from_chunk(client, left, ticker, doc_type, source_type, max_split_depth - 1)
            right_claims = extract_claims_from_chunk(client, right, ticker, doc_type, source_type, max_split_depth - 1)
            return left_claims + right_claims

        return []


# -----------------------------
# Reliability update (recency-weighted Brier)
# -----------------------------
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


# -----------------------------
# Ingest pipeline
# -----------------------------
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
    """Ingest a single source document and update the persistent knowledge base.

    Pipeline stages in this function:
    1) Register document metadata (`docs`) and normalize raw text.
    2) Extract and within-document deduplicate candidate claims.
    3) Match claims against existing facts (or create new facts).
    4) Update exposure/probability state and compute impact predictions.
    5) Persist provenance rows (`doc_claims`) and document aggregates (`doc_scores`).

    `as_of` controls time-aware metrics (awareness/decay). For walk-forward simulation,
    pass `as_of=timestamp` so the system only uses information available at that time.

    This version also writes human-readable pipeline artifacts after each major step.
    The goal is simple auditability: you can open the `pipeline/` folders and inspect
    exactly what the system saw and produced at each stage.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (set it in .env or your shell).")

    as_of = as_of or timestamp
    ensure_pipeline_dirs()

    client = OpenAI()
    global STRUCTURED_PARSE_ENABLED
    if STRUCTURED_PARSE_ENABLED:
        STRUCTURED_PARSE_ENABLED = _supports_structured_parse(client)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    doc_hash = sha256_file(path)
    doc_id = f"doc_{ticker}_{doc_hash[:12]}"
    artifact_stem = safe_file_stem(f"{path.stem}_{doc_hash[:8]}")

    conn.execute(
        "INSERT OR IGNORE INTO docs(doc_id,ticker,doc_type,source_type,timestamp,sha256,url) VALUES (?,?,?,?,?,?,?)",
        (doc_id, ticker, doc_type, source_type, timestamp.isoformat(), doc_hash, url),
    )
    conn.commit()

    # ------------------------------------------------------------------
    # Step 1: normalise the raw source into plain analysis text.
    # ------------------------------------------------------------------
    text = normalise_to_text(path)
    write_text_file(
        PIPELINE_STAGE_DIRS["normalised"] / f"{artifact_stem}.txt",
        text,
    )

    # ------------------------------------------------------------------
    # Step 2: chunk the document, then record both the full chunk list and the
    # subset actually sampled for the LLM.
    # ------------------------------------------------------------------
    all_chunks = chunk_text(text)
    sampled_chunks_with_idx = sample_chunks_with_indices(all_chunks, TS_MAX_CHUNKS)

    all_chunks_dir = PIPELINE_STAGE_DIRS["all_chunks"] / artifact_stem
    for chunk_no, chunk_text_value in enumerate(all_chunks, start=1):
        write_text_file(all_chunks_dir / f"chunk_{chunk_no:03d}.txt", chunk_text_value)
    write_json_file(
        all_chunks_dir / "chunk_index.json",
        [
            {
                "chunk_no": chunk_no,
                "char_count": len(chunk_text_value),
                "preview": chunk_text_value[:220],
            }
            for chunk_no, chunk_text_value in enumerate(all_chunks, start=1)
        ],
    )

    sampled_chunks_dir = PIPELINE_STAGE_DIRS["sampled_chunks"] / artifact_stem
    for sampled_no, (original_chunk_no, chunk_text_value) in enumerate(sampled_chunks_with_idx, start=1):
        write_text_file(
            sampled_chunks_dir / f"sampled_{sampled_no:03d}_from_chunk_{original_chunk_no:03d}.txt",
            chunk_text_value,
        )
    write_json_file(
        sampled_chunks_dir / "sampled_chunk_index.json",
        [
            {
                "sampled_no": sampled_no,
                "original_chunk_no": original_chunk_no,
                "char_count": len(chunk_text_value),
                "preview": chunk_text_value[:220],
            }
            for sampled_no, (original_chunk_no, chunk_text_value) in enumerate(sampled_chunks_with_idx, start=1)
        ],
    )

    existing_facts = fetch_facts_for_ticker(conn, ticker)

    # ------------------------------------------------------------------
    # Step 3: LLM extraction on each sampled chunk.
    # ------------------------------------------------------------------
    extracted: List[ExtractedClaim] = []
    chunk_claims_dir = PIPELINE_STAGE_DIRS["chunk_claims"] / artifact_stem
    for sampled_no, (original_chunk_no, chunk_text_value) in enumerate(sampled_chunks_with_idx, start=1):
        print(f"[ingest] Extracting claims from chunk {sampled_no}/{len(sampled_chunks_with_idx)} ...")
        chunk_claims = extract_claims_from_chunk(client, chunk_text_value, ticker, doc_type, source_type)
        extracted.extend(chunk_claims)
        write_json_file(
            chunk_claims_dir / f"sampled_{sampled_no:03d}_claims.json",
            {
                "sampled_no": sampled_no,
                "original_chunk_no": original_chunk_no,
                "claims": [model_to_dict(claim) for claim in chunk_claims],
            },
        )
        # small pause to avoid tiny bursts to the LLM and keep console readable
        import time
        time.sleep(0.01)

    # ------------------------------------------------------------------
    # Step 4: within-document deduplication of extracted claims.
    # ------------------------------------------------------------------
    merged: List[ExtractedClaim] = []
    seen_vecs: List[np.ndarray] = []
    for c in extracted:
        v = embed(client, c.claim)
        if any(cosine(v, sv) > 0.92 for sv in seen_vecs):
            continue
        seen_vecs.append(v)
        merged.append(c)

    write_json_file(
        PIPELINE_STAGE_DIRS["merged_claims"] / f"{artifact_stem}.json",
        [model_to_dict(claim) for claim in merged],
    )

    reach = float(SOURCE_REACH.get(source_type, 0.50))
    authority = float(np.clip(authority, 0.0, 1.0))

    pred_by_horizon = {b: 0.0 for b in HORIZON_BUCKETS}
    near_term_total = 0.0

    # For UI ranking and pipeline artifacts.
    claim_rows: List[Dict[str, Any]] = []
    matching_rows: List[Dict[str, Any]] = []

    n_new = n_known = n_reconfirmed = 0

    # ------------------------------------------------------------------
    # Step 5: match each merged claim against the KB and compute scores.
    # ------------------------------------------------------------------
    for c in merged:
        claim_vec = embed(client, c.claim)
        novelty, matched_id, best_sim, quantitative_delta = novelty_against_kb(c.claim, claim_vec, existing_facts)

        best_text = None
        fuzz_sim = 0.0
        if matched_id is not None:
            best_text = next(x["canonical_text"] for x in existing_facts if x["fact_id"] == matched_id)
            fuzz_sim = fuzz.token_set_ratio(c.claim, best_text) / 100.0

        same_fact = matched_id is not None and treat_as_same_fact(
            embed_sim=best_sim,
            fuzz_sim=fuzz_sim,
            quantitative_delta=quantitative_delta,
        )

        sign = polarity_to_sign(c.polarity)
        horizon = normalise_horizon(c.horizon_profile)

        # Small deterministic features derived from the claim text itself.
        # These are easy to inspect and help us move closer to the research goal
        # without changing the overall architecture.
        text_features = measure_features(c.claim)
        speech_act = infer_speech_act(c.claim, c.modality)
        score_materiality = float(min(1.0, c.materiality_0_1 * (1.0 + 0.50 * quantitative_delta)))
        score_surprise = float(min(1.0, c.surprise_0_1 + 0.35 * quantitative_delta))

        # Use a slightly more granular source id. This keeps the existing idea of
        # source reliability, but avoids pooling every management statement for a
        # ticker into one single bucket.
        source_id = f"{ticker}:{source_type}:{c.speaker_role}"
        rel = get_source_reliability(conn, source_id)
        p0 = float(np.clip(c.credibility_0_1, 0.0, 1.0))
        p_prag = pragmatics_adjust(p0, c.commitment_0_1, c.conditionality_0_1, c.is_forward_looking)
        p_true_new = apply_reliability(p_prag, rel)

        matching_rows.append({
            "claim": c.claim,
            "matched_fact_id": matched_id,
            "matched_text": best_text,
            "best_match_similarity": float(best_sim),
            "fuzz_similarity": float(fuzz_sim),
            "same_fact": bool(same_fact),
            "quantitative_delta": float(quantitative_delta),
            "speech_act": speech_act,
            "measurability_0_1": float(text_features["measurability_0_1"]),
            "concreteness_0_1": float(text_features["concreteness_0_1"]),
            "hedge_count": int(text_features["hedge_count"]),
            "commit_count": int(text_features["commit_count"]),
            "source_id": source_id,
            "source_reliability": float(rel),
        })

        if not same_fact:
            fact_id = f"fact_{ticker}_{hashlib.sha256(c.claim.encode('utf-8')).hexdigest()[:16]}"

            # Compute an enhanced, signed ts_coef that uses pragmatic p_true and horizon/time weighting.
            # We use p_true_new (already adjusted for pragmatics & source reliability) as the probability at issue.
            try:
                ts_coef = compute_ts_coef_enhanced(
                    sign=int(sign),
                    materiality=float(score_materiality),
                    novelty=float(novelty),
                    surprise=float(score_surprise),
                    credibility=float(c.credibility_0_1),
                    p_true=float(p_true_new),
                    horizon=horizon,
                    issued_at_iso=timestamp.isoformat(),
                    source_type=source_type,
                )
            except Exception:
                ts_coef = compute_ts_coef(sign, c.materiality_0_1, novelty, c.surprise_0_1)

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
                    fact_id, ticker, c.claim, json.dumps(claim_vec.tolist()),
                    int(sign), float(c.materiality_0_1), float(novelty), float(c.surprise_0_1), float(ts_coef), json.dumps(horizon),
                    source_id, c.speaker_role,
                    1 if c.is_forward_looking else 0, c.modality, float(c.commitment_0_1), float(c.conditionality_0_1), c.evidential_basis,
                    float(p0), float(p_prag), float(p_true_new), float(p_true_new),
                    timestamp.isoformat(), now, now
                ),
            )
            conn.commit()

            existing_facts.append({
                "fact_id": fact_id,
                "canonical_text": c.claim,
                "embedding": claim_vec,
                "ts_coef": float(ts_coef),
            })

            status = "NEW"
        else:
            fact_id = matched_id  # type: ignore[assignment]
            cur = conn.execute("SELECT ts_coef, horizon_json, p_true_latest, issued_at FROM facts WHERE fact_id = ?", (fact_id,))
            ts_coef_db, horizon_json_db, p_true_latest_db, issued_at_db = cur.fetchone()
            ts_coef = float(ts_coef_db)
            horizon = json.loads(horizon_json_db)

            # incorporate new evidence monotonically
            p_true_latest = combine_independent_evidence(float(p_true_latest_db), p_true_new, authority)
            conn.execute(
                "UPDATE facts SET p_true_latest=?, updated_at=? WHERE fact_id=?",
                (float(p_true_latest), utc_now().isoformat(), fact_id),
            )
            conn.commit()

            status = status_from_similarity(best_sim)

        if status == "NEW":
            n_new += 1
        elif status == "KNOWN":
            n_known += 1
        else:
            n_reconfirmed += 1

        # Delta-awareness at event time
        aw_before = awareness(conn, fact_id, as_of)
        exposure_id = f"exp_{fact_id}_{doc_id}"
        conn.execute(
            "INSERT OR IGNORE INTO exposures(exposure_id,fact_id,doc_id,source_type,reach,authority,timestamp) VALUES (?,?,?,?,?,?,?)",
            (exposure_id, fact_id, doc_id, source_type, reach, authority, timestamp.isoformat()),
        )
        conn.commit()
        aw_after = awareness(conn, fact_id, as_of)
        delta_aw = float(max(0.0, aw_after - aw_before))

        # Probability used for this event
        cur = conn.execute("SELECT p_true_latest, issued_at FROM facts WHERE fact_id = ?", (fact_id,))
        p_true_used, issued_at = cur.fetchone()
        p_true_used = float(p_true_used)

        # Per-horizon prediction contribution
        pred_h = {b: float(ts_coef * p_true_used * delta_aw * float(horizon[b])) for b in HORIZON_BUCKETS}
        pred_total = float(sum(pred_h.values()))

        for b in HORIZON_BUCKETS:
            pred_by_horizon[b] += pred_h[b]
        near_term_total += float(sum(pred_h[b] * NEAR_WEIGHTS[b] for b in HORIZON_BUCKETS))

        # For ranking in UI
        age_days = max(0.0, (as_of - dt.datetime.fromisoformat(issued_at)).total_seconds() / 86400.0)
        aw_now = aw_after
        impact_rank = impact_now(ts_coef, p_true_used, horizon, age_days, aw_now)

        doc_claim_id = f"dc_{doc_id}_{fact_id}_{hashlib.sha256(c.claim.encode('utf-8')).hexdigest()[:8]}"
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
                doc_claim_id, doc_id, fact_id,
                c.claim, c.quote, c.rationale, c.polarity,
                float(best_sim), status,
                float(delta_aw), float(p_true_used), json.dumps(pred_h), float(pred_total)
            ),
        )
        conn.commit()

        claim_rows.append({
            "fact_id": fact_id,
            "status": status,
            "best_match_similarity": float(best_sim),
            "delta_awareness": float(delta_aw),
            "ts_coef": float(ts_coef),
            "p_true": float(p_true_used),
            "impact_rank": float(impact_rank),
            "pred_total": float(pred_total),
            "pred_horizon": pred_h,
            "claim": c.claim,
            "quote": c.quote,
            "rationale": c.rationale,
            "pragmatics": {
                "is_forward_looking": bool(c.is_forward_looking),
                "modality": c.modality,
                "commitment_0_1": float(c.commitment_0_1),
                "conditionality_0_1": float(c.conditionality_0_1),
                "evidential_basis": c.evidential_basis,
                "speaker_role": c.speaker_role,
            },
            "speech_act": speech_act,
            "measurability_0_1": float(text_features["measurability_0_1"]),
            "concreteness_0_1": float(text_features["concreteness_0_1"]),
            "hedge_count": int(text_features["hedge_count"]),
            "commit_count": int(text_features["commit_count"]),
            "quantitative_delta": float(quantitative_delta),
        })

    write_json_file(
        PIPELINE_STAGE_DIRS["fact_matching"] / f"{artifact_stem}.json",
        matching_rows,
    )

    # Document-level score row
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
            doc_id, ticker, timestamp.isoformat(),
            json.dumps(pred_by_horizon),
            float(near_term_total),
            int(len(claim_rows)), int(n_new), int(n_known), int(n_reconfirmed)
        ),
    )
    conn.commit()
    conn.close()

    # Return top claims by impact_rank for showmanship
    claim_rows.sort(key=lambda x: abs(x["impact_rank"]), reverse=True)

    write_json_file(
        PIPELINE_STAGE_DIRS["doc_outputs"] / f"{artifact_stem}.json",
        {
            "doc_id": doc_id,
            "ticker": ticker,
            "doc_type": doc_type,
            "source_type": source_type,
            "timestamp": timestamp.isoformat(),
            "n_claims": len(claim_rows),
            "pred_by_horizon": pred_by_horizon,
            "pred_near_term": float(near_term_total),
            "top_claims": claim_rows[:15],
            "all_claim_rows": claim_rows,
        },
    )

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


# -----------------------------
# Convenience queries for UI / scripts
# -----------------------------
def get_doc_by_sha(sha256: str) -> Optional[Dict[str, Any]]:
    """Look up a document row by its SHA-256 fingerprint."""
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute(
        "SELECT doc_id, ticker, doc_type, source_type, timestamp, sha256, url FROM docs WHERE sha256 = ?",
        (sha256,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    keys = ["doc_id", "ticker", "doc_type", "source_type", "timestamp", "sha256", "url"]
    return dict(zip(keys, row))


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


def list_sources() -> List[Dict[str, Any]]:
    """List sources and their current reliability estimates."""
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.execute("SELECT source_id, reliability, updated_at FROM sources ORDER BY source_id ASC")
    out = [{"source_id": s, "reliability": float(r), "updated_at": t} for (s, r, t) in cur.fetchall()]
    conn.close()
    return out
