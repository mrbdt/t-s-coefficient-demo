"""
End-to-end ingestion runner for the T-S coefficient pipeline.

What this script does:
1) Reads a JSON manifest of source documents (10-K/10-Q/call transcripts/etc.).
2) Clears the existing `pipeline/` stage outputs so the new run is easy to inspect.
3) Downloads each source file into `pipeline/1_ingested_inputs/`.
4) Sends each document through `ingest_document`, which extracts claims, links them
   to the knowledge base, and computes valuation-impact style scores.
5) Prints an operator-friendly summary so an analyst can quickly inspect what changed.

This file is intentionally glue code: network IO + orchestration live here, while
business logic stays in `ts_system.py`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv

from ts_system import PIPELINE_STAGE_DIRS, clear_pipeline_outputs, ensure_pipeline_dirs, ingest_document

load_dotenv()

# All downloaded source files are stored in the first pipeline stage directory.
OUTDIR = PIPELINE_STAGE_DIRS["inputs"]
ensure_pipeline_dirs()

# SEC requests should use a descriptive user-agent string.
SEC_UA = os.environ.get("SEC_USER_AGENT", "TS-demo/1.0 (my.email@example.com)")


def load_docs_from_json(path: str | Path = "docs.json") -> List[Dict[str, object]]:
    """Load a list of document descriptors from a JSON manifest.

    Expected fields per item:
    - name
    - url

    Optional fields:
    - suffix
    - ticker
    - doc_type
    - source_type
    - timestamp
    - authority

    We keep this parser strict for required fields so bad manifests fail early.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Docs JSON not found: {manifest_path.resolve()}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs: List[Dict[str, object]] = []

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each item in docs JSON must be an object/dict.")

        name = item.get("name")
        url = item.get("url")
        if not name or not url:
            raise ValueError("Each doc entry must contain at least 'name' and 'url' fields.")

        suffix = item.get("suffix")
        if not suffix:
            inferred_suffix = Path(str(url)).suffix
            suffix = inferred_suffix if inferred_suffix else ".html"

        ticker = str(item.get("ticker", "UNKNOWN")).upper()
        doc_type = str(item.get("doc_type", "UNKNOWN"))
        source_type = str(item.get("source_type", "SEC_FILING"))
        authority = float(item.get("authority", 1.0))

        ts_str = item.get("timestamp")
        if ts_str:
            try:
                timestamp = dt.datetime.fromisoformat(str(ts_str))
            except Exception:
                timestamp = dt.datetime.fromisoformat(str(ts_str) + "+00:00")
        else:
            timestamp = dt.datetime.now(dt.timezone.utc)

        docs.append(
            {
                "name": str(name),
                "url": str(url),
                "suffix": str(suffix),
                "ticker": ticker,
                "doc_type": doc_type,
                "source_type": source_type,
                "timestamp": timestamp,
                "authority": authority,
            }
        )

    return docs


def download(url: str, outpath: Path) -> None:
    """Download a source file to disk.

    SEC endpoints prefer a descriptive User-Agent, so we add one for sec.gov URLs.
    """
    headers = {"User-Agent": SEC_UA} if "sec.gov" in url else {}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    outpath.write_bytes(response.content)

def main() -> None:
    """Run the walk-forward ingestion loop and print a compact analyst summary."""
    # Resolve the manifest at runtime so environment changes are picked up each time.
    docs_json_path = os.environ.get("DOCS_JSON_PATH", "docs.json")
    docs = load_docs_from_json(docs_json_path)

    # Clear previous pipeline stage outputs at the start of each run so the folder
    # tree always reflects just the current ingestion pass.
    clear_pipeline_outputs()
    ensure_pipeline_dirs()

    print("DB:", os.environ.get("TS_DB_PATH", "system_db.sqlite3"))
    print("LLM Model Used:", os.environ.get("OPENAI_MODEL"))
    print("Embed Model Used:", os.environ.get("OPENAI_EMBED_MODEL"))
    print("Docs manifest:", docs_json_path)
    print("Pipeline root:", OUTDIR.parent)

    total_docs = len(docs)

    # Process documents in manifest order so results resemble a walk-forward run.
    for i, d in enumerate(docs, start=1):
        path = OUTDIR / f"{i:02d}_{d['name']}{d['suffix']}"

        print(f"\n[{i}/{total_docs}] Downloading {d['name']} ...")
        download(str(d["url"]), path)
        time.sleep(0.8)  # polite throttle

        print(f"[{i}/{total_docs}] Ingesting {d['name']} @ {d['timestamp'].isoformat()} ...")
        res = ingest_document(
            path=path,
            ticker=str(d["ticker"]),
            doc_type=str(d["doc_type"]),
            source_type=str(d["source_type"]),
            timestamp=d["timestamp"],  # type: ignore[arg-type]
            authority=float(d["authority"]),
            url=str(d["url"]),
            as_of=d["timestamp"],  # type: ignore[arg-type]
        )

        print("Doc pred_by_horizon:", {k: round(v, 6) for k, v in res.pred_by_horizon.items()})
        print("Doc pred_near_term:", round(res.pred_near_term, 6))
        print("Top 5 claims by |impact_rank|:")
        for c in res.top_claims[:5]:
            if isinstance(c, dict):
                status = c.get("status", "UNKNOWN")
                impact_rank = float(c.get("impact_rank", 0.0))
                delta_aw = float(c.get("delta_awareness", 0.0))
                p_true = float(c.get("p_true", 0.0))
                ts_coef = float(c.get("ts_coef", 0.0))
                claim_text = str(c.get("claim", ""))[:140]
            else:
                print("[debug] Warning: top_claim element not a dict; showing raw repr")
                print(repr(c)[:400])
                status = "MALFORMED"
                impact_rank = 0.0
                delta_aw = 0.0
                p_true = 0.0
                ts_coef = 0.0
                claim_text = str(c)[:140]

            print(
                f"  - [{status}] impact_rank={impact_rank:+.4f} "
                f"delta_aw={delta_aw:.3f} p_true={p_true:.2f} ts_coef={ts_coef:+.3f} :: "
                f"{claim_text}"
            )


if __name__ == "__main__":
    main()
