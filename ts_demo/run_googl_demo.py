"""
End-to-end demo runner for the T-S coefficient pipeline.

What this script does:
1) Reads a JSON manifest of source documents (10-K/10-Q/call transcripts/etc.).
2) Downloads each file locally (if not already present).
3) Sends each document through `ingest_document`, which extracts claims, links them
   to the knowledge base, and computes valuation-impact style scores.
4) Prints an operator-friendly summary so an analyst can quickly inspect what changed.

This file is intentionally "glue code": it wires together IO + orchestration and keeps
business logic in `ts_system.py`.
"""

import datetime as dt
import os
import time
from pathlib import Path
import json

import requests
from dotenv import load_dotenv

from ts_system import ingest_document

load_dotenv()

OUTDIR = Path("googl_demo_inputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEC_UA = os.environ.get("SEC_USER_AGENT", "GOOGL-demo/1.0 (my.email@example.com)")

import json

def load_docs_from_json(path: str | Path = "googl_docs.json"):
    """
    Load a list of doc descriptors from a JSON file and normalise fields:
      - timestamp -> datetime with tzinfo
      - suffix defaulted from URL if missing
      - authority defaulted to 1.0
    Returns a list of dicts matching the previous DOCS structure.
    """
    # Resolve and validate the manifest path first so any config mistakes fail fast.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Docs JSON not found: {p.resolve()}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    docs = []
    # Parse each descriptor defensively so malformed entries are caught with clear errors.
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each item in docs JSON must be an object/dict.")
        # Required: name and url
        name = item.get("name")
        url = item.get("url")
        if not name or not url:
            raise ValueError("Each doc entry must contain at least 'name' and 'url' fields.")
        suffix = item.get("suffix")
        if not suffix:
            # infer from URL path
            suf = Path(url).suffix
            suffix = suf if suf else ".html"
        ticker = item.get("ticker", "UNKNOWN").upper()
        doc_type = item.get("doc_type", "UNKNOWN")
        source_type = item.get("source_type", "SEC_FILING")
        ts_str = item.get("timestamp")
        # Timestamps are expected in ISO format; we accept either timezone-aware
        # strings or naive ones (fallback assumes UTC).
        if ts_str:
            try:
                timestamp = dt.datetime.fromisoformat(ts_str)
            except Exception:
                # try naive parse (no timezone)
                timestamp = dt.datetime.fromisoformat(ts_str + "+00:00")
        else:
            timestamp = dt.datetime.now(dt.timezone.utc)
        authority = float(item.get("authority", 1.0))
        # Shape each manifest entry into the exact dict expected by the rest
        # of the ingestion flow.
        docs.append({
            "name": name,
            "url": url,
            "suffix": suffix,
            "ticker": ticker,
            "doc_type": doc_type,
            "source_type": source_type,
            "timestamp": timestamp,
            "authority": authority
        })
    return docs

# Load docs (default path: googl_docs.json at repo root)
DOCS_JSON_PATH = os.environ.get("DOCS_JSON_PATH", "docs.json")
DOCS = load_docs_from_json(DOCS_JSON_PATH)

def download(url: str, outpath: Path) -> None:
    """Download a source file to disk.

    SEC endpoints prefer a descriptive User-Agent, so we add one for sec.gov URLs.
    """
    headers = {"User-Agent": SEC_UA} if "sec.gov" in url else {}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    outpath.write_bytes(r.content)

def main() -> None:
    """Run the demo ingestion loop over every document in the manifest."""
    print("DB:", os.environ.get("TS_DB_PATH"))
    print("LLM Model Used:", os.environ.get("OPENAI_MODEL"))
    print("Embed Model Used:", os.environ.get("OPENAI_EMBED_MODEL"))
    # Process documents in chronological order supplied by the manifest so results
    # resemble a walk-forward analyst workflow.
    for i, d in enumerate(DOCS, start=1):
        path = OUTDIR / f"{i:02d}_{d['name']}{d['suffix']}"

        # Skip network work when the file has already been downloaded locally.
        if not path.exists():
            print(f"\n[{i}/7] Downloading {d['name']} ...")
            download(d["url"], path)
            time.sleep(0.8)  # polite throttle

        print(f"[{i}/7] Ingesting {d['name']} @ {d['timestamp'].isoformat()} ...")
        # `as_of` is set to the same document timestamp to emulate a realistic
        # "what was known at that time" run.
        res = ingest_document(
            path=path,
            ticker=d["ticker"],
            doc_type=d["doc_type"],
            source_type=d["source_type"],
            timestamp=d["timestamp"],
            authority=d["authority"],
            url=d["url"],
            as_of=d["timestamp"],  # walk-forward
        )

        print("Doc pred_by_horizon:", {k: round(v, 6) for k, v in res.pred_by_horizon.items()})
        print("Doc pred_near_term:", round(res.pred_near_term, 6))
        # We print only the top few claims for readability in terminal output.
        print("Top 5 claims by |impact_rank|:")
        for c in res.top_claims[:5]:
            # Defensive handling: sometimes an item may be a string (or otherwise malformed).
            if isinstance(c, dict):
                status = c.get("status", "UNKNOWN")
                impact_rank = float(c.get("impact_rank", 0.0))
                delta_aw = float(c.get("delta_awareness", 0.0))
                p_true = float(c.get("p_true", 0.0))
                ts_coef = float(c.get("ts_coef", 0.0))
                claim_text = str(c.get("claim", ""))[:140]
            else:
                # Debug print for unexpected formats (prints once)
                print("[debug] Warning: top_claim element not a dict; showing raw repr")
                print(repr(c)[:400])
                status = "MALFORMED"
                impact_rank = 0.0
                delta_aw = 0.0
                p_true = 0.0
                ts_coef = 0.0
                claim_text = str(c)[:140]

            print(f"  - [{status}] impact_rank={impact_rank:+.4f} "
                  f"delta_aw={delta_aw:.3f} p_true={p_true:.2f} ts_coef={ts_coef:+.3f} :: "
                  f"{claim_text}")

if __name__ == "__main__":
    main()
