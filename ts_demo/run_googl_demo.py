import datetime as dt
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from ts_system import ingest_document

load_dotenv()

OUTDIR = Path("googl_demo_inputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEC_UA = os.environ.get("SEC_USER_AGENT", "GOOGL-demo/1.0 (your.email@example.com)")

DOCS = [
    {
        "name": "GOOGL_2024_10K",
        "url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm",
        "suffix": ".html",
        "ticker": "GOOGL",
        "doc_type": "10-K",
        "source_type": "SEC_FILING",
        "timestamp": dt.datetime(2025, 2, 5, 0, 0, 0, tzinfo=dt.timezone.utc),
        "authority": 1.0,
    },
    {
        "name": "GOOGL_Q1_2025_EARNINGS_CALL",
        "url": "https://s206.q4cdn.com/479360582/files/doc_financials/2025/q1/2025-q1-earnings-transcript.pdf",
        "suffix": ".pdf",
        "ticker": "GOOGL",
        "doc_type": "EARNINGS_CALL_TRANSCRIPT",
        "source_type": "EARNINGS_CALL",
        "timestamp": dt.datetime(2025, 4, 24, 20, 30, 0, tzinfo=dt.timezone.utc),
        "authority": 0.95,
    },
    """
    {
        "name": "GOOGL_2025_Q1_10Q",
        "url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000043/goog-20250331.htm",
        "suffix": ".html",
        "ticker": "GOOGL",
        "doc_type": "10-Q",
        "source_type": "SEC_FILING",
        "timestamp": dt.datetime(2025, 4, 25, 0, 0, 0, tzinfo=dt.timezone.utc),
        "authority": 1.0,
    },
    {
        "name": "GOOGL_Q2_2025_EARNINGS_CALL",
        "url": "https://s206.q4cdn.com/479360582/files/doc_financials/2025/q2/2025-q2-earnings-transcript.pdf",
        "suffix": ".pdf",
        "ticker": "GOOGL",
        "doc_type": "EARNINGS_CALL_TRANSCRIPT",
        "source_type": "EARNINGS_CALL",
        "timestamp": dt.datetime(2025, 7, 23, 20, 30, 0, tzinfo=dt.timezone.utc),
        "authority": 0.95,
    },
    {
        "name": "GOOGL_2025_Q2_10Q",
        "url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000062/goog-20250630.htm",
        "suffix": ".html",
        "ticker": "GOOGL",
        "doc_type": "10-Q",
        "source_type": "SEC_FILING",
        "timestamp": dt.datetime(2025, 7, 24, 0, 0, 0, tzinfo=dt.timezone.utc),
        "authority": 1.0,
    },
    {
        "name": "GOOGL_Q3_2025_EARNINGS_CALL",
        "url": "https://s206.q4cdn.com/479360582/files/doc_events/2025/Oct/29/2025_Q3_Earnings_Transcript.pdf",
        "suffix": ".pdf",
        "ticker": "GOOGL",
        "doc_type": "EARNINGS_CALL_TRANSCRIPT",
        "source_type": "EARNINGS_CALL",
        "timestamp": dt.datetime(2025, 10, 29, 21, 30, 0, tzinfo=dt.timezone.utc),
        "authority": 0.95,
    },
    {
        "name": "GOOGL_2025_Q3_10Q",
        "url": "https://www.sec.gov/Archives/edgar/data/1652044/000165204425000091/goog-20250930.htm",
        "suffix": ".html",
        "ticker": "GOOGL",
        "doc_type": "10-Q",
        "source_type": "SEC_FILING",
        "timestamp": dt.datetime(2025, 10, 30, 0, 0, 0, tzinfo=dt.timezone.utc),
        "authority": 1.0,
    },
    """
]

def download(url: str, outpath: Path) -> None:
    headers = {"User-Agent": SEC_UA} if "sec.gov" in url else {}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    outpath.write_bytes(r.content)

def main() -> None:
    print("DB:", os.environ.get("TS_DB_PATH"))
    print("LLM Model Used:", os.environ.get("OPENAI_MODEL"))
    print("Embed Model Used:", os.environ.get("OPENAI_EMBED_MODEL"))
    for i, d in enumerate(DOCS, start=1):
        path = OUTDIR / f"{i:02d}_{d['name']}{d['suffix']}"

        if not path.exists():
            print(f"\n[{i}/7] Downloading {d['name']} ...")
            download(d["url"], path)
            time.sleep(0.8)  # polite throttle

        print(f"[{i}/7] Ingesting {d['name']} @ {d['timestamp'].isoformat()} ...")
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
        print("Top 5 claims by |impact_rank|:")
        for c in res.top_claims[:5]:
            print(
                f"  - [{c['status']}] impact_rank={c['impact_rank']:+.4f} "
                f"delta_aw={c['delta_awareness']:.3f} p_true={c['p_true']:.2f} ts_coef={c['ts_coef']:+.3f} :: "
                f"{c['claim'][:140]}"
            )

if __name__ == "__main__":
    main()
