from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "data" / "output" / "ts_kb_GOOGL_demo.sqlite3"
DB_PATH = os.getenv("TS_DB_PATH", str(DEFAULT_DB_PATH))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

TS_MAX_CHUNKS = int(os.getenv("TS_MAX_CHUNKS", "30"))
LAMBDA_DECAY = float(os.getenv("TS_LAMBDA_DECAY", "0.002"))
TAU_RELIABILITY_DAYS = float(os.getenv("TS_TAU_RELIABILITY_DAYS", "365"))

HORIZON_BUCKETS = ["1D", "1W", "1M", "3M", "1Y", "3Y"]
HORIZON_TRADING_DAYS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252, "3Y": 756}
NEAR_WEIGHTS = {"1D": 1.00, "1W": 0.75, "1M": 0.55, "3M": 0.40, "1Y": 0.25, "3Y": 0.15}
HALF_LIFE_DAYS = {"1D": 2, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "3Y": 900}

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

EMBED_SAME_STRICT = 0.93
EMBED_SAME_LOOSE = 0.85
FUZZ_SAME_STRICT = 0.92

def status_from_similarity(sim: float) -> str:
    if sim >= 0.95:
        return "RECONFIRMED"
    if sim >= 0.65:
        return "KNOWN"
    return "NEW"
