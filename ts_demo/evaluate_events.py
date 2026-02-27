import argparse
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ts_system import DB_PATH, HORIZON_BUCKETS, HORIZON_TRADING_DAYS, init_db

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
USER_INPUT_DIR = DATA_DIR / "input" / "user_input"
OUTPUT_DIR = DATA_DIR / "output"


def market_event_date(ts: dt.datetime) -> pd.Timestamp:
    """
    If event timestamp is after (approx) US close, shift to next trading day.
    We use a simple UTC threshold (20:00) for v1.
    """
    date = pd.Timestamp(ts.date())
    if ts.time() >= dt.time(20, 0):  # after close (approx)
        date += pd.Timedelta(days=1)
    return date


def fit_market_model(r_stock: np.ndarray, r_mkt: np.ndarray) -> tuple[float, float]:
    """
    OLS: r_stock = alpha + beta * r_mkt
    """
    beta = float(np.cov(r_stock, r_mkt, ddof=1)[0, 1] / (np.var(r_mkt, ddof=1) + 1e-12))
    alpha = float(np.mean(r_stock) - beta * np.mean(r_mkt))
    return alpha, beta


def load_eval_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {cfg_path.resolve()}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Evaluation config must be a JSON object.")
    ticker = str(cfg.get("ticker", "")).strip().upper()
    market = str(cfg.get("market", "")).strip().upper()
    if not ticker or not market:
        raise ValueError("Evaluation config must contain non-empty 'ticker' and 'market' fields.")
    return {"ticker": ticker, "market": market}


def default_output_path(ticker: str) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return str(OUTPUT_DIR / f"{ticker}-eval-{stamp}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--config", default=os.getenv("EVAL_CONFIG_PATH", str(USER_INPUT_DIR / "eval_config.json")))
    ap.add_argument("--ticker", default=None, help="Optional override for ticker in eval config.")
    ap.add_argument("--market", default=None, help="Optional override for market benchmark in eval config.")
    ap.add_argument("--out", default=None, help="Optional output CSV path. Defaults to timestamped file in data/output.")
    args = ap.parse_args()

    cfg = load_eval_config(args.config)
    ticker = (args.ticker or cfg["ticker"]).upper()
    market = (args.market or cfg["market"]).upper()
    out_path = args.out or default_output_path(ticker)

    conn = sqlite3.connect(args.db)
    init_db(conn)

    cur = conn.execute(
        "SELECT d.doc_id, d.timestamp, s.pred_horizon_json "
        "FROM docs d JOIN doc_scores s ON d.doc_id = s.doc_id "
        "WHERE d.ticker = ? ORDER BY d.timestamp ASC",
        (ticker,),
    )
    events = [(doc_id, dt.datetime.fromisoformat(ts), json.loads(pred_h)) for doc_id, ts, pred_h in cur.fetchall()]
    conn.close()

    if not events:
        raise SystemExit("No events found in DB. Run run_googl_demo.py first.")

    # Download prices covering all events
    start = min(ts for _, ts, _ in events) - dt.timedelta(days=300)
    end = max(ts for _, ts, _ in events) + dt.timedelta(days=180)

    px = yf.download([ticker, market], start=start.date().isoformat(), end=end.date().isoformat(), auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"].copy()
    else:
        close = px[["Close"]].copy()

    close = close.dropna(how="all")
    if ticker not in close.columns or market not in close.columns:
        raise SystemExit("Price download missing expected tickers. Check ticker symbols / network.")

    rets = close.pct_change().dropna()
    dates = rets.index

    # Prepare contamination flags
    print(f"Price download window: {start.date().isoformat()} → {end.date().isoformat()}")
    print(f"Processing {len(events)} events...")

    event_days = []
    for doc_id, ts, pred_h in events:
        d = market_event_date(ts)
        idx = dates.searchsorted(d)
        if idx >= len(dates):
            print(f"[eval] Event {doc_id} at {d.date()} after last price date; skipping.")
            continue
        event_days.append((doc_id, ts, pred_h, idx, dates[idx]))

    rows = []
    processed = 0
    skipped = 0
    for i, (doc_id, ts, pred_h, idx0, d0) in enumerate(event_days):
        print(f"[eval] ({i+1}/{len(event_days)}) processing event {doc_id} at trading date {d0.date()}...")
        # estimation window: [-120, -20]
        est_start = idx0 - 120
        est_end = idx0 - 20
        if est_start < 0 or est_end <= est_start:
            print(f"[eval]  - Insufficient estimation window for {doc_id}; skipping.")
            skipped += 1
            continue

        r_stock = rets[ticker].iloc[est_start:est_end].to_numpy()
        r_mkt = rets[market].iloc[est_start:est_end].to_numpy()
        alpha, beta = fit_market_model(r_stock, r_mkt)

        row = {
            "doc_id": doc_id,
            "event_timestamp_utc": ts.isoformat(),
            "event_trading_day": str(d0.date()),
        }

        # contamination: another event within the horizon window
        for b in HORIZON_BUCKETS:
            n = HORIZON_TRADING_DAYS[b]
            window_end = idx0 + n
            contaminated = any((j != i) and (idx0 < idxj <= window_end) for j, (_, _, _, idxj, _) in enumerate(event_days))
            row[f"contaminated_{b}"] = contaminated

        # realised CAR per horizon
        for b in HORIZON_BUCKETS:
            n = HORIZON_TRADING_DAYS[b]
            idx1 = min(len(rets) - 1, idx0 + n)
            r_s = rets[ticker].iloc[idx0:idx1].to_numpy()
            r_m = rets[market].iloc[idx0:idx1].to_numpy()
            ar = r_s - (alpha + beta * r_m)
            car = float(np.sum(ar))
            row[f"CAR_{b}"] = car
            row[f"PRED_{b}"] = float(pred_h.get(b, 0.0))

        rows.append(row)
        processed += 1

    print(f"[eval] Done. Processed: {processed}, Skipped: {skipped}, Total events considered: {len(event_days)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Not enough data to evaluate (estimation windows missing).")

    # Correlations (all vs uncontaminated)
    print("\n=== Correlations (Pearson) ===")
    for b in HORIZON_BUCKETS:
        if df[f"PRED_{b}"].std() < 1e-12 or df[f"CAR_{b}"].std() < 1e-12:
            continue
        pear = df[f"PRED_{b}"].corr(df[f"CAR_{b}"], method="pearson")
        print(f"{b}: {pear:.3f}")

    print("\n=== Correlations (Pearson, uncontaminated only) ===")
    for b in HORIZON_BUCKETS:
        sub = df[~df[f"contaminated_{b}"]].copy()
        if len(sub) < 3:
            continue
        if sub[f"PRED_{b}"].std() < 1e-12 or sub[f"CAR_{b}"].std() < 1e-12:
            continue
        pear = sub[f"PRED_{b}"].corr(sub[f"CAR_{b}"], method="pearson")
        print(f"{b}: {pear:.3f} (n={len(sub)})")

    # Directional hit rate
    print("\n=== Directional hit-rate (uncontaminated only) ===")
    for b in HORIZON_BUCKETS:
        sub = df[~df[f"contaminated_{b}"]].copy()
        if len(sub) < 3:
            continue
        pred_sign = np.sign(sub[f"PRED_{b}"])
        car_sign = np.sign(sub[f"CAR_{b}"])
        hit = float(np.mean(pred_sign == car_sign))
        print(f"{b}: {hit:.2%} (n={len(sub)})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
