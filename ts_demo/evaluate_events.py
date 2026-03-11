"""
Document-level event evaluation for the T-S prototype.

What this file does:
1) Read ingested document-level predictions from SQLite.
2) Map each document timestamp to a market event day.
3) Download price history for the company and a market benchmark.
4) Fit a simple market model before each event.
5) Compute realised abnormal returns (CARs) after each event.
6) Save the event-study output to `event_evaluation_results/` and register the
   run back into SQLite for traceability.
7) Optionally export a claim-level panel and a document-level feature panel for
   empirical work, without needing extra standalone scripts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ts_system import (
    EVENT_EVAL_DIR,
    HORIZON_BUCKETS,
    HORIZON_TRADING_DAYS,
    compare_quantitative_claims,
    infer_speech_act,
    init_db,
    measure_features,
)

load_dotenv()


def market_event_date(ts: dt.datetime) -> pd.Timestamp:
    """Map a UTC event timestamp to the trading day used in the event study.

    Current rule:
    - if the timestamp is at or after 20:00 UTC, treat it as an after-close event
      and shift it to the next trading day;
    - otherwise keep the same calendar date.

    This is intentionally simple. If you later want exchange-aware handling, this
    is the single place to upgrade.
    """
    date = pd.Timestamp(ts.date())
    if ts.time() >= dt.time(20, 0):
        date += pd.Timedelta(days=1)
    return date


def fit_market_model(r_stock: np.ndarray, r_mkt: np.ndarray) -> Tuple[float, float]:
    """Fit a one-factor market model on an estimation window.

    Model:
        r_stock = alpha + beta * r_market

    We use the closed-form OLS solution because it is short, readable, and more
    than enough for a demo-scale event study.
    """
    beta = float(np.cov(r_stock, r_mkt, ddof=1)[0, 1] / (np.var(r_mkt, ddof=1) + 1e-12))
    alpha = float(np.mean(r_stock) - beta * np.mean(r_mkt))
    return alpha, beta


def resolve_output_path(default_name: str, user_out: str) -> Path:
    """Resolve where an output file should be written.

    Behaviour:
    - if the user passes no `--out`, create a descriptive file name automatically;
    - if the user passes just a bare file name, place it inside
      `event_evaluation_results/`;
    - if the user passes a full path, respect it.
    """
    EVENT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not user_out:
        return EVENT_EVAL_DIR / default_name

    out_path = Path(user_out)
    if out_path.parent == Path("."):
        return EVENT_EVAL_DIR / out_path.name
    return out_path


def json_default(value: Any) -> Any:
    """Convert common numpy/pandas types into plain JSON-safe Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def build_summary(
    df: pd.DataFrame,
    ticker: str,
    market: str,
    out_path: Path,
    estimation_days: int,
    buffer_days: int,
    price_start: dt.datetime,
    price_end: dt.datetime,
    processed: int,
    skipped: int,
) -> Dict[str, Any]:
    """Build a small run summary used both for printing and DB registration."""
    summary: Dict[str, Any] = {
        "ticker": ticker,
        "market_ticker": market,
        "out_csv_path": str(out_path),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimation_days": int(estimation_days),
        "buffer_days": int(buffer_days),
        "price_window_start": price_start.date().isoformat(),
        "price_window_end": price_end.date().isoformat(),
        "processed_events": int(processed),
        "skipped_events": int(skipped),
        "row_count": int(len(df)),
        "pearson_all": {},
        "pearson_uncontaminated": {},
        "hit_rate_uncontaminated": {},
    }

    for bucket in HORIZON_BUCKETS:
        pred_col = f"PRED_{bucket}"
        car_col = f"CAR_{bucket}"
        contam_col = f"contaminated_{bucket}"

        if pred_col in df.columns and car_col in df.columns:
            if df[pred_col].std() >= 1e-12 and df[car_col].std() >= 1e-12:
                summary["pearson_all"][bucket] = float(df[pred_col].corr(df[car_col], method="pearson"))

        if contam_col in df.columns and pred_col in df.columns and car_col in df.columns:
            sub = df[~df[contam_col]].copy()
            if len(sub) >= 3 and sub[pred_col].std() >= 1e-12 and sub[car_col].std() >= 1e-12:
                summary["pearson_uncontaminated"][bucket] = {
                    "corr": float(sub[pred_col].corr(sub[car_col], method="pearson")),
                    "n": int(len(sub)),
                }

            if len(sub) >= 3:
                pred_sign = np.sign(sub[pred_col])
                car_sign = np.sign(sub[car_col])
                summary["hit_rate_uncontaminated"][bucket] = {
                    "hit_rate": float(np.mean(pred_sign == car_sign)),
                    "n": int(len(sub)),
                }

    return summary


def register_event_evaluation(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    ticker: str,
    market: str,
    out_path: Path,
    estimation_days: int,
    buffer_days: int,
    summary: Dict[str, Any],
) -> str:
    """Register the evaluation run and its rows in SQLite.

    We keep the DB registration deliberately simple:
    - one run row with metadata and summary JSON;
    - one row per evaluated document, stored as JSON for easy inspection.

    This avoids a big schema expansion while still giving you a durable audit log.
    """
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run_seed = f"{ticker}|{market}|{created_at}|{out_path}"
    eval_run_id = f"eval_{hashlib.sha256(run_seed.encode('utf-8')).hexdigest()[:12]}"

    conn.execute(
        """
        INSERT OR REPLACE INTO event_evaluation_runs(
            eval_run_id, ticker, market_ticker, created_at, out_csv_path,
            estimation_days, buffer_days, row_count, summary_json
        )
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            eval_run_id,
            ticker,
            market,
            created_at,
            str(out_path),
            int(estimation_days),
            int(buffer_days),
            int(len(df)),
            json.dumps(summary, ensure_ascii=False, default=json_default),
        ),
    )

    for row_no, row in enumerate(df.to_dict(orient="records"), start=1):
        safe_row = {}
        for key, value in row.items():
            if pd.isna(value):
                safe_row[key] = None
            else:
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_row[key] = value
                else:
                    safe_row[key] = json.loads(json.dumps(value, default=json_default))

        eval_row_id = f"evalrow_{eval_run_id}_{row_no:04d}"
        conn.execute(
            """
            INSERT OR REPLACE INTO event_evaluation_rows(eval_row_id, eval_run_id, doc_id, row_json)
            VALUES (?,?,?,?)
            """,
            (
                eval_row_id,
                eval_run_id,
                str(safe_row.get("doc_id", "")),
                json.dumps(safe_row, ensure_ascii=False, default=json_default),
            ),
        )

    conn.commit()
    return eval_run_id


def print_summary(summary: Dict[str, Any]) -> None:
    """Print the run summary in a compact, analyst-friendly format."""
    print("\n=== Correlations (Pearson) ===")
    for bucket, value in summary.get("pearson_all", {}).items():
        print(f"{bucket}: {value:.3f}")

    print("\n=== Correlations (Pearson, uncontaminated only) ===")
    for bucket, payload in summary.get("pearson_uncontaminated", {}).items():
        print(f"{bucket}: {payload['corr']:.3f} (n={payload['n']})")

    print("\n=== Directional hit-rate (uncontaminated only) ===")
    for bucket, payload in summary.get("hit_rate_uncontaminated", {}).items():
        print(f"{bucket}: {payload['hit_rate']:.2%} (n={payload['n']})")


def build_claim_panel(conn: sqlite3.Connection, ticker: str, eval_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per atomic claim for downstream empirical work.

    Why this sits in `evaluate_events.py`:
    - the claim panel is mainly useful once you also have event-window outcomes;
    - keeping it here avoids extra standalone scripts and keeps the codebase small.

    The panel mixes:
    - raw provenance from the DB,
    - deterministic measurability/concreteness features,
    - a coarse speech-act label,
    - optional merge-in of document-level CAR/PRED columns.
    """
    cur = conn.execute(
        """
        SELECT
            d.doc_id,
            d.ticker,
            d.doc_type,
            d.source_type,
            d.timestamp,
            dc.fact_id,
            dc.extracted_claim,
            dc.quote,
            dc.rationale,
            dc.status,
            dc.best_match_similarity,
            dc.delta_awareness,
            dc.p_true_used,
            dc.pred_total,
            dc.pred_horizon_json,
            f.canonical_text,
            f.sign,
            f.materiality,
            f.novelty,
            f.surprise,
            f.ts_coef,
            f.is_forward_looking,
            f.modality,
            f.commitment,
            f.conditionality,
            f.evidential_basis,
            f.speaker_role
        FROM doc_claims dc
        JOIN docs d ON dc.doc_id = d.doc_id
        JOIN facts f ON dc.fact_id = f.fact_id
        WHERE d.ticker = ?
        ORDER BY d.timestamp ASC, dc.doc_claim_id ASC
        """,
        (ticker,),
    )

    rows = []
    for row in cur.fetchall():
        (
            doc_id,
            ticker_val,
            doc_type,
            source_type,
            timestamp,
            fact_id,
            claim,
            quote,
            rationale,
            status,
            best_match_similarity,
            delta_awareness,
            p_true_used,
            pred_total,
            pred_horizon_json,
            canonical_text,
            sign,
            materiality,
            novelty,
            surprise,
            ts_coef,
            is_forward_looking,
            modality,
            commitment,
            conditionality,
            evidential_basis,
            speaker_role,
        ) = row

        feat = measure_features(claim)
        speech_act = infer_speech_act(claim, modality)
        quant_cmp = compare_quantitative_claims(claim, canonical_text)

        rows.append(
            {
                "doc_id": doc_id,
                "ticker": ticker_val,
                "doc_type": doc_type,
                "source_type": source_type,
                "timestamp": timestamp,
                "fact_id": fact_id,
                "claim": claim,
                "quote": quote,
                "rationale": rationale,
                "status": status,
                "best_match_similarity": float(best_match_similarity),
                "delta_awareness": float(delta_awareness),
                "p_true_used": float(p_true_used),
                "pred_total": float(pred_total),
                "pred_horizon_json": pred_horizon_json,
                "canonical_fact_text": canonical_text,
                "sign": int(sign),
                "materiality": float(materiality),
                "novelty": float(novelty),
                "surprise": float(surprise),
                "ts_coef": float(ts_coef),
                "is_forward_looking": int(is_forward_looking),
                "modality": modality,
                "speech_act": speech_act,
                "commitment_0_1": float(commitment),
                "conditionality_0_1": float(conditionality),
                "evidential_basis": evidential_basis,
                "speaker_role": speaker_role,
                "measurability_0_1": float(feat["measurability_0_1"]),
                "concreteness_0_1": float(feat["concreteness_0_1"]),
                "hedge_count": int(feat["hedge_count"]),
                "commit_count": int(feat["commit_count"]),
                "has_number": int(bool(feat["values"])),
                "has_currency": int(bool(feat["has_currency"])),
                "has_percent": int(bool(feat["has_percent"])),
                "has_date": int(bool(feat["has_date"])),
                "quantitative_delta": float(quant_cmp["delta_strength"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Convenient within-document controls.
    df["doc_claim_count"] = df.groupby("doc_id")["fact_id"].transform("count")
    doc_abs_pred = df.groupby("doc_id")["pred_total"].transform(lambda s: s.abs().sum() + 1e-12)
    df["pred_share_of_doc_abs"] = df["pred_total"].abs() / doc_abs_pred
    df["pred_z_within_doc"] = df.groupby("doc_id")["pred_total"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    )

    if eval_df is not None and not eval_df.empty:
        keep_cols = [
            c
            for c in eval_df.columns
            if c == "doc_id" or c.startswith("CAR_") or c.startswith("PRED_") or c.startswith("contaminated_")
        ]
        eval_small = eval_df[keep_cols].copy()
        df = df.merge(eval_small, on="doc_id", how="left")

    return df


def build_doc_feature_panel(claim_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the claim panel into one row per released document.

    This is the safer first-stage empirical panel because your realised outcomes
    currently live at the document/event level rather than the individual-claim level.

    We compute both unweighted and prediction-weighted aggregates so you can test:
    - broad document communication style;
    - style concentrated in the most model-relevant claims.
    """
    if claim_df.empty:
        return claim_df.copy()

    df = claim_df.copy()
    df["abs_pred_total"] = df["pred_total"].abs()
    df["weight"] = df.groupby("doc_id")["abs_pred_total"].transform(lambda s: s / (s.sum() + 1e-12))

    df["is_commissive"] = (df["speech_act"] == "COMMISSIVE").astype(int)
    df["is_assertive"] = (df["speech_act"] == "ASSERTIVE").astype(int)

    meta_first_cols = [
        "ticker", "doc_type", "source_type", "timestamp", "doc_claim_count",
    ]
    event_cols = [c for c in df.columns if c.startswith("CAR_") or c.startswith("PRED_") or c.startswith("contaminated_")]

    out_rows = []
    for doc_id, g in df.groupby("doc_id", sort=False):
        row = {"doc_id": doc_id}
        for col in meta_first_cols + event_cols:
            if col in g.columns:
                row[col] = g.iloc[0][col]

        row["n_claims"] = int(len(g))
        row["share_commissive"] = float(g["is_commissive"].mean())
        row["share_assertive"] = float(g["is_assertive"].mean())
        row["share_forward_looking"] = float(g["is_forward_looking"].mean())

        row["avg_measurability_0_1"] = float(g["measurability_0_1"].mean())
        row["avg_concreteness_0_1"] = float(g["concreteness_0_1"].mean())
        row["avg_commitment_0_1"] = float(g["commitment_0_1"].mean())
        row["avg_conditionality_0_1"] = float(g["conditionality_0_1"].mean())

        row["hedge_density_per_claim"] = float(g["hedge_count"].sum() / max(len(g), 1))
        row["commit_density_per_claim"] = float(g["commit_count"].sum() / max(len(g), 1))
        row["max_quantitative_delta"] = float(g["quantitative_delta"].max())
        row["mean_quantitative_delta"] = float(g["quantitative_delta"].mean())

        row["weighted_commissive_share"] = float((g["is_commissive"] * g["weight"]).sum())
        row["weighted_assertive_share"] = float((g["is_assertive"] * g["weight"]).sum())
        row["weighted_forward_looking_share"] = float((g["is_forward_looking"] * g["weight"]).sum())
        row["weighted_measurability_0_1"] = float((g["measurability_0_1"] * g["weight"]).sum())
        row["weighted_concreteness_0_1"] = float((g["concreteness_0_1"] * g["weight"]).sum())
        row["weighted_commitment_0_1"] = float((g["commitment_0_1"] * g["weight"]).sum())
        row["weighted_conditionality_0_1"] = float((g["conditionality_0_1"] * g["weight"]).sum())
        row["weighted_quantitative_delta"] = float((g["quantitative_delta"] * g["weight"]).sum())
        row["weighted_hedge_density"] = float((g["hedge_count"] * g["weight"]).sum())
        row["weighted_commit_density"] = float((g["commit_count"] * g["weight"]).sum())

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main() -> None:
    """Run the event-study evaluation from the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("TS_DB_PATH", "system_db.sqlite3"))
    ap.add_argument("--ticker", default="GOOGL")
    ap.add_argument("--market", default="SPY")
    ap.add_argument(
        "--out",
        default="",
        help="Optional output CSV name or path. Bare file names are placed inside event_evaluation_results/.",
    )
    ap.add_argument("--estimation-days", type=int, default=120)
    ap.add_argument("--buffer-days", type=int, default=20)

    # Optional downstream exports.
    ap.add_argument(
        "--export-claim-panel",
        action="store_true",
        help="Also export one row per atomic claim into event_evaluation_results/.",
    )
    ap.add_argument(
        "--claim-panel-out",
        default="",
        help="Optional claim-panel CSV name/path. Bare names go into event_evaluation_results/.",
    )
    ap.add_argument(
        "--export-doc-feature-panel",
        action="store_true",
        help="Also export one row per document with aggregated pragmatics features.",
    )
    ap.add_argument(
        "--doc-feature-panel-out",
        default="",
        help="Optional doc-feature-panel CSV name/path. Bare names go into event_evaluation_results/.",
    )
    args = ap.parse_args()

    event_out_path = resolve_output_path(
        default_name=f"{args.ticker.lower()}_vs_{args.market.lower()}_event_evaluation.csv",
        user_out=args.out,
    )

    conn = sqlite3.connect(args.db)
    init_db(conn)

    # ------------------------------------------------------------------
    # Read the event list from the existing ingest results.
    # ------------------------------------------------------------------
    cur = conn.execute(
        "SELECT d.doc_id, d.timestamp, s.pred_horizon_json "
        "FROM docs d JOIN doc_scores s ON d.doc_id = s.doc_id "
        "WHERE d.ticker = ? ORDER BY d.timestamp ASC",
        (args.ticker,),
    )
    events = [
        (doc_id, dt.datetime.fromisoformat(ts), json.loads(pred_h))
        for doc_id, ts, pred_h in cur.fetchall()
    ]

    if not events:
        conn.close()
        raise SystemExit("No events found in DB. Run run_program.py first.")

    # ------------------------------------------------------------------
    # Download prices for a window that comfortably covers both estimation and
    # post-event periods.
    # ------------------------------------------------------------------
    price_start = min(ts for _, ts, _ in events) - dt.timedelta(days=300)
    price_end = max(ts for _, ts, _ in events) + dt.timedelta(days=180)

    print(f"Price download window: {price_start.date().isoformat()} -> {price_end.date().isoformat()}")
    print(f"Processing {len(events)} events...")

    px = yf.download(
        [args.ticker, args.market],
        start=price_start.date().isoformat(),
        end=price_end.date().isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"].copy()
    else:
        close = px[["Close"]].copy()

    close = close.dropna(how="all")
    if args.ticker not in close.columns or args.market not in close.columns:
        conn.close()
        raise SystemExit("Price download missing expected tickers. Check ticker symbols / network.")

    rets = close.pct_change().dropna()
    dates = rets.index

    # ------------------------------------------------------------------
    # Resolve each document timestamp into a trading-day event anchor.
    # ------------------------------------------------------------------
    event_days = []
    for doc_id, ts, pred_h in events:
        event_day = market_event_date(ts)
        idx = dates.searchsorted(event_day)
        if idx >= len(dates):
            print(f"[eval] Event {doc_id} at {event_day.date()} after last price date; skipping.")
            continue
        event_days.append((doc_id, ts, pred_h, idx, dates[idx]))

    rows = []
    processed = 0
    skipped = 0

    # ------------------------------------------------------------------
    # Evaluate each event one by one.
    # ------------------------------------------------------------------
    for i, (doc_id, ts, pred_h, idx0, trading_day) in enumerate(event_days):
        print(f"[eval] ({i + 1}/{len(event_days)}) processing event {doc_id} at trading date {trading_day.date()}...")

        # Estimation window: [-(estimation_days), -(buffer_days)]
        est_start = idx0 - args.estimation_days
        est_end = idx0 - args.buffer_days
        if est_start < 0 or est_end <= est_start:
            print(f"[eval]  - Insufficient estimation window for {doc_id}; skipping.")
            skipped += 1
            continue

        r_stock = rets[args.ticker].iloc[est_start:est_end].to_numpy()
        r_mkt = rets[args.market].iloc[est_start:est_end].to_numpy()
        alpha, beta = fit_market_model(r_stock, r_mkt)

        row: Dict[str, Any] = {
            "doc_id": doc_id,
            "ticker": args.ticker,
            "market_ticker": args.market,
            "event_timestamp_utc": ts.isoformat(),
            "event_trading_day": str(trading_day.date()),
        }

        # Contamination flag: another event arrives before the horizon finishes.
        for bucket in HORIZON_BUCKETS:
            n_days = HORIZON_TRADING_DAYS[bucket]
            window_end = idx0 + n_days
            contaminated = any(
                (j != i) and (idx0 < idxj <= window_end)
                for j, (_, _, _, idxj, _) in enumerate(event_days)
            )
            row[f"contaminated_{bucket}"] = contaminated

        # Realised CAR per horizon.
        for bucket in HORIZON_BUCKETS:
            n_days = HORIZON_TRADING_DAYS[bucket]
            idx1 = min(len(rets) - 1, idx0 + n_days)
            r_s = rets[args.ticker].iloc[idx0:idx1].to_numpy()
            r_m = rets[args.market].iloc[idx0:idx1].to_numpy()
            abnormal_returns = r_s - (alpha + beta * r_m)
            car = float(np.sum(abnormal_returns))
            row[f"CAR_{bucket}"] = car
            row[f"PRED_{bucket}"] = float(pred_h.get(bucket, 0.0))

        rows.append(row)
        processed += 1

    print(f"[eval] Done. Processed: {processed}, Skipped: {skipped}, Total events considered: {len(event_days)}")

    df = pd.DataFrame(rows)
    if df.empty:
        conn.close()
        raise SystemExit("Not enough data to evaluate (estimation windows missing).")

    summary = build_summary(
        df=df,
        ticker=args.ticker,
        market=args.market,
        out_path=event_out_path,
        estimation_days=args.estimation_days,
        buffer_days=args.buffer_days,
        price_start=price_start,
        price_end=price_end,
        processed=processed,
        skipped=skipped,
    )

    # ------------------------------------------------------------------
    # Optional claim-level and doc-level panels.
    # ------------------------------------------------------------------
    claim_panel_path = None
    doc_feature_panel_path = None
    claim_df = None

    need_claim_panel = args.export_claim_panel or args.export_doc_feature_panel
    if need_claim_panel:
        claim_df = build_claim_panel(conn=conn, ticker=args.ticker, eval_df=df)
        if claim_df.empty:
            print("[eval] Claim panel requested, but no claims were found in the DB.")
        else:
            claim_panel_path = resolve_output_path(
                default_name=f"{args.ticker.lower()}_claim_panel.csv",
                user_out=args.claim_panel_out,
            )
            claim_panel_path.parent.mkdir(parents=True, exist_ok=True)
            claim_df.to_csv(claim_panel_path, index=False)
            print(f"[eval] Saved claim panel: {claim_panel_path}")

            if args.export_doc_feature_panel:
                doc_feature_df = build_doc_feature_panel(claim_df)
                doc_feature_panel_path = resolve_output_path(
                    default_name=f"{args.ticker.lower()}_doc_feature_panel.csv",
                    user_out=args.doc_feature_panel_out,
                )
                doc_feature_panel_path.parent.mkdir(parents=True, exist_ok=True)
                doc_feature_df.to_csv(doc_feature_panel_path, index=False)
                print(f"[eval] Saved doc feature panel: {doc_feature_panel_path}")

    if claim_panel_path is not None:
        summary["claim_panel_csv_path"] = str(claim_panel_path)
    if doc_feature_panel_path is not None:
        summary["doc_feature_panel_csv_path"] = str(doc_feature_panel_path)

    print_summary(summary)

    # ------------------------------------------------------------------
    # Save the run to disk: one CSV for row-level results and one small JSON
    # summary for quick inspection.
    # ------------------------------------------------------------------
    event_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(event_out_path, index=False)

    summary_path = event_out_path.with_name(event_out_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    eval_run_id = register_event_evaluation(
        conn=conn,
        df=df,
        ticker=args.ticker,
        market=args.market,
        out_path=event_out_path,
        estimation_days=args.estimation_days,
        buffer_days=args.buffer_days,
        summary=summary,
    )
    conn.close()

    print(f"\nSaved CSV: {event_out_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Registered evaluation run in DB: {eval_run_id}")


if __name__ == "__main__":
    main()
