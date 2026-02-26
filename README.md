# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

## Repository layout

- `ts_demo/core/` → shared core modules extracted from the old monolithic `ts_system.py`:
  - `config.py` (environment + constants)
  - `models.py` (pydantic claim schemas)
  - `text_processing.py` (document normalization/chunking)
  - `database.py`, `scoring.py`, `extraction.py`, `reliability.py`, `query.py`, `ingest_api.py` (split runtime logic by responsibility)
- `ts_demo/pipeline/` → ingestion pipeline stages + orchestrator.
- `ts_demo/data/input/` → downloaded/raw demo source documents.
- `ts_demo/data/output/` → generated artifacts (DB, eval CSV, docs manifest).
- `ts_demo/ts_system.py` → compatibility API used by UI/scripts, now delegating to modularized components.

## How to use
Activate the venv (if not already) (do so within the ts_demo folder):

```bash
source .venv/bin/activate
```

If not yet present, create a `.env` file for environment parameters. see `.env.example` for fields that needs to be filled in.

(optional) Re-create a clean DB, clear the inputs folder, clear the evaluation output (for a clear demo):

```bash
rm -f data/output/ts_kb_GOOGL_demo.sqlite3
rm -f data/output/googl_eval.csv
find data/input -mindepth 1 -delete
```

Run the ingestion and watch the terminal:

```bash
python run_googl_demo.py
```

Run evaluation:

```bash
python evaluate_events.py --ticker GOOGL --market SPY --out data/output/googl_eval.csv
```

Run UI:

```bash
streamlit run app.py
```
