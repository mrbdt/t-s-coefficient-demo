# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

## Repository layout

- `ts_demo/core/` → shared core modules extracted from the old monolithic `ts_system.py`:
  - `config.py` (environment + constants)
  - `models.py` (pydantic claim schemas)
  - `text_processing.py` (document normalization/chunking)
  - `database.py`, `scoring.py`, `extraction.py`, `reliability.py`, `query.py`, `ingest_api.py` (split runtime logic by responsibility)
- `ts_demo/pipeline/` → ingestion pipeline stages + orchestrator.
- `ts_demo/data/input/user_input/` → user-provided configuration files (`docs.json`, `eval_config.json`).
- `ts_demo/data/input/ingested/` → downloaded/raw source documents assessed by the pipeline.
- `ts_demo/data/output/` → generated artifacts (KB sqlite + evaluation CSVs).
- `ts_demo/ts_system.py` → compatibility API used by UI/scripts, now delegating to modularized components.

## How to use
Activate the venv (if not already) (do so within the ts_demo folder):

```bash
source .venv/bin/activate
```

Load environment variables so `TS_DB_PATH` and model settings are available:

```bash
set -a
source .env
set +a
```

If not yet present, create a `.env` file for environment parameters. see `.env.example` for fields that needs to be filled in.

(optional) Re-create a clean DB, clear the inputs folder, clear the evaluation output (for a clear demo):

```bash
rm -f "$TS_DB_PATH"
rm -f data/output/*-eval-*.csv data/output/googl_eval.csv
find data/input/ingested -mindepth 1 -delete
```

Run the ingestion and watch the terminal (from inside `ts_demo/`):

```bash
python run_googl_demo.py
```

Run evaluation:

```bash
python evaluate_events.py
```

Run UI:

```bash
streamlit run app.py
```
