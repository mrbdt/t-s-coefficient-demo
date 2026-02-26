# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

## Pipeline structure
The ingestion framework is now split into stage-specific files (all in `ts_demo/`) plus one orchestrator:

1. `ingestion_stage.py` → document registration / metadata persistence.
2. `normalization_stage.py` → raw-file normalization to plain text.
3. `chunking_stage.py` → chunk creation and sampling.
4. `claim_extraction_stage.py` → claim extraction + within-doc deduplication.
5. `claim_scoring_stage.py` → novelty, probability, and t-s coefficient scoring logic.
6. `fact_storage_stage.py` → fact upsert/update into the KB.
7. `awareness_exposure_stage.py` → exposure writes + awareness delta calculations.
8. `document_scoring_stage.py` → document-level aggregate score persistence.
9. `pipeline_main.py` → main orchestrator that calls each stage in order.

`ts_system.py` remains the compatibility API surface used by the UI and scripts, and now delegates ingestion to `pipeline_main.py`.

## How to use
Activate the venv (if not already) (do so within the ts_demo folder):

```bash
source .venv/bin/activate
```

If not yet present, create a `.env` file for environment parameters. see `.env.example` for fields that needs to be filled in.

(optional) Re-create a clean DB, clear the inputs folder, clear the evaluation output (for a clear demo):

```bash
rm -f ts_kb_GOOGL_demo.sqlite3
rm -f googl_eval.csv
find googl_demo_inputs -mindepth 1 -delete
```

Run the ingestion and watch the terminal:

```bash
python run_googl_demo.py
```

Run evaluation:

```bash
python evaluate_events.py --ticker GOOGL --market SPY --out googl_eval.csv
```

Run UI:

```bash
streamlit run app.py
```

NB: my current codebase contains some elements that are AI generated and not that well labelled as such. these elements will either be removed, altered or labelled as such in a future iteration.
