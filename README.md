# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

How to use:

Activate the venv (if not already) (do so within the ts_demo folder):
    source .venv/bin/activate

If not yet present, create a .env file for environment parameters. see .env.example for fields that needs to be filled in.

(optional) Re-create a clean DB, clear the inputs folder, clear the evaluation output (for a clear demo):
    rm -f system_db.sqlite3

    rm -f eval_results.csv

    find ingested_inputs -mindepth 1 -delete


Run the ingestion and watch the terminal:
    python run_ingest.py

Run evaluation:
    python evaluate_events.py --ticker GOOGL --market SPY --out eval_results.csv
    
Run UI:
    streamlit run app.py

NB: my current codebase contains some elements that are AI generated and not that well labelled as such. these elements will either be removed, altered or labelled as such in a future iteration.