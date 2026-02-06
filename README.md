# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

How to use:

Activate the venv (if not already) (do so within the ts_demo folder):
    source .venv/bin/activate

(optional) Re-create a clean DB, clear the inputs folder, clear the evaluation output (for a clear demo):
    rm -f ts_kb_GOOGL_demo.sqlite3

    rm -f googl_eval.csv

    find googl_demo_inputs -mindepth 1 -delete


Run the ingestion and watch the terminal:
    python run_googl_demo.py

Run evaluation:
    python evaluate_events.py --ticker GOOGL --market SPY --out googl_eval.csv
    
Run UI:
    streamlit run app.py

NB: my current codebase contains some elements that are AI generated and not that well labelled as such. these elements will either be removed, altered or labelled in a future iteration.