# t-s-coefficient-demo
mid-semester demo of the t-s-coefficient

How to use:

Activate the venv (if not already) (do so within the ts_demo folder):
    source .venv/bin/activate

Re-create a clean DB (for a clear demo):
    rm -f ts_kb_GOOGL_demo.sqlite3

Run the ingestion and watch the terminal:
    python run_googl_demo.py

Run evaluation:
    python evaluate_events.py --ticker GOOGL --market SPY --out googl_eval.csv
    
Run UI:
    streamlit run app.py
