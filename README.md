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


do now: 
git rm --cached .env
git commit -m "Stop tracking .env (local only)"
git push

***

I've sorted this with a new .gitignore.

now; here's a question: how is our model currently handling tables? may of these documents contain many tables, and these are just as important as natural language bits of the file.

also: your patch A ("make extract_claims_from_chunk robust and trim chunks") is a little too much eyeballing for my liking - I don't want to trim chunks as the whole point of this program is not to miss out on any information, and its better if we find and resolve the underlying problem causing the "[LLM] structured parse failed (BadRequestError), falling back to tolerant JSON parsing...". please find this out, and offer an alternative patch accordingly, giving the full patch including any helpers you previously identified. The bottom line is that you need to make sure no information in the document is lost.

***

you should have access to the repo now.

also, in .env.example, we have the following line:

# For SEC downloads, set a descriptive UA string
SEC_USER_AGENT="Name (your.email@example.com)"

what is this? I've not filled it in yet and yet the demo is working, so I'm wondering if it is really necessary...