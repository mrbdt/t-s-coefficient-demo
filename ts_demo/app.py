import os
import streamlit as st
from dotenv import load_dotenv

from ts_system import (
    list_docs,
    list_doc_claims,
    list_sources,
    list_unresolved_forward_looking,
    resolve_fact,
)

load_dotenv()

st.set_page_config(page_title="T‑S Demo", layout="wide")
st.title("T‑S Prototype (Pragmatics + Provability)")

db_path = os.getenv("TS_DB_PATH", "ts_kb.sqlite3")
st.caption(f"DB: {db_path}")

ticker = st.text_input("Ticker", "GOOGL").upper()

tab1, tab2, tab3 = st.tabs(["Documents", "Unresolved forward-looking", "Source reliability"])

with tab1:
    docs = list_docs(ticker)
    if not docs:
        st.info("No documents ingested yet. Run: python run_googl_demo.py")
    else:
        st.subheader("Ingested documents (walk-forward order)")
        for d in docs:
            with st.expander(f"{d['timestamp']} | {d['doc_type']} | {d['source_type']} | doc_id={d['doc_id']}"):
                st.write("Pred horizon:", d["pred_horizon"])
                st.write("Pred near-term:", d["pred_near_term"])
                st.write("New / Known / Reconfirmed:", d["n_new"], d["n_known"], d["n_reconfirmed"])

                claims = list_doc_claims(d["doc_id"])
                st.markdown("### Top claims (by |pred_total|)")
                for c in claims[:15]:
                    st.markdown(
                        f"**[{c['status']}]** pred_total=`{c['pred_total']:+.5f}` "
                        f"delta_aw=`{c['delta_awareness']:.3f}` p_true=`{c['p_true_used']:.2f}` "
                        f"sim=`{c['best_match_similarity']:.2f}`"
                    )
                    st.write(c["claim"])
                    st.caption(c["quote"])
                    st.write(c["pred_horizon"])
                    st.write(c["rationale"])
                    st.divider()

with tab2:
    st.subheader("Unresolved forward-looking claims")
    unresolved = list_unresolved_forward_looking(ticker, limit=50)
    if not unresolved:
        st.info("No unresolved forward-looking claims found.")
    else:
        for u in unresolved[:20]:
            with st.expander(f"{u['issued_at']} | p_issue={u['p_true_at_issue']:.2f} | {u['source_id']} | fact_id={u['fact_id']}"):
                st.write(u["claim"])
                st.write({
                    "modality": u["modality"],
                    "commitment_0_1": u["commitment_0_1"],
                    "conditionality_0_1": u["conditionality_0_1"],
                    "speaker_role": u["speaker_role"],
                })

        st.markdown("### Resolve a fact (manual adjudication)")
        fact_id = st.text_input("fact_id to resolve")
        outcome = st.selectbox("Outcome", ["TRUE", "FALSE"])
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.05)
        evidence = st.text_area("Evidence / justification (short)")
        if st.button("Submit resolution"):
            if not fact_id.strip() or not evidence.strip():
                st.error("Provide fact_id and evidence text.")
            else:
                resolve_fact(
                    fact_id=fact_id.strip(),
                    outcome=(outcome == "TRUE"),
                    confidence=float(confidence),
                    evidence=evidence.strip(),
                    method="MANUAL",
                )
                st.success("Resolution recorded and reliability updated.")

with tab3:
    st.subheader("Source reliability (recency-weighted)")
    sources = list_sources()
    if not sources:
        st.info("No sources yet.")
    else:
        st.dataframe(sources, use_container_width=True)
