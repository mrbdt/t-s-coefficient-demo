# ts_demo/app.py
import os
import sqlite3
import re
import html
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# reuse system helpers where useful
from ts_system import (
    list_docs,
    list_doc_claims,
    list_sources,
    list_unresolved_forward_looking,
    resolve_fact,
    normalise_to_text,
    sha256_file,
)

load_dotenv()

st.set_page_config(page_title="T-S Demo — Files & Facts", layout="wide")
st.title("T-S Prototype — Files, Facts & Provenance")

DB_PATH = os.getenv("TS_DB_PATH", "ts_kb.sqlite3")
st.caption(f"DB: {DB_PATH}")

# Directory where run_googl_demo downloads inputs
# Adjust this if you keep them under ts_demo/googl_demo_inputs
INPUT_DIR = Path("googl_demo_inputs")
if not INPUT_DIR.exists():
    # also try the ts_demo subdir pattern (some earlier runs used that)
    alt = Path("ts_demo") / "googl_demo_inputs"
    if alt.exists():
        INPUT_DIR = alt

# ---------- DB helpers ----------
def _get_conn():
    return sqlite3.connect(DB_PATH)

def get_doc_by_sha(sha: str):
    """Return doc row (doc_id, ticker, doc_type, source_type, timestamp, sha256, url) or None."""
    cur = _get_conn().cursor()
    cur.execute("SELECT doc_id,ticker,doc_type,source_type,timestamp,sha256,url FROM docs WHERE sha256 = ?", (sha,))
    row = cur.fetchone()
    cur.connection.close()
    if not row:
        return None
    keys = ["doc_id","ticker","doc_type","source_type","timestamp","sha256","url"]
    return dict(zip(keys, row))

def get_doc_by_id(doc_id: str):
    cur = _get_conn().cursor()
    cur.execute("SELECT doc_id,ticker,doc_type,source_type,timestamp,sha256,url FROM docs WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    cur.connection.close()
    if not row:
        return None
    keys = ["doc_id","ticker","doc_type","source_type","timestamp","sha256","url"]
    return dict(zip(keys, row))

def get_fact_overview(fact_id: str):
    conn = _get_conn()
    cur = conn.execute(
        "SELECT fact_id, canonical_text, ts_coef, p_true_latest, p_true_at_issue, issued_at, source_id, speaker_role, is_forward_looking, modality, commitment, conditionality, evidential_basis "
        "FROM facts WHERE fact_id = ?",
        (fact_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["fact_id","canonical_text","ts_coef","p_true_latest","p_true_at_issue","issued_at","source_id","speaker_role","is_forward_looking","modality","commitment","conditionality","evidential_basis"]
    return dict(zip(keys,row))

def get_fact_occurrences(fact_id: str):
    conn = _get_conn()
    cur = conn.execute(
        "SELECT d.doc_id, d.timestamp, d.doc_type, d.source_type, dc.status, dc.best_match_similarity, dc.delta_awareness, dc.pred_total "
        "FROM doc_claims dc JOIN docs d ON dc.doc_id = d.doc_id WHERE dc.fact_id = ? ORDER BY d.timestamp ASC",
        (fact_id,),
    )
    rows = cur.fetchall()
    conn.close()
    occs = []
    for r in rows:
        occs.append({
            "doc_id": r[0],
            "timestamp": r[1],
            "doc_type": r[2],
            "source_type": r[3],
            "status": r[4],
            "best_match_similarity": float(r[5] or 0.0),
            "delta_awareness": float(r[6] or 0.0),
            "pred_total": float(r[7] or 0.0),
        })
    return occs

def get_fact_resolutions(fact_id: str):
    conn = _get_conn()
    cur = conn.execute(
        "SELECT resolution_id, resolved_at, outcome, confidence, evidence, method, p_pred_at_issue FROM resolutions WHERE fact_id = ? ORDER BY resolved_at DESC",
        (fact_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"resolution_id": r[0], "resolved_at": r[1], "outcome": bool(r[2]), "confidence": float(r[3]), "evidence": r[4], "method": r[5], "p_pred_at_issue": float(r[6])} for r in rows]

# ---------- utilities ----------
def highlight_text(text: str, phrases: List[str]) -> str:
    """
    Robust highlighting that tolerates punctuation/line-break differences.

    Strategy:
      - For each phrase, extract word-tokens (\w+). If there are >=2 word tokens,
        build a pattern that matches those words in order, allowing arbitrary
        non-word characters between them (so punctuation and line breaks don't stop matches).
      - If phrase contains no word tokens, fall back to a direct literal match.
      - Sort phrases longest-first to avoid partial/shorter matches shadowing longer ones.
      - Replace matches in the *raw* text with placeholders, capture the exact matched substring
        (so we preserve original casing and punctuation), then escape the whole document
        and substitute placeholders with <mark>HTML of the escaped matched text.
    """
    if not text:
        return ""

    # If no phrases, just escape and return with line breaks converted
    if not phrases:
        return html.escape(text).replace("\n", "<br/>")

    # Prepare unique phrases, sorted by length (long->short)
    uniq_phrases = [p for p in sorted({(p or "").strip() for p in phrases if p and isinstance(p, str)}, key=len, reverse=True)]
    if not uniq_phrases:
        return html.escape(text).replace("\n", "<br/>")

    raw = text  # operate on original raw text
    placeholder_map = {}
    idx = [0]  # mutable counter used in nested repl

    for phrase in uniq_phrases:
        if not phrase:
            continue

        # Extract word tokens; prefer multi-word patterns
        words = re.findall(r"\w+", phrase, flags=re.UNICODE)
        pat = None
        try:
            if len(words) >= 2:
                # pattern: word1 \W+ word2 \W+ word3 ... with word boundaries
                escaped_words = [re.escape(w) for w in words]
                pattern = r"\b" + r"\W+".join(escaped_words) + r"\b"
                pat = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
            elif len(words) == 1:
                # single word: only match if the word is reasonably long (avoid ultra-common short words)
                if len(words[0]) >= 3:
                    pattern = r"\b" + re.escape(words[0]) + r"\b"
                    pat = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
                else:
                    # skip short single-word phrases as they will produce many false positives
                    pat = None
            else:
                # no word tokens, fallback to literal match of phrase
                pat = re.compile(re.escape(phrase), flags=re.IGNORECASE | re.DOTALL)
        except re.error:
            # If building the pattern fails for any reason, fallback to escaped literal
            try:
                pat = re.compile(re.escape(phrase), flags=re.IGNORECASE | re.DOTALL)
            except Exception:
                pat = None

        if not pat:
            continue

        # Replacement function: create a placeholder and store escaped matched text
        def _repl(m, idx=idx, pm=placeholder_map):
            ph = f"__HL_{idx[0]}__"
            matched = m.group(0)
            # Store the HTML-escaped matched substring wrapped in <mark>
            pm[ph] = f'<mark style="background:#fff3bf">{html.escape(matched)}</mark>'
            idx[0] += 1
            return ph

        # Substitute in raw text
        try:
            raw = pat.sub(_repl, raw)
        except Exception:
            # if substitution fails, continue without replacing this phrase
            continue

    # Escape the document (placeholders will be escaped too)
    esc = html.escape(raw)

    # Replace escaped placeholders with the safe <mark> HTML we recorded
    for ph, marked_html in placeholder_map.items():
        esc_ph = html.escape(ph)
        esc = esc.replace(esc_ph, marked_html)

    # Convert newlines to <br/>
    esc = esc.replace("\n", "<br/>")
    return esc

# ---------- UI state ----------
if "selected_fact" not in st.session_state:
    st.session_state["selected_fact"] = None
if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None

# ---------- Layout ----------
tab_files, tab_unresolved, tab_sources = st.tabs(["Files", "Unresolved forward-looking", "Source reliability"])

with tab_files:
    st.header("Input files and facts")
    cols = st.columns([1, 2, 2])
    left, middle, right = cols

    # LEFT: list files from input directory
    with left:
        st.subheader("Local input files")
        if not INPUT_DIR.exists():
            st.warning(f"Input directory not found: {INPUT_DIR.resolve()}")
        else:
            files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file()])
            if not files:
                st.info(f"No files found in {INPUT_DIR}. Run run_googl_demo.py first.")
            else:
                for f in files:
                    try:
                        sha = sha256_file(f)
                    except Exception:
                        sha = None
                    docrow = get_doc_by_sha(sha) if sha else None
                    label = f.name
                    if docrow:
                        label += f"  ({docrow['doc_type']} | {docrow['timestamp']})"
                    # clickable select file
                    if st.button(label, key=f"file_{f.name}"):
                        st.session_state["selected_doc"] = str(f.resolve())
                        # clear fact selection
                        st.session_state["selected_fact"] = None

                    # show small metadata
                    if docrow:
                        st.caption(f"doc_id: {docrow['doc_id']}  |  ticker: {docrow['ticker']}")

    # MIDDLE: show selected document (left) and EXTRACTED FACTS (right) in separate columns
    with middle:
        st.subheader("Document viewer & facts")
        sel_doc_path = st.session_state.get("selected_doc")
        if not sel_doc_path:
            st.info("Select a file from the left to view its content and extracted facts.")
        else:
            fpath = Path(sel_doc_path)
            st.markdown(f"**File:** `{fpath.name}`")
            # compute sha and match doc
            try:
                sha = sha256_file(fpath)
                docrow = get_doc_by_sha(sha)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                docrow = None

            # show doc metadata / warning
            if docrow:
                st.caption(f"doc_id: {docrow['doc_id']}  |  ticker: {docrow['ticker']}")
            else:
                st.caption("This file hasn't been ingested (no matching sha in DB).")

            # Get claims for this document (if any)
            claims = list_doc_claims(docrow["doc_id"]) if docrow else []

            # Read and normalise the document text
            try:
                raw_text = normalise_to_text(fpath)
            except Exception as e:
                raw_text = f"Error reading/normalising file: {e}"

            # Build list of quote snippets to highlight
            phrases = [c.get("quote","") for c in claims if c.get("quote")]

            # Layout: two columns side-by-side: left = text, right = facts list + details
            doc_col, facts_col = st.columns([2.5, 1.5])

            # LEFT: Document text (with highlights)
            with doc_col:
                st.markdown("### Document text (highlighted facts shown in yellow)", unsafe_allow_html=True)
                highlighted_html = highlight_text(raw_text, phrases)
                st.markdown(highlighted_html, unsafe_allow_html=True)

            # RIGHT: Extracted facts (compact list) and selection to view details
            with facts_col:
                st.markdown("### Extracted facts from this file")
                if not claims:
                    st.write("No extracted claims found for this document.")
                else:
                    # show list of claims as clickable rows
                    for i, c in enumerate(claims):
                        fact_id = c.get("fact_id")
                        status = c.get("status","UNKNOWN")
                        snippet = (c.get("claim") or "")[:180]
                        btn_key = f"claim_btn_{fpath.name}_{i}"
                        if st.button(f"[{status}] {snippet}", key=btn_key):
                            st.session_state["selected_fact"] = fact_id
                            st.session_state["selected_doc"] = str(fpath.resolve())
                        st.caption(f"p_true={c.get('p_true_used',0.0):.2f}  delta_aw={c.get('delta_awareness',0.0):.3f}  sim={c.get('best_match_similarity',0.0):.2f}")

    # RIGHT: fact detail panel
    with right:
        st.subheader("Fact details & provenance")
        sel_fact = st.session_state.get("selected_fact")
        if not sel_fact:
            st.info("Click a fact in the document pane to see provenance and details.")
        else:
            fact = get_fact_overview(sel_fact)
            if not fact:
                st.error("Fact not found in DB.")
            else:
                st.markdown(f"### Fact `{fact['fact_id']}`")
                st.markdown(f"**Text:** {fact['canonical_text']}")
                st.write("**TS_coef:**", fact.get("ts_coef"))
                st.write("**P_true (latest):**", fact.get("p_true_latest"))
                st.write("**P_true (at issue):**", fact.get("p_true_at_issue"))
                st.write("**Issued at:**", fact.get("issued_at"))
                st.write("**Source:**", fact.get("source_id"), "/", fact.get("speaker_role"))
                st.write("**Pragmatics:**", {
                    "is_forward_looking": bool(fact.get("is_forward_looking")),
                    "modality": fact.get("modality"),
                    "commitment": fact.get("commitment"),
                    "conditionality": fact.get("conditionality"),
                    "evidential_basis": fact.get("evidential_basis")
                })

                st.markdown("---")
                st.markdown("#### Occurrences (where this fact appears)")
                occs = get_fact_occurrences(sel_fact)
                if not occs:
                    st.write("No occurrences.")
                else:
                    # Show earliest occurrence first and mark which is current selected_doc if applicable
                    for o in occs:
                        with st.expander(f"{o['timestamp']} | {o['doc_type']} | {o['doc_id']}"):
                            st.write("status:", o["status"])
                            st.write("similarity:", f"{o['best_match_similarity']:.3f}")
                            st.write("delta_awareness:", f"{o['delta_awareness']:.3f}")
                            st.write("pred_total:", f"{o['pred_total']:.6f}")
                            # show who said it (source_id) and whether earlier than selected doc
                            st.write("doc reference:", o["doc_id"])
                            # link back to the documents pane by setting session state
                            if st.button(f"Open document {o['doc_id']}", key=f"open_doc_{o['doc_id']}"):
                                # fetch doc sha to map to local file if present
                                drow = get_doc_by_id(o['doc_id'])
                                if drow and drow.get("sha256"):
                                    # search for file with same sha
                                    candidate = None
                                    for f in INPUT_DIR.iterdir():
                                        try:
                                            if sha256_file(f) == drow["sha256"]:
                                                candidate = f
                                                break
                                        except Exception:
                                            continue
                                    if candidate:
                                        st.session_state["selected_doc"] = str(candidate.resolve())
                                    else:
                                        st.warning("Local file for that doc id not found in input directory.")
                                else:
                                    st.warning("Doc metadata not found in DB.")

                st.markdown("---")
                st.markdown("#### Resolutions / adjudications")
                res = get_fact_resolutions(sel_fact)
                if not res:
                    st.write("No resolutions recorded.")
                else:
                    for r in res:
                        st.write(f"- {r['resolved_at']}: outcome={r['outcome']} confidence={r['confidence']:.2f} method={r['method']}")
                        st.write("  ", r['evidence'])


with tab_unresolved:
    st.subheader("Unresolved forward-looking claims")
    unresolved = list_unresolved_forward_looking("GOOGL", limit=50)
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

with tab_sources:
    st.subheader("Source reliability (recency-weighted)")
    sources = list_sources()
    if not sources:
        st.info("No sources yet.")
    else:
        st.dataframe(sources, use_container_width=True)
