# ts_demo/app.py
import os
import sqlite3
import re
import html
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Try to import helper functions from ts_system; be robust if some helpers are missing
try:
    from ts_system import (
        list_docs,
        list_doc_claims,
        list_sources,
        list_unresolved_forward_looking,
        resolve_fact,
        normalise_to_text,
        sha256_file,
        get_doc_by_sha,
    )
except Exception:
    import ts_system as ts_sys  # type: ignore
    list_docs = getattr(ts_sys, "list_docs")
    list_doc_claims = getattr(ts_sys, "list_doc_claims")
    list_sources = getattr(ts_sys, "list_sources")
    list_unresolved_forward_looking = getattr(ts_sys, "list_unresolved_forward_looking")
    resolve_fact = getattr(ts_sys, "resolve_fact")
    normalise_to_text = getattr(ts_sys, "normalise_to_text")
    sha256_file = getattr(ts_sys, "sha256_file")
    get_doc_by_sha = getattr(ts_sys, "get_doc_by_sha", None)

load_dotenv()

st.set_page_config(page_title="T-S Demo — Files & Facts", layout="wide")
st.title("T-S Prototype — Files, Facts & Provenance")

DB_PATH = os.getenv("TS_DB_PATH", "ts_kb.sqlite3")
st.caption(f"DB: {DB_PATH}")

# Input dir
INPUT_DIR = Path("googl_demo_inputs")
if not INPUT_DIR.exists():
    alt = Path("ts_demo") / "googl_demo_inputs"
    if alt.exists():
        INPUT_DIR = alt

# ---------- DB helpers ----------
def _get_conn():
    return sqlite3.connect(DB_PATH)

def get_doc_by_sha_safe(sha: Optional[str]):
    """Return doc row or None. Use get_doc_by_sha if available, else SQL."""
    if not sha:
        return None
    if 'get_doc_by_sha' in globals() and callable(globals()['get_doc_by_sha']):
        try:
            return globals()['get_doc_by_sha'](sha)
        except Exception:
            pass
    conn = _get_conn()
    cur = conn.execute("SELECT doc_id,ticker,doc_type,source_type,timestamp,sha256,url FROM docs WHERE sha256 = ?", (sha,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["doc_id","ticker","doc_type","source_type","timestamp","sha256","url"]
    return dict(zip(keys, row))

def get_doc_by_id(doc_id: str):
    conn = _get_conn()
    cur = conn.execute("SELECT doc_id,ticker,doc_type,source_type,timestamp,sha256,url FROM docs WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
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

# ---------- text cleaning & robust highlighter ----------
def _clean_normalised_text_for_display(raw_text: str) -> str:
    """
    Conservative cleaning of normalized text for display:
      - unescape HTML entities first (so &lt;br/&gt; => <br/>),
      - extract any annotation text from KaTeX (<annotation>...</annotation>) if present,
      - remove KaTeX/mathml blocks and stray tags,
      - convert <br/> to newline,
      - collapse whitespace and trim.
    Returns plain text.
    """
    if not raw_text:
        return ""

    # Unescape HTML entities first so '&lt;br/&gt;' becomes '<br/>'
    text = html.unescape(raw_text)

    # Extract textual annotations from MathML/KaTeX if present (these often contain the human-readable text)
    # We'll keep what's inside <annotation>...</annotation> and drop the rest of KaTeX blocks.
    text = re.sub(r"<annotation[^>]*>(.*?)</annotation>", lambda m: m.group(1), text, flags=re.IGNORECASE | re.DOTALL)

    # Remove KaTeX / mathml blocks entirely (they introduce style/HTML that we don't want)
    text = re.sub(r'<span[^>]*class=["\'][^"\']*katex[^"\']*["\'][^>]*>.*?</span>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<math[^>]*>.*?</math>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<annotation[^>]*>.*?</annotation>', ' ', text, flags=re.IGNORECASE | re.DOTALL)

    # Normalise explicit <br/> to newline
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)

    # Remove any remaining tags (replace with space so words remain separated)
    text = re.sub(r"<[^>]+>", " ", text)

    # Unescape again (safety), collapse whitespace, preserve single newlines
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse multiple blank lines
    text = re.sub(r"[ \t]+", " ", text)  # collapse spaces/tabs
    return text.strip()

def highlight_text(text: str, phrases_with_ids: List[Tuple[str, str]]) -> str:
    """
    Highlight each phrase (with associated fact_id) in the cleaned text with a literal yellow <mark>.
    Each <mark> is wrapped in an <a href='?fact_id=...'> so clicking it will reload the app with the selected fact.
    Matching is tolerant: we match sequences of word tokens allowing punctuation/newlines between words.

    phrases_with_ids: list of (phrase_text, fact_id)
    """
    if not text:
        return ""

    clean_text = _clean_normalised_text_for_display(text)
    if not phrases_with_ids:
        return html.escape(clean_text).replace("\n", "<br/>")

    # prepare unique phrase -> fact mapping, sort by phrase length descending
    uniq_pairs = []
    seen = set()
    for ph, fid in phrases_with_ids:
        if not ph or not isinstance(ph, str):
            continue
        key = ph.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq_pairs.append((key, fid))
    uniq_pairs.sort(key=lambda p: len(p[0]), reverse=True)
    if not uniq_pairs:
        return html.escape(clean_text).replace("\n", "<br/>")

    raw = clean_text
    placeholder_map = {}
    counter = [0]

    for phrase, fact_id in uniq_pairs:
        # skip very short phrases
        if len(phrase.strip()) < 3:
            continue
        # tokenise into words
        words = re.findall(r"\w+", phrase, flags=re.UNICODE)
        if words and len(words) >= 2:
            pattern = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
        elif words and len(words) == 1:
            if len(words[0]) < 3:
                continue
            pattern = r"\b" + re.escape(words[0]) + r"\b"
        else:
            # fallback literal
            pattern = re.escape(phrase)

        try:
            pat = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        except re.error:
            # fallback to literal
            try:
                pat = re.compile(re.escape(phrase), flags=re.IGNORECASE | re.DOTALL)
            except Exception:
                continue

        def _repl(m, counter=counter, pm=placeholder_map, fid=fact_id):
            ph = f"__HL_{counter[0]}__"
            matched = m.group(0)
            # anchor with fact_id in query param; URL-encode it
            fid_enc = urllib.parse.quote_plus(str(fid))
            pm[ph] = f'<a href="?fact_id={fid_enc}"><mark style="background:#fff59d;color:#000;font-weight:normal">{html.escape(matched)}</mark></a>'
            counter[0] += 1
            return ph

        try:
            raw = pat.sub(_repl, raw)
        except Exception:
            continue

    # Escape the whole text and restore placeholders with the <mark> anchors
    esc = html.escape(raw)
    for ph, marked_html in placeholder_map.items():
        esc_ph = html.escape(ph)
        esc = esc.replace(esc_ph, marked_html)

    # Convert newlines to <br/>
    esc = esc.replace("\n", "<br/>")
    return esc

# ---------- query-param driven initial selection ----------
# If the app is opened with ?fact_id=..., select that fact automatically.
qparams = st.experimental_get_query_params()
initial_fact = qparams.get("fact_id", [None])[0]
if initial_fact:
    st.session_state["selected_fact"] = initial_fact

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

    # LEFT: files
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
                    docrow = get_doc_by_sha_safe(sha) if sha else None
                    label = f.name
                    if docrow:
                        label += f"  ({docrow['doc_type']} | {docrow['timestamp']})"
                    if st.button(label, key=f"file_{f.name}"):
                        st.session_state["selected_doc"] = str(f.resolve())
                        st.session_state["selected_fact"] = None
                    if docrow:
                        st.caption(f"doc_id: {docrow['doc_id']}  |  ticker: {docrow['ticker']}")

    # MIDDLE: viewer and facts
    with middle:
        st.subheader("Document viewer & facts")
        sel_doc_path = st.session_state.get("selected_doc")
        if not sel_doc_path:
            st.info("Select a file from the left to view its content and extracted facts.")
        else:
            fpath = Path(sel_doc_path)
            st.markdown(f"**File:** `{fpath.name}`")
            try:
                sha = sha256_file(fpath)
                docrow = get_doc_by_sha_safe(sha)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                docrow = None
            if docrow:
                st.caption(f"doc_id: {docrow['doc_id']}  |  ticker: {docrow['ticker']}")
            else:
                st.caption("This file hasn't been ingested (no matching sha in DB).")

            claims = list_doc_claims(docrow["doc_id"]) if docrow else []

            try:
                raw_text = normalise_to_text(fpath)
            except Exception as e:
                raw_text = f"Error reading/normalising file: {e}"

            phrases_with_ids = [(c.get("quote", ""), c.get("fact_id")) for c in claims if c.get("quote")]
            highlighted_html = highlight_text(raw_text, phrases_with_ids)

            doc_col, facts_col = st.columns([2.6, 1.4])

            with doc_col:
                st.markdown("### Document text (click highlighted text to open fact)", unsafe_allow_html=True)
                # Render only the HTML produced by highlight_text (which injects safe <a><mark> fragments)
                st.markdown(highlighted_html, unsafe_allow_html=True)

            with facts_col:
                st.markdown("### Extracted facts from this file")
                if not claims:
                    st.write("No extracted claims found for this document.")
                else:
                    for i, c in enumerate(claims):
                        fact_id = c.get("fact_id")
                        status = c.get("status", "UNKNOWN")
                        snippet = (c.get("claim") or "")[:200]
                        is_selected = (fact_id == st.session_state.get("selected_fact"))
                        if is_selected:
                            st.markdown(f"<div style='background:#fff59d;padding:8px;border-radius:4px'>{html.escape('['+status+'] '+snippet)}</div>", unsafe_allow_html=True)
                            if st.button("Deselect", key=f"deselect_{fpath.name}_{i}"):
                                st.session_state["selected_fact"] = None
                                # clear query param
                                st.experimental_set_query_params()
                        else:
                            if st.button(f"[{status}] {snippet}", key=f"claim_btn_{fpath.name}_{i}"):
                                st.session_state["selected_fact"] = fact_id
                                st.session_state["selected_doc"] = str(fpath.resolve())
                                # set query param so highlighted anchor clicks and list clicks appear in URL
                                st.experimental_set_query_params(fact_id=fact_id)
                        st.caption(f"p_true={c.get('p_true_used',0.0):.2f}  delta_aw={c.get('delta_awareness',0.0):.3f}  sim={c.get('best_match_similarity',0.0):.2f}")

    # RIGHT: details
    with right:
        st.subheader("Fact details & provenance")
        sel_fact = st.session_state.get("selected_fact")
        if not sel_fact:
            st.info("Click a fact in the middle column or a highlighted text in the document to see provenance and details.")
        else:
            fact = get_fact_overview(sel_fact)
            if not fact:
                st.error("Fact not found in DB.")
            else:
                st.markdown(f"<div style='background:#fff59d;padding:10px;border-radius:6px'>", unsafe_allow_html=True)
                st.markdown(f"### Fact `{fact['fact_id']}`", unsafe_allow_html=True)
                st.markdown(f"<b>Text:</b> {html.escape(fact.get('canonical_text',''))}", unsafe_allow_html=True)
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
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### Occurrences (where this fact appears)")
                occs = get_fact_occurrences(sel_fact)
                if not occs:
                    st.write("No occurrences.")
                else:
                    for o in occs:
                        with st.expander(f"{o['timestamp']} | {o['doc_type']} | {o['doc_id']}"):
                            st.write("status:", o["status"])
                            st.write("similarity:", f"{o['best_match_similarity']:.3f}")
                            st.write("delta_awareness:", f"{o['delta_awareness']:.3f}")
                            st.write("pred_total:", f"{o['pred_total']:.6f}")
                            st.write("doc reference:", o['doc_id'])
                            if st.button(f"Open document {o['doc_id']}", key=f"open_doc_{o['doc_id']}"):
                                drow = get_doc_by_id(o['doc_id'])
                                if drow and drow.get("sha256"):
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
