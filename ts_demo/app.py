# ts_demo/app.py
import os
import sqlite3
import re
import html
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
st.title("T-S Prototype — Files, Facts & Provenance (Docs → Facts)")

DB_PATH = os.getenv("TS_DB_PATH", "ts_kb.sqlite3")
st.caption(f"DB: {DB_PATH}")

INPUT_DIR = Path("googl_demo_inputs")
if not INPUT_DIR.exists():
    alt = Path("ts_demo") / "googl_demo_inputs"
    if alt.exists():
        INPUT_DIR = alt

# --- DB helpers ---------------------------------------------------------
def _get_conn():
    return sqlite3.connect(DB_PATH)

def get_doc_by_sha_safe(sha: Optional[str]):
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

# --- helper: find local file for doc_id ---------------------------------
def find_local_file_for_doc_id(doc_id: str) -> Optional[str]:
    drow = get_doc_by_id(doc_id)
    if not drow:
        return None
    sha = drow.get("sha256")
    if not sha:
        return None
    for f in INPUT_DIR.iterdir():
        if not f.is_file():
            continue
        try:
            if sha256_file(f) == sha:
                return str(f.resolve())
        except Exception:
            continue
    return None

# --- robust cleaning & tolerant highlighter -----------------------------
def _clean_normalised_text_for_display(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    # Extract annotations from KaTeX/MathML
    text = re.sub(r"<annotation[^>]*>(.*?)</annotation>", lambda m: m.group(1), text, flags=re.IGNORECASE | re.DOTALL)
    # Remove KaTeX/MathML blocks and known style spans
    text = re.sub(r'<(?:span|div)[^>]*class=["\'][^"\']*(katex|MathJax|math)[^"\']*["\'][^>]*>.*?</(?:span|div)>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<math[^>]*>.*?</math>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<annotation[^>]*>.*?</annotation>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove italics and style spans that cause red/italic text
    text = re.sub(r'</?(?:i|em|strong|b)[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<span[^>]*style=[^>]*>.*?</span>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    # Normalize brs -> newline
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # Fix glued tokens: digit→letter, letter→digit, lower→Upper
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Collapse whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _tokens_of(s: str) -> List[str]:
    return re.findall(r"\w+", s, flags=re.UNICODE)

def _first_matching_substring(clean_text: str, phrase: str) -> Optional[Tuple[str, int, int, str]]:
    words = _tokens_of(phrase)
    if words:
        # Try full sequence
        if len(words) >= 2:
            pattern = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
        else:
            if len(words[0]) < 3:
                pattern = None
            else:
                pattern = r"\b" + re.escape(words[0]) + r"\b"
        if pattern:
            try:
                m = re.search(pattern, clean_text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    return (m.group(0), m.start(), m.end(), pattern)
            except re.error:
                pass
        # Sliding window fallback
        L = len(words)
        max_window = min(8, L)
        for wlen in range(max_window, 2, -1):
            for start in range(0, L - wlen + 1):
                subw = words[start:start + wlen]
                pat = r"\b" + r"\W+".join(re.escape(w) for w in subw) + r"\b"
                try:
                    m = re.search(pat, clean_text, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        return (m.group(0), m.start(), m.end(), pat)
                except re.error:
                    continue
    # fallback literal match
    try:
        idx = clean_text.lower().find(phrase.lower())
        if idx >= 0:
            return (clean_text[idx:idx+len(phrase)], idx, idx+len(phrase), "literal")
    except Exception:
        pass
    return None

def highlight_text_with_debug(text: str, phrases_with_ids_and_docids: List[Tuple[str,str,Optional[str]]]) -> Tuple[str, List[Dict]]:
    clean_text = _clean_normalised_text_for_display(text)
    if not phrases_with_ids_and_docids:
        return (html.escape(clean_text).replace("\n", "<br/>"), [])

    uniq: List[Tuple[str,str,Optional[str]]] = []
    seen = set()
    for item in phrases_with_ids_and_docids:
        # each item: (phrase, fact_id, doc_id)
        if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        phrase = item[0]
        fid = item[1]
        docid = item[2] if len(item) > 2 else None
        key = (phrase or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append((key, fid, docid))
    uniq.sort(key=lambda p: len(p[0]), reverse=True)

    raw = clean_text
    placeholder_map: Dict[str,str] = {}
    counter = [0]
    debug_info = []

    for phrase, fid, docid in uniq:
        info = {"phrase": phrase, "fact_id": fid, "doc_id": docid, "matched": False, "matched_substring": None, "pattern": None, "excerpt": None}
        res = _first_matching_substring(raw, phrase)
        if res:
            matched, sidx, eidx, pattern = res
            ph = f"__HL_{counter[0]}__"
            fid_enc = urllib.parse.quote_plus(str(fid))
            doc_enc = urllib.parse.quote_plus(str(docid)) if docid else ""
            anchor = f'<a href="?fact_id={fid_enc}&doc_id={doc_enc}" target="_self" rel="noopener noreferrer"><mark style="background:#fff59d;color:#000;font-weight:normal">{html.escape(matched)}</mark></a>'
            # Replace only the first occurrence at sidx:eidx (careful with indices after replacements)
            try:
                raw = raw[:sidx] + ph + raw[eidx:]
            except Exception:
                raw = raw.replace(matched, ph, 1)
            placeholder_map[ph] = anchor
            counter[0] += 1
            info.update({"matched": True, "matched_substring": matched, "pattern": pattern, "excerpt": raw[max(0,sidx-80):min(len(raw), eidx+80)]})
        else:
            info.update({"matched": False})
        debug_info.append(info)

    esc = html.escape(raw)
    for ph, anchor in placeholder_map.items():
        esc_ph = html.escape(ph)
        esc = esc.replace(esc_ph, anchor)
    esc = esc.replace("\n", "<br/>")
    wrapper = "<div style='color:#000 !important; font-style:normal !important; font-weight:normal !important;'>" + esc + "</div>"
    return wrapper, debug_info

# --- restore selection from query params ---------------------------------
qparams = st.experimental_get_query_params()
initial_fact = qparams.get("fact_id", [None])[0]
initial_doc = qparams.get("doc_id", [None])[0]
if initial_fact:
    st.session_state["selected_fact"] = initial_fact
if initial_doc:
    # find local file for doc_id and set selected_doc if available
    lf = find_local_file_for_doc_id(initial_doc)
    if lf:
        st.session_state["selected_doc"] = lf

# --- session state -------------------------------------------------------
if "selected_fact" not in st.session_state:
    st.session_state["selected_fact"] = None
if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None

# --- layout --------------------------------------------------------------
tab_files, tab_unresolved, tab_sources = st.tabs(["Files", "Unresolved forward-looking", "Source reliability"])

with tab_files:
    st.header("Input files and facts")
    cols = st.columns([1, 2, 2])
    left, middle, right = cols

    # LEFT files
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
                        st.experimental_set_query_params()  # clear query params
                    if docrow:
                        st.caption(f"doc_id: {docrow['doc_id']}  |  ticker: {docrow['ticker']}")

    # MIDDLE viewer & facts
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

            # build phrase,fact_id,doc_id tuples (docrow may be None)
            doc_id_for_phrases = docrow["doc_id"] if docrow else None
            phrases_with_ids_and_docids = [(c.get("quote",""), c.get("fact_id"), doc_id_for_phrases) for c in claims if c.get("quote")]
            highlighted_html, debug_info = highlight_text_with_debug(raw_text, phrases_with_ids_and_docids)

            doc_col, facts_col = st.columns([2.6, 1.4])

            with doc_col:
                st.markdown("### Document text (click highlighted text to open fact)", unsafe_allow_html=True)
                if st.checkbox("Show debug: cleaned text & match report", value=False):
                    st.markdown("**CLEANED TEXT (head)**")
                    clean_text = _clean_normalised_text_for_display(raw_text)
                    st.text_area("cleaned", clean_text[:4000], height=240)
                    st.markdown("**Match report** (phrase → matched substring or failed)")
                    for info in debug_info:
                        st.write(info)
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
                                st.experimental_set_query_params()
                        else:
                            if st.button(f"[{status}] {snippet}", key=f"claim_btn_{fpath.name}_{i}"):
                                st.session_state["selected_fact"] = fact_id
                                st.session_state["selected_doc"] = str(fpath.resolve())
                                st.experimental_set_query_params(fact_id=fact_id, doc_id=docrow["doc_id"] if docrow else "")
                        st.caption(f"p_true={c.get('p_true_used',0.0):.2f}  delta_aw={c.get('delta_awareness',0.0):.3f}  sim={c.get('best_match_similarity',0.0):.2f}")

    # RIGHT details
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
