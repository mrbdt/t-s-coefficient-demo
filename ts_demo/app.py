"""
Streamlit UI for exploring the T-S prototype knowledge base.

The app is intentionally split into three analyst-facing workflows:
1) "Files": inspect document text, see extracted claims, and open detailed provenance.
2) "Unresolved forward-looking": find predictions that still need adjudication.
3) "Source reliability": inspect reliability scores learned from resolutions.

Most helper functions in this file are about one of two concerns:
- Fetching well-shaped records from SQLite for display.
- Cleaning / highlighting document text so extracted claims are easy to click and audit.
"""

# ts_demo/app.py
import os
import sqlite3
import re
import html
import urllib.parse
import textwrap
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
    """Create a short-lived SQLite connection.

    We intentionally open/close connections per operation in this demo app to keep
    state handling simple and avoid long-lived connection surprises in Streamlit.
    """
    return sqlite3.connect(DB_PATH)

def get_doc_by_sha_safe(sha: Optional[str]):
    """Lookup document metadata by SHA-256, with graceful fallback.

    Why this exists:
    - In some environments, `ts_system.get_doc_by_sha` may not be available.
    - We still want the app to keep working, so we fallback to a direct SQL query.
    """
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
    """Fetch one row from `docs` and return it as a dict for UI use."""
    conn = _get_conn()
    cur = conn.execute("SELECT doc_id,ticker,doc_type,source_type,timestamp,sha256,url FROM docs WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["doc_id","ticker","doc_type","source_type","timestamp","sha256","url"]
    return dict(zip(keys, row))

def get_fact_overview(fact_id: str):
    """Fetch the main fact record shown in the right-side detail panel."""
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
    """Return every document occurrence for a fact, ordered chronologically."""
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
    """Return adjudication history for a fact (newest first)."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT resolution_id, resolved_at, outcome, confidence, evidence, method, p_pred_at_issue FROM resolutions WHERE fact_id = ? ORDER BY resolved_at DESC",
        (fact_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"resolution_id": r[0], "resolved_at": r[1], "outcome": bool(r[2]), "confidence": float(r[3]), "evidence": r[4], "method": r[5], "p_pred_at_issue": float(r[6])} for r in rows]

def get_ts_coefs_for_fact_ids(fact_ids: List[str]) -> Dict[str, float]:
    """
    Fetch ts_coef for a list of fact_ids in one query.
    Returns dict: fact_id -> ts_coef
    """
    ids = [x for x in fact_ids if x]
    if not ids:
        return {}
    uniq = sorted(set(ids))
    placeholders = ",".join(["?"] * len(uniq))
    q = f"SELECT fact_id, ts_coef FROM facts WHERE fact_id IN ({placeholders})"
    conn = _get_conn()
    cur = conn.execute(q, tuple(uniq))
    rows = cur.fetchall()
    conn.close()
    out: Dict[str, float] = {}
    for fid, ts in rows:
        try:
            out[str(fid)] = float(ts)
        except Exception:
            out[str(fid)] = 0.0
    for fid in uniq:
        out.setdefault(fid, 0.0)
    return out

# --- helper: find local file for doc_id ---------------------------------
def find_local_file_for_doc_id(doc_id: str) -> Optional[str]:
    """Map a DB doc_id back to a local file path by matching SHA-256 hashes."""
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
    """Prepare normalized text for robust rendering/highlighting.

    The ingestion pipeline can produce HTML-rich text (tables, inline math/XBRL remnants,
    style spans). This cleaner strips noisy markup and normalizes whitespace so matching
    extracted quotes is more reliable and display is easier to read.
    """
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
    """Tokenize text into alphanumeric words for tolerant matching."""
    return re.findall(r"\w+", s, flags=re.UNICODE)

def _first_matching_substring(clean_text: str, phrase: str) -> Optional[Tuple[str, int, int, str]]:
    """Find a best-effort substring match of `phrase` inside `clean_text`.

    Matching strategy is deliberately tolerant:
    1) Try a full token sequence regex.
    2) Fall back to sliding token windows (helps when quote extraction is slightly off).
    3) Last resort literal case-insensitive substring.
    """
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

# --- TS coefficient colour scale (NOT blue/gray/red) ---------------------
# Diverging palette:
#   negative -> violet tint
#   zero     -> very light gray
#   positive -> amber/gold tint
TS_NEG_RGB = (196, 181, 253)  # violet-300 (#c4b5fd)
TS_NEU_RGB = (229, 231, 235)  # gray-100  (#f3f4f6)
TS_POS_RGB = (253, 230, 138)  # amber-200 (#fde68a)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _interp_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = _clamp(float(t), 0.0, 1.0)
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(_clamp(rgb[0], 0, 255), _clamp(rgb[1], 0, 255), _clamp(rgb[2], 0, 255))

def ts_coef_to_bg_hex(ts_coef: Optional[float], max_abs: float) -> str:
    """
    Map ts_coef to a background colour using a symmetric document-local scale.
      x = ts_coef / max_abs, clamped to [-1, 1]
    Negative interpolates TS_NEG_RGB -> TS_NEU_RGB
    Positive interpolates TS_NEU_RGB -> TS_POS_RGB
    """
    if ts_coef is None:
        return _rgb_to_hex(TS_NEU_RGB)
    try:
        ts = float(ts_coef)
    except Exception:
        return _rgb_to_hex(TS_NEU_RGB)

    max_abs = float(max_abs or 0.0)
    if max_abs <= 1e-12:
        return _rgb_to_hex(TS_NEU_RGB)

    x = _clamp(ts / max_abs, -1.0, 1.0)
    if x < 0.0:
        t = x + 1.0
        rgb = _interp_rgb(TS_NEG_RGB, TS_NEU_RGB, t)
    else:
        t = x
        rgb = _interp_rgb(TS_NEU_RGB, TS_POS_RGB, t)

    return _rgb_to_hex(rgb)

def ts_legend_html(max_abs: float) -> str:
    max_abs = float(max_abs or 0.0)
    neg_hex = _rgb_to_hex(TS_NEG_RGB)
    neu_hex = _rgb_to_hex(TS_NEU_RGB)
    pos_hex = _rgb_to_hex(TS_POS_RGB)

    left_label = f"{-max_abs:.6g}"
    right_label = f"{max_abs:.6g}"

    # IMPORTANT: dedent to avoid Markdown interpreting it as a code block
    return textwrap.dedent(f"""
    <div style="margin: 0.25rem 0 0.75rem 0;">
      <div style="display:flex; align-items:baseline; justify-content:space-between; gap: 0.75rem;">
        <div style="font-size: 0.86rem; color: #444;">
          <b>T-S coefficient highlight scale</b>
          <span style="color:#666;">(scaled to ±max |ts_coef| in this document)</span>
        </div>
        <div style="font-size: 0.80rem; color:#555;">
          max |ts_coef| = <b>{max_abs:.6g}</b>
        </div>
      </div>

      <div style="margin-top: 0.35rem;">
        <div style="
            height: 14px;
            border-radius: 7px;
            border: 1px solid rgba(0,0,0,0.14);
            background: linear-gradient(90deg, {neg_hex} 0%, {neu_hex} 50%, {pos_hex} 100%);
        "></div>

        <div style="display:flex; justify-content:space-between; font-size: 0.74rem; color:#666; margin-top: 0.18rem;">
          <span>{left_label}</span>
          <span>0</span>
          <span>{right_label}</span>
        </div>

        <div style="font-size: 0.74rem; color:#666; margin-top: 0.18rem;">
          <span style="display:inline-block; width: 10px; height: 10px; background:{neg_hex}; border:1px solid rgba(0,0,0,0.10); border-radius:2px; margin-right:6px;"></span>
          more negative
          <span style="display:inline-block; width: 10px; height: 10px; background:{pos_hex}; border:1px solid rgba(0,0,0,0.10); border-radius:2px; margin:0 6px 0 14px;"></span>
          more positive
        </div>
      </div>
    </div>
    """).strip()

def highlight_text_with_debug(
    text: str,
    phrases_with_ids_and_docids_and_ts: List[Tuple[str, str, Optional[str], Optional[float]]],
    ts_max_abs: float,
) -> Tuple[str, List[Dict]]:
    """Highlight extracted claim quotes in document text and collect match diagnostics.

    Returns:
    - HTML string with clickable highlights that deep-link to fact details.
    - Per-phrase debug metadata (matched substring, pattern used, etc.).
    """
    clean_text = _clean_normalised_text_for_display(text)
    if not phrases_with_ids_and_docids_and_ts:
        return (html.escape(clean_text).replace("\n", "<br/>"), [])

    uniq: List[Tuple[str, str, Optional[str], Optional[float]]] = []
    seen = set()
    for item in phrases_with_ids_and_docids_and_ts:
        if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        phrase = item[0]
        fid = item[1]
        docid = item[2] if len(item) > 2 else None
        ts_coef = item[3] if len(item) > 3 else None

        key = (phrase or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append((key, fid, docid, ts_coef))

    uniq.sort(key=lambda p: len(p[0]), reverse=True)

    raw = clean_text
    placeholder_map: Dict[str, str] = {}
    counter = [0]
    debug_info = []

    for phrase, fid, docid, ts_coef in uniq:
        bg_hex = ts_coef_to_bg_hex(ts_coef, ts_max_abs)
        try:
            title = f"ts_coef={float(ts_coef):.6g}"
        except Exception:
            title = "ts_coef=0"

        info = {
            "phrase": phrase,
            "fact_id": fid,
            "doc_id": docid,
            "ts_coef": float(ts_coef or 0.0),
            "bg_hex": bg_hex,
            "matched": False,
            "matched_substring": None,
            "pattern": None,
            "excerpt": None,
        }

        res = _first_matching_substring(raw, phrase)
        if res:
            matched, sidx, eidx, pattern = res
            ph = f"__HL_{counter[0]}__"
            fid_enc = urllib.parse.quote_plus(str(fid))
            doc_enc = urllib.parse.quote_plus(str(docid)) if docid else ""

            anchor = (
                f'<a href="?fact_id={fid_enc}&doc_id={doc_enc}" '
                f'target="_self" rel="noopener noreferrer" '
                f'style="text-decoration:none;">'
                f'<mark title="{html.escape(title)}" '
                f'style="background:{bg_hex}; color:#000; font-weight:normal; padding: 0 2px; border-radius: 2px; '
                f'box-shadow: inset 0 0 0 1px rgba(0,0,0,0.10);">'
                f'{html.escape(matched)}'
                f'</mark></a>'
            )

            try:
                raw = raw[:sidx] + ph + raw[eidx:]
            except Exception:
                raw = raw.replace(matched, ph, 1)

            placeholder_map[ph] = anchor
            counter[0] += 1
            info.update({"matched": True, "matched_substring": matched, "pattern": pattern, "excerpt": raw[max(0, sidx-80):min(len(raw), eidx+80)]})
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
# This allows deep links such as ?fact_id=...&doc_id=..., so opening a URL can
# directly focus the app on a specific fact/document pair.
qparams = st.experimental_get_query_params()
initial_fact = qparams.get("fact_id", [None])[0]
initial_doc = qparams.get("doc_id", [None])[0]
if initial_fact:
    st.session_state["selected_fact"] = initial_fact
if initial_doc:
    lf = find_local_file_for_doc_id(initial_doc)
    if lf:
        st.session_state["selected_doc"] = lf

# --- session state -------------------------------------------------------
# Streamlit reruns top-to-bottom on interactions; session_state preserves the
# currently selected document/fact across reruns.
if "selected_fact" not in st.session_state:
    st.session_state["selected_fact"] = None
if "selected_doc" not in st.session_state:
    st.session_state["selected_doc"] = None

# --- layout --------------------------------------------------------------
tab_files, tab_unresolved, tab_sources = st.tabs(["Files", "Unresolved forward-looking", "Source reliability"])

with tab_files:
    st.header("Input files and facts")
    # Three-column analyst workflow:
    # - left: pick a local source file
    # - middle: read text + see extracted claims
    # - right: inspect full provenance for selected fact
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
                        st.experimental_set_query_params()
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

            # `claims` are doc-specific links to canonical facts plus per-doc scoring metadata.
            claims = list_doc_claims(docrow["doc_id"]) if docrow else []

            try:
                raw_text = normalise_to_text(fpath)
            except Exception as e:
                raw_text = f"Error reading/normalising file: {e}"

            # Pull ts_coef in one batch query so the colour scale is normalized
            # against the currently opened document's own claim range.
            fact_ids_for_doc = [c.get("fact_id") for c in claims if c.get("fact_id")]
            ts_map = get_ts_coefs_for_fact_ids([fid for fid in fact_ids_for_doc if fid])

            ts_max_abs = 0.0
            if ts_map:
                try:
                    ts_max_abs = max(abs(float(v or 0.0)) for v in ts_map.values())
                except Exception:
                    ts_max_abs = 0.0

            doc_id_for_phrases = docrow["doc_id"] if docrow else None
            phrases_with_ids_and_docids_and_ts = []
            for c in claims:
                quote = c.get("quote", "")
                fid = c.get("fact_id")
                if quote and fid:
                    phrases_with_ids_and_docids_and_ts.append((quote, fid, doc_id_for_phrases, ts_map.get(fid, 0.0)))

            highlighted_html, debug_info = highlight_text_with_debug(
                raw_text,
                phrases_with_ids_and_docids_and_ts,
                ts_max_abs=ts_max_abs,
            )

            doc_col, facts_col = st.columns([2.6, 1.4])

            with doc_col:
                st.markdown("### Document text (click highlighted text to open fact)", unsafe_allow_html=True)

                # Legend / key on top of this section
                if claims:
                    st.markdown(ts_legend_html(ts_max_abs), unsafe_allow_html=True)

                # Optional debug view is useful when a quote wasn't highlighted as expected.
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
                    # Each claim row gets a compact colour swatch tied to ts_coef so an analyst
                    # can visually scan positive/negative importance before opening details.
                    for i, c in enumerate(claims):
                        fact_id = c.get("fact_id")
                        status = c.get("status", "UNKNOWN")
                        snippet = (c.get("claim") or "")[:200]
                        is_selected = (fact_id == st.session_state.get("selected_fact"))

                        ts_val = ts_map.get(fact_id, 0.0) if fact_id else 0.0
                        swatch_hex = ts_coef_to_bg_hex(ts_val, ts_max_abs)

                        # Row layout: [tiny colour swatch] [select/deselect button]
                        sw_col, btn_col = st.columns([0.12, 0.88], gap="small")
                        with sw_col:
                            st.markdown(
                                f"<div style='margin-top: 0.55rem;'>"
                                f"<span title='ts_coef={ts_val:.6g}' "
                                f"style='display:inline-block;width:12px;height:12px;"
                                f"background:{swatch_hex};border:1px solid rgba(0,0,0,0.12);"
                                f"border-radius:3px;'></span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with btn_col:
                            if is_selected:
                                st.markdown(
                                    f"<div style='background:#fff59d;padding:8px;border-radius:4px'>"
                                    f"{html.escape('['+status+'] '+snippet)}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                if st.button("Deselect", key=f"deselect_{fpath.name}_{i}"):
                                    st.session_state["selected_fact"] = None
                                    st.experimental_set_query_params()
                            else:
                                # Plain string label (NO HTML)
                                if st.button(f"[{status}] {snippet}", key=f"claim_btn_{fpath.name}_{i}"):
                                    st.session_state["selected_fact"] = fact_id
                                    st.session_state["selected_doc"] = str(fpath.resolve())
                                    st.experimental_set_query_params(fact_id=fact_id, doc_id=docrow["doc_id"] if docrow else "")

                        st.caption(
                            f"ts_coef={ts_val:.6g}  "
                            f"p_true={c.get('p_true_used',0.0):.2f}  "
                            f"delta_aw={c.get('delta_awareness',0.0):.3f}  "
                            f"sim={c.get('best_match_similarity',0.0):.2f}"
                        )

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
