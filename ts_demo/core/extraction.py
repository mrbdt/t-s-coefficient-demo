from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
from openai import OpenAI

from core.config import HORIZON_BUCKETS, OPENAI_EMBED_MODEL, OPENAI_MODEL
from core.models import ExtractedClaim, ExtractedClaims

def embed(client: OpenAI, text: str) -> np.ndarray:
    """Create a unit-normalized embedding vector for similarity comparisons."""
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text, encoding_format="float")
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    n = float(np.linalg.norm(vec) + 1e-12)
    return vec / n

STRUCTURED_PARSE_ENABLED = True

def _supports_structured_parse(client: OpenAI) -> bool:
    """
    Quick probe to check if responses.parse(...) with our Pydantic schema is supported.
    """
    try:
        resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "Return an empty claims list."},
                {"role": "user", "content": "TEXT: test"},
            ],
            text_format=ExtractedClaims,
        )
        _ = resp.output_parsed
        return True
    except Exception as e:
        print(f"[LLM] Structured parse probe failed ({type(e).__name__}): {e}")
        return False

def _safe_json_load(raw: str) -> dict:
    """Best-effort JSON parser for imperfect model outputs.

    We first try strict JSON, then attempt to recover the first JSON object-shaped
    substring. If both fail, we return an empty claims payload rather than crashing.
    """
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"claims": []}
        return {"claims": []}

def _normalize_evidential(v: str) -> str:
    if not v or not isinstance(v, str):
        return "UNSPECIFIED"
    vv = v.strip().upper()
    if "REPORT" in vv or "NUMBER" in vv or "REPORTED" in vv:
        return "REPORTED_NUMBER"
    if "OPERATION" in vv or "OBSERV" in vv:
        return "OPERATIONAL_OBSERVATION"
    if "INTERNAL" in vv or "METRIC" in vv:
        return "INTERNAL_METRIC"
    return "UNSPECIFIED"

def _normalize_modality(v: Optional[str]) -> str:
    if not v:
        return "ASSERTION"
    vv = v.strip().upper()
    for opt in ["FORECAST","INTENTION","CONDITIONAL","RISK","OPINION","ASSERTION"]:
        if opt in vv:
            return opt
    return "ASSERTION"

def _normalize_speaker_role(v: Optional[str]) -> str:
    if not v:
        return "OTHER"
    vv = v.strip().upper()
    if "COMPANY" in vv or "SEC" in vv or "FORM" in vv:
        return "COMPANY_OFFICIAL"
    if "MANAGEMENT" in vv or "CEO" in vv or "CFO" in vv:
        return "MANAGEMENT"
    if "ANALYST" in vv or "SELL" in vv or "BUY" in vv:
        return "ANALYST"
    return "OTHER"

def parse_raw_claims(data: dict, chunk: str) -> List[ExtractedClaim]:
    """
    Defensive mapping from arbitrary JSON returned by the model into our ExtractedClaim objects.
    Uses sensible defaults for missing fields and normalises enums.
    """
    out: List[ExtractedClaim] = []
    if not isinstance(data, dict):
        return out
    raw_claims = data.get("claims") or data.get("items") or []
    if not isinstance(raw_claims, list):
        return out

    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = rc.get("claim") or rc.get("text") or (chunk[:400] + ("..." if len(chunk) > 400 else ""))
        # Basic defaults
        polarity = (rc.get("polarity") or "NEUTRAL").upper()
        if polarity not in {"BULLISH","BEARISH","MIXED","NEUTRAL"}:
            # try to infer from words
            pt = claim_text.lower()
            if any(w in pt for w in ["increase","upside","beat","outperform","grow","gain","benefit"]):
                polarity = "BULLISH"
            elif any(w in pt for w in ["decline","drop","miss","slowdown","risk","loss","downside","concern"]):
                polarity = "BEARISH"
            else:
                polarity = "NEUTRAL"

        # numeric fields with safe clamping
        def _safe_float(k, default):
            try:
                v = rc.get(k, default)
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        materiality = max(0.0, min(1.0, _safe_float("materiality_0_1", 0.15)))
        credibility = max(0.0, min(1.0, _safe_float("credibility_0_1", 0.6)))
        surprise = max(0.0, min(1.0, _safe_float("surprise_0_1", 0.25)))

        # horizon_profile default / normalise
        hp = rc.get("horizon_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        # fallback distribution
        if sum([float(v) for v in hp.values()]) <= 0:
            hp = {"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05}
        # normalise
        s = float(sum(float(hp.get(k,0.0)) for k in HORIZON_BUCKETS))
        if s <= 0:
            s = 1.0
        horizon_profile = {k: float(hp.get(k,0.0))/s for k in HORIZON_BUCKETS}

        rationale = rc.get("rationale") or rc.get("why") or ""
        quote = rc.get("quote") or rc.get("snippet") or (claim_text if len(claim_text) < 600 else claim_text[:600] + "...")

        # pragmatics defaults
        is_forward = bool(rc.get("is_forward_looking", False))
        modality = _normalize_modality(rc.get("modality"))
        commitment = max(0.0, min(1.0, _safe_float("commitment_0_1", 0.5)))
        conditionality = max(0.0, min(1.0, _safe_float("conditionality_0_1", 0.0)))
        evidential = _normalize_evidential(rc.get("evidential_basis") or rc.get("evidence") or "")
        speaker_role = _normalize_speaker_role(rc.get("speaker_role"))

        # Build ExtractedClaim object with fallback values where possible
        try:
            ec = ExtractedClaim(
                claim=str(claim_text),
                polarity=polarity,  # pydantic will coerce or validate
                materiality_0_1=float(materiality),
                credibility_0_1=float(credibility),
                surprise_0_1=float(surprise),
                horizon_profile=horizon_profile,
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=bool(is_forward),
                modality=modality,
                commitment_0_1=float(commitment),
                conditionality_0_1=float(conditionality),
                evidential_basis=evidential,
                speaker_role=speaker_role,
            )
            out.append(ec)
        except Exception as e:
            # Last-resort fallback: use dict and leave pydantic to convert later if needed
            out.append(ExtractedClaim(
                claim=str(claim_text),
                polarity="NEUTRAL",
                materiality_0_1=0.15,
                credibility_0_1=0.6,
                surprise_0_1=0.25,
                horizon_profile={"1D": 0.2, "1W": 0.25, "1M": 0.25, "3M": 0.15, "1Y":0.10, "3Y":0.05},
                rationale=str(rationale),
                quote=str(quote),

                is_forward_looking=False,
                modality="ASSERTION",
                commitment_0_1=0.5,
                conditionality_0_1=0.0,
                evidential_basis="UNSPECIFIED",
                speaker_role="OTHER",
            ))

    return out

def extract_claims_from_chunk(
    client: OpenAI,
    chunk: str,
    ticker: str,
    doc_type: str,
    source_type: str,
    max_split_depth: int = 2
) -> List[ExtractedClaim]:
    """Extract valuation-relevant atomic claims from a text chunk.

    Strategy:
    1) Prefer structured parsing into `ExtractedClaims` for schema safety.
    2) If unavailable/failing, fallback to JSON mode and coerce via `parse_raw_claims`.
    3) If chunk is still too noisy/long, caller may recursively split and retry.

    This function intentionally never mutates KB state; it only returns parsed claims.
    """
    instructions = (
        "You are an expert equity research analyst.\n"
        "Extract ONLY price-relevant, atomic claims.\n"
        "Ignore boilerplate and generic risk language unless it is clearly new/changed.\n"
        "Each claim must be standalone and specific; include key numbers AND period when present.\n"
        "\n"
        "For EACH claim, output the following fields:\n"
        "- claim (string)\n"
        "- polarity (one of: BULLISH, BEARISH, MIXED, NEUTRAL)\n"
        "- materiality_0_1 (0..1)\n"
        "- credibility_0_1 (0..1)\n"
        "- surprise_0_1 (0..1)\n"
        "- horizon_profile (object with keys: 1D, 1W, 1M, 3M, 1Y, 3Y; values sum to ~1)\n"
        "- rationale (string)\n"
        "- quote (string; exact supporting snippet)\n"
        "- is_forward_looking (boolean)\n"
        "- modality (one of: ASSERTION, FORECAST, INTENTION, CONDITIONAL, RISK, OPINION)\n"
        "- commitment_0_1 (0..1; \"will\" > \"expect\" > \"could\")\n"
        "- conditionality_0_1 (0..1; \"subject to\"/\"if\" increases)\n"
        "- evidential_basis (one of: REPORTED_NUMBER, OPERATIONAL_OBSERVATION, INTERNAL_METRIC, UNSPECIFIED)\n"
        "- speaker_role (one of: COMPANY_OFFICIAL, MANAGEMENT, ANALYST, OTHER; COMPANY_OFFICIAL for SEC)\n"
        "\n"
        "Return between 0 and 8 claims. If nothing material, return an empty list."
    )

    user = (
        f"TICKER: {ticker}\nDOC_TYPE: {doc_type}\nSOURCE_TYPE: {source_type}\n\n"
        f"TEXT:\n{chunk}"
    )

    global STRUCTURED_PARSE_ENABLED
    if STRUCTURED_PARSE_ENABLED:
        try:
            print("[LLM] structured parse request...")
            resp = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user},
                ],
                text_format=ExtractedClaims,
            )
            parsed: ExtractedClaims = resp.output_parsed
            return parsed.claims
        except Exception as e:
            print(f"[LLM] structured parse failed ({type(e).__name__}): {e}")
            STRUCTURED_PARSE_ENABLED = False
            print("[LLM] disabling structured parse for remainder of run; will use JSON fallback.")

    # Fallback to JSON-only responses.create
    try:
        print("[LLM] calling responses.create with JSON-only instruction...")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": instructions + "\nReturn ONLY valid JSON matching: {\"claims\": [ ... ]}. Do not include any extra keys outside 'claims'."},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_object"}},
        )
        raw = (resp.output_text or "").strip()
        data = _safe_json_load(raw)
        parsed_claims = parse_raw_claims(data, chunk)
        return parsed_claims

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[LLM] responses.create failed: {err_msg}")

        msg = str(e).lower()
        token_error_indicators = ["context length", "max_tokens", "input length", "exceeds", "token limit", "context_length"]
        if any(k in msg for k in token_error_indicators) and max_split_depth > 0:
            print("[LLM] Detected token/context error — splitting chunk and retrying (preserving content).")
            mid = len(chunk) // 2
            left = chunk[:mid]
            right = chunk[mid:]
            left_claims = extract_claims_from_chunk(client, left, ticker, doc_type, source_type, max_split_depth - 1)
            right_claims = extract_claims_from_chunk(client, right, ticker, doc_type, source_type, max_split_depth - 1)
            return left_claims + right_claims

        return []
