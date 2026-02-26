from __future__ import annotations

import time
from typing import List

import numpy as np
from openai import OpenAI

from ts_system import ExtractedClaim, cosine, embed, extract_claims_from_chunk


def extract_and_deduplicate_claims(
    client: OpenAI,
    chunks: List[str],
    ticker: str,
    doc_type: str,
    source_type: str,
) -> List[ExtractedClaim]:
    """Extract claim candidates per chunk and deduplicate within-document."""
    extracted: List[ExtractedClaim] = []
    for i, ch in enumerate(chunks):
        print(f"[ingest] Extracting claims from chunk {i+1}/{len(chunks)} ...")
        extracted.extend(extract_claims_from_chunk(client, ch, ticker, doc_type, source_type))
        time.sleep(0.01)

    merged: List[ExtractedClaim] = []
    seen_vecs: List[np.ndarray] = []
    for claim in extracted:
        v = embed(client, claim.claim)
        if any(cosine(v, sv) > 0.92 for sv in seen_vecs):
            continue
        seen_vecs.append(v)
        merged.append(claim)
    return merged
