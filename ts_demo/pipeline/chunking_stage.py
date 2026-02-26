from __future__ import annotations

from typing import List

from ts_system import TS_MAX_CHUNKS, chunk_text, sample_chunks


def build_chunks(text: str) -> List[str]:
    """Chunk normalized text and apply max-chunk sampling policy."""
    return sample_chunks(chunk_text(text), TS_MAX_CHUNKS)
