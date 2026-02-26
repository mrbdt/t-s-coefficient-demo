from __future__ import annotations

from pathlib import Path

from ts_system import normalise_to_text


def normalize_document(path: Path) -> str:
    """Normalize a source document into plain analysis text."""
    return normalise_to_text(path)
