from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List

import pdfplumber
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def table_to_markdown_from_bs4(table_tag) -> str:
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(" ", strip=True).replace("\n", " ").strip() for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r += [""] * (max_cols - len(r))
    header = rows[0]
    body = rows[1:] if all((h == "" or any(ch.isalpha() for ch in h)) for h in header) else rows
    if body is rows:
        header = [f"col{i+1}" for i in range(max_cols)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * max_cols) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def table_to_markdown_from_list(table: List[List[str]]) -> str:
    if not table:
        return ""
    max_cols = max(len(r) for r in table)
    rows = [[(x or "").strip().replace("\n", " ") for x in r] + [""] * (max_cols - len(r)) for r in table]
    header = rows[0] if any(cell.isalpha() for cell in " ".join(rows[0])) else [f"col{i+1}" for i in range(max_cols)]
    body = rows[1:] if header == rows[0] else rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * max_cols) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def normalise_to_text(path: Path | str) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        parts: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                for t in (page.extract_tables() or []):
                    md = table_to_markdown_from_list(t)
                    if md:
                        parts.append(f"[START_TABLE page={page_no}]\n{md}\n[END_TABLE]\n")
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
        return "\n\n".join(parts)
    if ext in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for table in list(soup.find_all("table")):
            md = table_to_markdown_from_bs4(table)
            if md:
                p = soup.new_tag("p")
                p.string = f"\n\n[START_TABLE]\n{md}\n[END_TABLE]\n\n"
                table.replace_with(p)
        text = soup.get_text("\n")
        text = re.sub(r"[\s\n\r]+", " ", text).strip()
        return text.replace("[START_TABLE]", "\n[START_TABLE]").replace("[END_TABLE]", "[END_TABLE]\n")
    if ext == ".docx":
        from docx import Document as DocxDocument
        return "\n\n".join([p.text.strip() for p in DocxDocument(str(path)).paragraphs if p.text.strip()])
    return path.read_text(errors="ignore")


def chunk_text(text: str, max_chars: int = 7000, overlap: int = 700) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def sample_chunks(chunks: List[str], k: int) -> List[str]:
    if len(chunks) <= k:
        return chunks
    a = chunks[: max(1, k // 3)]
    mid_start = max(0, len(chunks) // 2 - max(1, k // 6))
    b = chunks[mid_start: mid_start + max(1, k // 3)]
    c = chunks[-max(1, k // 3):]
    return (a + b + c)[:k]
