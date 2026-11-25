from __future__ import annotations

"""Legacy-compatible markdown chunking for corrosion RAG.

This module restores the important behavior of the previous small-model
pipeline: tables are protected as standalone retrieval units and normal text is
split only after the table blocks have been removed.  It intentionally avoids a
hard dependency on LangChain so the refactored multi-model matrix can run in a
minimal environment.
"""

import re
from typing import Any, Dict, List, Tuple


_MD_TABLE_RE = re.compile(
    r"((?:^|\n)\|[^\n]+\|\s*\n\|[\s:\\-|]+\|\s*\n(?:\|[^\n]+\|\s*\n?)*)",
    re.MULTILINE,
)
_HTML_TABLE_RE = re.compile(
    r"(\*{0,2}Table\s*\d+[^\n]*\*{0,2}\n)?(<table[\s\S]*?</table>)",
    re.IGNORECASE,
)
_NORMALIZED_ROWS_RE = re.compile(
    r"(\[Normalized HTML Table Rows\][\s\S]*?\[End Normalized HTML Table Rows\])",
    re.IGNORECASE,
)
_LABELED_TABLE_RE = re.compile(
    r"(===\s*Table\s*\d+\s*===[\s\S]*?(?=\n===|\n##|\n#|\Z))",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"__TABLE_PLACEHOLDER_(\d+)__")
_JUNK_RE = re.compile(r"^\s*((\[参见表格\s*\d+\]|\[.*?\])\s*)+$", re.DOTALL)


def _guess_table_label(text: str) -> str:
    m = re.search(r"\bTable\s*\d+[^\n]*", text or "", re.IGNORECASE)
    return m.group(0).strip(" *#") if m else ""


def _protect_tables(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace table-like blocks with placeholders and return captured tables."""
    tables: List[Dict[str, Any]] = []

    def add_table(content: str, table_format: str, label: str = "") -> str:
        idx = len(tables)
        content = (content or "").strip()
        tables.append({
            "index": idx,
            "content": content,
            "table_format": table_format,
            "caption": label or _guess_table_label(content),
        })
        return f"\n__TABLE_PLACEHOLDER_{idx}__\n"

    # Order matters: labeled and normalized blocks may contain pipe rows; protect
    # them before the generic markdown pipe-table regex.
    text = _LABELED_TABLE_RE.sub(lambda m: add_table(m.group(1), "labeled_table"), text)
    text = _HTML_TABLE_RE.sub(
        lambda m: add_table(((m.group(1) or "") + m.group(2)), "html_table", _guess_table_label(m.group(1) or "")),
        text,
    )
    text = _NORMALIZED_ROWS_RE.sub(lambda m: add_table(m.group(1), "normalized_rows"), text)
    text = _MD_TABLE_RE.sub(lambda m: add_table(m.group(1), "markdown_table"), text)
    return text, tables


def _section_blocks(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    blocks: List[Tuple[str, str]] = []
    section = ""
    buf: List[str] = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m and buf:
            blocks.append((section, "\n".join(buf)))
            buf = []
            section = m.group(2).strip()
        elif m:
            section = m.group(2).strip()
        buf.append(line)
    if buf:
        blocks.append((section, "\n".join(buf)))
    return blocks


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    cur = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            step = max(1, chunk_size - chunk_overlap)
            start = 0
            while start < len(para):
                piece = para[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(para):
                    break
                start += step
            continue
        if cur and len(cur) + len(para) + 2 > chunk_size:
            chunks.append(cur.strip())
            if chunk_overlap > 0 and chunks[-1]:
                tail = chunks[-1][-chunk_overlap:]
                cur = (tail + "\n\n" + para).strip()
            else:
                cur = para
        else:
            cur = (cur + "\n\n" + para).strip() if cur else para
    if cur:
        chunks.append(cur.strip())
    return chunks


def _table_to_chunk(paper_id: str, table: Dict[str, Any], idx: int, section: str = "") -> Dict[str, Any]:
    unit_id = f"{paper_id}#chunk_{idx}"
    return {
        "paper_id": paper_id,
        "chunk_id": unit_id,
        "unit_id": unit_id,
        "unit_type": "chunk",
        "chunk_type": "table",
        "content_type": "table",
        "table_format": table.get("table_format", "table"),
        "table_index": table.get("index", idx),
        "caption": table.get("caption", ""),
        "section": section or "table",
        "section_title": section or "table",
        "text": table.get("content", ""),
        "metadata": {
            "content_type": "table",
            "table_format": table.get("table_format", "table"),
            "table_index": table.get("index", idx),
            "caption": table.get("caption", ""),
        },
    }


def split_markdown(text: str, paper_id: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Split a markdown paper into legacy-style text/table chunks.

    Critical compatibility properties:
    - HTML, markdown, labelled, and normalized table-row blocks are standalone chunks.
    - Text chunks contain only lightweight table placeholders, never half-tables.
    - Chunk ids are ordered within a paper, so window expansion cannot cross papers.
    """
    protected, tables = _protect_tables(text)
    table_map = {t["index"]: t for t in tables}

    chunks: List[Dict[str, Any]] = []
    inserted_tables = set()

    def next_idx() -> int:
        return len(chunks)

    for section, block in _section_blocks(protected):
        parts = _PLACEHOLDER_RE.split(block)
        text_buf = ""
        for part_i, part in enumerate(parts):
            if part_i % 2 == 1:
                tbl_idx = int(part)
                # Flush text before the table so the table appears at the correct
                # neighborhood for window expansion.
                for piece in _split_long_text(text_buf, chunk_size, chunk_overlap):
                    if not _JUNK_RE.fullmatch(piece.strip()):
                        idx = next_idx(); cid = f"{paper_id}#chunk_{idx}"
                        chunks.append({
                            "paper_id": paper_id,
                            "chunk_id": cid,
                            "unit_id": cid,
                            "unit_type": "chunk",
                            "chunk_type": "text",
                            "content_type": "text",
                            "section": section,
                            "section_title": section,
                            "text": piece,
                            "metadata": {"content_type": "text", "section_title": section},
                        })
                text_buf = f"[参见表格 {tbl_idx}]"
                if tbl_idx not in inserted_tables and tbl_idx in table_map:
                    chunks.append(_table_to_chunk(paper_id, table_map[tbl_idx], next_idx(), section))
                    inserted_tables.add(tbl_idx)
            else:
                text_buf = (text_buf + "\n" + part).strip() if text_buf else part
        for piece in _split_long_text(text_buf, chunk_size, chunk_overlap):
            if not _JUNK_RE.fullmatch(piece.strip()):
                idx = next_idx(); cid = f"{paper_id}#chunk_{idx}"
                chunks.append({
                    "paper_id": paper_id,
                    "chunk_id": cid,
                    "unit_id": cid,
                    "unit_type": "chunk",
                    "chunk_type": "text",
                    "content_type": "text",
                    "section": section,
                    "section_title": section,
                    "text": piece,
                    "metadata": {"content_type": "text", "section_title": section},
                })

    # Fallback: append any table not referenced by a text block.
    for tbl_idx in sorted(table_map):
        if tbl_idx not in inserted_tables:
            chunks.append(_table_to_chunk(paper_id, table_map[tbl_idx], next_idx(), "table"))

    total = len(chunks)
    for i, c in enumerate(chunks):
        c["chunk_index_in_doc"] = i
        c["chunk_index"] = i
        c["total_chunks_in_doc"] = total
        c["total_chunks"] = total
        c.setdefault("metadata", {})["chunk_index_in_doc"] = i
        c["metadata"]["total_chunks_in_doc"] = total
    return chunks


def is_markdown_table_block(text: str) -> bool:
    s = text or ""
    return bool(_MD_TABLE_RE.search(s) or re.search(r"<table[\s\S]*?</table>", s, re.I) or "[Normalized HTML Table Rows]" in s)
