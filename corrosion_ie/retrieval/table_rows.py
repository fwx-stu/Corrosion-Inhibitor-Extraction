from __future__ import annotations

"""Table-row extraction used by table-first and evidence-aware strategies.

The refactored v2 parser only accepted markdown rows starting with ``|``.  This
version supports all table formats produced by the current data directory:
markdown pipe tables, bare normalized pipe rows, HTML tables, and explicitly
marked ``[Normalized HTML Table Rows]`` blocks.
"""

from html.parser import HTMLParser
import re
from typing import Any, Dict, List, Tuple


_NORMALIZED_BLOCK_RE = re.compile(
    r"\[Normalized HTML Table Rows\]([\s\S]*?)\[End Normalized HTML Table Rows\]",
    re.IGNORECASE,
)
_HTML_TABLE_RE = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)


def _clean_cell(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def _split_pipe_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|") and line.endswith("|"):
        line = line[1:-1]
    return [_clean_cell(c) for c in line.split("|")]


def _looks_like_pipe_row(line: str) -> bool:
    s = line.strip()
    if not s or "|" not in s:
        return False
    cells = _split_pipe_row(s)
    return len(cells) >= 2


def _is_separator(line: str) -> bool:
    cells = _split_pipe_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c is not None)


def _guess_caption(before: str, after: str) -> str:
    candidates = []
    for line in before.splitlines()[-8:] + after.splitlines()[:4]:
        s = _clean_cell(line.strip().strip("#* "))
        if not s or len(s) > 300 or "<table" in s.lower():
            continue
        if re.search(r"\b(table|tab\.?|表)\b", s, re.I):
            candidates.append(s)
    return candidates[-1] if candidates else ""


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._cur_row: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._cur_row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._in_cell:
            self._cur_row.append(_clean_cell(" ".join(self._cell_parts)))
            self._in_cell = False
            self._cell_parts = []
        elif tag == "tr":
            if any(c for c in self._cur_row):
                self.rows.append(self._cur_row)
            self._cur_row = []


def _rows_from_html(html: str) -> List[List[str]]:
    p = _HTMLTableParser()
    try:
        p.feed(html)
    except Exception:
        return []
    return p.rows


def _pipe_blocks_from_text(text: str) -> List[Tuple[int, int, List[str]]]:
    """Return (start_line, end_line_exclusive, lines) for pipe row blocks."""
    lines = text.splitlines()
    blocks: List[Tuple[int, int, List[str]]] = []
    i = 0
    while i < len(lines):
        if not _looks_like_pipe_row(lines[i]):
            i += 1
            continue
        start = i
        block = []
        while i < len(lines) and _looks_like_pipe_row(lines[i]):
            block.append(lines[i])
            i += 1
        if len(block) >= 2:
            blocks.append((start, i, block))
    return blocks


def _records_from_matrix(
    matrix: List[List[str]],
    paper_id: str,
    table_idx: int,
    caption: str,
    before: str,
    after: str,
    max_rows: int = 120,
    source_format: str = "table",
) -> List[Dict[str, Any]]:
    matrix = [[_clean_cell(c) for c in row] for row in matrix if any(_clean_cell(c) for c in row)]
    if len(matrix) < 2:
        return []

    header = matrix[0]
    data_start = 1
    # Markdown separator row after header.
    if len(matrix) > 1 and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in matrix[1] if c is not None):
        data_start = 2

    # Avoid pathological empty/duplicate headers.
    header = [h if h else f"col_{i + 1}" for i, h in enumerate(header)]
    seen_headers: Dict[str, int] = {}
    norm_header: List[str] = []
    for h in header:
        base = h
        seen_headers[base] = seen_headers.get(base, 0) + 1
        norm_header.append(base if seen_headers[base] == 1 else f"{base}_{seen_headers[base]}")
    header = norm_header

    out = []
    for ridx, cells in enumerate(matrix[data_start:data_start + max_rows], 1):
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]
        columns = dict(zip(header, cells))
        row_text = "; ".join(f"{k}: {v}" for k, v in columns.items() if str(v).strip())
        if not row_text:
            continue
        # Keep row evidence compact. Repeating 1,800 characters of nearby
        # context for every row can consume the entire evidence pack before
        # retrieved experimental-section chunks are reached. Context remains in
        # metadata and is represented separately by local retrieval units.
        text = f"caption: {caption}\nrow: {row_text}" if caption else f"row: {row_text}"
        out.append({
            "paper_id": paper_id,
            "unit_id": f"{paper_id}::table_{table_idx:03d}_row_{ridx:03d}",
            "unit_type": "table_row",
            "caption": caption,
            "columns": columns,
            "text": text,
            "section": "table",
            "table_index": table_idx,
            "row_index": ridx,
            "source_format": source_format,
            "nearby_context_before": before,
            "nearby_context_after": after,
        })
    return out


def table_to_rows(
    block_lines: List[str],
    paper_id: str,
    table_idx: int,
    caption: str,
    before: str,
    after: str,
    max_rows: int = 120,
) -> List[Dict[str, Any]]:
    matrix = [_split_pipe_row(x) for x in block_lines if _looks_like_pipe_row(x)]
    return _records_from_matrix(matrix, paper_id, table_idx, caption, before, after, max_rows, "pipe")


def extract_markdown_tables(text: str, paper_id: str, context_chars: int = 1200, max_rows: int = 120) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    table_idx = 0
    lines = text.splitlines()

    consumed_spans: List[Tuple[int, int]] = []
    normalized_char_spans = [(m.start(), m.end()) for m in _NORMALIZED_BLOCK_RE.finditer(text)]

    # 1) Explicit normalized blocks.
    for m in _NORMALIZED_BLOCK_RE.finditer(text):
        before_text = text[max(0, m.start() - context_chars):m.start()]
        after_text = text[m.end():m.end() + context_chars // 2]
        caption = _guess_caption(before_text, after_text)
        block_lines = [x for x in m.group(1).splitlines() if _looks_like_pipe_row(x)]
        rows = table_to_rows(block_lines, paper_id, table_idx, caption, before_text[-context_chars:], after_text[:context_chars // 2], max_rows)
        for r in rows:
            r["source_format"] = "normalized_rows"
        tables.extend(rows)
        table_idx += 1
        start_line = text[:m.start()].count("\n")
        end_line = text[:m.end()].count("\n") + 1
        consumed_spans.append((start_line, end_line))

    # 2) HTML tables. If a normalized block immediately follows the HTML
    # table, prefer the normalized rows because they preserve rowspan/colspan
    # alignment better than a plain HTMLParser matrix.
    for m in _HTML_TABLE_RE.finditer(text):
        if any(m.end() <= ns <= m.end() + 4000 for ns, _ne in normalized_char_spans):
            continue
        before_text = text[max(0, m.start() - context_chars):m.start()]
        after_text = text[m.end():m.end() + context_chars // 2]
        caption = _guess_caption(before_text, after_text)
        matrix = _rows_from_html(m.group(1))
        tables.extend(_records_from_matrix(matrix, paper_id, table_idx, caption, before_text[-context_chars:], after_text[:context_chars // 2], max_rows, "html_table"))
        table_idx += 1
        start_line = text[:m.start()].count("\n")
        end_line = text[:m.end()].count("\n") + 1
        consumed_spans.append((start_line, end_line))

    def in_consumed(start: int, end: int) -> bool:
        return any(not (end <= a or start >= b) for a, b in consumed_spans)

    # 3) Generic markdown/bare pipe blocks not already handled.
    for start, end, block in _pipe_blocks_from_text(text):
        if in_consumed(start, end):
            continue
        before = "\n".join(lines[max(0, start - 8):start])[-context_chars:]
        after = "\n".join(lines[end:min(len(lines), end + 4)])[:context_chars // 2]
        caption = _guess_caption(before, after)
        tables.extend(table_to_rows(block, paper_id, table_idx, caption, before, after, max_rows=max_rows))
        table_idx += 1

    return tables



# Classification deliberately excludes ``nearby_context_*``.  The original
# implementation classified a table row from the full evidence text, so a
# percentage or corrosion term in an adjacent paragraph could turn an HOMO or
# adsorption table into a performance table.  The classifier uses from the table
# schema, row values and a short caption only.
def _classification_parts(row: Dict[str, Any]) -> Tuple[str, str, str]:
    columns = row.get("columns") if isinstance(row.get("columns"), dict) else {}
    headers = " | ".join(str(k) for k in columns.keys())
    values = " | ".join(str(v) for v in columns.values())
    caption = _clean_cell(str(row.get("caption") or ""))
    # A malformed conversion can place an entire neighbouring HTML table on a
    # single "caption" line. Such text is context, not a real short caption.
    if "<table" in caption.lower() or len(caption) > 300:
        caption = ""
    return (_clean_cell(headers).lower(), _clean_cell(values).lower(), caption.lower())


_QUANTUM_HEADER_RE = re.compile(
    r"(?:e\s*[_\\{}()\-]?\s*homo|e\s*[_\\{}()\-]?\s*lumo|\bhomo\b|\blumo\b|energy\s*gap|"
    r"\bdelta\s*e\b|\bΔ\s*e\b|dipole\s*moment|global\s*(?:hardness|softness|electrophilicity)|"
    r"ionization\s*potential|electron\s*affinity)",
    re.I,
)

# Explicit adsorption quantities.  ``binding energy`` by itself is deliberately
# excluded here because XPS peak-position tables use the same phrase.  Generic
# binding energy is handled separately and requires computational adsorption
# context plus the absence of XPS markers.
_ADSORPTION_EXPLICIT_RE = re.compile(
    r"(?:delta\s*g\s*[_\\{}()\-]?\s*ads|Δ\s*g\s*[_\\{}()\-]?\s*ads|\bg\s*[_\\{}()\-]?\s*ads\b|"
    r"adsorption\s*(?:free\s*)?energy|gibbs\s*free\s*energy\s*(?:of\s*)?adsorption|"
    r"\be\s*[_\\{}()\-]?\s*ads\b|\beads\b|"
    r"\be\s*[_\\{}()\-]?\s*(?:bind|binding)\b|\bebinding\b|"
    r"\bk\s*[_\\{}()\-]?\s*ads\b|langmuir\s*(?:constant|parameter))",
    re.I,
)
_GENERIC_BINDING_ENERGY_RE = re.compile(r"\bbinding\s+energ(?:y|ies)\b", re.I)
_XPS_CONTEXT_RE = re.compile(
    r"(?:\bxps\b|x-ray\s+photoelectron|photoelectron\s+spectr|"
    r"\bassignment\b|\bpeak(?:s|\s+position)?\b|deconvolut|"
    r"\b(?:c|n|o|s|p|cl|fe|zn|cu|ni|cr|mn)\s*(?:1s|2p|3p|3d)\s*(?:1/2|3/2|5/2)?\b|"
    r"\b(?:2p|3p|3d|1s)\s*(?:1/2|3/2|5/2)?\b)",
    re.I,
)
_ADSORPTION_COMPUTATIONAL_CONTEXT_RE = re.compile(
    r"(?:\bdft\b|density\s+functional|molecular\s+dynamics|\bmd\b|monte\s+carlo|"
    r"simulation|computational|adsorb(?:ed|ate|ing|tion)|surface|"
    r"fe\s*\(\s*110\s*\)|metal\s*[-–—]?\s*inhibitor|interaction\s+energy|"
    r"deformation\s+energy|rigid\s+adsorption\s+energy)",
    re.I,
)
_PERFORMANCE_HEADER_RE = re.compile(
    r"(?:inhibition\s*efficien|inhibitory\s*efficien|protection\s*efficien|\befficiency\b|"
    r"\bc?ie\b(?:\s*[/(_-]?\s*%\s*\)?)?|\bie\s*(?:pdp|eis|wl|pp)?\s*%?|%\s*ie\b|"
    r"[ηƞ]\s*(?:pdp|eis|wl|pp|rp)?\s*(?:[/(_-]?\s*%\s*\)?)?|%\s*[ηƞ]|"
    r"\be\s*\(\s*%\s*\)|\be\s*%|surface\s*coverage|"
    r"corrosion\s*inhibition\s*\(\s*%\s*\))",
    re.I,
)


def _schema_normalize(text: str) -> str:
    """Normalize common Markdown/LaTeX spellings used in table headers."""
    value = str(text or "")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("$", "").replace("`", "")
    value = re.sub(r"[_^]", "_", value)
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


def _is_xps_binding_context(text: str) -> bool:
    return bool(_XPS_CONTEXT_RE.search(_schema_normalize(text)))


def _has_adsorption_schema(text: str) -> bool:
    norm = _schema_normalize(text)
    if _ADSORPTION_EXPLICIT_RE.search(norm):
        return True
    if not _GENERIC_BINDING_ENERGY_RE.search(norm):
        return False
    if _is_xps_binding_context(norm):
        return False
    return bool(_ADSORPTION_COMPUTATIONAL_CONTEXT_RE.search(norm))


def classify_table_row(row: Dict[str, Any]) -> str | None:
    """Return the single supported row group supported by the table itself.

    Classification uses only headers, current-row values and a short caption.
    Quantum and explicit adsorption schemas take precedence over performance.
    XPS binding-energy peak tables are explicitly rejected from adsorption.
    Generic percent signs, EIS, Rct and Icorr are intentionally insufficient.
    """
    headers, values, caption = _classification_parts(row)
    headers = _schema_normalize(headers)
    values = _schema_normalize(values)
    caption = _schema_normalize(caption)
    schema_text = f"{headers}\n{caption}"
    row_text = f"{headers}\n{values}\n{caption}"

    quantum_score = (
        3 * len(_QUANTUM_HEADER_RE.findall(headers))
        + 2 * len(_QUANTUM_HEADER_RE.findall(values))
        + len(_QUANTUM_HEADER_RE.findall(caption))
    )

    # Explicit Eads/ΔGads/adsorption-energy labels can appear either in a
    # header or in a Parameter column.  Plain "binding energy" is accepted only
    # for computational adsorption context and never for XPS peak assignments.
    adsorption_score = 0
    if _has_adsorption_schema(headers):
        adsorption_score += 3
    if _has_adsorption_schema(values):
        adsorption_score += 2
    if _has_adsorption_schema(caption):
        adsorption_score += 1
    if (
        _GENERIC_BINDING_ENERGY_RE.search(row_text)
        and _ADSORPTION_COMPUTATIONAL_CONTEXT_RE.search(row_text)
        and not _is_xps_binding_context(row_text)
    ):
        adsorption_score = max(adsorption_score, 3)
    if _is_xps_binding_context(row_text) and not _ADSORPTION_EXPLICIT_RE.search(row_text):
        adsorption_score = 0

    performance_score = (
        3 * len(_PERFORMANCE_HEADER_RE.findall(headers))
        + 2 * len(_PERFORMANCE_HEADER_RE.findall(values))
        + len(_PERFORMANCE_HEADER_RE.findall(caption))
    )
    if performance_score and re.search(r"(?<!\d)\d+(?:\.\d+)?\s*%(?!\d)", values):
        performance_score += 1

    scores = {
        "quantum": quantum_score,
        "adsorption": adsorption_score,
        "performance": performance_score,
    }
    best_score = max(scores.values())
    if best_score <= 0:
        return None
    # Stable tie-break: specialised schemas win over performance.
    for group in ("quantum", "adsorption", "performance"):
        if scores[group] == best_score:
            return group
    return None

def is_likely_performance_row(row: Dict[str, Any]) -> bool:
    return classify_table_row(row) == "performance"


def is_likely_quantum_row(row: Dict[str, Any]) -> bool:
    return classify_table_row(row) == "quantum"


def is_likely_adsorption_row(row: Dict[str, Any]) -> bool:
    return classify_table_row(row) == "adsorption"
