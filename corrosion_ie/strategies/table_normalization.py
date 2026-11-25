from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from bs4 import BeautifulSoup  # type: ignore

from ..schemas import EvidenceUnit


_WS_RE = re.compile(r"\s+")
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.I | re.S)
_MARKDOWN_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")

_REFERENCE_CAPTION_RE = re.compile(
    r"(?:"
    r"compar(?:e|ed|ing|ison)(?:\s+of|\s+between|\s+with)?\b.*(?:previous|reported|published|other|literature|inhibitor|derivative|plant)|"
    r"literature\s+(?:review|survey|comparison|data)|"
    r"reviews?\s+on\s+the\s+corrosion\s+inhibition|"
    r"previous(?:ly)?\s+(?:reported|published)|"
    r"reported\s+(?:inhibitors?|data|values?)\s+in\s+(?:the\s+)?literature|"
    r"similar\s+(?:compounds?|derivatives?|inhibitors?)|"
    r"other\s+(?:organic\s+)?inhibitors?|"
    r"according\s+to\s+(?:the\s+)?literature|"
    r"reference\s+compounds?|published\s+data"
    r")",
    re.I | re.S,
)
_REFERENCE_HEADER_RE = re.compile(
    r"^(?:refs?\.?|ref(?:erence)?['’]s|references?|citation|citations|authors?|year|doi|literature)$",
    re.I,
)
_REFERENCE_ROW_RE = re.compile(
    r"(?:\[(?:\d{1,3}(?:\s*[-,]\s*\d{1,3})*)\]|"
    r"\((?:19|20)\d{2}[a-z]?\)|\b(?:19|20)\d{2}\b|"
    r"\bet\s+al\.?\b|\bdoi\s*:|https?://|"
    r"\brefs?\.?\s*\d+)",
    re.I,
)

_REVIEW_DOCUMENT_RE = re.compile(
    r"(?:\b(?:this|the\s+present)\s+review\b|"
    r"\breview\s+article\b|"
    r"\bwe\s+(?:review|survey)\b|"
    r"\bprovides?\s+(?:a\s+)?(?:comprehensive\s+)?review\b|"
    r"\bcomprehensive\s+review\b)",
    re.I,
)


def detect_review_article(text: str) -> bool:
    """Detect a review article from title/abstract-level self-description.

    Only the beginning of the target paper is inspected so that phrases such as
    ``literature review`` in an introduction of a primary study do not trigger
    review mode.
    """

    head = (text or "")[:12000]
    return bool(_REVIEW_DOCUMENT_RE.search(head))


_HEADER_TERM_RE = re.compile(
    r"(?:^|\b)(?:inhibitors?|compounds?|sample|substrate|metal|medium|solution|acid|"
    r"concentration|conc\.?|dosage|temperature|temp\.?|t\s*\(|ti\s*\(|n°|time|immersion|method|"
    r"efficien|ie\s*%?|η|ƞ|corrosion|cr\b|crc\b|rct\b|icorr\b|ecorr\b|slope|intercept|"
    r"k[_ -]?ads|g[_ -]?ads|h[_ -]?ads|s[_ -]?ads|delta|Δ|ea\b|r\s*[²2]|"
    r"homo|lumo|adsorption|isotherm|parameter|reference|refs?\.?|author|year)(?:\b|$)",
    re.I,
)
_TEMPERATURE_HEADER_RE = re.compile(r"^\s*[+-]?\d{2,4}(?:\.\d+)?\s*(?:k|°\s*c|deg\.?\s*c)\s*$", re.I)
_NUMERIC_DATA_RE = re.compile(
    r"^\s*[~≈<>≤≥]?\s*[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    r"(?:\s*(?:±|\+/-)\s*\d+(?:[.,]\d+)?)?"
    r"(?:\s*(?:×|x)\s*10\s*[+-]?\d+)?"
    r"(?:\s*[-–—]\s*[+-]?\d+(?:[.,]\d+)?)?"
    r"(?:\s*[-%a-zA-Z°/²³μΩ.]+)?\s*$",
    re.I,
)
_LOGICAL_TABLE_LABEL_RE = re.compile(
    r"(?:langmuir|freundlich|temkin|frumkin|el[-\s]?awady|flory[-\s]?huggins|"
    r"dubinin|redlich|sips|toth|kinetic|thermodynamic|potentiodynamic|impedance|eis)",
    re.I,
)
_GENERIC_HEADER_LABEL_RE = re.compile(
    r"^(?:inhibitors?|isotherms?\s+parameters?|parameters?|temperature|temp\.?|t\s*\([^)]*\))$",
    re.I,
)

_CURRENT_STUDY_RE = re.compile(
    r"(?:this|present|current)\s+(?:study|work|investigation)|"
    r"(?:our|the\s+present)\s+(?:study|work|compound|inhibitor)|"
    r"herein|proposed\s+(?:inhibitor|compound|method)",
    re.I,
)


@dataclass
class TableGrid:
    caption: str
    headers: List[str]
    rows: List[List[str]]
    source_format: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return len(self.headers)

    def render_markdown(self) -> str:
        def esc(value: Any) -> str:
            text = _clean_text(str(value or ""))
            return text.replace("|", "\\|").replace("\n", " ")

        parts: List[str] = []
        if self.caption:
            parts.append(f"caption: {esc(self.caption)}")
        if not self.headers:
            return "\n".join(parts)
        parts.append("| " + " | ".join(esc(x) for x in self.headers) + " |")
        parts.append("| " + " | ".join("---" for _ in self.headers) + " |")
        for row in self.rows:
            padded = list(row[: self.width]) + [""] * max(0, self.width - len(row))
            parts.append("| " + " | ".join(esc(x) for x in padded) + " |")
        return "\n".join(parts)


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\xa0", " ")
    return _WS_RE.sub(" ", value).strip()


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(str(value or default)))
    except Exception:
        return default


def _extract_preamble(text: str, table_start: int, fallback_caption: str) -> str:
    if fallback_caption:
        return _clean_text(fallback_caption)
    before = text[:table_start]
    # Do not accidentally use the previous HTML table as the next caption.
    # Search a small backward window for a plausible plain-text caption/heading.
    for raw_line in reversed(before.splitlines()[-30:]):
        line = _clean_text(raw_line)
        if not line:
            continue
        lower = line.lower()
        if any(tag in lower for tag in ("<table", "</table", "<tr", "<td", "<th")):
            continue
        if len(line) > 1200:
            continue
        return line[-800:]
    return ""


def _html_matrix(text: str, caption: str = "") -> Tuple[str, List[List[str]], List[bool], List[bool], List[bool], Dict[str, Any]] | None:
    """Expand the first HTML table into a rectangular matrix.

    Rowspan and colspan values are materialized into every covered cell so that
    logical rows remain self-contained when a large table is fragmented later.
    """

    match = _HTML_TABLE_RE.search(text or "")
    if not match:
        return None
    effective_caption = _extract_preamble(text or "", match.start(), caption)
    soup = BeautifulSoup(match.group(0), "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    row_maps: List[Dict[int, str]] = []
    row_has_th: List[bool] = []
    row_has_span: List[bool] = []
    row_received_carry: List[bool] = []
    carried: Dict[Tuple[int, int], str] = {}
    max_col = 0

    trs = table.find_all("tr")
    for r_idx, tr in enumerate(trs):
        values: Dict[int, str] = {c: value for (rr, c), value in carried.items() if rr == r_idx}
        received_carry = bool(values)
        col = 0
        cells = tr.find_all(["th", "td"], recursive=False)
        has_th = any(cell.name == "th" for cell in cells)
        has_span = False
        for cell in cells:
            while col in values:
                col += 1
            value = _clean_text(cell.get_text(" ", strip=True))
            rowspan = _safe_int(cell.get("rowspan"), 1)
            colspan = _safe_int(cell.get("colspan"), 1)
            has_span = has_span or rowspan > 1 or colspan > 1
            for dc in range(colspan):
                values[col + dc] = value
                for dr in range(1, rowspan):
                    carried[(r_idx + dr, col + dc)] = value
            col += colspan
        if values:
            max_col = max(max_col, max(values) + 1)
            row_maps.append(values)
            row_has_th.append(has_th)
            row_has_span.append(has_span)
            row_received_carry.append(received_carry)

    if not row_maps or max_col <= 0:
        return None
    matrix = [[_clean_text(row.get(c, "")) for c in range(max_col)] for row in row_maps]
    diagnostics = {
        "rowspan_count": len(re.findall(r"\browspan\s*=", match.group(0), re.I)),
        "colspan_count": len(re.findall(r"\bcolspan\s*=", match.group(0), re.I)),
        "raw_row_count": len(matrix),
        "raw_column_count": max_col,
    }
    return effective_caption, matrix, row_has_th, row_has_span, row_received_carry, diagnostics


def _nonempty_cells(row: Sequence[str]) -> List[str]:
    return [_clean_text(value) for value in row if _clean_text(value)]


def _header_signal_count(row: Sequence[str]) -> int:
    return sum(bool(_HEADER_TERM_RE.search(value)) for value in _nonempty_cells(row))


def _numeric_data_count(row: Sequence[str]) -> int:
    return sum(bool(_NUMERIC_DATA_RE.match(value)) and not bool(_TEMPERATURE_HEADER_RE.match(value)) for value in _nonempty_cells(row))


def _temperature_header_count(row: Sequence[str]) -> int:
    return sum(bool(_TEMPERATURE_HEADER_RE.match(value)) for value in _nonempty_cells(row))


def _is_header_like(row: Sequence[str], *, has_th: bool = False, has_span: bool = False, first: bool = False) -> bool:
    values = _nonempty_cells(row)
    if not values:
        return False
    if has_th or first:
        return True
    n = len(values)
    header_hits = _header_signal_count(values)
    numeric_hits = _numeric_data_count(values)
    temp_hits = _temperature_header_count(values)
    unique = len({value.casefold() for value in values})

    # Spanning rows are commonly group headings (temperature bands, isotherm
    # names, electrochemical methods).  Numeric-looking temperature labels are
    # still header content in this context.
    if has_span:
        if temp_hits >= max(1, n // 2):
            return True
        if header_hits >= 2 and numeric_hits <= max(1, n // 2):
            return True
        if numeric_hits == 0 and unique <= max(3, n // 2):
            return True
    if header_hits >= 2 and numeric_hits <= max(1, n // 2):
        return True
    if header_hits >= 1 and numeric_hits == 0:
        return True
    if temp_hits == n:
        return True
    # A short repeated label expanded from colspan is a group heading.
    if unique <= 2 and numeric_hits == 0 and max(len(value) for value in values) <= 80:
        return True
    return False


def _detect_header_block(
    matrix: Sequence[Sequence[str]],
    row_has_th: Sequence[bool],
    row_has_span: Sequence[bool],
    row_received_carry: Sequence[bool],
    start: int = 0,
    *,
    repeated: bool = False,
    max_rows: int = 5,
) -> int:
    if start >= len(matrix):
        return 0
    count = 0
    previous_had_span = False
    for idx in range(start, min(len(matrix), start + max_rows)):
        first = idx == start and not repeated
        eligible_continuation = bool(
            first
            or row_has_th[idx]
            or row_received_carry[idx]
            or previous_had_span
            or row_has_span[idx]
        )
        if not eligible_continuation:
            break
        if not _is_header_like(
            matrix[idx],
            has_th=bool(row_has_th[idx]),
            has_span=bool(row_has_span[idx]),
            first=first,
        ):
            break
        count += 1
        previous_had_span = bool(row_has_span[idx])
    if not repeated:
        return max(1, count)
    if count <= 0 or start + count >= len(matrix):
        return 0
    next_idx = start + count
    next_row = matrix[next_idx]
    next_eligible = bool(
        row_has_th[next_idx]
        or row_received_carry[next_idx]
        or previous_had_span
        or row_has_span[next_idx]
    )
    if next_eligible and _is_header_like(
        next_row,
        has_th=bool(row_has_th[next_idx]),
        has_span=bool(row_has_span[next_idx]),
    ):
        return 0
    return count


def _looks_like_repeated_header_start(
    matrix: Sequence[Sequence[str]],
    row_has_th: Sequence[bool],
    row_has_span: Sequence[bool],
    row_received_carry: Sequence[bool],
    idx: int,
) -> bool:
    if idx <= 0 or idx >= len(matrix) - 1:
        return False
    row = matrix[idx]
    if not _is_header_like(row, has_th=bool(row_has_th[idx]), has_span=bool(row_has_span[idx])):
        return False
    values = _nonempty_cells(row)
    if not values:
        return False
    strong = bool(row_has_th[idx] or row_has_span[idx] or _header_signal_count(values) >= 2)
    if not strong:
        return False
    return _detect_header_block(matrix, row_has_th, row_has_span, row_received_carry, idx, repeated=True) > 0


def _logical_label(header_rows: Sequence[Sequence[str]]) -> str:
    candidates: List[str] = []
    for row in header_rows:
        for value in _nonempty_cells(row):
            if _LOGICAL_TABLE_LABEL_RE.search(value):
                candidates.append(value)
    if candidates:
        return max(candidates, key=len)[:160]
    # Use a non-generic colspan/group label when it is repeated over columns.
    for row in header_rows:
        values = _nonempty_cells(row)
        counts: Dict[str, int] = {}
        originals: Dict[str, str] = {}
        for value in values:
            key = value.casefold()
            counts[key] = counts.get(key, 0) + 1
            originals[key] = value
        repeated = [originals[key] for key, count in counts.items() if count >= 2 and not _GENERIC_HEADER_LABEL_RE.match(originals[key])]
        if repeated:
            return max(repeated, key=len)[:160]
    return ""


def _build_grid(
    *,
    caption: str,
    header_rows: Sequence[Sequence[str]],
    data_rows: Sequence[Sequence[str]],
    source_format: str,
    diagnostics: Dict[str, Any],
    logical_index: int,
    logical_count: int,
) -> TableGrid | None:
    if not header_rows:
        return None
    width = max(len(row) for row in [*header_rows, *data_rows])
    padded_headers = [list(row) + [""] * (width - len(row)) for row in header_rows]
    padded_data = [list(row) + [""] * (width - len(row)) for row in data_rows]
    headers: List[str] = []
    for c in range(width):
        seen: List[str] = []
        for row in padded_headers:
            value = _clean_text(row[c])
            if value and value.casefold() not in {x.casefold() for x in seen}:
                seen.append(value)
        headers.append(" / ".join(seen) if seen else f"column_{c + 1}")

    keep_cols = [
        c for c in range(width)
        if headers[c] != f"column_{c + 1}" or any(_clean_text(row[c]) for row in padded_data)
    ]
    if not keep_cols:
        return None
    headers = [headers[c] for c in keep_cols]
    cleaned_rows: List[List[str]] = []
    for row in padded_data:
        cleaned = [_clean_text(row[c]) for c in keep_cols]
        if any(cleaned):
            cleaned_rows.append(cleaned)
    if not cleaned_rows:
        return None

    label = _logical_label(header_rows)
    effective_caption = _clean_text(caption)
    if logical_count > 1 and label and label.casefold() not in effective_caption.casefold():
        effective_caption = f"{effective_caption} [logical subtable: {label}]".strip()
    return TableGrid(
        caption=effective_caption,
        headers=headers,
        rows=cleaned_rows,
        source_format=source_format,
        diagnostics={
            **diagnostics,
            "header_row_count": len(header_rows),
            "data_row_count": len(cleaned_rows),
            "column_count": len(headers),
            "logical_subtable_index": logical_index,
            "logical_subtable_count": logical_count,
            "logical_subtable_label": label,
        },
    )


def html_tables_to_grids(text: str, caption: str = "") -> List[TableGrid]:
    parsed = _html_matrix(text, caption=caption)
    if parsed is None:
        return []
    effective_caption, matrix, row_has_th, row_has_span, row_received_carry, diagnostics = parsed
    if len(matrix) == 1:
        return []

    first_header_count = _detect_header_block(matrix, row_has_th, row_has_span, row_received_carry, 0)
    first_header_count = min(first_header_count, len(matrix) - 1)
    segments: List[Tuple[List[List[str]], List[List[str]]]] = []
    current_headers = [list(row) for row in matrix[:first_header_count]]
    current_rows: List[List[str]] = []
    idx = first_header_count
    while idx < len(matrix):
        if current_rows and _looks_like_repeated_header_start(matrix, row_has_th, row_has_span, row_received_carry, idx):
            segments.append((current_headers, current_rows))
            block = _detect_header_block(matrix, row_has_th, row_has_span, row_received_carry, idx, repeated=True)
            current_headers = [list(row) for row in matrix[idx: idx + block]]
            current_rows = []
            idx += block
            continue
        current_rows.append(list(matrix[idx]))
        idx += 1
    if current_rows:
        segments.append((current_headers, current_rows))

    grids: List[TableGrid] = []
    logical_count = len(segments)
    for logical_index, (header_rows, data_rows) in enumerate(segments, start=1):
        grid = _build_grid(
            caption=effective_caption,
            header_rows=header_rows,
            data_rows=data_rows,
            source_format="html",
            diagnostics={**diagnostics, "repeated_header_splits": max(0, logical_count - 1)},
            logical_index=logical_index,
            logical_count=logical_count,
        )
        if grid is not None:
            grids.append(grid)
    return grids


def html_table_to_grid(text: str, caption: str = "") -> TableGrid | None:
    """Backward-compatible single-grid view.

    Production table-aware packing uses :func:`html_tables_to_grids` and keeps
    every logical subtable.  Callers that expect one grid receive the first one.
    """

    grids = html_tables_to_grids(text, caption=caption)
    return grids[0] if grids else None


def _split_markdown_row(line: str) -> List[str]:
    line = line.strip().strip("|")
    # A backslash-escaped pipe is preserved as text.
    parts = re.split(r"(?<!\\)\|", line)
    return [_clean_text(part.replace("\\|", "|")) for part in parts]


def markdown_table_to_grid(text: str, caption: str = "") -> TableGrid | None:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    table_indices = [i for i, line in enumerate(lines) if _MARKDOWN_ROW_RE.match(line)]
    if len(table_indices) < 2:
        return None
    start = table_indices[0]
    contiguous: List[str] = []
    expected = start
    for idx in table_indices:
        if idx != expected and contiguous:
            break
        contiguous.append(lines[idx])
        expected = idx + 1
    if len(contiguous) < 2:
        return None

    separator_index = next(
        (i for i, line in enumerate(contiguous) if _MARKDOWN_SEPARATOR_RE.match(line)),
        None,
    )
    if separator_index is not None and separator_index > 0:
        # Markdown exported from XML often preserves multi-level headers as
        # several pipe rows followed by one separator row.  Every row before
        # the separator belongs to the header; the separator itself is never
        # data.
        header_lines = contiguous[:separator_index]
        data_lines = contiguous[separator_index + 1:]
    else:
        header_lines = contiguous[:1]
        data_lines = contiguous[1:]

    header_rows = [_split_markdown_row(line) for line in header_lines]
    data_rows = [_split_markdown_row(line) for line in data_lines]

    # Several XML-to-Markdown converters place only the top header row before
    # the Markdown separator and emit lower header levels immediately after it.
    # Promote only strongly header-like leading rows; numeric experimental rows
    # remain data.
    promoted = 0
    while data_rows and promoted < 4:
        values = _nonempty_cells(data_rows[0])
        header_hits = _header_signal_count(values)
        numeric_hits = _numeric_data_count(values)
        strong_lower_header = bool(
            values
            and header_hits >= 2
            and numeric_hits <= max(1, len(values) // 4)
        )
        if not strong_lower_header:
            break
        header_rows.append(data_rows.pop(0))
        promoted += 1

    effective_caption = caption or _extract_preamble(text or "", sum(len(x) + 1 for x in lines[:start]), "")
    return _build_grid(
        caption=_clean_text(effective_caption),
        header_rows=header_rows,
        data_rows=data_rows,
        source_format="markdown",
        diagnostics={
            "raw_row_count": len(contiguous),
            "raw_column_count": max((len(row) for row in [*header_rows, *data_rows]), default=0),
            "repeated_header_splits": 0,
            "post_separator_header_rows_promoted": promoted,
        },
        logical_index=1,
        logical_count=1,
    )


def table_to_grids(text: str, caption: str = "") -> List[TableGrid]:
    html_grids = html_tables_to_grids(text, caption=caption)
    if html_grids:
        return html_grids
    markdown = markdown_table_to_grid(text, caption=caption)
    return [markdown] if markdown is not None else []


def table_to_grid(text: str, caption: str = "") -> TableGrid | None:
    grids = table_to_grids(text, caption=caption)
    return grids[0] if grids else None


def _reference_column_indices(headers: Sequence[str]) -> List[int]:
    return [i for i, header in enumerate(headers) if _REFERENCE_HEADER_RE.match(_clean_text(header))]


def _row_has_reference(row: Sequence[str], ref_cols: Sequence[int]) -> bool:
    # Citation-like text elsewhere in a row is not sufficient: current-study
    # adsorption tables legitimately contain labels such as ``Langmuir [61]``,
    # and microscopy tables can contain acquisition dates.  Only dedicated
    # reference/citation columns identify literature rows.
    return any(i < len(row) and _clean_text(row[i]) for i in ref_cols)


def _row_is_current(row: Sequence[str]) -> bool:
    return bool(_CURRENT_STUDY_RE.search(" | ".join(row)))


def classify_reference_grid(grid: TableGrid) -> Dict[str, Any]:
    caption_ref = bool(_REFERENCE_CAPTION_RE.search(grid.caption or ""))
    ref_cols = _reference_column_indices(grid.headers)

    # Comparison tables often put a citation/"Our work" marker only on the
    # first row of a multi-row inhibitor group.  Blank reference cells below
    # that row are continuations and must inherit the preceding group status.
    statuses: List[str] = []
    previous = "unknown"
    for row in grid.rows:
        current = _row_is_current(row)
        ref_cell_values = [
            _clean_text(row[i]) for i in ref_cols if i < len(row) and _clean_text(row[i])
        ]
        explicit_ref = bool(ref_cell_values)
        if current:
            status = "current"
        elif explicit_ref:
            status = "reference"
        elif ref_cols and previous in {"current", "reference"}:
            status = previous
        elif caption_ref:
            # A strongly literature-labelled table without a dedicated reference
            # column is treated as literature, except continuation rows after an
            # explicit current-study marker.
            status = "current" if previous == "current" else "reference"
        else:
            status = "unknown"
        statuses.append(status)
        if status != "unknown":
            previous = status

    reference_rows = [i for i, status in enumerate(statuses) if status == "reference"]
    current_rows = [i for i, status in enumerate(statuses) if status == "current"]
    ratio = len(reference_rows) / max(1, len(grid.rows))

    kind = "none"
    if current_rows and (caption_ref or ref_cols or reference_rows):
        kind = "mixed"
    elif caption_ref or (ref_cols and ratio >= 0.25) or ratio >= 0.75:
        kind = "pure"

    return {
        "kind": kind,
        "caption_reference_signal": caption_ref,
        "reference_column_indices": ref_cols,
        "reference_row_indices": reference_rows,
        "current_study_row_indices": current_rows,
        "row_reference_status": statuses,
        "reference_row_ratio": round(ratio, 6),
    }


def sanitize_reference_grid(grid: TableGrid) -> Tuple[TableGrid | None, Dict[str, Any]]:
    info = classify_reference_grid(grid)
    if info["kind"] == "pure":
        return None, {**info, "action": "exclude_pure_reference_table"}
    if info["kind"] != "mixed":
        return grid, {**info, "action": "keep"}

    current = set(info["current_study_row_indices"])
    rows = [row for idx, row in enumerate(grid.rows) if idx in current]
    if not rows:
        return None, {**info, "action": "exclude_mixed_without_current_rows"}
    sanitized = TableGrid(
        caption=grid.caption,
        headers=list(grid.headers),
        rows=rows,
        source_format=grid.source_format,
        diagnostics={**grid.diagnostics, "reference_rows_removed": len(grid.rows) - len(rows)},
    )
    return sanitized, {**info, "action": "keep_current_study_rows_only", "rows_kept": len(rows)}


def clone_table_evidence(
    ev: EvidenceUnit,
    grid: TableGrid,
    extra_metadata: Dict[str, Any] | None = None,
    *,
    unit_id: str | None = None,
) -> EvidenceUnit:
    metadata = dict(ev.metadata or {})
    metadata.update({
        "table_normalized": True,
        "table_source_format": grid.source_format,
        "table_headers": list(grid.headers),
        "table_row_count": len(grid.rows),
        "table_column_count": len(grid.headers),
        **grid.diagnostics,
        **(extra_metadata or {}),
    })
    return EvidenceUnit(
        paper_id=ev.paper_id,
        unit_id=unit_id or ev.unit_id,
        unit_type="table",
        text=grid.render_markdown(),
        query_group=ev.query_group,
        query=ev.query,
        rank=ev.rank,
        score=ev.score,
        section=ev.section,
        caption=grid.caption or ev.caption,
        columns=dict(ev.columns or {}),
        metadata=metadata,
    )


def normalize_and_sanitize_tables(
    ev: EvidenceUnit,
    *,
    allow_reference_tables: bool = False,
) -> Tuple[List[EvidenceUnit], Dict[str, Any]]:
    """Normalize every logical subtable contained in one source table unit."""

    grids = table_to_grids(ev.text or "", caption=ev.caption or "")
    if not grids:
        return [ev], {"action": "not_parsed", "kind": "unknown", "logical_table_count": 0}

    kept: List[EvidenceUnit] = []
    infos: List[Dict[str, Any]] = []
    for idx, grid in enumerate(grids, start=1):
        classification = classify_reference_grid(grid)
        if allow_reference_tables and classification.get("kind") in {"pure", "mixed"}:
            info = {
                **classification,
                "action": "keep_review_reference_table",
                "review_reference_table_allowed": True,
                "rows_kept": len(grid.rows),
            }
            sanitized = grid
        else:
            sanitized, info = sanitize_reference_grid(grid)
        info = {
            **info,
            "logical_subtable_index": idx,
            "logical_subtable_count": len(grids),
            "logical_subtable_label": grid.diagnostics.get("logical_subtable_label", ""),
        }
        infos.append(info)
        if sanitized is None:
            continue
        logical_id = ev.unit_id if len(grids) == 1 else f"{ev.unit_id}::logical_{idx:03d}"
        kept.append(clone_table_evidence(
            ev,
            sanitized,
            {
                "reference_table": info,
                "logical_parent_unit_id": ev.unit_id,
                "logical_subtable_index": idx,
                "logical_subtable_count": len(grids),
            },
            unit_id=logical_id,
        ))

    if len(grids) == 1:
        return kept, infos[0]
    kinds = {str(info.get("kind") or "unknown") for info in infos}
    aggregate_kind = "mixed" if "mixed" in kinds or len(kinds) > 1 else next(iter(kinds), "unknown")
    return kept, {
        "action": "split_logical_tables",
        "kind": aggregate_kind,
        "logical_table_count": len(grids),
        "logical_tables_kept": len(kept),
        "logical_tables_excluded": len(grids) - len(kept),
        "logical_table_infos": infos,
    }


def normalize_and_sanitize_table(
    ev: EvidenceUnit,
    *,
    allow_reference_tables: bool = False,
) -> Tuple[EvidenceUnit | None, Dict[str, Any]]:
    """Backward-compatible first logical table wrapper."""

    kept, info = normalize_and_sanitize_tables(ev, allow_reference_tables=allow_reference_tables)
    return (kept[0] if kept else None), info

