#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.strategies.table_normalization import html_tables_to_grids, markdown_table_to_grid

_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.I | re.S)
_MARKDOWN_BLOCK_RE = re.compile(r"(?:^|\n)((?:\s*\|[^\n]*\|\s*\n?){2,})", re.M)
_MARKDOWN_SEPARATOR_ANY_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
_EXACT_HEADER_RE = re.compile(
    r"\s*(?:inhibitors?|compound|temperature|temp\.?|t\s*\([^)]*\)|concentration|conc\.?|"
    r"slope|intercept|k_?ads|Δ?g(?:0|°)?ads.*|r[²2]|ie%?|efficiency|reference|refs?\.?)\s*",
    re.I,
)


def nearest_caption(text: str, start: int) -> str:
    for line in reversed(text[:start].splitlines()[-20:]):
        if line.strip():
            return line.strip()[:1000]
    return ""


def scan(md_dirs: List[Path]) -> Dict[str, Any]:
    raw_tables = 0
    raw_html_tables = 0
    raw_markdown_tables = 0
    logical_tables = 0
    logical_html_tables = 0
    logical_markdown_tables = 0
    parse_failures: List[Dict[str, Any]] = []
    suspicious_header_rows: List[Dict[str, Any]] = []
    split_tables: List[Dict[str, Any]] = []
    rowspan_tables = 0
    colspan_tables = 0

    for md_dir in md_dirs:
        for path in sorted(md_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for table_index, match in enumerate(_TABLE_RE.finditer(text), start=1):
                raw_tables += 1
                raw_html_tables += 1
                html = match.group(0)
                rowspan_tables += int(bool(re.search(r"\browspan\s*=", html, re.I)))
                colspan_tables += int(bool(re.search(r"\bcolspan\s*=", html, re.I)))
                caption = nearest_caption(text, match.start())
                grids = html_tables_to_grids(html, caption=caption)
                if not grids:
                    parse_failures.append({
                        "paper_id": path.stem,
                        "table_index": table_index,
                        "caption": caption,
                    })
                    continue
                logical_tables += len(grids)
                logical_html_tables += len(grids)
                if len(grids) > 1:
                    split_tables.append({
                        "paper_id": path.stem,
                        "table_index": table_index,
                        "logical_table_count": len(grids),
                        "labels": [grid.diagnostics.get("logical_subtable_label", "") for grid in grids],
                        "caption": caption,
                    })
                for logical_index, grid in enumerate(grids, start=1):
                    if not grid.rows:
                        parse_failures.append({
                            "paper_id": path.stem,
                            "table_index": table_index,
                            "logical_index": logical_index,
                            "caption": caption,
                            "reason": "empty logical table",
                        })
                        continue
                    first = grid.rows[0]
                    exact = sum(bool(_EXACT_HEADER_RE.fullmatch(cell or "")) for cell in first)
                    if exact >= 2:
                        suspicious_header_rows.append({
                            "paper_id": path.stem,
                            "table_index": table_index,
                            "logical_index": logical_index,
                            "first_data_row": first,
                            "headers": grid.headers,
                            "caption": caption,
                        })

            for table_index, match in enumerate(_MARKDOWN_BLOCK_RE.finditer(text), start=1):
                block = match.group(1)
                if not _MARKDOWN_SEPARATOR_ANY_RE.search(block):
                    continue
                raw_tables += 1
                raw_markdown_tables += 1
                caption = nearest_caption(text, match.start())
                grid = markdown_table_to_grid(block, caption=caption)
                if grid is None or not grid.rows:
                    parse_failures.append({
                        "paper_id": path.stem,
                        "table_index": table_index,
                        "source_format": "markdown",
                        "caption": caption,
                    })
                    continue
                logical_tables += 1
                logical_markdown_tables += 1
                first = grid.rows[0]
                exact = sum(bool(_EXACT_HEADER_RE.fullmatch(cell or "")) for cell in first)
                if exact >= 2:
                    suspicious_header_rows.append({
                        "paper_id": path.stem,
                        "table_index": table_index,
                        "logical_index": 1,
                        "source_format": "markdown",
                        "first_data_row": first,
                        "headers": grid.headers,
                        "caption": caption,
                    })

    return {
        "raw_source_tables": raw_tables,
        "raw_html_tables": raw_html_tables,
        "raw_markdown_tables": raw_markdown_tables,
        "logical_tables": logical_tables,
        "logical_html_tables": logical_html_tables,
        "logical_markdown_tables": logical_markdown_tables,
        "parse_failures": parse_failures,
        "suspicious_header_rows": suspicious_header_rows,
        "source_tables_split": len(split_tables),
        "split_tables": split_tables,
        "tables_with_rowspan": rowspan_tables,
        "tables_with_colspan": colspan_tables,
        "valid": not parse_failures and not suspicious_header_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit HTML table normalization over one or more Markdown directories.")
    ap.add_argument("--md-dir", action="append", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("table_normalization_audit.json"))
    args = ap.parse_args()
    result = scan(args.md_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"split_tables", "parse_failures", "suspicious_header_rows"}}, indent=2))
    print("[table-audit]", args.output)
    if not result["valid"]:
        print("[table-audit-error] parse_failures=", len(result["parse_failures"]))
        print("[table-audit-error] suspicious_header_rows=", len(result["suspicious_header_rows"]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
