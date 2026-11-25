#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.retrieval.chunking import split_markdown
from corrosion_ie.retrieval.legacy_exact_chunking import split_markdown_legacy_exact


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare old exact and v2 Markdown chunkers.")
    ap.add_argument("--md-dir", default="input/md")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--out", default="chunker_comparison.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.md_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        old = split_markdown_legacy_exact(
            text, path.stem, args.chunk_size, args.chunk_overlap
        )
        new = split_markdown(text, path.stem, args.chunk_size, args.chunk_overlap)
        rows.append(
            {
                "paper_id": path.stem,
                "legacy_exact_chunks": len(old),
                "v2_chunks": len(new),
                "difference": len(new) - len(old),
                "legacy_table_chunks": sum(c.get("content_type") == "table" for c in old),
                "v2_table_chunks": sum(c.get("content_type") == "table" for c in new),
            }
        )

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"papers: {len(rows)}")
    if rows:
        old_counts = [r["legacy_exact_chunks"] for r in rows]
        new_counts = [r["v2_chunks"] for r in rows]
        print(f"legacy_exact chunks: mean={mean(old_counts):.2f}, median={median(old_counts):.0f}")
        print(f"v2 chunks:           mean={mean(new_counts):.2f}, median={median(new_counts):.0f}")
        print("largest differences:")
        for row in sorted(rows, key=lambda x: abs(x["difference"]), reverse=True)[:10]:
            print(
                f"  {row['paper_id']}: legacy={row['legacy_exact_chunks']} "
                f"v2={row['v2_chunks']} diff={row['difference']:+d}"
            )
    print(f"csv: {out.resolve()}")


if __name__ == "__main__":
    main()
