#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.io_utils import ensure_dir, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reproducible paper-id splits from a gold JSONL file.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25, help="Fraction or absolute count if >=1.")
    args = ap.parse_args()
    ids = []
    seen = set()
    for obj in read_jsonl(args.gold):
        pid = obj.get("paper_id") or obj.get("id") or obj.get("source_file")
        if pid and pid not in seen:
            ids.append(pid); seen.add(pid)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_test = int(args.test_size) if args.test_size >= 1 else max(1, round(len(ids) * args.test_size))
    split = {"seed": args.seed, "train_ids": sorted(ids[n_test:]), "test_ids": sorted(ids[:n_test])}
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[splits] total={len(ids)} train={len(split['train_ids'])} test={len(split['test_ids'])} -> {out}")

if __name__ == "__main__":
    main()
