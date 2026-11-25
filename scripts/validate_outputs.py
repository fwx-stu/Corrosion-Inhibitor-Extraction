#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.io_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate run output files and JSONL readability.")
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    problems = []
    for run_dir in sorted(run_root.glob("*/*/*/r*")):
        for name in ["predictions.jsonl", "evidence.jsonl", "raw_outputs.jsonl", "run_meta.json"]:
            p = run_dir / name
            if not p.exists():
                problems.append((str(run_dir), f"missing {name}"))
        p = run_dir / "predictions.jsonl"
        if p.exists():
            try:
                n = sum(1 for _ in read_jsonl(p))
                if n == 0:
                    problems.append((str(run_dir), "empty predictions.jsonl"))
            except Exception as e:
                problems.append((str(run_dir), f"bad predictions.jsonl: {e}"))
    if problems:
        print("[validate] problems:")
        for a, b in problems:
            print(f"  {a}: {b}")
        raise SystemExit(1)
    print("[validate] OK")

if __name__ == "__main__":
    main()
