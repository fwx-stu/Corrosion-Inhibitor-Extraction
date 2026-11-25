#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.evidence_diagnostics import diagnose_run_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute evidence recall diagnostics from evidence.jsonl and gold records.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_root = Path(args.run_root) if args.run_root else cfg.run_root
    diagnose_run_root(cfg.gold_path, run_root)
    print(f"[diagnose] done -> {run_root / 'evidence_diagnostics.csv'}")

if __name__ == "__main__":
    main()
