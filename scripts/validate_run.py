#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.runner import validate_run_directory


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate one run directory.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--expected-docs", type=int, default=None)
    ap.add_argument("--enforce-quality", action="store_true")
    args = ap.parse_args()

    valid, details = validate_run_directory(
        Path(args.run_dir),
        enforce_quality=args.enforce_quality,
    )
    if args.expected_docs is not None and details.get("docs_total") != args.expected_docs:
        valid = False
        details["expected_docs"] = args.expected_docs
        details["reason"] = "run_meta docs_total differs from --expected-docs"
    print(json.dumps(details, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(2)
    print("[validate-run] complete")


if __name__ == "__main__":
    main()
