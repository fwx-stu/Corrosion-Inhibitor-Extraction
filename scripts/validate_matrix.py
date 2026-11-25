#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.runner import validate_run_directory


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate every run listed in matrix_manifest.json.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--expected-docs", type=int, default=None)
    ap.add_argument(
        "--enforce-quality",
        action="store_true",
        help="Also require run_meta.json quality_gate_pass=true.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.run_root) if args.run_root else cfg.run_root
    manifest = json.loads((root / "matrix_manifest.json").read_text(encoding="utf-8"))
    expected = args.expected_docs if args.expected_docs is not None else cfg.expected_paper_count
    checked = []
    failures = []
    for job in manifest.get("jobs", []):
        valid, details = validate_run_directory(
            Path(job["run_dir"]),
            enforce_quality=args.enforce_quality,
        )
        if expected is not None and details.get("docs_total") != expected:
            valid = False
            details["reason"] = "docs_total differs from expected count"
        details.update({**job, "expected_docs": expected, "valid": valid})
        checked.append(details)
        if not valid:
            failures.append(details)
    result = {
        "jobs_expected": manifest.get("jobs_expected"),
        "jobs_valid": len(checked) - len(failures),
        "jobs_failed": len(failures),
        "expected_docs_per_job": expected,
        "quality_enforced": args.enforce_quality,
        "runs": checked,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)
    print(f"[validate-matrix] complete: {len(checked)}/{len(checked)} runs valid")


if __name__ == "__main__":
    main()
