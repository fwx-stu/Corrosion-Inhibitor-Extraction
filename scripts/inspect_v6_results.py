#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.eval.integrity import evaluation_integrity, run_integrity


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect V6 matrix completion and metric ranking.")
    ap.add_argument(
        "--run-root",
        default=str(ROOT / "output/exp_v2/corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6"),
    )
    args = ap.parse_args()
    run_root = Path(args.run_root)
    manifest_path = run_root / "matrix_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])

    complete = 0
    partial = 0
    evaluated = 0
    coverage_sum = 0.0
    metadata_runs = 0
    for job in jobs:
        run_dir = Path(job["run_dir"])
        status = run_integrity(run_dir)
        if status["meta_exists"]:
            metadata_runs += 1
            coverage_sum += status["document_coverage"]
        complete += int(status["complete"])
        partial += int(status["meta_exists"] and not status["complete"])
        evaluated += int(evaluation_integrity(run_dir)[0])

    print(f"jobs expected: {len(jobs)}")
    print(f"runs structurally complete: {complete}")
    print(f"runs partial: {partial}")
    print(f"runs currently evaluated: {evaluated}")
    if metadata_runs:
        print(f"mean document coverage: {coverage_sum / metadata_runs:.4f}")

    path = run_root / "aggregate_metrics.csv"
    if not path.exists():
        print(f"aggregate not available: {path}")
        return
    print("\nmodel-strategy ranking (composite is supporting evidence):")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row.get("field_micro_f1_mean") or 0), reverse=True)
    for row in rows:
        print(
            f"{row.get('model_key',''):14s} {row.get('strategy',''):22s} "
            f"n={row.get('n_runs','')} "
            f"field_f1={float(row.get('field_micro_f1_mean') or 0):.4f} "
            f"record_f1={float(row.get('record_f1_mean') or 0):.4f} "
            f"null_acc={float(row.get('null_accuracy_mean') or 0):.4f} "
            f"composite={float(row.get('composite_score_mean') or 0):.4f} "
            f"coverage={float(row.get('document_coverage_mean') or 0):.4f}"
        )


if __name__ == "__main__":
    main()
