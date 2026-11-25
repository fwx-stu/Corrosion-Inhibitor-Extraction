#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata, read_json, run_integrity
from corrosion_ie.io_utils import read_csv, write_text


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value or "")


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the complete markdown experiment report.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    root = Path(args.run_root)
    expected_evaluation = expected_evaluation_metadata(load_config(args.config)) if args.config else None
    lines = ["# Experiment report", "", f"Run root: `{root}`", ""]

    aggregate_path = root / "aggregate_metrics.csv"
    aggregate = read_csv(aggregate_path) if aggregate_path.exists() else []
    lines += [
        "## Aggregate metrics", "",
        "Field Micro F1 and Record F1 are the primary extraction metrics. Null Accuracy and the equal-weight composite are supporting diagnostics rather than stand-alone ranking criteria.", "",
    ]
    aggregate_columns = [
        "model_key", "strategy", "n_runs",
        "field_micro_precision_mean", "field_micro_recall_mean", "field_micro_f1_mean", "field_micro_f1_std",
        "record_precision_mean", "record_recall_mean", "record_f1_mean", "record_f1_std",
        "null_accuracy_mean", "composite_score_mean", "composite_score_std",
        "document_coverage_mean", "run_complete_mean", "quality_gate_pass_mean",
    ]
    lines += markdown_table(aggregate, aggregate_columns) if aggregate else ["Aggregate metrics are not available."]

    manifest_path = root / "matrix_manifest.json"
    expected_jobs = 0
    structural_complete_jobs = 0
    partial_jobs = 0
    current_evaluations = 0
    stale_or_missing_evaluations = 0
    jobs: List[Dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = list(manifest.get("jobs", []))
        expected_jobs = int(manifest.get("jobs_expected", len(jobs)))
        for job in jobs:
            run_dir = Path(job["run_dir"])
            status = run_integrity(run_dir)
            structural_complete_jobs += int(status["complete"])
            partial_jobs += int(status["meta_exists"] and not status["complete"])
            current, _, _ = evaluation_integrity(run_dir, expected_evaluation)
            current_evaluations += int(current)
            stale_or_missing_evaluations += int(not current)

    matrix_failures_path = root / "matrix_failures.jsonl"
    failed_jobs = 0
    if matrix_failures_path.exists():
        failed_jobs = sum(1 for line in matrix_failures_path.read_text(encoding="utf-8").splitlines() if line.strip())
    lines += [
        "", "## Matrix completion", "",
        f"- Expected jobs: {expected_jobs}",
        f"- Structurally complete jobs: {structural_complete_jobs}",
        f"- Partial jobs with run metadata: {partial_jobs}",
        f"- Current evaluated jobs: {current_evaluations}",
        f"- Missing or stale evaluations: {stale_or_missing_evaluations}",
        f"- Logged matrix-level failures: {failed_jobs}",
    ]

    quality_rows: List[Dict[str, Any]] = []
    for job in jobs:
        run_dir = Path(job["run_dir"])
        meta = read_json(run_dir / "run_meta.json")
        if not meta:
            continue
        metrics = meta.get("quality_metrics") or {}
        evaluation_current, evaluation_reason, _ = evaluation_integrity(run_dir, expected_evaluation)
        quality_rows.append({
            "model": job["model_key"],
            "strategy": job["strategy"],
            "prompt_variant": job["prompt_variant"],
            "repeat": f"r{int(job['seed'])}",
            "docs_total": meta.get("docs_total"),
            "docs_ok": meta.get("docs_ok"),
            "docs_failed": meta.get("docs_failed"),
            "document_coverage": metrics.get("document_coverage", meta.get("document_coverage")),
            "complete": meta.get("complete"),
            "quality_gate": meta.get("quality_gate_pass"),
            "parse_error_rate": metrics.get("parse_error_rate"),
            "empty_prediction_rate": metrics.get("empty_prediction_rate"),
            "input_over_budget": metrics.get("input_over_budget_count"),
            "input_truncations": metrics.get("input_truncation_count"),
            "output_truncation_rate": metrics.get("output_truncation_rate"),
            "evaluation_current": evaluation_current,
            "evaluation_status": evaluation_reason,
        })
    lines += ["", "## Run quality", ""]
    quality_columns = [
        "model", "strategy", "prompt_variant", "repeat",
        "docs_total", "docs_ok", "docs_failed", "document_coverage", "complete", "quality_gate",
        "parse_error_rate", "empty_prediction_rate", "input_over_budget", "input_truncations",
        "output_truncation_rate", "evaluation_current", "evaluation_status",
    ]
    lines += markdown_table(quality_rows, quality_columns) if quality_rows else ["Quality metadata are not available."]

    ci_path = root / "condition_metrics_with_bootstrap_ci.csv"
    if ci_path.exists():
        ci_rows = read_csv(ci_path)
        lines += ["", "## Bootstrap confidence intervals", ""]
        ci_columns = [
            "model_key", "strategy", "prompt_variant", "repeat", "paper_count",
            "field_micro_f1", "field_micro_f1_ci_low", "field_micro_f1_ci_high",
            "record_f1", "record_f1_ci_low", "record_f1_ci_high",
            "null_accuracy", "null_accuracy_ci_low", "null_accuracy_ci_high",
        ]
        lines += markdown_table(ci_rows, ci_columns) if ci_rows else ["No complete condition was available for bootstrap analysis."]

    comparisons_path = root / "paired_statistical_comparisons.csv"
    if comparisons_path.exists():
        comparison_rows = read_csv(comparisons_path)
        lines += ["", "## Paired comparisons", ""]
        comparison_columns = [
            "comparison_family", "comparison", "model_or_strategy", "prompt_variant", "repeat",
            "metric", "value_low", "value_high", "difference_high_minus_low",
            "ci_low", "ci_high", "p_value", "p_holm",
        ]
        lines += markdown_table(comparison_rows, comparison_columns) if comparison_rows else ["No valid paired comparison was available."]

    lines += ["", "## Additional diagnostics", ""]
    for name in [
        "aggregation_exclusions.json", "truncation_summary.csv", "evidence_consistency.json",
        "document_features.csv", "stratified_condition_metrics.csv", "stratified_strategy_differences.csv",
        "evidence_diagnostics.csv", "statistical_analysis_meta.json", "v6_pipeline_status.tsv",
    ]:
        if (root / name).exists():
            lines.append(f"- `{name}`")

    output = root / "final_report.md"
    write_text(output, "\n".join(lines) + "\n")
    print(f"[report] {output}")


if __name__ == "__main__":
    main()
