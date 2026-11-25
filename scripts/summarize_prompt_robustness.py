#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata, read_json, run_integrity
from corrosion_ie.io_utils import read_csv, write_csv, write_text

METRICS = [
    "field_micro_precision", "field_micro_recall", "field_micro_f1",
    "record_precision", "record_recall", "record_f1", "null_accuracy",
    "composite_score",
]


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_conditions(run_root: Path, composite_weights: dict[str, float], expected_evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = run_root / "matrix_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"matrix manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        run_dir = Path(job["run_dir"])
        summary = run_dir / "evaluation" / "summary_metrics.csv"
        current, reason, _ = evaluation_integrity(run_dir, expected_evaluation)
        if not current:
            print(f"[prompt-summary-warn] invalid or stale evaluation, skipped: {run_dir}: {reason}")
            continue
        content = read_csv(summary)
        if len(content) != 1:
            print(f"[prompt-summary-warn] expected one summary row, skipped: {summary}; got {len(content)}")
            continue
        row = {
            **content[0],
            "model_key": job["model_key"],
            "strategy": job["strategy"],
            "prompt_variant": job["prompt_variant"],
            "repeat": f"r{int(job['seed'])}",
        }
        values = {metric: safe_float(row.get(metric)) for metric in composite_weights}
        if all(value is not None for value in values.values()):
            row["composite_score"] = sum(float(values[metric]) * weight for metric, weight in composite_weights.items())
        status = run_integrity(run_dir)
        meta = read_json(run_dir / "run_meta.json")
        quality = meta.get("quality_metrics") or {}
        row.update({
            "run_complete": status["complete"],
            "quality_gate_pass": status["quality_gate_pass"],
            "document_coverage": status["document_coverage"],
            "docs_failed": status["docs_failed"],
            "parse_error_rate": quality.get("parse_error_rate"),
            "output_truncation_rate": quality.get("output_truncation_rate"),
        })
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], expected_prompts: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_key"]), str(row["strategy"]))].append(row)
    expected_set = set(expected_prompts)
    output: list[dict[str, Any]] = []
    for (model, strategy), items in sorted(groups.items()):
        # Keep at most one deterministic result per prompt. A partial group is
        # summarized instead of aborting all other model-strategy groups.
        by_prompt: dict[str, dict[str, Any]] = {}
        for item in items:
            prompt = str(item["prompt_variant"])
            if prompt not in by_prompt:
                by_prompt[prompt] = item
        prompts = sorted(by_prompt)
        missing = sorted(expected_set - set(prompts))
        extra = sorted(set(prompts) - expected_set)
        available = [by_prompt[p] for p in expected_prompts if p in by_prompt]
        if not available:
            continue

        out: dict[str, Any] = {
            "model_key": model,
            "strategy": strategy,
            "n_prompt_conditions": len(available),
            "expected_prompt_conditions": len(expected_prompts),
            "prompt_variants": "|".join(str(item["prompt_variant"]) for item in available),
            "missing_prompt_variants": "|".join(missing),
            "extra_prompt_variants": "|".join(extra),
            "complete_prompt_set": not missing and not extra,
            "aggregation_unit": "prompt_policy",
        }
        for metric in METRICS:
            typed = [
                (str(item["prompt_variant"]), float(value))
                for item in available
                if (value := safe_float(item.get(metric))) is not None
            ]
            if not typed:
                continue
            values = [v for _, v in typed]
            best_prompt, best_value = max(typed, key=lambda x: x[1])
            worst_prompt, worst_value = min(typed, key=lambda x: x[1])
            out[f"{metric}_prompt_mean"] = mean(values)
            out[f"{metric}_prompt_std"] = stdev(values) if len(values) >= 2 else 0.0
            out[f"{metric}_prompt_min"] = worst_value
            out[f"{metric}_prompt_max"] = best_value
            out[f"{metric}_prompt_range"] = best_value - worst_value
            out[f"{metric}_best_prompt"] = best_prompt
            out[f"{metric}_worst_prompt"] = worst_prompt
        output.append(out)
    return output


def make_markdown(run_root: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Three-policy robustness report", "", f"Run root: `{run_root}`", "",
        "Each model-strategy group contains recall, balanced, and precision under deterministic decoding.",
        "The policy SD measures prompt-policy sensitivity; it is not a 95% confidence interval.", "",
        "## Policy-condition results", "",
        "| Model | Strategy | Policy | Field Micro F1 | Record F1 | Null Accuracy | Composite |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["model_key"], r["strategy"], r["prompt_variant"])):
        lines.append(
            f"| {row['model_key']} | {row['strategy']} | {row['prompt_variant']} | "
            f"{float(row.get('field_micro_f1', 0)):.4f} | {float(row.get('record_f1', 0)):.4f} | "
            f"{float(row.get('null_accuracy', 0)):.4f} | {float(row.get('composite_score', 0)):.4f} |"
        )
    lines += ["", "## Mean and policy sensitivity", "",
              "| Model | Strategy | Policies | Complete | Field F1 mean | Field F1 SD | Record F1 mean | Null accuracy mean | Composite mean |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['model_key']} | {row['strategy']} | {row['n_prompt_conditions']} | "
            f"{row['complete_prompt_set']} | {float(row.get('field_micro_f1_prompt_mean', 0)):.4f} | "
            f"{float(row.get('field_micro_f1_prompt_std', 0)):.4f} | "
            f"{float(row.get('record_f1_prompt_mean', 0)):.4f} | "
            f"{float(row.get('null_accuracy_prompt_mean', 0)):.4f} | "
            f"{float(row.get('composite_score_prompt_mean', 0)):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Average recall/balanced/precision within each model-strategy group.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_root = Path(args.run_root) if args.run_root else cfg.run_root
    expected_prompts = list(cfg.prompt_variants)
    if expected_prompts != ["recall", "balanced", "precision"]:
        raise SystemExit(f"[summary-error] expected policies recall,balanced,precision; got {expected_prompts}")
    composite_cfg = cfg.evaluation.get("composite_score", {}) or {}
    raw_weights = composite_cfg.get("weights") or {
        "field_micro_f1": 1.0 / 3.0,
        "record_f1": 1.0 / 3.0,
        "null_accuracy": 1.0 / 3.0,
    }
    positive = {key: float(value) for key, value in raw_weights.items() if float(value) > 0}
    total = sum(positive.values()) or 1.0
    composite_weights = {key: value / total for key, value in positive.items()}
    rows = load_conditions(run_root, composite_weights, expected_evaluation_metadata(cfg))
    summary = summarize(rows, expected_prompts)
    write_csv(run_root / "prompt_condition_metrics.csv", rows)
    write_csv(run_root / "prompt_robustness_summary.csv", summary)
    write_text(run_root / "prompt_robustness_report.md", make_markdown(run_root, rows, summary))
    print(f"[prompt-summary] conditions={len(rows)} groups={len(summary)}")


if __name__ == "__main__":
    main()
