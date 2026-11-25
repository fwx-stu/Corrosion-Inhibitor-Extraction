from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

from .integrity import evaluation_integrity, read_json, run_integrity
from ..io_utils import read_csv, write_csv, write_json

SUMMARY_NUMERIC = [
    "record_precision", "record_recall", "record_f1",
    "field_micro_precision", "field_micro_recall", "field_micro_f1",
    "null_accuracy", "composite_score", "pred_record_total", "gold_record_total",
    "document_coverage", "docs_failed", "quality_gate_pass", "run_complete",
    "parse_error_rate", "empty_prediction_rate", "output_truncation_rate",
]
PER_FIELD_NUMERIC = ["precision", "recall", "f1", "tp", "fp", "fn", "pred_count", "gold_count"]


def discover_run_dirs(run_root: Path) -> List[Path]:
    manifest = run_root / "matrix_manifest.json"
    if manifest.exists():
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        paths: List[Path] = []
        for job in obj.get("jobs", []):
            stored = Path(str(job.get("run_dir") or ""))
            if stored.exists():
                paths.append(stored)
                continue
            paths.append(
                run_root
                / str(job["model_key"])
                / str(job["strategy"])
                / str(job["prompt_variant"])
                / f"r{int(job['seed'])}"
            )
        return paths
    return sorted({p.parent.parent for p in run_root.glob("*/*/*/r*/evaluation/summary_metrics.csv")})


def parse_run_path(path: Path, run_root: Path) -> Dict[str, str]:
    rel = path.relative_to(run_root)
    return {
        "model_key": rel.parts[0],
        "strategy": rel.parts[1],
        "prompt_variant": rel.parts[2],
        "repeat": rel.parts[3],
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_rows(
    rows: List[Dict[str, Any]],
    group_fields: List[str],
    numeric_cols: List[str],
    glm_baseline_f1: float | None = None,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in group_fields)].append(row)

    output: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        result: Dict[str, Any] = {field: key[index] for index, field in enumerate(group_fields)}
        result["n_runs"] = len(items)
        for column in numeric_cols:
            values = [_safe_float(item.get(column, "")) for item in items]
            values = [value for value in values if value is not None]
            if values:
                result[f"{column}_mean"] = mean(values)
                result[f"{column}_std"] = stdev(values) if len(values) >= 2 else 0.0
        if glm_baseline_f1 is not None and "field_micro_f1_mean" in result:
            result["gap_to_glm_field_micro_f1"] = glm_baseline_f1 - result["field_micro_f1_mean"]
        output.append(result)
    return output


def _normalized_composite_weights(weights: Dict[str, float] | None) -> Dict[str, float]:
    default = {
        "field_micro_f1": 1.0 / 3.0,
        "record_f1": 1.0 / 3.0,
        "null_accuracy": 1.0 / 3.0,
    }
    if not weights:
        return default
    cleaned: Dict[str, float] = {}
    for key in default:
        try:
            value = float(weights.get(key, 0.0))
        except Exception:
            value = 0.0
        if value > 0:
            cleaned[key] = value
    total = sum(cleaned.values())
    if total <= 0:
        return default
    return {key: value / total for key, value in cleaned.items()}


def _add_composite_score(row: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    values: Dict[str, float] = {}
    for metric in weights:
        value = _safe_float(row.get(metric, ""))
        if value is None:
            return dict(row)
        values[metric] = value
    output = dict(row)
    output["composite_score"] = sum(values[key] * weights[key] for key in weights)
    return output


def _quality_columns(run_dir: Path) -> Dict[str, Any]:
    status = run_integrity(run_dir)
    meta = read_json(run_dir / "run_meta.json")
    quality = meta.get("quality_metrics") or {}
    return {
        "run_complete": int(bool(status["complete"])),
        "quality_gate_pass": int(bool(status["quality_gate_pass"])),
        "document_coverage": status["document_coverage"],
        "docs_total": status["docs_total"],
        "docs_ok": status["docs_ok"],
        "docs_failed": status["docs_failed"],
        "parse_error_rate": quality.get("parse_error_rate"),
        "empty_prediction_rate": quality.get("empty_prediction_rate"),
        "output_truncation_rate": quality.get("output_truncation_rate"),
    }


def aggregate(
    run_root: str | Path,
    glm_baseline_f1: float | None = None,
    group_by: str = "model,strategy",
    composite_weights: Dict[str, float] | None = None,
    expected_evaluation: Dict[str, Any] | None = None,
) -> None:
    run_root = Path(run_root)
    weights = _normalized_composite_weights(composite_weights)
    exclusions: List[Dict[str, Any]] = []
    valid_runs: List[Path] = []
    for run_dir in discover_run_dirs(run_root):
        valid, reason, meta = evaluation_integrity(run_dir, expected_evaluation)
        if valid:
            valid_runs.append(run_dir)
        else:
            exclusions.append({"run_dir": str(run_dir), "reason": reason, "evaluation_meta": meta})
    evaluation_exclusions = list(exclusions)
    per_field_exclusions: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []
    for run_dir in valid_runs:
        summary = run_dir / "evaluation" / "summary_metrics.csv"
        meta = parse_run_path(summary, run_root)
        quality = _quality_columns(run_dir)
        for row in read_csv(summary):
            rows.append(_add_composite_score({**row, **meta, **quality}, weights))
    write_csv(run_root / "all_run_summary_metrics.csv", rows)

    alias = {"model": "model_key", "prompt": "prompt_variant", "seed": "repeat"}
    group_fields = [alias.get(value.strip(), value.strip()) for value in group_by.split(",") if value.strip()]
    if not group_fields:
        group_fields = ["model_key", "strategy"]
    write_csv(
        run_root / "aggregate_metrics.csv",
        _aggregate_rows(rows, group_fields, SUMMARY_NUMERIC, glm_baseline_f1),
    )
    write_csv(
        run_root / "aggregate_metrics_by_prompt.csv",
        _aggregate_rows(rows, ["model_key", "strategy", "prompt_variant"], SUMMARY_NUMERIC, glm_baseline_f1),
    )

    per_field_rows: List[Dict[str, Any]] = []
    for run_dir in valid_runs:
        path = run_dir / "evaluation" / "per_field_metrics.csv"
        if not path.exists():
            per_field_exclusions.append({"run_dir": str(run_dir), "reason": "missing per_field_metrics.csv"})
            continue
        meta = parse_run_path(path, run_root)
        quality = _quality_columns(run_dir)
        for row in read_csv(path):
            per_field_rows.append({**row, **meta, **quality})
    write_csv(run_root / "all_run_per_field_metrics.csv", per_field_rows)

    per_field_group = list(group_fields)
    if "field" not in per_field_group:
        per_field_group.append("field")
    write_csv(
        run_root / "aggregate_per_field_metrics.csv",
        _aggregate_rows(per_field_rows, per_field_group, PER_FIELD_NUMERIC),
    )
    write_csv(
        run_root / "aggregate_per_field_metrics_by_prompt.csv",
        _aggregate_rows(per_field_rows, ["model_key", "strategy", "prompt_variant", "field"], PER_FIELD_NUMERIC),
    )
    write_json(run_root / "aggregation_exclusions.json", {
        "valid_evaluated_runs": len(valid_runs),
        "excluded_runs": len(evaluation_exclusions),
        "evaluation_exclusions": evaluation_exclusions,
        "per_field_exclusions": per_field_exclusions,
    })
