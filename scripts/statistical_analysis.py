#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata

STRATEGY_PAIRS = [
    ("q0_safe_full", "q1_budgeted_rag", "Q1-Q0"),
    ("q1_budgeted_rag", "q2_table_aware_rag", "Q2-Q1"),
    ("q0_safe_full", "q2_table_aware_rag", "Q2-Q0"),
    # Reviewer controls.  These are evaluated against the already completed
    # Q1/Q2 runs, so the original model generations never need to be repeated.
    ("b0_uniform_token_matched", "q1_budgeted_rag", "Q1-B0_uniform"),
    ("b1_section_aware_token_matched", "q2_table_aware_rag", "Q2-B1_section"),
    ("q1_budgeted_rag", "a1_q1_table_budget", "A1_table-Q1"),
    ("q1_budgeted_rag", "a2_q1_condition_anchors", "A2_anchor-Q1"),
    ("a1_q1_table_budget", "q2_table_aware_rag", "Q2-A1_table"),
    ("a2_q1_condition_anchors", "q2_table_aware_rag", "Q2-A2_anchor"),
]
MODEL_PAIRS = [
    ("qwen2_5_3b", "qwen2_5_7b", "7B-3B"),
    ("qwen2_5_7b", "qwen2_5_14b", "14B-7B"),
    ("qwen2_5_3b", "qwen2_5_14b", "14B-3B"),
]
METRICS = ("record_f1", "field_micro_f1", "null_accuracy")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def prf(tp: float, pred: float, gold: float) -> Tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


@dataclass(frozen=True)
class Condition:
    model: str
    strategy: str
    prompt: str
    repeat: str
    run_dir: Path

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return self.model, self.strategy, self.prompt, self.repeat


def discover_conditions(run_root: Path) -> List[Condition]:
    manifest = run_root / "matrix_manifest.json"
    conditions: List[Condition] = []
    if manifest.exists():
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        for job in obj.get("jobs", []):
            stored = Path(str(job.get("run_dir") or ""))
            if stored.exists():
                run_dir = stored
            else:
                run_dir = (
                    run_root
                    / str(job["model_key"])
                    / str(job["strategy"])
                    / str(job["prompt_variant"])
                    / f"r{int(job['seed'])}"
                )
            conditions.append(Condition(
                str(job["model_key"]), str(job["strategy"]), str(job["prompt_variant"]),
                f"r{int(job['seed'])}", run_dir,
            ))
        return conditions
    for path in sorted(run_root.glob("*/*/*/r*/evaluation/per_paper_metrics.csv")):
        rel = path.relative_to(run_root)
        conditions.append(Condition(rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3], path.parent.parent))
    return conditions


def load_sufficient(condition: Condition) -> Dict[str, np.ndarray]:
    paper_path = condition.run_dir / "evaluation" / "per_paper_metrics.csv"
    field_path = condition.run_dir / "evaluation" / "per_paper_field_metrics.csv"
    if not paper_path.exists() or not field_path.exists():
        raise FileNotFoundError(f"missing evaluation files for {condition.run_dir}")

    papers = read_csv(paper_path)
    fields = read_csv(field_path)
    paper_ids = sorted({row["paper_id"] for row in papers})
    index = {pid: i for i, pid in enumerate(paper_ids)}
    n = len(paper_ids)
    out = {
        "paper_ids": np.asarray(paper_ids, dtype=object),
        "record_tp": np.zeros(n),
        "record_pred": np.zeros(n),
        "record_gold": np.zeros(n),
        "field_tp": np.zeros(n),
        "field_pred": np.zeros(n),
        "field_gold": np.zeros(n),
        "null_correct": np.zeros(n),
        "null_total": np.zeros(n),
    }
    for row in papers:
        i = index[row["paper_id"]]
        out["record_tp"][i] = f(row.get("record_tp"))
        out["record_pred"][i] = f(row.get("pred_record_count"))
        out["record_gold"][i] = f(row.get("gold_record_count"))
        out["null_correct"][i] = f(row.get("correct_null_count"))
        out["null_total"][i] = f(row.get("gold_null_count_matched"))
    for row in fields:
        i = index[row["paper_id"]]
        out["field_tp"][i] += f(row.get("tp"))
        out["field_pred"][i] += f(row.get("pred_count"))
        out["field_gold"][i] += f(row.get("gold_count"))
    return out


def metric(suff: Dict[str, np.ndarray], name: str, indices: np.ndarray | None = None) -> float:
    idx = indices if indices is not None else slice(None)
    if name == "record_f1":
        return prf(suff["record_tp"][idx].sum(), suff["record_pred"][idx].sum(), suff["record_gold"][idx].sum())[2]
    if name == "field_micro_f1":
        return prf(suff["field_tp"][idx].sum(), suff["field_pred"][idx].sum(), suff["field_gold"][idx].sum())[2]
    if name == "null_accuracy":
        total = suff["null_total"][idx].sum()
        return float(suff["null_correct"][idx].sum() / total) if total else float("nan")
    raise KeyError(name)


def align(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    a_ids = list(a["paper_ids"])
    b_ids = list(b["paper_ids"])
    if set(a_ids) != set(b_ids):
        only_a = sorted(set(a_ids) - set(b_ids))
        only_b = sorted(set(b_ids) - set(a_ids))
        raise ValueError(
            f"paired comparison requires identical paper IDs; only_low={only_a[:10]}, only_high={only_b[:10]}"
        )
    common = sorted(set(a_ids))
    if not common:
        raise ValueError("no paper IDs for paired comparison")
    ai = {pid: i for i, pid in enumerate(a_ids)}
    bi = {pid: i for i, pid in enumerate(b_ids)}

    def take(src: Dict[str, np.ndarray], mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
        indices = np.asarray([mapping[pid] for pid in common], dtype=int)
        return {key: (np.asarray(common, dtype=object) if key == "paper_ids" else value[indices]) for key, value in src.items()}
    return take(a, ai), take(b, bi)


def bootstrap_condition(suff: Dict[str, np.ndarray], metric_name: str, iterations: int, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(suff["paper_ids"])
    vals = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        vals[i] = metric(suff, metric_name, idx)
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (float("nan"), float("nan"))




def align_many(items: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    if not items:
        return []
    common = set(items[0]["paper_ids"])
    for item in items[1:]:
        common &= set(item["paper_ids"])
    ordered = sorted(common)
    if not ordered:
        raise ValueError("no common paper IDs across prompt conditions")
    aligned: List[Dict[str, np.ndarray]] = []
    for item in items:
        mapping = {pid: index for index, pid in enumerate(item["paper_ids"])}
        indices = np.asarray([mapping[pid] for pid in ordered], dtype=int)
        aligned.append({
            key: (np.asarray(ordered, dtype=object) if key == "paper_ids" else value[indices])
            for key, value in item.items()
        })
    return aligned


def bootstrap_prompt_mean(
    items: Sequence[Dict[str, np.ndarray]],
    metric_name: str,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, int]:
    aligned = align_many(items)
    n = len(aligned[0]["paper_ids"])
    estimate = float(np.nanmean([metric(item, metric_name) for item in aligned]))
    values = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, n, size=n)
        values[index] = float(np.nanmean([metric(item, metric_name, sample) for item in aligned]))
    values = values[np.isfinite(values)]
    if not len(values):
        return estimate, float("nan"), float("nan"), n
    return estimate, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), n


def bootstrap_difference(
    low: Dict[str, np.ndarray], high: Dict[str, np.ndarray], metric_name: str,
    iterations: int, rng: np.random.Generator,
) -> Tuple[float, float]:
    n = len(low["paper_ids"])
    vals = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        vals[i] = metric(high, metric_name, idx) - metric(low, metric_name, idx)
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (float("nan"), float("nan"))


def swapped_suff(low: Dict[str, np.ndarray], high: Dict[str, np.ndarray], mask: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    a: Dict[str, np.ndarray] = {"paper_ids": low["paper_ids"]}
    b: Dict[str, np.ndarray] = {"paper_ids": high["paper_ids"]}
    for key in low:
        if key == "paper_ids":
            continue
        a[key] = np.where(mask, high[key], low[key])
        b[key] = np.where(mask, low[key], high[key])
    return a, b


def paired_permutation_p(
    low: Dict[str, np.ndarray], high: Dict[str, np.ndarray], metric_name: str,
    iterations: int, rng: np.random.Generator,
) -> float:
    observed = metric(high, metric_name) - metric(low, metric_name)
    if not math.isfinite(observed):
        return float("nan")
    n = len(low["paper_ids"])
    extreme = 0
    for _ in range(iterations):
        mask = rng.random(n) < 0.5
        perm_low, perm_high = swapped_suff(low, high, mask)
        value = metric(perm_high, metric_name) - metric(perm_low, metric_name)
        extreme += int(math.isfinite(value) and abs(value) >= abs(observed) - 1e-15)
    return (extreme + 1) / (iterations + 1)


def holm_adjust(rows: List[Dict[str, Any]]) -> None:
    """Holm-correct separately for strategy and model comparison families."""
    families = sorted({str(row.get("comparison_family")) for row in rows})
    for family in families:
        for metric_name in METRICS:
            indices = [
                i for i, row in enumerate(rows)
                if row.get("comparison_family") == family
                and row["metric"] == metric_name
                and math.isfinite(float(row["p_value"]))
            ]
            ranked = sorted(indices, key=lambda i: float(rows[i]["p_value"]))
            running = 0.0
            m = len(ranked)
            for rank, idx in enumerate(ranked):
                adjusted = min(1.0, (m - rank) * float(rows[idx]["p_value"]))
                running = max(running, adjusted)
                rows[idx]["p_holm"] = running
            for i, row in enumerate(rows):
                if (
                    row.get("comparison_family") == family
                    and row["metric"] == metric_name
                    and "p_holm" not in row
                ):
                    row["p_holm"] = ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-level bootstrap, paired permutation tests, and Holm correction.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    ap.add_argument(
        "--additional-run-root",
        action="append",
        default=[],
        help="Additional completed run root; repeat this option as needed.",
    )
    ap.add_argument("--bootstrap-iterations", type=int, default=None)
    ap.add_argument("--permutation-iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_root = Path(args.run_root) if args.run_root else cfg.run_root
    eval_cfg = cfg.evaluation
    b_iter = int(args.bootstrap_iterations or eval_cfg.get("bootstrap_iterations", 10000))
    p_iter = int(args.permutation_iterations or eval_cfg.get("permutation_iterations", 10000))
    seed = int(args.seed or eval_cfg.get("bootstrap_seed", 20260710))
    rng = np.random.default_rng(seed)

    expected_evaluation = expected_evaluation_metadata(cfg)
    expected_evaluation.pop("docs_dir", None)
    run_roots = [run_root] + [Path(value) for value in args.additional_run_root]
    discovered_conditions = [
        condition
        for root in run_roots
        for condition in discover_conditions(root)
    ]
    conditions: List[Condition] = []
    data: Dict[Tuple[str, str, str, str], Dict[str, np.ndarray]] = {}
    skipped_conditions: List[Dict[str, Any]] = []
    seen_condition_keys: set[Tuple[str, str, str, str]] = set()
    for condition in discovered_conditions:
        if condition.key in seen_condition_keys:
            skipped_conditions.append({
                "condition": "|".join(condition.key),
                "run_dir": str(condition.run_dir),
                "reason": "duplicate condition key across run roots",
            })
            continue
        seen_condition_keys.add(condition.key)
        current, reason, _ = evaluation_integrity(condition.run_dir, expected_evaluation)
        if not current:
            skipped_conditions.append({
                "condition": "|".join(condition.key),
                "run_dir": str(condition.run_dir),
                "reason": reason,
            })
            print(f"[statistics-warn] skipped {condition.run_dir}: {reason}")
            continue
        try:
            data[condition.key] = load_sufficient(condition)
            conditions.append(condition)
        except Exception as exc:
            skipped_conditions.append({
                "condition": "|".join(condition.key),
                "run_dir": str(condition.run_dir),
                "reason": repr(exc),
            })
            print(f"[statistics-warn] skipped {condition.run_dir}: {exc!r}")

    condition_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        suff = data[condition.key]
        row: Dict[str, Any] = {
            "model_key": condition.model,
            "strategy": condition.strategy,
            "prompt_variant": condition.prompt,
            "repeat": condition.repeat,
            "paper_count": len(suff["paper_ids"]),
        }
        for metric_name in METRICS:
            value = metric(suff, metric_name)
            lo, hi = bootstrap_condition(suff, metric_name, b_iter, rng)
            row[metric_name] = value
            row[f"{metric_name}_ci_low"] = lo
            row[f"{metric_name}_ci_high"] = hi
        condition_rows.append(row)

    comparisons: List[Dict[str, Any]] = []
    skipped_comparisons: List[Dict[str, Any]] = []

    def add_comparison(low_key: Tuple[str, str, str, str], high_key: Tuple[str, str, str, str], label: str, family: str) -> None:
        if low_key not in data or high_key not in data:
            return
        try:
            low, high = align(data[low_key], data[high_key])
        except Exception as exc:
            skipped_comparisons.append({
                "comparison": label,
                "family": family,
                "condition_low": "|".join(low_key),
                "condition_high": "|".join(high_key),
                "reason": repr(exc),
            })
            print(f"[statistics-warn] skipped comparison {label}: {exc!r}")
            return
        for metric_name in METRICS:
            low_value = metric(low, metric_name)
            high_value = metric(high, metric_name)
            diff = high_value - low_value
            ci_low, ci_high = bootstrap_difference(low, high, metric_name, b_iter, rng)
            p_value = paired_permutation_p(low, high, metric_name, p_iter, rng)
            comparisons.append({
                "comparison_family": family,
                "comparison": label,
                "model_or_strategy": high_key[0] if family == "strategy" else high_key[1],
                "prompt_variant": high_key[2],
                "repeat": high_key[3],
                "condition_low": "|".join(low_key),
                "condition_high": "|".join(high_key),
                "paper_count": len(low["paper_ids"]),
                "metric": metric_name,
                "value_low": low_value,
                "value_high": high_value,
                "difference_high_minus_low": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            })

    prompts_repeats = sorted({(c.prompt, c.repeat) for c in conditions})
    models = sorted({c.model for c in conditions})
    strategies = sorted({c.strategy for c in conditions})
    for prompt, repeat in prompts_repeats:
        for model in models:
            for low, high, label in STRATEGY_PAIRS:
                add_comparison((model, low, prompt, repeat), (model, high, prompt, repeat), label, "strategy")
        for strategy in strategies:
            for low, high, label in MODEL_PAIRS:
                add_comparison((low, strategy, prompt, repeat), (high, strategy, prompt, repeat), label, "model")

    prompt_mean_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Condition]] = defaultdict(list)
    for condition in conditions:
        grouped[(condition.model, condition.strategy)].append(condition)
    for (model, strategy), group in sorted(grouped.items()):
        sufficient = [data[condition.key] for condition in group]
        for metric_name in METRICS:
            try:
                estimate, lower, upper, paper_count = bootstrap_prompt_mean(
                    sufficient, metric_name, b_iter, rng
                )
            except Exception as exc:
                skipped_comparisons.append({
                    "comparison": "prompt_mean_bootstrap",
                    "family": "prompt_mean",
                    "condition_low": "",
                    "condition_high": f"{model}|{strategy}",
                    "reason": repr(exc),
                })
                print(f"[statistics-warn] prompt-mean CI skipped for {model}/{strategy}: {exc!r}")
                continue
            prompt_mean_rows.append({
                "model_key": model,
                "strategy": strategy,
                "metric": metric_name,
                "prompt_mean_estimate": estimate,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "paper_count": paper_count,
                "n_conditions": len(group),
                "prompt_variants": "|".join(sorted({condition.prompt for condition in group})),
            })

    holm_adjust(comparisons)
    write_csv(run_root / "condition_metrics_with_bootstrap_ci.csv", condition_rows)
    write_csv(run_root / "paired_statistical_comparisons.csv", comparisons)
    write_csv(run_root / "bootstrap_ci_prompt_mean.csv", prompt_mean_rows)
    summary = {
        "run_root": str(run_root),
        "input_run_roots": [str(root) for root in run_roots],
        "conditions_discovered": len(discovered_conditions),
        "conditions": len(condition_rows),
        "conditions_skipped": skipped_conditions,
        "comparisons": len(comparisons),
        "comparisons_skipped": skipped_comparisons,
        "prompt_mean_rows": len(prompt_mean_rows),
        "bootstrap_iterations": b_iter,
        "permutation_iterations": p_iter,
        "seed": seed,
        "prompt_mean_supported": True,
        "outputs": [
            str(run_root / "condition_metrics_with_bootstrap_ci.csv"),
            str(run_root / "paired_statistical_comparisons.csv"),
            str(run_root / "bootstrap_ci_prompt_mean.csv"),
        ],
    }
    (run_root / "statistical_analysis_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
