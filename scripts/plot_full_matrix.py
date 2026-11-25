#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate prompt-mean full-matrix figures.")
    ap.add_argument("--run-root", required=True, type=Path)
    args = ap.parse_args()
    root = args.run_root
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise SystemExit("matplotlib and numpy are required") from exc

    source = root / "bootstrap_ci_prompt_mean.csv"
    rows = read_csv(source)
    if not rows:
        raise SystemExit(f"no prompt-mean bootstrap rows in {source}; run statistical_analysis.py first")

    model_order = {"qwen2_5_3b": 0, "qwen2_5_7b": 1, "qwen2_5_14b": 2}
    strategy_order = {"q0_safe_full": 0, "q1_budgeted_rag": 1, "q2_table_aware_rag": 2}
    rows.sort(key=lambda row: (
        model_order.get(row.get("model_key", ""), 99),
        strategy_order.get(row.get("strategy", ""), 99),
        row.get("metric", ""),
    ))

    for metric, title, output_name in [
        ("field_micro_f1", "Prompt-mean Field Micro F1 with paper-level 95% CI", "full_matrix_field_f1_ci.png"),
        ("record_f1", "Prompt-mean Record F1 with paper-level 95% CI", "full_matrix_record_f1_ci.png"),
        ("null_accuracy", "Prompt-mean Null Accuracy with paper-level 95% CI", "full_matrix_null_accuracy_ci.png"),
    ]:
        data = [row for row in rows if row.get("metric") == metric]
        data = [row for row in data if math.isfinite(number(row.get("prompt_mean_estimate", "")))]
        if not data:
            print(f"[plot-warn] no usable rows for {metric}; skipped")
            continue
        labels = [f"{row['model_key']}\n{row['strategy']}" for row in data]
        values = [number(row["prompt_mean_estimate"]) for row in data]
        lower = [max(0.0, value - number(row["ci_lower_95"])) for value, row in zip(values, data)]
        upper = [max(0.0, number(row["ci_upper_95"]) - value) for value, row in zip(values, data)]
        x = np.arange(len(data))
        fig, axis = plt.subplots(figsize=(max(10, len(data) * 1.25), 5.8))
        axis.errorbar(x, values, yerr=[lower, upper], fmt="o", capsize=4)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylim(bottom=0)
        fig.tight_layout()
        output = root / output_name
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {output}")


if __name__ == "__main__":
    main()
