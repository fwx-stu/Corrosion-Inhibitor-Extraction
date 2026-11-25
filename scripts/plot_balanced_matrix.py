#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot balanced-only condition metrics with paper-level bootstrap 95% CIs.")
    ap.add_argument("--run-root", required=True, type=Path)
    args = ap.parse_args()
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise SystemExit("matplotlib and numpy are required") from exc

    path = args.run_root / "condition_metrics_with_bootstrap_ci.csv"
    rows = [row for row in read_csv(path) if row.get("prompt_variant") == "balanced"]
    if not rows:
        raise SystemExit(f"no balanced rows in {path}; run statistical_analysis.py first")
    model_order = {"qwen2_5_3b": 0, "qwen2_5_7b": 1, "qwen2_5_14b": 2}
    strategy_order = {"q0_safe_full": 0, "q1_budgeted_rag": 1, "q2_table_aware_rag": 2}
    rows.sort(key=lambda row: (model_order.get(row["model_key"], 99), strategy_order.get(row["strategy"], 99)))

    for metric, title, filename in [
        ("field_micro_f1", "Balanced-policy Field Micro F1", "balanced_field_micro_f1_ci.png"),
        ("record_f1", "Balanced-policy Record F1", "balanced_record_f1_ci.png"),
        ("null_accuracy", "Balanced-policy Null Accuracy", "balanced_null_accuracy_ci.png"),
    ]:
        labels = [f"{row['model_key']}\n{row['strategy']}" for row in rows]
        values = [float(row[metric]) for row in rows]
        low = [value - float(row[f"{metric}_ci_low"]) for value, row in zip(values, rows)]
        high = [float(row[f"{metric}_ci_high"]) - value for value, row in zip(values, rows)]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.25), 5.8))
        ax.errorbar(x, values, yerr=[low, high], fmt="o", capsize=4)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        out = args.run_root / filename
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}")


if __name__ == "__main__":
    main()
