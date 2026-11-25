#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.io_utils import read_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot prompt-condition F1 and prompt mean ± SD.")
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise SystemExit(
            "[plot-error] matplotlib is required. Install with: python -m pip install matplotlib"
        ) from exc

    conditions = read_csv(run_root / "prompt_condition_metrics.csv")
    summary = read_csv(run_root / "prompt_robustness_summary.csv")
    if not conditions or not summary:
        raise SystemExit("[plot-error] run summarize_prompt_robustness.py first")

    prompts = []
    for row in conditions:
        prompt = row["prompt_variant"]
        if prompt not in prompts:
            prompts.append(prompt)
    groups = sorted({(row["model_key"], row["strategy"]) for row in conditions})
    labels = [f"{model}\n{strategy}" for model, strategy in groups]

    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, len(prompts))
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5.2))
    for prompt_index, prompt in enumerate(prompts):
        values = []
        for model, strategy in groups:
            matches = [
                row for row in conditions
                if row["model_key"] == model
                and row["strategy"] == strategy
                and row["prompt_variant"] == prompt
            ]
            values.append(float(matches[0]["field_micro_f1"]) if matches else float("nan"))
        offset = (prompt_index - (len(prompts) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=prompt)
    ax.set_ylabel("Field Micro F1")
    ax.set_title("Performance under three prompt conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out1 = run_root / "prompt_condition_field_f1.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_map = {(row["model_key"], row["strategy"]): row for row in summary}
    means = [float(summary_map[group]["field_micro_f1_prompt_mean"]) for group in groups]
    stds = [float(summary_map[group]["field_micro_f1_prompt_std"]) for group in groups]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5.2))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=5)
    ax.set_ylabel("Field Micro F1")
    ax.set_title("Prompt-robust mean ± SD across three prompts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out2 = run_root / "prompt_robustness_mean_sd.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] {out1}")
    print(f"[plot] {out2}")


if __name__ == "__main__":
    main()
