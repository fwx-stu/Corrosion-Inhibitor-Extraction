#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.aggregate import aggregate
from corrosion_ie.eval.integrity import expected_evaluation_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate evaluation metrics.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--glm-baseline-f1", type=float, default=None)
    ap.add_argument("--group-by", default="model,strategy", help="Comma-separated grouping fields. Aliases: model, strategy, prompt, seed.")
    args = ap.parse_args()
    glm = args.glm_baseline_f1
    composite_weights = None
    expected_evaluation = None
    if args.config:
        cfg = load_config(args.config)
        glm = glm if glm is not None else cfg.evaluation.get("glm_baseline_f1")
        composite_cfg = cfg.evaluation.get("composite_score", {}) or {}
        if composite_cfg.get("enabled", True):
            composite_weights = composite_cfg.get("weights")
        expected_evaluation = expected_evaluation_metadata(cfg)
    aggregate(
        args.run_root,
        glm_baseline_f1=glm,
        group_by=args.group_by,
        composite_weights=composite_weights,
        expected_evaluation=expected_evaluation,
    )
    print(f"[aggregate] done -> {args.run_root}")

if __name__ == "__main__":
    main()
