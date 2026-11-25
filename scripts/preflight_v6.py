#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config, resolve_model_config


def model_marker(path_value: Any) -> bool:
    if not path_value:
        return False
    path = Path(str(path_value)).expanduser()
    if (path / "config.json").exists():
        return True
    snapshots = path / "snapshots"
    return snapshots.exists() and any((item / "config.json").exists() for item in snapshots.iterdir() if item.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict static preflight for V6 matrices.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--expected-jobs", type=int, required=True)
    ap.add_argument("--skip-models", action="store_true", help="Static package testing only.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    errors: list[str] = []
    jobs = len(cfg.models) * len(cfg.strategies) * len(cfg.prompt_variants) * len(cfg.repeat_seeds)
    if jobs != args.expected_jobs:
        errors.append(f"matrix has {jobs} jobs; expected {args.expected_jobs}")
    if cfg.raw.get("run", {}).get("stop_on_error") is not False:
        errors.append("run.stop_on_error must be false")
    if cfg.raw.get("run", {}).get("fail_on_quality_gate") is not False:
        errors.append("run.fail_on_quality_gate must be false")
    if cfg.expected_paper_count is None:
        errors.append("expected_paper_count is required")

    required_files = [
        cfg.root / "prompts/templates/q0_full.txt",
        cfg.root / "prompts/templates/q1_rag.txt",
        cfg.root / "prompts/policies/recall.txt",
        cfg.root / "prompts/policies/balanced.txt",
        cfg.root / "prompts/policies/precision.txt",
        cfg.root / "scripts/run_matrix.py",
        cfg.root / "scripts/evaluate_predictions.py",
        cfg.root / "scripts/aggregate_runs.py",
        cfg.root / "scripts/statistical_analysis.py",
        cfg.root / "vendor/legacy_evaluate.py",
    ]
    for path in required_files:
        if not path.is_file():
            errors.append(f"missing required file: {path}")

    artifacts: Dict[str, Any] = {}
    if not args.skip_models:
        for model_key in cfg.models:
            model = resolve_model_config(model_key, cfg.config_path)
            resolved = model.get("resolved_path")
            artifacts[model_key] = resolved
            if not model_marker(resolved):
                errors.append(f"no valid local model for {model_key}: {resolved or 'unresolved'}")
        retrieval = cfg.retrieval
        for label, key in [("embedding", "embedding_model"), ("reranker", "reranker_model")]:
            value = retrieval.get(key)
            artifacts[label] = value
            if label == "reranker" and not retrieval.get("enable_reranker", True):
                continue
            if not model_marker(value):
                errors.append(f"no valid local {label} model: {value or 'unresolved'}")

    result = {
        "config": str(cfg.config_path),
        "experiment_id": cfg.experiment_id,
        "dataset": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "expected_papers": cfg.expected_paper_count,
        "jobs": jobs,
        "models": cfg.models,
        "strategies": cfg.strategies,
        "prompts": cfg.prompt_variants,
        "seeds": cfg.repeat_seeds,
        "retrieval_device": cfg.retrieval.get("device"),
        "artifacts": artifacts,
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)
    print("[v6-preflight-ok]")


if __name__ == "__main__":
    main()
