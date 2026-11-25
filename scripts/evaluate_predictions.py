#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata, sha256_file
from corrosion_ie.eval.legacy_wrapper import evaluate_prediction
from corrosion_ie.io_utils import append_jsonl, write_json


def manifest_run_dirs(run_root: Path) -> list[Path]:
    manifest_path = run_root / "matrix_manifest.json"
    if not manifest_path.exists():
        return [p.parent for p in sorted(run_root.glob("*/*/*/r*/predictions.jsonl"))]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for job in manifest.get("jobs", []):
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


def require_hungarian_backend() -> str:
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "[evaluate-error] scipy is required for deterministic Hungarian record matching. "
            "Install requirements.txt before formal evaluation."
        ) from exc
    return "scipy_hungarian"


def _atomic_replace_directory(source: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + f".old.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate runs listed in the current matrix manifest.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--only-missing", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    record_matching_backend = require_hungarian_backend()
    run_root = Path(args.run_root) if args.run_root else cfg.run_root
    docs_dir_value = cfg.evaluation.get("docs_dir") or str(cfg.md_dir)
    docs_dir = (cfg.root / docs_dir_value) if not Path(str(docs_dir_value)).is_absolute() else Path(str(docs_dir_value))
    run_dirs = manifest_run_dirs(run_root)
    print(f"[evaluate] manifest runs={len(run_dirs)} run_root={run_root}")

    static_expected = expected_evaluation_metadata(cfg)

    for run_dir in run_dirs:
        pred = run_dir / "predictions.jsonl"
        if not pred.exists():
            append_jsonl(run_root / "evaluation_failures.jsonl", {
                "run_dir": str(run_dir),
                "error": f"missing predictions: {pred}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[evaluate-warn] missing predictions, skipped: {pred}")
            continue

        out_dir = run_dir / "evaluation"
        expected = {
            "prediction_sha256": sha256_file(pred),
            **static_expected,
            "record_matching_backend": record_matching_backend,
        }
        if args.only_missing:
            current, _, stored = evaluation_integrity(run_dir, static_expected)
            if current and all(stored.get(key) == value for key, value in expected.items()):
                print(f"[evaluate-skip] {run_dir.relative_to(run_root)}")
                continue

        rel = run_dir.relative_to(run_root)
        strategy_label = "__".join(rel.parts[:4])
        tmp_dir = run_dir / f"evaluation.tmp.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        print(f"[evaluate] {rel}")
        try:
            evaluate_prediction(cfg.gold_path, pred, tmp_dir, strategy=strategy_label, docs_dir=docs_dir)
            write_json(tmp_dir / "evaluation_meta.json", expected)
            if not (tmp_dir / "summary_metrics.csv").exists():
                raise RuntimeError("evaluator did not create summary_metrics.csv")
            _atomic_replace_directory(tmp_dir, out_dir)
        except Exception as exc:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            append_jsonl(run_root / "evaluation_failures.jsonl", {
                "run_dir": str(run_dir),
                "error": repr(exc),
                "prediction_sha256": expected["prediction_sha256"],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[evaluate-warn] failed and skipped: {rel}: {exc!r}")
            continue


if __name__ == "__main__":
    main()
