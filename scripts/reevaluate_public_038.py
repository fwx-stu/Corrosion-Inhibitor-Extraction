#!/usr/bin/env python3
"""Re-evaluate saved predictions from 8b-v6-reviewer-controls-v1(2).

Only this process receives the frozen threshold. Existing project code,
predictions, evaluation directories and calibration labels are not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "public-saved-predictions-038-v1"
# Reviewed evaluator: field weights, normalization and matching order are fixed.
EVALUATOR_SHA256 = "1bcbd9c106490a662bd380a8579e05ab676f56cf45b9f7dadcc773d5ac5ce4a9"
EXPERIMENTS = [
    ("corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6", "dev37", 27,
     "experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"),
    ("corrosion_q0q1q2_test50_balanced_opt_final_v6", "test50", 9,
     "experiment_q0q1q2_test50_balanced_opt_v6.yaml"),
    ("corrosion_reviewer_controls_test50_v1", "test50", 12,
     "experiment_reviewer_controls_test50_v1_gpu_3b7b.yaml"),
]
REQUIRED_CSV = ["summary_metrics.csv", "per_field_metrics.csv",
                "per_paper_metrics.csv", "per_paper_field_metrics.csv"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def local_frozen_path(root, stored):
    """Relocate the recorded server path without following another installation."""
    parts = Path(stored).parts
    for anchor in ("output", "input", "work"):
        if anchor in parts:
            return root.joinpath(*parts[parts.index(anchor):])
    raise ValueError(f"Cannot relocate frozen source: {stored}")


def paper_ids(path):
    ids = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                obj = json.loads(line)
                if not obj.get("paper_id"):
                    raise ValueError(f"Missing paper_id in {path}")
                ids.append(str(obj["paper_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate paper IDs in {path}")
    return set(ids)


def run_step(root, out, label, script, arguments):
    log = out / "logs" / (label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[postprocess] {label}; log={log}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    command = [sys.executable, str(root / "scripts" / script), *map(str, arguments)]
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=root, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-3000:])
        raise RuntimeError(f"{label} failed; see {log}")


def manuscript_holm(test_root, expected_comparisons=None):
    """Keep original per-metric correction; add the manuscript's 81/63 families."""
    rows = read_csv(test_root / "paired_statistical_comparisons.csv")
    counts = Counter(r["comparison_family"] for r in rows)
    if expected_comparisons and counts != Counter(strategy=81, model=63):
        raise RuntimeError(f"Expected 81 strategy + 63 model comparisons, got {dict(counts)}")
    for row in rows:
        row["p_holm_legacy_by_metric"] = row["p_holm"]
        row["p_holm"] = ""
        row["holm_scope"] = "all three metrics within comparison_family"
    for family in counts:
        members = [r for r in rows if r["comparison_family"] == family]
        if any(not math.isfinite(float(r["p_value"])) for r in members):
            raise ValueError(f"Non-finite permutation p value in {family}")
        members.sort(key=lambda r: float(r["p_value"]))
        running = 0.0
        for rank, row in enumerate(members):
            running = max(running, min(1.0, (len(members) - rank) * float(row["p_value"])))
            row["p_holm"] = running
            row["holm_family_size"] = len(members)
    write_csv(test_root / "paired_comparisons_manuscript_holm.csv", rows)
    return dict(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--frozen", default="work/record_threshold/results/selected_threshold_frozen.json")
    ap.add_argument("--output-root", default="reproduced/qwen")
    ap.add_argument("--check-only", action="store_true", help="Validate files only; do not evaluate.")
    ap.add_argument("--smoke-test", action="store_true", help="One full-paper run per experiment; NOT formal results.")
    ap.add_argument("--bootstrap-iterations", type=int, default=10000)
    ap.add_argument("--permutation-iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()
    if args.bootstrap_iterations <= 0 or args.permutation_iterations <= 0:
        raise ValueError("Resampling iterations must be positive")
    root = Path(args.project_root).expanduser().resolve()
    frozen_path = (root / args.frozen).resolve()
    out = (root / args.output_root).resolve()
    if args.smoke_test:
        out = out.with_name(out.name + "_smoke")
    if out == root or out in root.parents or (root / "output").resolve() in [out, *out.parents]:
        raise ValueError("Use a separate work/ output directory")
    if sha(root / "vendor/legacy_evaluate.py") != EVALUATOR_SHA256:
        raise ValueError("Evaluator differs from the reviewed installation. Use the uploaded original evaluator; do not manually edit its threshold.")
    frozen = read_json(frozen_path)
    threshold = float(frozen["selected_threshold"])
    if (threshold != 0.38 or frozen.get("dataset") != "dev37"
            or frozen.get("test50_used") is not False
            or frozen.get("primary_matching_order") != "threshold_before_hungarian"):
        raise ValueError("Frozen protocol must be dev37-only, threshold 0.38, threshold_before_hungarian")
    if sha(root / "input/dev37/gold/dev37.jsonl") != frozen["gold_sha256"]:
        raise ValueError("dev37 Gold differs from calibration")
    if sha(root / "scripts/calibrate_record_threshold.py") != frozen["script_sha256"]:
        raise ValueError("Calibration script differs from the frozen version")
    for item in frozen["prediction_files"]:
        path = local_frozen_path(root, item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"Calibration prediction changed: {path}")
    annotations = local_frozen_path(root, frozen["annotations_file"])
    if annotations.exists() and sha(annotations) != frozen["annotations_sha256"]:
        raise ValueError("Calibration workbook bytes differ from the frozen workbook")

    sys.path.insert(0, str(root))
    from scipy.optimize import linear_sum_assignment
    from corrosion_ie.config import load_config
    from corrosion_ie.eval.legacy_wrapper import _load_legacy
    from corrosion_ie.eval.aggregate import aggregate
    linear_sum_assignment([[0.0]])
    legacy = _load_legacy()
    if legacy.config is not None:
        raise ValueError("Unexpected external 'config' module changes evaluator defaults; use the project environment without a conflicting config.py")
    legacy.RECORD_MATCH_THRESHOLD = threshold
    datasets = {}
    for dataset, count in (("dev37", 37), ("test50", 50)):
        gold = root / f"input/{dataset}/gold/{dataset}.jsonl"
        docs = root / f"input/{dataset}/md"
        ids = paper_ids(gold)
        if len(ids) != count:
            raise ValueError(f"Incomplete {dataset} Gold or Markdown")
        docs_hash = sha(root / "input/paper_manifest.csv")
        datasets[dataset] = (gold, docs, ids, docs_hash)

    plans = []
    for name, dataset, expected_runs, config_name in EXPERIMENTS:
        source = root / "output/exp_v2" / name
        manifest = read_json(source / "matrix_manifest.json")
        config = root / "configs" / config_name
        if manifest.get("dataset_name") != dataset or len(manifest["jobs"]) != expected_runs:
            raise ValueError(f"Unexpected experiment manifest: {name}")
        jobs = []
        for job in manifest["jobs"]:
            rel = Path(str(job["model_key"])) / job["strategy"] / job["prompt_variant"] / f"r{int(job['seed'])}"
            pred = source / rel / "predictions.jsonl"
            # Always select the current installation, ignoring old absolute run_dir values.
            if paper_ids(pred) != datasets[dataset][2]:
                raise ValueError(f"Incomplete or wrong paper set: {pred}")
            jobs.append((job, rel, pred))
        if len({str(rel) for _, rel, _ in jobs}) != expected_runs:
            raise ValueError(f"Duplicate manifest conditions: {name}")
        print(f"[check] {dataset}: {expected_runs} runs x {len(datasets[dataset][2])} papers; {name}", flush=True)
        selected_jobs = jobs
        if args.smoke_test:
            smoke_strategy = "a1_q1_table_budget" if expected_runs == 12 else "q1_budgeted_rag"
            selected_jobs = [item for item in jobs if item[0]["model_key"] == "qwen2_5_14b"
                             and item[0]["strategy"] == smoke_strategy
                             and item[0]["prompt_variant"] == "balanced"][:1]
            if not selected_jobs:
                raise ValueError(f"Cannot select complete smoke condition from {name}")
        plans.append((name, dataset, config, source, manifest, selected_jobs))
    print("[check] frozen threshold=0.38; 48 formal runs; 4 dev37 smoke runs excluded", flush=True)
    if args.check_only:
        return

    marker = out / "reevaluation_manifest.json"
    identity = {"tool_version": VERSION, "source_project": str(root),
                "frozen_sha256": sha(frozen_path), "smoke_test": args.smoke_test}
    if out.exists() and any(out.iterdir()):
        if not marker.exists() or any(read_json(marker).get(k) != v for k, v in identity.items()):
            raise ValueError(f"Output contains unrelated results: {out}; choose a new --output-root")
    out.mkdir(parents=True, exist_ok=True)
    batch = {**identity, "status": "running", "record_threshold": threshold,
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "matching_order": frozen["primary_matching_order"],
             "evaluator_sha256": EVALUATOR_SHA256,
             "runner_sha256": sha(Path(__file__)),
             "bootstrap_iterations": args.bootstrap_iterations,
             "permutation_iterations": args.permutation_iterations, "seed": args.seed,
             "scope": "48 formal Qwen runs only; E0-E5 and GLM reference are not included",
             "pending_manuscript_analyses": ["E0-E5 and GLM reference re-evaluation",
                 "GLM paired comparison (20000 iterations, seed 20260717)",
                 "interaction models, exact-record and threshold-sensitivity diagnostics",
                 "publication figure regeneration"]}
    write_json(marker, batch)
    calibration_dir = out / "calibration"
    calibration_dir.mkdir(exist_ok=True)
    for src in sorted(frozen_path.parent.iterdir()):
        if src.is_file() and src.suffix in (".json", ".csv", ".png"):
            shutil.copy2(src, calibration_dir / src.name)
    if annotations.exists():
        shutil.copy2(annotations, calibration_dir / annotations.name)

    all_rows, differences = [], []
    completed_runs = 0
    for name, dataset, config, source, manifest, jobs in plans:
        new_root = out / name
        new_root.mkdir(exist_ok=True)
        gold, docs, ids, docs_hash = datasets[dataset]
        gold_map = legacy.load_gold(gold)
        lengths = {r["paper_id"]: int(r["document_chars"]) for r in read_csv(root / "input/paper_manifest.csv") if r["split"] == dataset}
        assert set(lengths) == set(gold_map)
        new_manifest = {**manifest, "source_manifest": str(source / "matrix_manifest.json"),
                        "config_path": str(config), "jobs_expected": len(jobs), "jobs": []}
        for job, rel, pred in jobs:
            new_run = new_root / rel
            new_run.mkdir(parents=True, exist_ok=True)
            new_pred = new_run / "predictions.jsonl"
            expected = {"prediction_sha256": sha(pred), "gold_sha256": sha(gold),
                        "evaluator_sha256": EVALUATOR_SHA256, "docs_dir": str(docs),
                        "docs_sha256": docs_hash, "record_matching_backend": "scipy_hungarian",
                        "record_threshold": threshold, "matching_order": frozen["primary_matching_order"],
                        "frozen_sha256": sha(frozen_path), "runner_version": VERSION}
            if not new_pred.exists() or sha(new_pred) != expected["prediction_sha256"]:
                shutil.copy2(pred, new_pred)
            for file_name in ("run_meta.json", "status.json"):
                if (pred.parent / file_name).exists():
                    shutil.copy2(pred.parent / file_name, new_run / file_name)
            eval_dir = new_run / "evaluation"
            meta = eval_dir / "evaluation_meta.json"
            valid = meta.exists() and all((eval_dir / f).exists() for f in REQUIRED_CSV)
            if valid:
                stored = read_json(meta)
                valid = all(stored.get(k) == v for k, v in expected.items())
                valid = valid and all(sha(eval_dir / f) == stored.get("output_sha256", {}).get(f) for f in REQUIRED_CSV)
            print(f"[{'reuse' if valid else 'evaluate'}] {completed_runs + 1}: {name}/{rel}", flush=True)
            if not valid:
                result = legacy.evaluate_strategy("__".join(rel.parts), new_pred, gold_map, doc_length_map=lengths)
                legacy.write_evaluation_outputs(result, eval_dir)
                if len(result["per_paper"]) != len(ids):
                    raise RuntimeError(f"Wrong evaluation paper count: {rel}")
                write_json(meta, {**expected, "source_prediction": str(pred),
                                 "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "output_sha256": {f: sha(eval_dir / f) for f in REQUIRED_CSV}})
            current = read_csv(eval_dir / "summary_metrics.csv")[0]
            tags = {"experiment": name, "dataset": dataset, "model_key": job["model_key"],
                    "evidence_strategy": job["strategy"], "prompt_variant": job["prompt_variant"],
                    "seed": job["seed"], "record_threshold": threshold}
            all_rows.append({**tags, **current})
            previous = pred.parent / "evaluation/summary_metrics.csv"
            if previous.exists():
                old = read_csv(previous)[0]
                for metric in ("record_f1", "field_micro_f1", "null_accuracy"):
                    differences.append({**tags, "metric": metric,
                        "previous_saved_value": old[metric], "new_value": current[metric],
                        "difference": float(current[metric]) - float(old[metric])})
            new_manifest["jobs"].append({**job, "run_dir": str(new_run), "source_run_dir": str(pred.parent)})
            if sha(pred) != expected["prediction_sha256"]:
                raise RuntimeError(f"Prediction changed during evaluation: {pred}")
            completed_runs += 1
        write_json(new_root / "matrix_manifest.json", new_manifest)
        cfg = load_config(config)
        weights = (cfg.evaluation.get("composite_score") or {}).get("weights")
        aggregate(new_root, group_by="model,strategy,prompt", composite_weights=weights,
                  expected_evaluation={"record_threshold": threshold, "frozen_sha256": sha(frozen_path)})
        exclusions = read_json(new_root / "aggregation_exclusions.json")
        if exclusions["excluded_runs"] or exclusions["per_field_exclusions"]:
            raise RuntimeError(f"Aggregation excluded runs: {new_root}")
    write_csv(out / "all_formal_summary_t038.csv", all_rows)
    write_csv(out / "comparison_with_previous_evaluation.csv", differences)

    dev_name, test_name, control_name = [x[0] for x in EXPERIMENTS]
    dev_cfg, test_cfg = [root / "configs" / x[3] for x in EXPERIMENTS[:2]]
    if not args.smoke_test:
        run_step(root, out, "dev37_prompt_summary", "summarize_prompt_robustness.py",
                 ["--config", dev_cfg, "--run-root", out / dev_name])
    for name, cfg in ((dev_name, dev_cfg), (test_name, test_cfg)):
        run_step(root, out, name + "_strata", "analyze_document_strata.py",
                 ["--config", cfg, "--run-root", out / name])
    # Preserve the original postprocessing order: standalone primary test50,
    # then reviewer controls first + primary test50 in the expanded analysis.
    for name, cfg, total in ((dev_name, dev_cfg, 27), (test_name, test_cfg, 9),
                             (control_name, test_cfg, 21)):
        command = ["--config", cfg, "--run-root", out / name,
                   "--bootstrap-iterations", args.bootstrap_iterations,
                   "--permutation-iterations", args.permutation_iterations, "--seed", args.seed]
        if name == control_name:
            command.extend(["--additional-run-root", out / test_name])
        run_step(root, out, name + "_statistics", "statistical_analysis.py", command)
        statistics = read_json(out / name / "statistical_analysis_meta.json")
        expected_count = (2 if name == control_name else 1) if args.smoke_test else total
        if statistics["conditions"] != expected_count or statistics["conditions_skipped"] or statistics["comparisons_skipped"]:
            raise RuntimeError(f"Incomplete statistical analysis: {out / name}")
    batch["test50_comparison_counts"] = manuscript_holm(out / control_name, not args.smoke_test)
    batch.update(status="complete", runs_evaluated=completed_runs,
                 finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(marker, batch)
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(out, arcname=out.name)
    print(f"[done] threshold={threshold}; evaluated={completed_runs}; smoke_test={args.smoke_test}", flush=True)
    print(f"[upload] {archive}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
