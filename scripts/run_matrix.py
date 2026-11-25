#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.io_utils import append_jsonl, ensure_dir, write_json
from corrosion_ie.llm.clients import clear_local_model_cache
from corrosion_ie.runner import iter_matrix, run_single_experiment
from corrosion_ie.schemas import RunIdentity

FORMAL_FORBIDDEN_GENERATION_ENV = (
    "GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_NEW_TOKENS",
    "GEN_MAX_INPUT_TOKENS", "GEN_DO_SAMPLE",
)


def split_csv(x):
    if not x:
        return None
    return [i.strip() for i in x.split(",") if i.strip()]


def summarize_jobs(jobs):
    model_counts = Counter(j[0] for j in jobs)
    strat_counts = Counter(j[1] for j in jobs)
    prompt_counts = Counter(j[2] for j in jobs)
    seed_counts = Counter(j[3] for j in jobs)
    print(f"[matrix] jobs={len(jobs)}")
    print("[matrix] models=", ", ".join(f"{k}:{v}" for k, v in model_counts.items()))
    print("[matrix] strategies=", ", ".join(f"{k}:{v}" for k, v in strat_counts.items()))
    print("[matrix] prompts=", ", ".join(f"{k}:{v}" for k, v in prompt_counts.items()))
    print("[matrix] seeds=", ", ".join(str(k) for k in seed_counts.keys()))


def merge_matrix_manifest(path: Path, manifest: dict) -> dict:
    """Merge split hardware batches without changing legacy behavior.

    This is opt-in through ``merge_matrix_manifest: true``.  It lets the new
    reviewer-control matrix use the same experiment directory for the GPU
    3B/7B batch and the CPU 14B batch while keeping all earlier configurations
    on the original overwrite semantics.
    """

    if not path.exists():
        return manifest
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot merge existing matrix manifest {path}: {exc}") from exc
    if str(previous.get("experiment_id")) != str(manifest.get("experiment_id")):
        raise ValueError(
            "matrix manifest experiment_id mismatch: "
            f"{previous.get('experiment_id')} != {manifest.get('experiment_id')}"
        )
    for key in ("dataset_name", "dataset_role", "expected_paper_count"):
        if previous.get(key) != manifest.get(key):
            raise ValueError(
                f"matrix manifest {key} mismatch: "
                f"{previous.get(key)!r} != {manifest.get(key)!r}"
            )

    def identity(job: dict) -> tuple:
        return (
            str(job.get("model_key")),
            str(job.get("strategy")),
            str(job.get("prompt_variant")),
            int(job.get("seed")),
        )

    merged_jobs = {
        identity(job): job
        for job in list(previous.get("jobs") or []) + list(manifest.get("jobs") or [])
    }
    output = dict(manifest)
    output["config_paths"] = list(dict.fromkeys(
        list(previous.get("config_paths") or [previous.get("config_path")])
        + list(manifest.get("config_paths") or [manifest.get("config_path")])
    ))
    output["jobs"] = [merged_jobs[key] for key in sorted(merged_jobs)]
    output["jobs_expected"] = len(output["jobs"])
    output["manifest_mode"] = "merged_split_hardware_batches_v1"
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a configurable model × strategy × prompt-policy matrix.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=None, help="Comma-separated model keys. Default: config models.")
    ap.add_argument("--strategies", default=None, help="Comma-separated strategy names. Default: config strategies.")
    ap.add_argument("--prompts", default=None, help="Comma-separated prompt variants. Default: config prompt_variants.")
    ap.add_argument("--seeds", default=None, help="Comma-separated seeds. Default: config repeat_seeds.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paper-id", default=None, help="Run one paper ID.")
    ap.add_argument("--paper-ids", default=None, help="Comma-separated paper IDs for targeted smoke tests.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print matrix summary only.")
    ap.add_argument("--list-jobs", action="store_true", help="Print every expanded (model, strategy, prompt, seed) job.")
    ap.add_argument("--dry-run-client", action="store_true", help="Smoke-test pipeline without loading LLM.")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Stop the matrix after a completed condition fails its predeclared quality gate.",
    )
    ap.add_argument("--progress", choices=["auto", "tqdm", "plain", "none"], default="auto", help="Document-level progress display mode.")
    ap.add_argument("--verbose-docs", action="store_true", help="Print one line per document even when tqdm is enabled.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    formal_test = cfg.dataset_role.lower() == "test" and not args.dry_run and not args.dry_run_client
    if formal_test:
        if os.getenv("CONFIRM_FROZEN_TEST", "") != "YES":
            raise SystemExit(
                "[test-protection] Formal test is locked. Export CONFIRM_FROZEN_TEST=YES "
                "after freezing code, prompts, gold, paper IDs, models, and environment."
            )
        overrides = {
            "models": args.models,
            "strategies": args.strategies,
            "prompts": args.prompts,
            "seeds": args.seeds,
            "limit": args.limit,
            "paper_id": args.paper_id,
            "paper_ids": args.paper_ids,
        }
        active = {key: value for key, value in overrides.items() if value is not None}
        if active:
            raise SystemExit(f"[test-protection] formal test matrix cannot be overridden: {active}")
        forbidden_env = {name: os.getenv(name) for name in FORMAL_FORBIDDEN_GENERATION_ENV if os.getenv(name) is not None}
        if forbidden_env:
            raise SystemExit(f"[test-protection] GEN_* overrides are forbidden in formal test: {sorted(forbidden_env)}")
        if args.force and os.getenv("ALLOW_TEST_RESET", "") != "YES":
            raise SystemExit(
                "[test-protection] --force is disabled for formal test. "
                "Use a new experiment_id; ALLOW_TEST_RESET=YES is for documented recovery only."
            )
        protocol_manifest = os.getenv("PROTOCOL_MANIFEST", "").strip()
        if not protocol_manifest:
            raise SystemExit("[test-protection] PROTOCOL_MANIFEST is required for formal test")
        manifest_path = Path(protocol_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise SystemExit(f"[test-protection] protocol manifest does not exist: {manifest_path}")
        verify_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "freeze_protocol.py"),
            "--config", str(Path(args.config).resolve()),
            "--verify", str(manifest_path),
        ]
        print(f"[test-protection] verifying frozen protocol: {manifest_path}", flush=True)
        verification = subprocess.run(verify_cmd, cwd=ROOT)
        if verification.returncode != 0:
            raise SystemExit("[test-protection] frozen protocol verification failed")
    paper_ids = split_csv(args.paper_ids)
    if args.paper_id and paper_ids:
        raise SystemExit("use either --paper-id or --paper-ids, not both")
    models = split_csv(args.models)
    strategies = split_csv(args.strategies)
    prompts = split_csv(args.prompts)
    seeds = [int(x) for x in split_csv(args.seeds)] if args.seeds else None
    jobs = list(iter_matrix(cfg, models=models, strategies=strategies, prompts=prompts, seeds=seeds))
    print(f"[matrix] run_root={cfg.run_root}")
    summarize_jobs(jobs)
    ensure_dir(cfg.run_root)
    manifest = {
        "experiment_id": cfg.experiment_id,
        "dataset_name": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "expected_paper_count": cfg.expected_paper_count,
        "config_path": str(Path(args.config).resolve()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs_expected": len(jobs),
        "jobs": [
            {
                "model_key": model,
                "strategy": strategy,
                "prompt_variant": prompt,
                "seed": int(seed),
                "run_dir": str(RunIdentity(cfg.experiment_id, model, strategy, prompt, int(seed)).output_dir(cfg.output_root)),
            }
            for model, strategy, prompt, seed in jobs
        ],
    }
    manifest_path = cfg.run_root / "matrix_manifest.json"
    if bool(cfg.raw.get("merge_matrix_manifest", False)):
        manifest = merge_matrix_manifest(manifest_path, manifest)
    write_json(manifest_path, manifest)
    print(f"[matrix] manifest={manifest_path} jobs_listed={manifest['jobs_expected']}")
    if args.list_jobs:
        for idx, j in enumerate(jobs, start=1):
            print(f"  [{idx:03d}/{len(jobs):03d}] {j}")
    if args.dry_run:
        return

    previous_model = None
    for idx, (model, strat, prompt, seed) in enumerate(jobs, start=1):
        if previous_model is not None and model != previous_model:
            print(f"[matrix] releasing local model cache before switching {previous_model} -> {model}", flush=True)
            clear_local_model_cache()
        previous_model = model
        print(f"\n[job {idx:03d}/{len(jobs):03d}] model={model} strategy={strat} prompt={prompt} seed={seed}", flush=True)
        effective_stop_on_error = (
            args.stop_on_error
            or bool((cfg.raw.get("run", {}) or {}).get("stop_on_error", False))
        )
        effective_fail_on_quality_gate = (
            args.fail_on_quality_gate
            or bool((cfg.raw.get("run", {}) or {}).get("fail_on_quality_gate", False))
        )
        try:
            run_single_experiment(
                cfg, model, strat, prompt, seed,
                force=args.force,
                limit=args.limit,
                paper_id=args.paper_id,
                paper_ids=paper_ids,
                stop_on_error=effective_stop_on_error,
                fail_on_quality_gate=effective_fail_on_quality_gate,
                dry_run=args.dry_run_client,
                progress=args.progress,
                verbose_docs=args.verbose_docs,
            )
        except Exception as exc:
            failure = {
                "job_index": idx,
                "jobs_total": len(jobs),
                "model_key": model,
                "strategy": strat,
                "prompt_variant": prompt,
                "seed": int(seed),
                "error": repr(exc),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(cfg.run_root / "matrix_failures.jsonl", failure)
            print(
                f"[matrix-job-error] job={idx}/{len(jobs)} "
                f"model={model} strategy={strat} prompt={prompt} seed={seed} "
                f"error={exc!r}; continuing with the next job",
                flush=True,
            )
            clear_local_model_cache()
            previous_model = None
            if effective_stop_on_error or effective_fail_on_quality_gate:
                raise
            continue
    clear_local_model_cache()

if __name__ == "__main__":
    main()
