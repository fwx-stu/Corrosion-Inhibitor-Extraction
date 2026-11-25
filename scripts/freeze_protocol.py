#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config, resolve_model_config
from corrosion_ie.io_utils import write_json
from corrosion_ie.llm.clients import generation_from_model_cfg
from corrosion_ie.runner import (
    _artifact_signature,
    _file_sha256,
    _runtime_environment,
    _selected_files_sha256,
    _strategy_auxiliary_files,
    _tree_sha256,
)


def stable_hash(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


FORMAL_FORBIDDEN_GENERATION_ENV = ("GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_NEW_TOKENS", "GEN_MAX_INPUT_TOKENS", "GEN_DO_SAMPLE")


def _model_artifact_signature(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Hash only a resolved local model directory.

    Hugging Face identifiers such as ``Qwen/Qwen2.5-7B-Instruct`` are model
    IDs, not filesystem paths. Treating them as relative paths can accidentally
    scan an unrelated directory with the same name.
    """
    model_id = model_cfg.get("model_id")
    resolved = model_cfg.get("resolved_path")
    if not resolved:
        return {"model_id": model_id, "path": None, "exists": False, "source": "model_id"}
    path = Path(str(resolved)).expanduser()
    if not path.exists():
        return {"model_id": model_id, "path": str(path.resolve()), "exists": False, "source": "resolved_path"}
    result = _artifact_signature(path, hash_weights=True, progress_label=f"generation:{model_id}")
    result["model_id"] = model_id
    result["source"] = "resolved_path"
    return result


def _local_artifact_signature(path_value: Any) -> Dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False}
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return {"path": str(path.resolve()), "exists": False}
    return _artifact_signature(path, hash_weights=True, progress_label=f"retrieval:{path.name}")




def _pip_freeze_sha256() -> str:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot capture pip freeze for protocol manifest: {exc}") from exc
    lines = sorted(line.strip() for line in output.splitlines() if line.strip())
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()

def build_manifest(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    if cfg.dataset_role.lower() == "test":
        active = {name: os.getenv(name) for name in FORMAL_FORBIDDEN_GENERATION_ENV if os.getenv(name) is not None}
        if active:
            raise RuntimeError(f"GEN_* overrides are forbidden while freezing/verifying formal test: {sorted(active)}")
    dataset_fp = cfg.md_dir.parent / "dataset_fingerprint.json"
    retrieval = cfg.retrieval
    models: Dict[str, Any] = {}
    print(f"[protocol] resolving {len(cfg.models)} generation models", flush=True)
    for key in cfg.models:
        print(f"[protocol] model {key}", flush=True)
        model_cfg = resolve_model_config(key, cfg.config_path)
        if cfg.generation:
            model_cfg = dict(model_cfg)
            model_cfg["generation"] = {
                **dict(model_cfg.get("generation", {}) or {}),
                **dict(cfg.generation),
            }
        artifact = _model_artifact_signature(model_cfg)
        if cfg.dataset_role.lower() == "test" and bool(cfg.raw.get("formal_require_local_artifacts", True)) and not artifact.get("exists"):
            raise RuntimeError(
                f"formal test requires a resolved local model artifact for {key}; "
                f"set {model_cfg.get('local_path_env') or 'the model path environment variable'} or MODELS_DIR"
            )
        models[key] = {
            "config": model_cfg,
            "artifact": artifact,
            "effective_generation": generation_from_model_cfg(model_cfg, seed=int(cfg.repeat_seeds[0])).__dict__,
        }

    print("[protocol] hashing retrieval artifacts", flush=True)
    retrieval_artifacts = {
        "embedding": _local_artifact_signature(retrieval.get("embedding_model")),
        "reranker": _local_artifact_signature(retrieval.get("reranker_model")),
    }
    if cfg.dataset_role.lower() == "test" and bool(cfg.raw.get("formal_require_local_artifacts", True)):
        missing_retrieval = [name for name, artifact in retrieval_artifacts.items() if not artifact.get("exists")]
        if missing_retrieval:
            raise RuntimeError(
                "formal test requires resolved local retrieval artifacts: " + ", ".join(missing_retrieval)
                + "; set EMBEDDING_MODEL/RERANKER_MODEL or MODELS_DIR"
            )

    print("[protocol] hashing source, prompts, environment, and dataset", flush=True)
    strategy_auxiliary_sha256: Dict[str, Dict[str, str]] = {}
    for strategy in cfg.strategies:
        paths = _strategy_auxiliary_files(cfg.root, cfg.strategy_params(strategy))
        strategy_auxiliary_sha256[strategy] = {
            str(path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path): _file_sha256(path)
            for path in paths
        }
    payload: Dict[str, Any] = {
        "schema": "8b-opt-protocol-manifest-v6",
        "experiment_id": cfg.experiment_id,
        "dataset_name": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "config_path": str(cfg.config_path),
        "config_sha256": _file_sha256(cfg.config_path),
        "fields_sha256": _file_sha256(cfg.root / "configs" / "fields.yaml"),
        "model_registry_sha256": _file_sha256(cfg.root / "configs" / "model_registry.yaml"),
        "source_tree_sha256": _tree_sha256(cfg.root / "corrosion_ie", patterns=("*.py",)),
        "prompt_tree_sha256": _tree_sha256(cfg.root / "prompts", patterns=("*.txt",)),
        "strategy_auxiliary_sha256": strategy_auxiliary_sha256,
        "script_sha256": _selected_files_sha256(cfg.root, [
            cfg.root / "scripts" / "run_matrix.py",
            cfg.root / "scripts" / "run_extraction.py",
            cfg.root / "scripts" / "evaluate_predictions.py",
            cfg.root / "scripts" / "aggregate_runs.py",
            cfg.root / "scripts" / "summarize_prompt_robustness.py",
            cfg.root / "scripts" / "statistical_analysis.py",
            cfg.root / "scripts" / "analyze_document_strata.py",
            cfg.root / "scripts" / "plot_balanced_matrix.py",
            cfg.root / "scripts" / "plot_full_matrix.py",
            cfg.root / "scripts" / "plot_prompt_robustness.py",
            cfg.root / "scripts" / "validate_dataset.py",
            cfg.root / "scripts" / "validate_matrix.py",
            cfg.root / "scripts" / "validate_outputs.py",
            cfg.root / "scripts" / "validate_split_isolation.py",
            cfg.root / "scripts" / "freeze_protocol.py",
            cfg.root / "scripts" / "preflight_v6.py",
            cfg.root / "scripts" / "prepare_v6_test50_protocol.sh",
            cfg.root / "scripts" / "run_v6_test50.sh",
            cfg.root / "scripts" / "capture_environment.py",
            cfg.root / "scripts" / "verify_evidence_consistency.py",
            cfg.root / "scripts" / "summarize_truncation.py",
            cfg.root / "scripts" / "make_report.py",
            cfg.root / "scripts" / "inspect_v6_results.py",
            cfg.root / "corrosion_ie" / "eval" / "integrity.py",
            cfg.root / "vendor" / "legacy_evaluate.py",
            cfg.root / "requirements.txt",
        ]),
        "evaluator_sha256": _file_sha256(cfg.root / "vendor" / "legacy_evaluate.py"),
        "requirements_sha256": _file_sha256(cfg.root / "requirements.txt"),
        "pip_freeze_sha256": _pip_freeze_sha256(),
        "environment_lock_sha256": _tree_sha256(
            cfg.root / "environment_lock", patterns=("*.txt", "*.json", "*.yml", "*.yaml")
        ) if (cfg.root / "environment_lock").exists() else None,
        "dataset_fingerprint_path": str(dataset_fp),
        "dataset_fingerprint_sha256": _file_sha256(dataset_fp) if dataset_fp.exists() else None,
        "gold_sha256": _file_sha256(cfg.gold_path),
        "paper_sha256": {p.name: _file_sha256(p) for p in sorted(cfg.md_dir.glob("*.md"))},
        "models": models,
        "retrieval_models": retrieval_artifacts,
        "runtime_environment": _runtime_environment(),
    }
    payload["protocol_sha256"] = stable_hash(payload)
    return payload


def compare(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(set(expected) | set(actual))
    differences = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key) and key not in {"created_at"}
    }
    return {"match": not differences, "differences": differences}


def main() -> None:
    ap = argparse.ArgumentParser(description="Write or verify a frozen protocol/environment manifest.")
    ap.add_argument("--config", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", default=None, help="Manifest output path.")
    group.add_argument("--verify", default=None, help="Existing manifest to verify.")
    args = ap.parse_args()

    actual = build_manifest(args.config)
    if args.write:
        actual["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, actual)
        print(json.dumps({"written": str(out), "protocol_sha256": actual["protocol_sha256"]}, indent=2))
        return

    path = Path(args.verify)
    if not path.exists():
        raise SystemExit(f"[protocol-error] manifest does not exist: {path}")
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[protocol-error] cannot read manifest {path}: {exc}") from exc
    result = compare(expected, actual)
    print(json.dumps({"manifest": str(path), **result}, ensure_ascii=False, indent=2))
    if not result["match"]:
        raise SystemExit(2)
    print(f"[protocol-ok] {actual['protocol_sha256']}")


if __name__ == "__main__":
    main()
