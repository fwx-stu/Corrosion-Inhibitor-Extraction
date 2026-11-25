from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ExperimentConfig, resolve_model_config
from .debug.trace import debug_output_file_names, write_debug_trace_outputs
from .io_utils import append_jsonl, ensure_dir, read_jsonl, write_json
from .llm.clients import build_client, generation_from_model_cfg
from .schemas import RunIdentity
from .strategies.common import load_fields_config
from .strategies.factory import build_strategy, normalize_strategy_name


def load_md_docs(md_dir: Path, limit: Optional[int] = None, paper_ids: Optional[List[str]] = None) -> List[Path]:
    files = sorted(md_dir.glob("*.md"))
    if paper_ids:
        keep = set(paper_ids)
        files = [p for p in files if p.stem in keep]
    if limit:
        files = files[:limit]
    return files


class QualityGateError(RuntimeError):
    """A complete run failed a predeclared quality threshold and must be reviewed."""


_VERIFIED_FORMAL_MANIFESTS: set[tuple[str, str]] = set()
_FORMAL_FORBIDDEN_GENERATION_ENV = (
    "GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_NEW_TOKENS",
    "GEN_MAX_INPUT_TOKENS", "GEN_DO_SAMPLE",
)
_ARTIFACT_SIGNATURE_CACHE: Dict[Tuple[str, bool], Dict[str, Any]] = {}


def _verify_formal_protocol_manifest(cfg: ExperimentConfig) -> None:
    value = os.getenv("PROTOCOL_MANIFEST", "").strip()
    if not value:
        raise RuntimeError(
            "Formal test requires PROTOCOL_MANIFEST pointing to a manifest created by "
            "scripts/freeze_protocol.py --write."
        )
    manifest = Path(value).expanduser()
    if not manifest.is_absolute():
        manifest = (cfg.root / manifest).resolve()
    else:
        manifest = manifest.resolve()
    if not manifest.exists():
        raise RuntimeError(f"formal-test protocol manifest does not exist: {manifest}")
    key = (str(cfg.config_path.resolve()), str(manifest))
    if key in _VERIFIED_FORMAL_MANIFESTS:
        return
    script = cfg.root / "scripts" / "freeze_protocol.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(cfg.config_path),
            "--verify",
            str(manifest),
        ],
        cwd=str(cfg.root),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"formal-test protocol verification failed: {detail}")
    _VERIFIED_FORMAL_MANIFESTS.add(key)


def _gold_nonempty_paper_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for row in read_jsonl(path):
        pid = str(row.get("paper_id") or "")
        if pid and isinstance(row.get("records"), list) and len(row.get("records") or []) > 0:
            out.add(pid)
    return out


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _runtime_environment() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _safe_package_version("torch"),
        "transformers": _safe_package_version("transformers"),
        "sentence_transformers": _safe_package_version("sentence-transformers"),
        "numpy": _safe_package_version("numpy"),
        "scipy": _safe_package_version("scipy"),
    }
    try:
        import torch  # type: ignore
        result.update({
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        result["torch_runtime_error"] = repr(exc)
    return result


def _artifact_signature(path_value: Any, *, hash_weights: bool = False, progress_label: str | None = None) -> Dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False}
    path = Path(str(path_value)).expanduser().resolve()
    cache_key = (str(path), bool(hash_weights))
    if cache_key in _ARTIFACT_SIGNATURE_CACHE:
        return dict(_ARTIFACT_SIGNATURE_CACHE[cache_key])
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    if path.is_file():
        result.update({"size": path.stat().st_size, "sha256": _file_sha256(path)})
        return result
    small_hashes: Dict[str, str] = {}
    weight_manifest: List[Dict[str, Any]] = []
    for file in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = file.relative_to(path).as_posix()
        lower = file.name.lower()
        size = file.stat().st_size
        if lower.endswith((".safetensors", ".bin", ".pt", ".pth")):
            item = {"name": rel, "size": size}
            if hash_weights:
                if progress_label:
                    print(f"[protocol-hash] {progress_label}: {rel} ({size} bytes)", flush=True)
                item["sha256"] = _file_sha256(file)
            weight_manifest.append(item)
        elif size <= 32 * 1024 * 1024 and lower.endswith((".json", ".txt", ".model", ".py", ".yaml", ".yml")):
            small_hashes[rel] = _file_sha256(file)
    result["small_file_sha256"] = small_hashes
    result["weight_manifest"] = weight_manifest
    result["directory_signature"] = _stable_fingerprint({"small": small_hashes, "weights": weight_manifest})
    _ARTIFACT_SIGNATURE_CACHE[cache_key] = dict(result)
    return result


def _row_quality_flags(row: Dict[str, Any], nonempty_gold_ids: set[str] | None = None) -> Dict[str, bool]:
    records = ((row.get("result") or {}).get("records") or [])
    metadata = row.get("metadata") or {}
    generation = metadata.get("generation") or {}
    pid = str(row.get("paper_id") or "")
    empty_eligible = nonempty_gold_ids is None or pid in nonempty_gold_ids
    return {
        "parse_error": bool(row.get("parse_error")),
        "empty_prediction": bool(empty_eligible and len(records) == 0),
        "empty_prediction_eligible": bool(empty_eligible),
        "input_over_budget": bool(generation.get("input_over_budget")),
        "input_truncated": bool(generation.get("truncated")),
        "output_truncated": bool(generation.get("reached_max_new_tokens")) or generation.get("finish_reason") == "length",
    }


def validate_run_directory(
    run_dir: Path,
    expected_fingerprint: str | None = None,
    enforce_quality: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate structural completeness and optionally the configured quality gate."""
    pred_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "run_meta.json"
    details: Dict[str, Any] = {"run_dir": str(run_dir)}
    if not pred_path.exists() or not meta_path.exists():
        details["reason"] = "missing predictions.jsonl or run_meta.json"
        return False, details
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = list(read_jsonl(pred_path))
    except Exception as exc:
        details["reason"] = f"unreadable output: {exc!r}"
        return False, details
    paper_ids = [str(row.get("paper_id") or "") for row in rows]
    unique_ids = {pid for pid in paper_ids if pid}
    docs_total = int(meta.get("docs_total", -1))
    docs_ok = int(meta.get("docs_ok", -1))
    docs_failed = int(meta.get("docs_failed", -1))
    stored_fingerprint = str(meta.get("config_fingerprint") or "")
    fingerprint_matches = expected_fingerprint is None or stored_fingerprint == expected_fingerprint
    structural_complete = (
        fingerprint_matches
        and docs_total >= 0
        and docs_ok == docs_total
        and docs_failed == 0
        and len(rows) == docs_total
        and len(unique_ids) == docs_total
        and len(paper_ids) == len(unique_ids)
    )
    flags = [_row_quality_flags(row) for row in rows]
    quality_counts = {
        key: sum(int(flag[key]) for flag in flags)
        for key in ["parse_error", "empty_prediction", "input_over_budget", "input_truncated", "output_truncated"]
    }
    quality_pass = bool(meta.get("quality_gate_pass", True))
    valid = structural_complete and (quality_pass if enforce_quality else True)
    details.update({
        "docs_total": docs_total,
        "docs_ok": docs_ok,
        "docs_failed": docs_failed,
        "prediction_rows": len(rows),
        "unique_paper_ids": len(unique_ids),
        "stored_fingerprint": stored_fingerprint,
        "expected_fingerprint": expected_fingerprint,
        "fingerprint_matches": fingerprint_matches,
        "structural_complete": structural_complete,
        "quality_gate_pass": quality_pass,
        "quality_counts": quality_counts,
        "complete": valid,
    })
    if not fingerprint_matches:
        details["reason"] = "stored result was produced by a different effective configuration"
    elif not structural_complete:
        details["reason"] = "counts do not describe a complete one-row-per-paper run"
    elif enforce_quality and not quality_pass:
        details["reason"] = "run is structurally complete but failed the configured quality gate"
    return valid, details


def is_complete(run_dir: Path, expected_fingerprint: str | None = None) -> bool:
    valid, _ = validate_run_directory(run_dir, expected_fingerprint=expected_fingerprint)
    return valid


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tree_sha256(root: Path, patterns: tuple[str, ...] = ("*.py", "*.txt", "*.yaml")) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return ""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    for path in sorted({p.resolve() for p in files if p.is_file()}):
        rel = path.relative_to(root.resolve()).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(_file_sha256(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _selected_files_sha256(root: Path, files: Sequence[Path]) -> str:
    h = hashlib.sha256()
    root = root.resolve()
    for path in sorted({p.resolve() for p in files if p.exists() and p.is_file()}):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_file_sha256(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _relevant_protocol_files(root: Path, strategy_name: str, prompt_variant: str) -> Tuple[List[Path], List[Path]]:
    common = [
        root / "corrosion_ie" / "llm" / "clients.py",
        root / "corrosion_ie" / "prompts.py",
        root / "corrosion_ie" / "strategies" / "base.py",
        root / "corrosion_ie" / "strategies" / "common.py",
        root / "corrosion_ie" / "strategies" / "token_packing.py",
        root / "corrosion_ie" / "strategies" / "table_normalization.py",
        root / "corrosion_ie" / "postprocess" / "json_parser.py",
        root / "corrosion_ie" / "postprocess" / "normalizer.py",
        root / "corrosion_ie" / "postprocess" / "grounding_filter.py",
        root / "corrosion_ie" / "schemas.py",
    ]
    policy = root / "prompts" / "policies" / f"{prompt_variant}.txt"
    if strategy_name == "q0_safe_full":
        sources = common + [root / "corrosion_ie" / "strategies" / "q0_full.py"]
        prompts = [policy, root / "prompts" / "templates" / "q0_full.txt"]
    elif strategy_name == "q1_budgeted_rag":
        sources = common + [
            root / "corrosion_ie" / "strategies" / "q1_budgeted_rag.py",
            root / "corrosion_ie" / "retrieval" / "legacy_hybrid.py",
            root / "corrosion_ie" / "retrieval" / "legacy_exact_chunking.py",
            root / "corrosion_ie" / "retrieval" / "local.py",
        ]
        prompts = [policy, root / "prompts" / "templates" / "q1_rag.txt"]
    elif strategy_name == "q2_table_aware_rag":
        sources = common + [
            root / "corrosion_ie" / "strategies" / "q2_table_aware_rag.py",
            root / "corrosion_ie" / "retrieval" / "legacy_hybrid.py",
            root / "corrosion_ie" / "retrieval" / "legacy_exact_chunking.py",
            root / "corrosion_ie" / "retrieval" / "local.py",
            root / "corrosion_ie" / "retrieval" / "table_rows.py",
        ]
        prompts = [policy, root / "prompts" / "templates" / "q1_rag.txt"]
    elif strategy_name in {"b0_uniform_token_matched", "b1_section_aware_token_matched"}:
        sources = common + [
            root / "corrosion_ie" / "strategies" / "non_rag_compression.py",
            root / "corrosion_ie" / "retrieval" / "chunker_factory.py",
            root / "corrosion_ie" / "retrieval" / "legacy_exact_chunking.py",
        ]
        prompts = [policy, root / "prompts" / "templates" / "q1_rag.txt"]
    elif strategy_name in {"a1_q1_table_budget", "a2_q1_condition_anchors"}:
        sources = common + [
            root / "corrosion_ie" / "strategies" / "q2_component_ablations.py",
            root / "corrosion_ie" / "retrieval" / "legacy_hybrid.py",
            root / "corrosion_ie" / "retrieval" / "legacy_exact_chunking.py",
            root / "corrosion_ie" / "retrieval" / "local.py",
            root / "corrosion_ie" / "retrieval" / "table_rows.py",
        ]
        prompts = [policy, root / "prompts" / "templates" / "q1_rag.txt"]
    else:
        raise KeyError(f"unsupported active strategy for protocol fingerprint: {strategy_name}")
    return sources, prompts


def _strategy_auxiliary_files(root: Path, strategy_cfg: Dict[str, Any]) -> List[Path]:
    """Resolve data files whose contents define a strategy protocol.

    Existing Q0–Q2 configurations have no auxiliary paths and are unchanged.
    Reviewer token-matched controls use a frozen per-paper budget manifest,
    which must be fingerprinted together with source code and prompts.
    """

    files: List[Path] = []
    for key, value in sorted((strategy_cfg or {}).items()):
        if not key.endswith("_path") or not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"strategy auxiliary file does not exist: {key}={path}")
        files.append(path)
    return files


def _stable_fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _cleanup_output_files(run_dir: Path) -> None:
    """Remove partial JSONL/status files before re-running an incomplete run.

    The runner writes JSONL files incrementally. If a previous process was
    interrupted before run_meta.json was written, appending to those files would
    duplicate papers or mix outputs from different generation settings.
    """
    for name in [
        "predictions.jsonl",
        "evidence.jsonl",
        "raw_outputs.jsonl",
        "failures.jsonl",
        "status.json",
        "run_meta.json",
        "resolved_config.json",
    ] + debug_output_file_names():
        p = run_dir / name
        if p.exists():
            p.unlink()
    evaluation_dir = run_dir / "evaluation"
    if evaluation_dir.exists():
        shutil.rmtree(evaluation_dir)




def _rewrite_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with canonical rows."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _prepare_partial_resume(
    pred_path: Path,
    evidence_path: Path,
    raw_path: Path,
) -> set[str]:
    """Return papers with both prediction and raw-output rows.

    Evidence can legitimately be empty for a paper, so it cannot be required as
    a completion marker. The runner writes raw output only after all evidence rows;
    therefore a prediction+raw pair is the canonical completed-paper marker.
    """
    if not (pred_path.exists() and evidence_path.exists() and raw_path.exists()):
        return set()
    predictions = list(read_jsonl(pred_path))
    raw_rows = list(read_jsonl(raw_path))
    evidence_rows = list(read_jsonl(evidence_path))

    pred_ids = [str(row.get("paper_id") or "") for row in predictions]
    raw_ids = [str(row.get("paper_id") or "") for row in raw_rows]
    if (
        any(not pid for pid in pred_ids + raw_ids)
        or len(pred_ids) != len(set(pred_ids))
        or len(raw_ids) != len(set(raw_ids))
    ):
        raise ValueError("duplicate or empty paper IDs in partial canonical outputs")

    raw_set = set(raw_ids)
    completed_order = [pid for pid in pred_ids if pid in raw_set]
    completed = set(completed_order)
    if not completed:
        return set()

    pred_by_id = {str(row["paper_id"]): row for row in predictions}
    raw_by_id = {str(row["paper_id"]): row for row in raw_rows}
    _rewrite_jsonl(pred_path, [pred_by_id[pid] for pid in completed_order])
    _rewrite_jsonl(raw_path, [raw_by_id[pid] for pid in completed_order])
    _rewrite_jsonl(
        evidence_path,
        [row for row in evidence_rows if str(row.get("paper_id") or "") in completed],
    )
    return completed

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_sec(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def _progress_mode(mode: str) -> str:
    mode = (mode or "auto").lower().strip()
    if mode not in {"auto", "tqdm", "plain", "none"}:
        raise ValueError("progress must be one of: auto, tqdm, plain, none")
    if mode == "auto":
        return "tqdm" if sys.stderr.isatty() else "plain"
    return mode


def _maybe_tqdm(items: Sequence[Path], desc: str, mode: str):
    """Return an iterator and optional tqdm bar.

    In log files or notebook-like terminals, tqdm can be unreadable.  In those
    cases use --progress plain, which prints one completed-document line.
    """
    mode = _progress_mode(mode)
    if mode != "tqdm":
        return iter(items), None, mode
    try:
        from tqdm.auto import tqdm  # type: ignore
        bar = tqdm(items, total=len(items), desc=desc, unit="doc", dynamic_ncols=True)
        return iter(bar), bar, mode
    except Exception:
        return iter(items), None, "plain"


def run_single_experiment(
    cfg: ExperimentConfig,
    model_key: str,
    strategy_name: str,
    prompt_variant: str,
    seed: int,
    force: bool = False,
    limit: Optional[int] = None,
    paper_id: Optional[str] = None,
    paper_ids: Optional[List[str]] = None,
    stop_on_error: bool = False,
    fail_on_quality_gate: bool = False,
    dry_run: bool = False,
    progress: str = "auto",
    verbose_docs: bool = False,
) -> Path:
    strategy_name = normalize_strategy_name(strategy_name)
    selected_paper_ids = list(paper_ids or ([] if paper_id is None else [paper_id]))
    formal_test = cfg.dataset_role.lower() == "test" and limit is None and not selected_paper_ids and not dry_run
    if formal_test and os.getenv("CONFIRM_FROZEN_TEST", "") != "YES":
        raise RuntimeError(
            "Formal test is locked. Export CONFIRM_FROZEN_TEST=YES after freezing the protocol."
        )
    if formal_test:
        forbidden = {name: os.getenv(name) for name in _FORMAL_FORBIDDEN_GENERATION_ENV if os.getenv(name) is not None}
        if forbidden:
            raise RuntimeError(f"GEN_* overrides are forbidden in formal test: {sorted(forbidden)}")
        _verify_formal_protocol_manifest(cfg)
    if formal_test and force and os.getenv("ALLOW_TEST_RESET", "") != "YES":
        raise RuntimeError("force reruns are disabled for formal test unless ALLOW_TEST_RESET=YES")
    run_id = RunIdentity(cfg.experiment_id, model_key, strategy_name, prompt_variant, int(seed))
    run_dir = ensure_dir(run_id.output_dir(cfg.output_root))
    pred_path = run_dir / "predictions.jsonl"
    evidence_path = run_dir / "evidence.jsonl"
    raw_path = run_dir / "raw_outputs.jsonl"
    fail_path = run_dir / "failures.jsonl"
    status_path = run_dir / "status.json"

    model_cfg = resolve_model_config(model_key, cfg.config_path)
    if cfg.generation:
        model_cfg = dict(model_cfg)
        model_cfg["generation"] = {
            **dict(model_cfg.get("generation", {}) or {}),
            **dict(cfg.generation),
        }
    gen = generation_from_model_cfg(model_cfg, int(seed))
    generation_config = asdict(gen)
    fields_cfg = load_fields_config(cfg.config_path)
    prompts_dir = cfg.config_path.parent.parent / "prompts"
    strat_cfg = cfg.strategy_params(strategy_name)
    if cfg.table_norm_dir is not None:
        strat_cfg["_table_norm_dir"] = str(cfg.table_norm_dir)
    retrieval_cfg = dict(cfg.retrieval)
    retrieval_cfg["_persistent_cache_root"] = str(cfg.root / "cache" / "retrieval" / cfg.dataset_name)
    docs = load_md_docs(cfg.md_dir, limit=limit, paper_ids=selected_paper_ids or None)
    if selected_paper_ids:
        found = {p.stem for p in docs}
        missing = sorted(set(selected_paper_ids) - found)
        if missing:
            raise RuntimeError(f"requested paper IDs not found under {cfg.md_dir}: {missing}")
    if limit is None and not selected_paper_ids and cfg.expected_paper_count is not None:
        if len(docs) != cfg.expected_paper_count:
            raise RuntimeError(
                f"dataset {cfg.dataset_name} expected {cfg.expected_paper_count} Markdown files; "
                f"found {len(docs)} under {cfg.md_dir}. Run scripts/validate_dataset.py first."
            )

    relevant_sources, relevant_prompts = _relevant_protocol_files(cfg.root, strategy_name, prompt_variant)
    strategy_auxiliary_files = _strategy_auxiliary_files(cfg.root, strat_cfg)
    relevant_sources = list(relevant_sources) + strategy_auxiliary_files
    prompt_tree_sha256 = _tree_sha256(cfg.root / "prompts", patterns=("*.txt",))
    source_tree_sha256 = _tree_sha256(cfg.root / "corrosion_ie", patterns=("*.py",))
    selected_source_sha256 = _selected_files_sha256(cfg.root, relevant_sources)
    selected_prompt_sha256 = _selected_files_sha256(cfg.root, relevant_prompts)
    support_config_sha256 = _selected_files_sha256(cfg.root, [
        cfg.config_path,
        cfg.root / "configs" / "fields.yaml",
        cfg.root / "configs" / "model_registry.yaml",
    ])
    input_hashes = {md.name: _file_sha256(md) for md in docs}
    table_norm_hashes = {}
    runtime_environment = _runtime_environment()
    resolved_generation_path = model_cfg.get("resolved_path")
    if resolved_generation_path:
        generation_artifact = _artifact_signature(resolved_generation_path)
        generation_artifact["model_id"] = model_cfg.get("model_id")
        generation_artifact["source"] = "resolved_path"
    else:
        generation_artifact = {
            "model_id": model_cfg.get("model_id"),
            "path": None,
            "exists": False,
            "source": "model_id",
        }
    model_artifacts = {
        "generation_model": generation_artifact,
        "embedding_model": _artifact_signature(retrieval_cfg.get("embedding_model")),
        "reranker_model": _artifact_signature(retrieval_cfg.get("reranker_model")),
    }
    fingerprint_payload = {
        "experiment_id": cfg.experiment_id,
        "dataset_name": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "expected_paper_count": cfg.expected_paper_count,
        "model_key": model_key,
        "model_config": model_cfg,
        "generation_config": generation_config,
        "runtime_environment": runtime_environment,
        "model_artifacts": model_artifacts,
        "retrieval_config": retrieval_cfg,
        "strategy": strategy_name,
        "strategy_config": strat_cfg,
        "quality_gate_config": dict(cfg.raw.get("quality_gate", {}) or {}),
        "prompt_variant": prompt_variant,
        "prompt_tree_sha256": prompt_tree_sha256,
        "source_tree_sha256": source_tree_sha256,
        "selected_source_sha256": selected_source_sha256,
        "strategy_auxiliary_sha256": {
            str(path.relative_to(cfg.root) if path.is_relative_to(cfg.root) else path): _file_sha256(path)
            for path in strategy_auxiliary_files
        },
        "selected_prompt_sha256": selected_prompt_sha256,
        "support_config_sha256": support_config_sha256,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "limit": limit,
        "paper_id": paper_id,
        "paper_ids": selected_paper_ids,
        "input_hashes": input_hashes,
        "table_norm_hashes": table_norm_hashes,
    }
    config_fingerprint = _stable_fingerprint(fingerprint_payload)

    existing_valid, existing_details = validate_run_directory(
        run_dir, expected_fingerprint=config_fingerprint, enforce_quality=False
    )
    if existing_valid and not force:
        if fail_on_quality_gate and not existing_details.get("quality_gate_pass", True):
            raise QualityGateError(
                f"completed run failed its quality gate and is preserved at {run_dir}; "
                "review it and use a new experiment_id after any protocol change"
            )
        print(f"[skip] {run_id.run_name} fingerprint={config_fingerprint[:12]} -> {run_dir}", flush=True)
        return run_dir

    resumed_ids: set[str] = set()
    if force:
        _cleanup_output_files(run_dir)
    elif any(p.exists() for p in [pred_path, evidence_path, raw_path, fail_path, status_path, run_dir / "run_meta.json"]):
        stored = {}
        resolved_path = run_dir / "resolved_config.json"
        if resolved_path.exists():
            try:
                stored = json.loads(resolved_path.read_text(encoding="utf-8"))
            except Exception:
                stored = {}
        if stored.get("config_fingerprint") == config_fingerprint and pred_path.exists():
            try:
                resumed_ids = _prepare_partial_resume(pred_path, evidence_path, raw_path)
                if not resumed_ids:
                    raise ValueError("no paper has a complete prediction/raw-output pair")
                # Failed or status rows from an interrupted attempt are not
                # part of the canonical completed-paper set and are retried.
                for transient in [fail_path, status_path, run_dir / "run_meta.json"]:
                    if transient.exists():
                        transient.unlink()
                evaluation_dir = run_dir / "evaluation"
                if evaluation_dir.exists():
                    shutil.rmtree(evaluation_dir)
                print(
                    f"[resume] fingerprint matches; preserving {len(resumed_ids)} completed papers -> {run_dir}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[rerun] partial outputs are not resumable ({exc!r}); clearing -> {run_dir}", flush=True)
                resumed_ids = set()
                _cleanup_output_files(run_dir)
        else:
            print(
                f"[rerun] incomplete or stale result detected; clearing outputs -> {run_dir}",
                flush=True,
            )
            _cleanup_output_files(run_dir)

    client = build_client(model_cfg, dry_run=dry_run)
    strategy = build_strategy(
        strategy_name, strat_cfg, fields_cfg, prompts_dir, prompt_variant, retrieval_cfg
    )

    write_json(run_dir / "resolved_config.json", {
        "config_fingerprint": config_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "config_path": str(cfg.config_path),
    })

    print(
        f"[run-start] {run_id.run_name} docs={len(docs)} "
        f"fingerprint={config_fingerprint[:12]} out={run_dir}",
        flush=True,
    )

    start = time.time()
    nonempty_gold_ids = _gold_nonempty_paper_ids(cfg.gold_path)
    existing_predictions = list(read_jsonl(pred_path)) if resumed_ids and pred_path.exists() else []
    ok = len(existing_predictions)
    failed = 0
    total_records = sum(len(((row.get("result") or {}).get("records") or [])) for row in existing_predictions)
    total_evidence = sum(1 for _ in read_jsonl(evidence_path)) if resumed_ids and evidence_path.exists() else 0
    existing_flags = [_row_quality_flags(row, nonempty_gold_ids) for row in existing_predictions]
    parse_errors = sum(int(flag["parse_error"]) for flag in existing_flags)
    empty_predictions = sum(int(flag["empty_prediction"]) for flag in existing_flags)
    empty_prediction_eligible = sum(int(flag["empty_prediction_eligible"]) for flag in existing_flags)
    input_over_budget = sum(int(flag["input_over_budget"]) for flag in existing_flags)
    input_truncations = sum(int(flag["input_truncated"]) for flag in existing_flags)
    output_truncations = sum(int(flag["output_truncated"]) for flag in existing_flags)
    doc_times: List[float] = []
    docs_to_run = [md for md in docs if md.stem not in resumed_ids]

    iterator, bar, actual_mode = _maybe_tqdm(docs_to_run, desc=run_id.run_name, mode=progress)
    for local_idx, md in enumerate(iterator, start=1):
        idx = len(resumed_ids) + local_idx
        pid = md.stem
        doc_start = time.time()
        records_n = 0
        evidence_n = 0
        parse_error = None
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
            result = strategy.run(pid, text, client, model_cfg.get("display_name") or model_cfg.get("model_id", model_key), gen)
            public_metadata = {k: v for k, v in (result.metadata or {}).items() if k != "debug_trace"}
            prediction_obj = result.to_prediction_dict()
            prediction_obj["metadata"] = public_metadata
            append_jsonl(pred_path, prediction_obj)
            evs = strategy.evidence_log()
            for ev in evs:
                append_jsonl(evidence_path, ev.to_dict())
            append_jsonl(raw_path, {
                "paper_id": pid,
                "strategy": strategy_name,
                "prompt_variant": prompt_variant,
                "seed": seed,
                "generation_config": generation_config,
                "raw_output": result.raw_output,
                "parse_error": result.parse_error,
                "metadata": public_metadata,
            })
            write_debug_trace_outputs(run_dir, result.metadata)
            ok += 1
            records_n = len(result.records or [])
            evidence_n = len(evs)
            parse_error = result.parse_error
            if parse_error:
                parse_errors += 1
            if pid in nonempty_gold_ids:
                empty_prediction_eligible += 1
                if records_n == 0:
                    empty_predictions += 1
            generation_meta = public_metadata.get("generation") or {}
            input_over_budget += int(bool(generation_meta.get("input_over_budget")))
            input_truncations += int(bool(generation_meta.get("truncated")))
            output_truncations += int(
                bool(generation_meta.get("reached_max_new_tokens"))
                or generation_meta.get("finish_reason") == "length"
            )
            total_records += records_n
            total_evidence += evidence_n
        except Exception as e:
            failed += 1
            obj = {"paper_id": pid, "error": repr(e), "strategy": strategy_name, "model_key": model_key, "prompt_variant": prompt_variant, "seed": seed}
            append_jsonl(fail_path, obj)
            parse_error = repr(e)
            print(f"[error] {pid}: {e!r}", flush=True)
            if stop_on_error:
                raise
        finally:
            dt = time.time() - doc_start
            doc_times.append(dt)
            elapsed = time.time() - start
            avg = elapsed / max(1, local_idx)
            eta = avg * max(0, len(docs_to_run) - local_idx)
            status = {
                "updated_at": _now(),
                "run_name": run_id.run_name,
                "paper_id": pid,
                "doc_index": idx,
                "docs_total": len(docs),
                "docs_ok": ok,
                "docs_failed": failed,
                "parse_errors": parse_errors,
                "empty_predictions": empty_predictions,
                "empty_prediction_eligible": empty_prediction_eligible,
                "input_over_budget": input_over_budget,
                "input_truncations": input_truncations,
                "output_truncations": output_truncations,
                "last_doc_sec": round(dt, 3),
                "elapsed_sec": round(elapsed, 3),
                "eta_sec": round(eta, 3),
                "records_last": records_n,
                "records_total": total_records,
                "evidence_last": evidence_n,
                "evidence_total": total_evidence,
                "last_parse_error": parse_error,
            }
            write_json(status_path, status)
            if bar is not None:
                bar.set_postfix({
                    "ok": ok,
                    "fail": failed,
                    "rec": records_n,
                    "ev": evidence_n,
                    "err": parse_errors,
                    "eta": _fmt_sec(eta),
                })
            elif actual_mode == "plain" or verbose_docs:
                print(
                    f"[doc {idx:03d}/{len(docs):03d}] {pid} "
                    f"ok={ok} fail={failed} records={records_n} evidence={evidence_n} "
                    f"parse_error={'yes' if parse_error else 'no'} dt={_fmt_sec(dt)} eta={_fmt_sec(eta)}",
                    flush=True,
                )
    if bar is not None:
        bar.close()

    elapsed = time.time() - start
    quality_cfg = dict(cfg.raw.get("quality_gate", {}) or {})
    denominator = max(1, ok)
    empty_denominator = max(1, empty_prediction_eligible)
    docs_total = len(docs)
    document_coverage = ok / max(1, docs_total)
    document_failure_rate = failed / max(1, docs_total)
    structural_complete = bool(failed == 0 and ok == docs_total)
    quality_metrics = {
        "document_failure_count": failed,
        "document_failure_rate": round(document_failure_rate, 6),
        "document_coverage": round(document_coverage, 6),
        "structural_complete": structural_complete,
        "parse_error_count": parse_errors,
        "parse_error_rate": round(parse_errors / denominator, 6),
        "empty_prediction_count": empty_predictions,
        "empty_prediction_eligible_count": empty_prediction_eligible,
        "empty_prediction_rate": round(empty_predictions / empty_denominator, 6),
        "input_over_budget_count": input_over_budget,
        "input_truncation_count": input_truncations,
        "output_truncation_count": output_truncations,
        "output_truncation_rate": round(output_truncations / denominator, 6),
    }
    quality_checks = {
        "max_document_failure_rate": document_failure_rate <= float(quality_cfg.get("max_document_failure_rate", 1.0)),
        "max_parse_error_rate": quality_metrics["parse_error_rate"] <= float(quality_cfg.get("max_parse_error_rate", 1.0)),
        "max_empty_prediction_rate": quality_metrics["empty_prediction_rate"] <= float(quality_cfg.get("max_empty_prediction_rate", 1.0)),
        "max_input_over_budget": input_over_budget <= int(quality_cfg.get("max_input_over_budget", 0)),
        "max_input_truncations": input_truncations <= int(quality_cfg.get("max_input_truncations", 0)),
        "max_output_truncation_rate": quality_metrics["output_truncation_rate"] <= float(quality_cfg.get("max_output_truncation_rate", 1.0)),
    }
    quality_gate_pass = bool(all(quality_checks.values()))
    meta = {
        "config_fingerprint": config_fingerprint,
        "experiment_id": cfg.experiment_id,
        "dataset_name": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "expected_paper_count": cfg.expected_paper_count,
        "model_key": model_key,
        "model_config": model_cfg,
        "generation_config": generation_config,
        "runtime_environment": runtime_environment,
        "model_artifacts": model_artifacts,
        "retrieval_config": retrieval_cfg,
        "strategy_config": strat_cfg,
        "strategy": strategy_name,
        "prompt_variant": prompt_variant,
        "seed": seed,
        "md_dir": str(cfg.md_dir),
        "gold_path": str(cfg.gold_path),
        "run_dir": str(run_dir),
        "docs_total": len(docs),
        "docs_resumed": len(resumed_ids),
        "docs_executed_this_process": len(docs_to_run),
        "docs_ok": ok,
        "docs_failed": failed,
        "document_coverage": round(document_coverage, 6),
        "document_failure_rate": round(document_failure_rate, 6),
        "structural_complete": structural_complete,
        "parse_errors": parse_errors,
        "empty_predictions": empty_predictions,
        "empty_prediction_eligible": empty_prediction_eligible,
        "input_over_budget": input_over_budget,
        "input_truncations": input_truncations,
        "output_truncations": output_truncations,
        "quality_gate_config": quality_cfg,
        "quality_metrics": quality_metrics,
        "quality_checks": quality_checks,
        "quality_gate_pass": quality_gate_pass,
        "records_total": total_records,
        "evidence_total": total_evidence,
        "avg_doc_sec_this_process": round(sum(doc_times) / max(1, len(doc_times)), 3),
        "elapsed_sec": round(elapsed, 3),
        "complete": structural_complete,
    }
    write_json(run_dir / "run_meta.json", meta)
    print(
        f"[run-done] {run_id.run_name} ok={ok}/{len(docs)} fail={failed} "
        f"parse_errors={parse_errors} empty={empty_predictions} output_trunc={output_truncations} "
        f"coverage={document_coverage:.3f} complete={structural_complete} "
        f"quality={'PASS' if quality_gate_pass else 'FAIL'} records={total_records} evidence={total_evidence} "
        f"elapsed={_fmt_sec(elapsed)} out={run_dir}",
        flush=True,
    )
    if fail_on_quality_gate and not quality_gate_pass:
        raise QualityGateError(
            f"run completed but failed quality gate: {run_dir}; checks={quality_checks}. "
            "Results were preserved. Do not overwrite them; revise the protocol under a new experiment_id."
        )
    return run_dir


def iter_matrix(cfg: ExperimentConfig, models: Optional[List[str]] = None, strategies: Optional[List[str]] = None, prompts: Optional[List[str]] = None, seeds: Optional[List[int]] = None):
    for model in models or cfg.models:
        for strat in strategies or cfg.strategies:
            for prompt in prompts or cfg.prompt_variants:
                for seed in seeds or cfg.repeat_seeds:
                    yield model, normalize_strategy_name(strat), prompt, int(seed)
