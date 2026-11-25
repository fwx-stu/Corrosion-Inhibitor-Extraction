from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def run_integrity(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    pred_path = run_dir / "predictions.jsonl"
    meta = read_json(meta_path) if meta_path.exists() else {}
    docs_total = int(meta.get("docs_total", 0) or 0)
    docs_ok = int(meta.get("docs_ok", 0) or 0)
    docs_failed = int(meta.get("docs_failed", 0) or 0)
    complete = bool(
        meta
        and pred_path.exists()
        and meta.get("complete") is True
        and docs_total > 0
        and docs_ok == docs_total
        and docs_failed == 0
    )
    coverage = float(meta.get("document_coverage", docs_ok / docs_total if docs_total else 0.0) or 0.0)
    return {
        "meta_exists": bool(meta),
        "predictions_exists": pred_path.exists(),
        "complete": complete,
        "docs_total": docs_total,
        "docs_ok": docs_ok,
        "docs_failed": docs_failed,
        "document_coverage": coverage,
        "quality_gate_pass": bool(meta.get("quality_gate_pass", False)) if meta else False,
    }


def expected_evaluation_metadata(cfg: Any) -> Dict[str, Any]:
    docs_dir_value = cfg.evaluation.get("docs_dir") or str(cfg.md_dir)
    docs_dir = (cfg.root / docs_dir_value) if not Path(str(docs_dir_value)).is_absolute() else Path(str(docs_dir_value))
    return {
        "gold_sha256": sha256_file(cfg.gold_path),
        "evaluator_sha256": sha256_file(cfg.root / "vendor" / "legacy_evaluate.py"),
        "docs_dir": str(docs_dir),
        "record_matching_backend": "scipy_hungarian",
    }


def evaluation_integrity(
    run_dir: Path,
    expected_static: Mapping[str, Any] | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    pred = run_dir / "predictions.jsonl"
    out_dir = run_dir / "evaluation"
    summary = out_dir / "summary_metrics.csv"
    meta_path = out_dir / "evaluation_meta.json"
    if not pred.exists():
        return False, "missing predictions.jsonl", {}
    if not summary.exists():
        return False, "missing evaluation/summary_metrics.csv", {}
    if not meta_path.exists():
        return False, "missing evaluation/evaluation_meta.json", {}
    meta = read_json(meta_path)
    if not meta:
        return False, "unreadable evaluation_meta.json", {}
    expected_hash = str(meta.get("prediction_sha256") or "")
    actual_hash = sha256_file(pred)
    if expected_hash != actual_hash:
        return False, "evaluation prediction hash is stale", meta
    if expected_static:
        mismatches = [
            key for key, value in expected_static.items()
            if meta.get(key) != value
        ]
        if mismatches:
            return False, "evaluation protocol metadata is stale: " + ",".join(mismatches), meta
    return True, "ok", meta
