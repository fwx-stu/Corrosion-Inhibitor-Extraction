#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def sha(path: Path) -> str:
    """Semantic JSONL hash, insensitive to object key order."""
    h = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            h.update(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def tokenizer_signature(run_dir: Path) -> str | None:
    resolved = run_dir / "resolved_config.json"
    if not resolved.exists():
        return None
    try:
        obj = json.loads(resolved.read_text(encoding="utf-8"))
        artifact = (((obj.get("fingerprint_payload") or {}).get("model_artifacts") or {}).get("generation_model") or {})
        small = artifact.get("small_file_sha256") or {}
        selected: Dict[str, Any] = {}
        for name, digest in small.items():
            lower = str(name).lower()
            if any(token in lower for token in (
                "tokenizer", "tokenization", "vocab", "merges", "special_tokens", "added_tokens",
            )):
                selected[str(name)] = digest
        if not selected:
            return None
        return hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify deterministic packed evidence across prompt/model conditions."
    )
    ap.add_argument("--run-root", required=True, type=Path)
    args = ap.parse_args()
    manifest_path = args.run_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    warnings: list[str] = []
    within_groups = defaultdict(list)
    for job in manifest["jobs"]:
        within_groups[(job["model_key"], job["strategy"])].append(job)

    within_rows = []
    for (model, strategy), jobs in sorted(within_groups.items()):
        hashes = {}
        for job in jobs:
            path = Path(job["run_dir"]) / "evidence.jsonl"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            hashes[job["prompt_variant"]] = sha(path)
        expected = {str(job["prompt_variant"]) for job in jobs}
        identical = len(hashes) == len(expected) and len(set(hashes.values())) == 1
        within_rows.append({
            "model": model,
            "strategy": strategy,
            "identical_across_policies": identical,
            "hashes": hashes,
        })
        if not identical:
            failures.append(f"{model}/{strategy}: evidence differs across policies")

    cross = defaultdict(dict)
    tokenizer_sigs = defaultdict(dict)
    for job in manifest["jobs"]:
        run_dir = Path(job["run_dir"])
        path = run_dir / "evidence.jsonl"
        key = (job["strategy"], job["prompt_variant"])
        if path.exists():
            cross[key][job["model_key"]] = sha(path)
            tokenizer_sigs[key][job["model_key"]] = tokenizer_signature(run_dir)

    cross_rows = []
    for (strategy, policy), hashes in sorted(cross.items()):
        signatures = tokenizer_sigs[(strategy, policy)]
        identical = len(set(hashes.values())) == 1
        known = [sig for sig in signatures.values() if sig]
        same_tokenizer = bool(known) and len(known) == len(signatures) and len(set(known)) == 1
        status = "pass"
        if not identical and same_tokenizer:
            status = "fail"
            failures.append(
                f"{strategy}/{policy}: packed evidence differs across models despite identical tokenizer signatures"
            )
        elif not identical:
            status = "warning"
            warnings.append(
                f"{strategy}/{policy}: packed evidence differs across models; tokenizer signatures differ or are unavailable"
            )
        cross_rows.append({
            "strategy": strategy,
            "policy": policy,
            "identical_across_models": identical,
            "tokenizer_signatures_identical": same_tokenizer,
            "status": status,
            "hashes": hashes,
            "tokenizer_signatures": signatures,
        })

    out = args.run_root / "evidence_consistency.json"
    out.write_text(json.dumps({
        "within_model_policy_check": within_rows,
        "across_model_check": cross_rows,
        "valid": not failures,
        "errors": failures,
        "warnings": warnings,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for row in within_rows:
        print(f"[{'PASS' if row['identical_across_policies'] else 'FAIL'}] {row['model']}/{row['strategy']}")
    for row in cross_rows:
        print(f"[{row['status'].upper()}] {row['strategy']}/{row['policy']} across models")
    for item in warnings:
        print("[warning]", item)
    if failures:
        for item in failures:
            print("[error]", item)
        return 2
    print("[evidence-ok]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
