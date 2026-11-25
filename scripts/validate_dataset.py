#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrosion_ie.config import load_config
from corrosion_ie.io_utils import write_json
from corrosion_ie.schemas import FIELD_NAMES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_gold_ids(path: Path) -> list[str]:
    """Read and minimally validate the flat gold JSONL schema.

    Test-set migration stays simple: only Markdown files and one gold JSONL are
    required. The checks prevent silent typos or placeholder all-null records.
    """
    ids: list[str] = []
    allowed = set(FIELD_NAMES)
    if not path.exists():
        raise ValueError(f"gold file not found: {path}")
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
        pid = str(obj.get("paper_id") or "").strip()
        if not pid:
            raise ValueError(f"missing paper_id in {path}:{lineno}")
        records = obj.get("records")
        if not isinstance(records, list):
            raise ValueError(f"records must be a list in {path}:{lineno}")
        seen_records: set[str] = set()
        for ridx, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"record must be an object in {path}:{lineno} record {ridx}")
            unknown = sorted(set(record) - allowed)
            if unknown:
                raise ValueError(f"unknown gold fields in {path}:{lineno} record {ridx}: {unknown}")
            if record and not any(record.get(field) not in [None, "", [], {}] for field in FIELD_NAMES):
                raise ValueError(f"all-null placeholder record in {path}:{lineno} record {ridx}")
            key = json.dumps({field: record.get(field) for field in FIELD_NAMES}, ensure_ascii=False, sort_keys=True)
            # Existing development gold can contain repeated records after
            # source-table normalization. Keep validation migration-friendly;
            # duplicates are handled by the evaluator rather than blocking use.
            seen_records.add(key)
        ids.append(pid)
    return ids


def read_manifest_ids(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "paper_id" not in (rows[0].keys() if rows else []):
        # Header-only manifests are common before upload; report a clear error.
        if path.read_text(encoding="utf-8-sig").strip() == "paper_id":
            return []
        raise ValueError(f"manifest must contain a paper_id column: {path}")
    return [str(r.get("paper_id") or "").strip() for r in rows if str(r.get("paper_id") or "").strip()]


def duplicate_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for value in values:
        if value in seen:
            dup.add(value)
        seen.add(value)
    return sorted(dup)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Markdown and gold JSONL before inference.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--write-fingerprint", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    expected = cfg.expected_paper_count
    if expected is None:
        raise SystemExit("[dataset-error] expected_paper_count is required in the config")

    md_files = sorted(p for p in cfg.md_dir.glob("*.md") if p.is_file())
    md_ids = [p.stem for p in md_files]
    gold_ids = read_gold_ids(cfg.gold_path)
    manifest_ids = read_manifest_ids(cfg.manifest_path) if cfg.manifest_path else []
    table_files = sorted(cfg.table_norm_dir.glob("*.md")) if cfg.table_norm_dir and cfg.table_norm_dir.exists() else []
    table_ids = [p.stem for p in table_files]

    errors: list[str] = []
    for label, ids in [("md", md_ids), ("gold", gold_ids), ("manifest", manifest_ids), ("table_norm", table_ids)]:
        dups = duplicate_values(ids)
        if dups:
            errors.append(f"{label}: duplicate paper_ids: {dups[:10]}")

    if len(md_ids) != expected:
        errors.append(f"md: expected {expected}, found {len(md_ids)}")
    if len(gold_ids) != expected:
        errors.append(f"gold: expected {expected}, found {len(gold_ids)}")
    if cfg.manifest_path and len(manifest_ids) != expected:
        errors.append(f"manifest: expected {expected}, found {len(manifest_ids)}")
    if cfg.require_table_norm and len(table_ids) != expected:
        errors.append(f"table_norm: expected {expected}, found {len(table_ids)}")

    md_set, gold_set = set(md_ids), set(gold_ids)
    if md_set != gold_set:
        errors.append(
            f"md/gold paper_id mismatch: md_only={sorted(md_set-gold_set)[:10]}, "
            f"gold_only={sorted(gold_set-md_set)[:10]}"
        )
    if cfg.manifest_path and set(manifest_ids) != md_set:
        errors.append(
            f"manifest/md paper_id mismatch: manifest_only={sorted(set(manifest_ids)-md_set)[:10]}, "
            f"md_only={sorted(md_set-set(manifest_ids))[:10]}"
        )
    if cfg.require_table_norm and set(table_ids) != md_set:
        errors.append(
            f"table_norm/md paper_id mismatch: table_only={sorted(set(table_ids)-md_set)[:10]}, "
            f"md_only={sorted(md_set-set(table_ids))[:10]}"
        )

    empty_md = [p.name for p in md_files if p.stat().st_size == 0]
    if empty_md:
        errors.append(f"empty Markdown files: {empty_md[:10]}")

    def portable_path(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(cfg.root.resolve()).as_posix()
        except Exception:
            return str(path)

    summary = {
        "dataset_name": cfg.dataset_name,
        "dataset_role": cfg.dataset_role,
        "expected_paper_count": expected,
        "md_count": len(md_ids),
        "gold_count": len(gold_ids),
        "manifest_count": len(manifest_ids),
        "table_norm_count": len(table_ids),
        "md_dir": portable_path(cfg.md_dir),
        "gold_path": portable_path(cfg.gold_path),
        "manifest_path": portable_path(cfg.manifest_path),
        "table_norm_dir": portable_path(cfg.table_norm_dir),
        "valid": not errors,
        "errors": errors,
    }

    if not errors:
        payload = {
            "dataset_name": cfg.dataset_name,
            "paper_ids": sorted(md_ids),
            "gold_sha256": sha256_file(cfg.gold_path),
            "manifest_sha256": sha256_file(cfg.manifest_path) if cfg.manifest_path else None,
            "md_sha256": {p.name: sha256_file(p) for p in md_files},
            "table_norm_sha256": {p.name: sha256_file(p) for p in table_files},
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        summary["dataset_fingerprint"] = hashlib.sha256(encoded).hexdigest()
        if args.write_fingerprint:
            out = cfg.md_dir.parent / "dataset_fingerprint.json"
            write_json(out, {**summary, "fingerprint_payload": payload})
            summary["fingerprint_path"] = str(out)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)
    print(f"[dataset-ok] {cfg.dataset_name}: {expected}/{expected} papers")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[dataset-error] {exc}")
        raise SystemExit(2)
