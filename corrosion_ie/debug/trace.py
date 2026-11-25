from __future__ import annotations

"""Lightweight debug trace support for extraction runs.

The trace layer is deliberately outside strategy logic.  Strategies add named
payloads to an in-memory ``DebugTrace``; the runner writes those payloads into
stable JSONL/CSV sidecars inside the existing run directory.  This keeps the
command-line entry points and normal prediction/evaluation files unchanged.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..io_utils import append_jsonl, ensure_dir

DEBUG_JSONL_FILES = [
    "retrieval_trace.jsonl",
    "evidence_selected.jsonl",
    "row_table_evidence.jsonl",
    "prompt_snapshot.jsonl",
    "parsed_before_filter.jsonl",
    "grounding_filter_report.jsonl",
    "predictions_after_filter.jsonl",
]
DEBUG_CSV_FILES = ["paper_debug_summary.csv"]


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse a permissive config boolean with a safe fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def cfg_bool(cfg: Mapping[str, Any] | None, key: str, default: bool, aliases: Iterable[str] = ()) -> bool:
    cfg = cfg or {}
    if key in cfg:
        return as_bool(cfg.get(key), default)
    for alias in aliases:
        if alias in cfg:
            return as_bool(cfg.get(alias), default)
    return default


def cfg_str(cfg: Mapping[str, Any] | None, key: str, default: str) -> str:
    cfg = cfg or {}
    value = cfg.get(key, default)
    return str(value if value is not None else default).strip() or default


def feature_flags(*cfgs: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return config-controlled feature flags with backward-compatible defaults.

    Later mappings override earlier mappings.  This lets a strategy-specific
    config override the shared retrieval config without breaking old configs
    that only define the new flags under ``retrieval``.

    Safe framework defaults:
    - enable_grounding_filter: false (the old global substring filter deleted valid fields)
    - enable_row_level_table_evidence: false (disabled in the frozen Q0/Q1/Q2 protocol)
    - enable_debug_trace: true
    - merge_row_table_candidates: false
    """
    cfg: Dict[str, Any] = {}
    for c in cfgs:
        if isinstance(c, Mapping):
            cfg.update(c)
    return {
        "enable_grounding_filter": cfg_bool(cfg, "enable_grounding_filter", False),
        "enable_row_level_table_evidence": cfg_bool(
            cfg,
            "enable_row_level_table_evidence",
            False,
            aliases=("enable_row_table_evidence",),
        ),
        "merge_row_table_candidates": cfg_bool(cfg, "merge_row_table_candidates", False),
        "enable_debug_trace": cfg_bool(cfg, "enable_debug_trace", True),
        "debug_trace_level": cfg_str(cfg, "debug_trace_level", "basic"),
    }


class DebugTrace:
    """In-memory event collector used by strategies and postprocessors."""

    VALID_LEVELS = {"summary", "basic", "full"}
    SUMMARY_FILES = {"retrieval_trace.jsonl", "grounding_filter_report.jsonl"}
    HEAVY_KEYS = {"text", "system_prompt", "user_prompt", "records", "record", "dropped_records", "metadata", "columns"}

    def __init__(self, paper_id: str, enabled: bool = True, level: str = "basic") -> None:
        self.paper_id = paper_id
        self.enabled = bool(enabled)
        level = str(level or "basic").strip().lower()
        self.level = level if level in self.VALID_LEVELS else "basic"
        self.events: Dict[str, List[Dict[str, Any]]] = {name: [] for name in DEBUG_JSONL_FILES}
        self.summary: Dict[str, Any] = {"paper_id": paper_id}

    def _sanitize(self, file_name: str, obj: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        payload = dict(obj)
        if self.level == "full":
            return payload
        if self.level == "summary" and file_name not in self.SUMMARY_FILES:
            return None

        compact: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in self.HEAVY_KEYS:
                if isinstance(value, str):
                    compact[f"{key}_char_count"] = len(value)
                elif isinstance(value, list):
                    compact[f"{key}_count"] = len(value)
                elif isinstance(value, Mapping):
                    compact[f"{key}_key_count"] = len(value)
                else:
                    compact[f"{key}_present"] = value is not None
                continue
            compact[key] = value
        return compact

    def add(self, file_name: str, obj: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload = self._sanitize(file_name, obj)
        if payload is None:
            return
        payload.setdefault("paper_id", self.paper_id)
        payload.setdefault("debug_trace_level", self.level)
        self.events.setdefault(file_name, []).append(payload)

    def add_many(self, file_name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.add(file_name, row)

    def set_summary(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self.summary.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "level": self.level,
            "events": self.events,
            "summary": self.summary,
        }


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    existing_header: List[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
    keys = list(existing_header)
    for k in row.keys():
        if k not in keys:
            keys.append(k)

    # If a later row has new columns, rewrite the file with the expanded header.
    if existing_header and keys != existing_header:
        old_rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for old in old_rows:
                writer.writerow(old)
            writer.writerow(dict(row))
        return

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not existing_header:
            writer.writeheader()
        writer.writerow(dict(row))


def write_debug_trace_outputs(run_dir: Path, result_metadata: Mapping[str, Any] | None) -> None:
    """Write debug sidecars if a result contains a debug trace payload.

    All expected sidecar files are created when tracing is enabled, even when a
    given paper has no row-level table evidence or no filtered records.
    """
    meta = result_metadata or {}
    trace = meta.get("debug_trace")
    if not isinstance(trace, Mapping) or not trace.get("enabled", False):
        return

    ensure_dir(run_dir)
    for file_name in DEBUG_JSONL_FILES:
        (run_dir / file_name).touch(exist_ok=True)

    events = trace.get("events") or {}
    if isinstance(events, Mapping):
        for file_name in DEBUG_JSONL_FILES:
            rows = events.get(file_name) or []
            for row in rows:
                if isinstance(row, Mapping):
                    append_jsonl(run_dir / file_name, dict(row))

    summary = trace.get("summary") or {}
    if isinstance(summary, Mapping) and summary:
        _append_csv(run_dir / "paper_debug_summary.csv", dict(summary))
    else:
        _append_csv(run_dir / "paper_debug_summary.csv", {"paper_id": meta.get("paper_id", "")})


def debug_output_file_names() -> List[str]:
    return DEBUG_JSONL_FILES + DEBUG_CSV_FILES
