from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..io_utils import read_jsonl, write_csv
from ..schemas import FIELD_NAMES


def _records_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(obj.get("records"), list):
        return obj["records"]
    res = obj.get("result")
    if isinstance(res, dict):
        if isinstance(res.get("records"), list):
            return res["records"]
        nested = res.get("corrosion_inhibition_analysis")
        if isinstance(nested, dict) and isinstance(nested.get("records"), list):
            return nested["records"]
    return []


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def value_in_text(value: Any, text: str) -> bool:
    if value in [None, ""]:
        return True
    v = _norm(value)
    e = _norm(text)
    if not v:
        return True
    if v in e:
        return True
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return bool(nums) and all(n in text for n in nums[:2])


def load_gold_records(gold_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for obj in read_jsonl(gold_path):
        pid = obj.get("paper_id") or obj.get("id") or obj.get("source_file")
        if not pid:
            continue
        out[pid].extend(_records_from_obj(obj))
    return dict(out)


def diagnose_run(gold_path: str | Path, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    gold = load_gold_records(gold_path)
    ev_text_by_pid = defaultdict(str)
    for ev in read_jsonl(run_dir / "evidence.jsonl"):
        ev_text_by_pid[ev.get("paper_id", "")] += "\n" + str(ev.get("text", ""))

    field_gold = {f: 0 for f in FIELD_NAMES}
    field_hit = {f: 0 for f in FIELD_NAMES}
    for pid, records in gold.items():
        text = ev_text_by_pid.get(pid, "")
        for record in records:
            for field in FIELD_NAMES:
                value = record.get(field)
                if value in [None, "", [], {}]:
                    continue
                field_gold[field] += 1
                if value_in_text(value, text):
                    field_hit[field] += 1

    predicted_non_null = 0
    predicted_linked = 0
    unsupported_flags = 0
    pred_path = run_dir / "predictions.jsonl"
    if pred_path.exists():
        for obj in read_jsonl(pred_path):
            for record in _records_from_obj(obj):
                mapping = record.get("field_evidence") if isinstance(record.get("field_evidence"), dict) else {}
                flags = {str(x) for x in (record.get("verifier_flags") or [])}
                audit = record.get("verifier_audit") if isinstance(record.get("verifier_audit"), dict) else {}
                audit_fields = audit.get("fields") if isinstance(audit.get("fields"), dict) else {}
                if audit_fields:
                    for field, status in audit_fields.items():
                        if field not in FIELD_NAMES or not isinstance(status, dict) or not status.get("predicted"):
                            continue
                        predicted_non_null += 1
                        if status.get("supported"):
                            predicted_linked += 1
                        if status.get("unsupported"):
                            unsupported_flags += 1
                    continue
                for field in FIELD_NAMES:
                    if record.get(field) in [None, "", [], {}]:
                        continue
                    predicted_non_null += 1
                    if mapping.get(field):
                        predicted_linked += 1
                    if f"unsupported:{field}" in flags:
                        unsupported_flags += 1

    out: Dict[str, Any] = {}
    for field in FIELD_NAMES:
        denom = field_gold[field]
        out[f"{field}_evidence_recall"] = field_hit[field] / denom if denom else ""
    out["gold_non_null_fields"] = sum(field_gold.values())
    out["evidence_supported_fields"] = sum(field_hit.values())
    out["micro_evidence_recall"] = sum(field_hit.values()) / max(1, sum(field_gold.values()))
    out["predicted_non_null_fields"] = predicted_non_null
    out["predicted_fields_with_evidence_link"] = predicted_linked
    out["prediction_evidence_support_rate"] = predicted_linked / max(1, predicted_non_null)
    out["predicted_unsupported_flags"] = unsupported_flags
    out["prediction_unsupported_flag_rate"] = unsupported_flags / max(1, predicted_non_null)
    return out


def diagnose_run_root(gold_path: str | Path, run_root: str | Path) -> None:
    run_root = Path(run_root)
    rows = []
    manifest_path = run_root / "matrix_manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dirs = [Path(job["run_dir"]) for job in manifest.get("jobs", [])]
    else:
        run_dirs = [p.parent for p in sorted(run_root.glob("*/*/*/r*/evidence.jsonl"))]
    for run_dir in run_dirs:
        if not (run_dir / "evidence.jsonl").exists():
            continue
        rel = run_dir.relative_to(run_root)
        meta = {
            "model_key": rel.parts[0],
            "strategy": rel.parts[1],
            "prompt_variant": rel.parts[2],
            "repeat": rel.parts[3],
        }
        rows.append({**meta, **diagnose_run(gold_path, run_dir)})
    write_csv(run_root / "evidence_diagnostics.csv", rows)
