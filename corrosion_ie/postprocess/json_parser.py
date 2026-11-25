from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import FIELD_NAMES


_GROUP_FIELD_MAP = {
    "inh": "inhibitor_name",
    "inhibitor": "inhibitor_name",
    "inhibitor_name": "inhibitor_name",
    "sub": "substrate",
    "substrate": "substrate",
    "med": "corrosive_medium",
    "medium": "corrosive_medium",
    "corrosive_medium": "corrosive_medium",
    "homo": "homo",
    "lumo": "lumo",
    "dg": "delta_g_ads",
    "delta_g_ads": "delta_g_ads",
    "ΔG_ads": "delta_g_ads",
    "eads": "adsorption_energy",
    "adsorption_energy": "adsorption_energy",
}

_DATA_FIELD_MAP = {
    "med_c": "medium_concentration",
    "medium_concentration": "medium_concentration",
    "inh_c": "inhibitor_concentration",
    "inhibitor_concentration": "inhibitor_concentration",
    "temp": "temperature",
    "temperature": "temperature",
    "time": "immersion_time",
    "immersion_time": "immersion_time",
    "method": "test_method",
    "test_method": "test_method",
    "ie": "inhibition_efficiency",
    "inhibition_efficiency": "inhibition_efficiency",
}


def _compact_row_to_record(row: Any) -> Dict[str, Any] | None:
    if not isinstance(row, list):
        return None
    # The compact protocol has 13 fixed columns. Small local models sometimes
    # omit only trailing null quantum/adsorption fields. Rows with the first
    # nine columns present can be padded safely because the missing positions
    # are unambiguously trailing fields in the frozen schema.
    if not (9 <= len(row) <= len(FIELD_NAMES)):
        return None
    if any(isinstance(value, (dict, list)) for value in row):
        return None
    # Some models repeat the column names as a header row despite instructions.
    if len(row) == len(FIELD_NAMES) and [str(x or "").strip() for x in row] == FIELD_NAMES:
        return None
    padded = list(row) + [None] * (len(FIELD_NAMES) - len(row))
    return {field: padded[idx] for idx, field in enumerate(FIELD_NAMES)}


def _compact_rows_to_records(rows: Any) -> List[Dict[str, Any]] | None:
    if not isinstance(rows, list):
        return None
    converted: List[Dict[str, Any]] = []
    for row in rows:
        record = _compact_row_to_record(row)
        if record is None:
            if rows and all(isinstance(x, dict) for x in rows):
                return None
            # Ignore a single accidental header row, but reject arbitrary shapes.
            if isinstance(row, list) and [str(x or "").strip() for x in row] == FIELD_NAMES:
                continue
            return None
        converted.append(record)
    return converted


def _flatten_inhibitors(obj: Any, paper_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    inhibitors = obj.get("inhibitors")
    if not isinstance(inhibitors, list):
        for key in ["corrosion_inhibition_analysis", "result", "data", "output"]:
            val = obj.get(key)
            if isinstance(val, dict) and isinstance(val.get("inhibitors"), list):
                inhibitors = val.get("inhibitors")
                break
    if not isinstance(inhibitors, list):
        return None

    records: List[Dict[str, Any]] = []
    for group in inhibitors:
        if not isinstance(group, dict):
            continue
        base: Dict[str, Any] = {}
        for src, dst in _GROUP_FIELD_MAP.items():
            if src in group and group.get(src) is not None:
                base[dst] = group.get(src)
        data_items = group.get("data")
        if not isinstance(data_items, list) or not data_items:
            data_items = [{}]
        for item in data_items:
            if not isinstance(item, dict):
                item = {}
            rec = dict(base)
            for src, dst in {**_GROUP_FIELD_MAP, **_DATA_FIELD_MAP}.items():
                if src in item and item.get(src) is not None:
                    rec[dst] = item.get(src)
            if rec:
                records.append(rec)
    return {"paper_id": paper_id, "records": records}


def strip_fences(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _as_records_obj(obj: Any, paper_id: str) -> Optional[Dict[str, Any]]:
    """Convert supported complete JSON shapes into the canonical record object."""
    if isinstance(obj, list):
        compact = _compact_rows_to_records(obj)
        if compact is not None:
            return {"paper_id": paper_id, "records": compact}
        if all(isinstance(row, dict) for row in obj):
            return {"paper_id": paper_id, "records": obj}
        return None

    if isinstance(obj, dict):
        grouped = _flatten_inhibitors(obj, paper_id)
        if grouped is not None:
            return grouped

        if isinstance(obj.get("records"), list):
            compact = _compact_rows_to_records(obj.get("records"))
            if compact is not None:
                return {"paper_id": str(obj.get("paper_id") or paper_id), "records": compact}
            obj.setdefault("paper_id", paper_id)
            return obj

        for key in ["corrosion_inhibition_analysis", "result", "data", "output"]:
            val = obj.get(key)
            if isinstance(val, dict):
                grouped = _flatten_inhibitors(val, paper_id)
                if grouped is not None:
                    return grouped
                if isinstance(val.get("records"), list):
                    compact = _compact_rows_to_records(val.get("records"))
                    return {
                        "paper_id": paper_id,
                        "records": compact if compact is not None else val["records"],
                    }

        field_set = set(FIELD_NAMES)
        if len(field_set & set(obj.keys())) >= 3:
            return {"paper_id": paper_id, "records": [obj]}
    return None


def _try_loads(s: str, paper_id: str) -> Optional[Dict[str, Any]]:
    try:
        return _as_records_obj(json.loads(s), paper_id)
    except Exception:
        return None


def _extract_balanced_values(s: str, opening: str, closing: str) -> List[Any]:
    values: List[Any] = []
    n = len(s)
    for start, ch0 in enumerate(s):
        if ch0 != opening:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, n):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    try:
                        values.append(json.loads(s[start:i + 1]))
                    except Exception:
                        pass
                    break
    return values


def _extract_balanced_dicts(s: str) -> List[Dict[str, Any]]:
    values = [value for value in _extract_balanced_values(s, "{", "}") if isinstance(value, dict)]
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for obj in values:
        key = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(obj)
    return out


def _balanced_json_objects(s: str) -> List[Dict[str, Any]]:
    field_set = set(FIELD_NAMES)
    return [obj for obj in _extract_balanced_dicts(s) if len(field_set & set(obj.keys())) >= 3]


def _balanced_compact_rows(s: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for value in _extract_balanced_values(s, "[", "]"):
        record = _compact_row_to_record(value)
        if record is not None:
            records.append(record)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


def _balanced_grouped_inhibitor_records(s: str, paper_id: str) -> List[Dict[str, Any]]:
    group_keys = set(_GROUP_FIELD_MAP) | {"data"}
    row_only_keys = set(_DATA_FIELD_MAP) - set(_GROUP_FIELD_MAP)
    records: List[Dict[str, Any]] = []
    for obj in _extract_balanced_dicts(s):
        keys = set(obj.keys())
        if not (keys & group_keys) or keys <= row_only_keys:
            continue
        flattened = _flatten_inhibitors({"inhibitors": [obj]}, paper_id)
        if flattened and flattened.get("records"):
            records.extend(flattened["records"])
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for rec in records:
        key = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out


def parse_llm_json(raw: str, paper_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse complete legacy/compact JSON and salvage closed rows if truncated."""
    raw = raw or ""
    raw_clean = strip_fences(raw)

    parsed = _try_loads(raw_clean, paper_id)
    if parsed is not None:
        return parsed, None

    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw, re.I)
    if m:
        parsed = _try_loads(m.group(1).strip(), paper_id)
        if parsed is not None:
            return parsed, None

    # Compact fixed-column rows are the frozen compact-row output protocol.  Salvage them
    # before object salvage because an incomplete outer array can still contain
    # many complete 13-value rows.
    compact = _balanced_compact_rows(raw_clean)
    if compact:
        return (
            {"paper_id": paper_id, "records": compact},
            f"salvaged {len(compact)} compact rows from truncated/invalid JSON",
        )

    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        matches = list(re.finditer(pattern, raw_clean))
        if matches:
            longest = max(matches, key=lambda x: x.end() - x.start())
            parsed = _try_loads(longest.group(0), paper_id)
            if parsed is not None:
                return parsed, None

    salvaged = _balanced_json_objects(raw_clean)
    if salvaged:
        return (
            {"paper_id": paper_id, "records": salvaged},
            f"salvaged {len(salvaged)} complete records from truncated/invalid JSON",
        )

    grouped = _balanced_grouped_inhibitor_records(raw_clean, paper_id)
    if grouped:
        return (
            {"paper_id": paper_id, "records": grouped},
            f"salvaged {len(grouped)} grouped LLM records from truncated/invalid JSON",
        )

    return {"paper_id": paper_id, "records": []}, f"JSON parse error: {raw[:200]}"
