#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腐蚀缓蚀文献挖掘结果评估脚本
================================

适配格式：
  gold / pred 每行一个 JSON，例如：
  {"paper_id": "xml_2166", "records": [{...}, {...}]}

也兼容 LLM 常见输出：
  {"paper_id": "xml_2166", "result": {"corrosion_inhibition_analysis": {"records": [...]}}}
  {"paper_id": "xml_2166", "result": "{...JSON字符串...}"}

核心设计：
  1. 先把 LLM 的“分组合并”记录展开成扁平 records。
  2. 每篇论文内部做一对一记录匹配：优先用 scipy 的匈牙利算法；没有 scipy 时自动退化为贪心匹配。
  3. 数值字段严格匹配：温度、时间、浓度、缓蚀效率、HOMO/LUMO、吸附能等必须数值和单位规范化后完全相等。
  4. 文本字段可控模糊匹配：缓蚀剂名、基底、腐蚀介质、测试方法允许同义词、缩写、括号差异、轻微写法差异。
  5. 主指标只保留 precision / recall / F1 / null_accuracy。
  6. 保留分桶诊断：
     - per_record_count_metrics.csv
     - per_record_count_field_metrics.csv
     - per_doc_length_metrics.csv
     - per_doc_length_field_metrics.csv

常用命令：
  python evaluate.py --gold train.jsonl --pred llm_results.jsonl --output evaluation --strategy LLM

或扫描 results 目录：
  python evaluate.py --gold train.jsonl --results results --output results/evaluation

如果要启用文档长度分桶：
  python evaluate.py --gold train.jsonl --pred llm_results.jsonl --output evaluation --docs-dir path/to/docs
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


getcontext().prec = 28

try:
    import config  # type: ignore
except Exception:
    config = None


# ============================================================
# 1. 基本配置
# ============================================================

DEFAULT_EVAL_FIELDS = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
]

EVAL_FIELDS: List[str] = list(getattr(config, "EVAL_FIELDS", DEFAULT_EVAL_FIELDS))

RECORD_MATCH_THRESHOLD: float = float(getattr(config, "RECORD_MATCH_THRESHOLD", 0.55))

MAX_GROUP_EXPANSION_PER_RECORD: int = int(
    getattr(config, "MAX_GROUP_EXPANSION_PER_RECORD", 300)
)

DEDUP_NORMALIZED_RECORDS: bool = bool(
    getattr(config, "DEDUP_NORMALIZED_RECORDS", True)
)

COUNT_EXTRA_PRED_PAPERS_AS_FP: bool = bool(
    getattr(config, "COUNT_EXTRA_PRED_PAPERS_AS_FP", True)
)

DEFAULT_DOCS_DIR = str(
    getattr(
        config,
        "DOCS_DIR",
        getattr(config, "MARKDOWN_DIR", getattr(config, "MD_DIR", "")),
    )
    if config
    else ""
)

DOC_LENGTH_BINS: List[float] = list(
    getattr(config, "DOC_LENGTH_BINS", [0, 5000, 10000, 20000, 40000, float("inf")])
)

RECORD_COUNT_BINS: List[float] = list(
    getattr(config, "RECORD_COUNT_BINS", [0, 1, 2, 4, 8, float("inf")])
)

NULL_STRINGS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nan",
    "not reported",
    "not specified",
    "not mentioned",
    "unknown",
    "-",
    "--",
    "—",
    "/",
}

# 仅用于数值/枚举型字段的扩展空值等价集合。
# 语义上表示「论文中无可抽取的定量数据」，与真正的空值等价。
# 不要用于 inhibitor_name / corrosive_medium 等自然语言描述字段。
NUMERIC_EMPTY_EQUIVALENTS: frozenset = NULL_STRINGS | {
    "not applicable",
    "not available",
    "not provided",
    "not quantified",
    "not measured",
    "not determined",
    "unreported",
    "未报告",
    "未提及",
    "未说明",
    "未给出",
    "未定量",
    "无",
    "空",
}

TEXT_FIELDS = {"inhibitor_name", "substrate", "corrosive_medium", "test_method"}

NUMERIC_FIELDS = {
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
}

VARIATION_FIELDS = {
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
    "homo",
    "lumo",
    "delta_g_ads",
    "adsorption_energy",
}

FIELD_WEIGHTS = {
    "inhibitor_name": 2.0,
    "substrate": 1.4,
    "corrosive_medium": 1.4,
    "medium_concentration": 1.2,
    "inhibitor_concentration": 1.8,
    "temperature": 1.1,
    "immersion_time": 1.0,
    "test_method": 1.6,
    "inhibition_efficiency": 1.4,
    "homo": 0.7,
    "lumo": 0.7,
    "delta_g_ads": 0.8,
    "adsorption_energy": 0.8,
}

TEXT_SYNONYMS = {
    # methods
    "eis": "electrochemical impedance spectroscopy",
    "electrochemical impedance": "electrochemical impedance spectroscopy",
    "electrochemical impedance spectra": "electrochemical impedance spectroscopy",
    "electrochemical impedance spectroscopy measurements": "electrochemical impedance spectroscopy",
    "pdp": "potentiodynamic polarization",
    "tafel polarization": "potentiodynamic polarization",
    "tafel extrapolation": "potentiodynamic polarization",
    "polarization": "potentiodynamic polarization",
    "potentiodynamic polarization curves": "potentiodynamic polarization",
    "wl": "weight loss",
    "mass loss": "weight loss",
    "gravimetric": "weight loss",
    "gravimetric method": "weight loss",
    "weight loss method": "weight loss",
    "lpr": "linear polarization resistance",
    "rp": "linear polarization resistance",
    "svet": "scanning vibrating electrode technique",
    "dft": "density functional theory",
    "density functional theory calculation": "density functional theory",
    "density functional theory calculations": "density functional theory",
    # media/materials
    "hydrochloric acid": "hcl",
    "hydrochloric acid solution": "hcl",
    "sulfuric acid": "h2so4",
    "sulphuric acid": "h2so4",
    "sulfuric acid solution": "h2so4",
    "sulphuric acid solution": "h2so4",
    "phosphoric acid solution": "phosphoric acid",
    "sodium chloride": "nacl",
    "nacl solution": "nacl",
    "sodium chloride solution": "nacl",
    "seawater solution": "seawater",
    "aluminium": "aluminum",
    "ms": "mild steel",
    "mild steel": "mild steel",
    "cs": "carbon steel",
    "carbon steel": "carbon steel",
}

INHIBITOR_SYNONYM_GROUPS: List[List[str]] = [
    ["meftal-p", "mefenamic acid", "meftal-p drug"],
    ["congo red dye (crd)", "crd", "congo red dye", "congo red"],
    ["ginger extract (ge)", "ge", "ginger extract"],
    ["allium sativum l.", "allium sativum", "garlic extract"],
    ["al3+ doped stearic acid (sa-al)", "sa-al"],
    ["stearic acid (sa)", "sa", "stearic acid"],
    ["propylamine", "propylamine (c3h7nh2)", "propylamine (c3h9n)"],
    ["butyrate ion (c4h7o2-)", "calcium butyrate ((c3h7coo)2ca)", "calcium butyrate"],
]

_INHIBITOR_CANONICAL: Dict[str, str] = {}
for group in INHIBITOR_SYNONYM_GROUPS:
    canonical = group[0].lower()
    for alias in group:
        _INHIBITOR_CANONICAL[alias.lower()] = canonical


# ============================================================
# 2. I/O 工具
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows

    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[warn] 跳过非法 JSON: {p}:{lineno}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                print(f"[warn] 跳过非 dict JSON: {p}:{lineno}")

    return rows


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p = Path(path)
    ensure_dir(p.parent)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_paper_id(obj: Dict[str, Any]) -> Optional[str]:
    for key in ("paper_id", "id", "paperId", "paperID", "doi"):
        if obj.get(key):
            return str(obj[key])

    result = obj.get("result")
    if isinstance(result, dict):
        for key in ("paper_id", "id", "paperId", "paperID", "doi"):
            if result.get(key):
                return str(result[key])

    return None


def _try_parse_json_string(v: Any) -> Any:
    if not isinstance(v, str):
        return v

    text = v.strip()
    if not text:
        return v

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        return v


# ============================================================
# 3. records 提取与分组合并展开
# ============================================================

def _find_records(obj: Any) -> Optional[List[Any]]:
    obj = _try_parse_json_string(obj)

    if isinstance(obj, dict):
        cia = obj.get("corrosion_inhibition_analysis")
        if isinstance(cia, dict) and isinstance(cia.get("records"), list):
            return cia["records"]

        for key in ("records", "record", "extractions", "items", "data"):
            val = obj.get(key)
            if isinstance(val, list):
                return val

        for v in obj.values():
            found = _find_records(v)
            if found is not None:
                return found

    elif isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return obj

        for item in obj:
            found = _find_records(item)
            if found is not None:
                return found

    return None


def extract_records(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("records"), list):
        return _normalize_records(obj["records"])

    result = _try_parse_json_string(obj.get("result"))
    if isinstance(result, dict):
        found = _find_records(result)
        if found is not None:
            return _normalize_records(found)

    found = _find_records(obj)
    return _normalize_records(found) if found is not None else []


def _is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip().lower() in NULL_STRINGS
    return False


def normalize_empty(v: Any) -> Any:
    """将数值/枚举型字段中语义等价于「无数据」的值统一归一化为 None。

    仅应在 NUMERIC_FIELDS 的比较路径中调用，不适用于需要保留
    自然语言描述的文本字段（inhibitor_name、corrosive_medium 等）。
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    text = str(v).strip()
    text_stripped = text.lower().strip(" .;:，。；：")
    if text_stripped in NUMERIC_EMPTY_EQUIVALENTS:
        return None
    return v


def is_empty_value(v: Any, field: str) -> bool:
    """统一的空值判断：数值/枚举字段额外识别语义空值（如 'not quantified'）。"""
    if field in NUMERIC_FIELDS and normalize_empty(v) is None:
        return True
    return normalize_text(v) is None


def _split_multivalue_string(s: str, field: str) -> Optional[List[str]]:
    text = s.strip()
    if not text:
        return None

    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("(") and text.endswith(")")
    ):
        try:
            candidate = json.loads(text.replace("'", '"'))
            if isinstance(candidate, list):
                vals = [str(x).strip() for x in candidate if not _is_nullish(x)]
                if len(vals) > 1:
                    return vals
        except Exception:
            pass

    if any(sep in text for sep in [";", "|", "\n"]):
        parts = re.split(r"\s*(?:;|\||\n)+\s*", text)
        parts = [
            p.strip()
            for p in parts
            if p.strip() and p.strip().lower() not in NULL_STRINGS
        ]
        if len(parts) > 1:
            return parts

    if field in VARIATION_FIELDS and re.search(r"\b\d+\)\s*", text):
        parts = re.split(r"\s*\b\d+\)\s*", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts

    return None


def _as_value_list(v: Any, field: str) -> List[Any]:
    if _is_nullish(v):
        return [None]

    if isinstance(v, list):
        vals: List[Any] = []
        for item in v:
            if isinstance(item, list):
                vals.extend(_as_value_list(item, field))
            elif not _is_nullish(item):
                vals.append(item)
        return vals or [None]

    if isinstance(v, tuple):
        vals = [x for x in v if not _is_nullish(x)]
        return vals or [None]

    if isinstance(v, dict):
        for key in ("value", "values", "text", "name"):
            if key in v:
                return _as_value_list(v[key], field)
        return [json.dumps(v, ensure_ascii=False, sort_keys=True)]

    if isinstance(v, str):
        split = _split_multivalue_string(v, field)
        if split:
            return split

    return [v]


def _expand_grouped_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    nested = rec.get("records") or rec.get("items") or rec.get("measurements")
    if isinstance(nested, list) and nested and all(isinstance(x, dict) for x in nested):
        parent = {k: v for k, v in rec.items() if k not in {"records", "items", "measurements"}}
        expanded: List[Dict[str, Any]] = []
        for child in nested:
            merged = {**parent, **child}
            expanded.extend(_expand_grouped_record(merged))
        return expanded

    value_lists = {field: _as_value_list(rec.get(field), field) for field in EVAL_FIELDS}
    multi_lengths = {
        field: len(vals) for field, vals in value_lists.items() if len(vals) > 1
    }

    if not multi_lengths:
        return [{field: value_lists[field][0] for field in EVAL_FIELDS}]

    unique_lengths = sorted(set(multi_lengths.values()))
    max_len = max(unique_lengths)

    if len(unique_lengths) == 1 or all(length in {1, max_len} for length in multi_lengths.values()):
        rows: List[Dict[str, Any]] = []
        for idx in range(max_len):
            row: Dict[str, Any] = {}
            for field in EVAL_FIELDS:
                vals = value_lists[field]
                row[field] = vals[idx] if len(vals) > 1 else vals[0]
            rows.append(row)
        return rows

    multi_fields = [f for f, vals in value_lists.items() if len(vals) > 1]
    product_size = math.prod(len(value_lists[f]) for f in multi_fields)

    if product_size <= MAX_GROUP_EXPANSION_PER_RECORD:
        rows = []
        for combo in itertools.product(*(value_lists[f] for f in multi_fields)):
            row = {field: value_lists[field][0] for field in EVAL_FIELDS}
            for field, value in zip(multi_fields, combo):
                row[field] = value
            rows.append(row)
        return rows

    rows = []
    for idx in range(max_len):
        row = {}
        for field in EVAL_FIELDS:
            vals = value_lists[field]
            row[field] = vals[idx % len(vals)] if len(vals) > 1 else vals[0]
        rows.append(row)

    return rows


def _has_any_eval_value(row: Dict[str, Any]) -> bool:
    return any(not is_empty_value(row.get(field), field) for field in EVAL_FIELDS)


def _normalize_records(records: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue

        for flat in _expand_grouped_record(rec):
            row = {field: flat.get(field) for field in EVAL_FIELDS}
            if _has_any_eval_value(row):
                normalized.append(row)

    if DEDUP_NORMALIZED_RECORDS:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in normalized:
            key = tuple(canonical_for_key(row.get(field), field) for field in EVAL_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        normalized = deduped

    return normalized


# ============================================================
# 4. 文本归一化与模糊匹配
# ============================================================

def _basic_cleanup(v: Any) -> Optional[str]:
    if _is_nullish(v):
        return None

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)

    s = str(v).strip()
    if s.lower() in NULL_STRINGS:
        return None

    s = s.lower()
    s = (
        s.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("·", " ")
        .replace("，", ",")
        .replace("；", ";")
        .replace("：", ":")
    )
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _replace_common_unit_text(s: str) -> str:
    replacements = [
        (r"\bdegrees?\s*c(?:elsius)?\b", "c"),
        (r"°\s*c\b", "c"),
        (r"℃", "c"),
        (r"\bkelvin\b", "k"),
        (r"\bhrs?\b", "h"),
        (r"\bhours?\b", "h"),
        (r"\bminutes?\b", "min"),
        (r"\bmins?\b", "min"),
        (r"\bseconds?\b", "s"),
        (r"\bdays?\b", "d"),
        (r"\bmol\s*[ ]?l\s*[-^]?1\b", "mol/l"),
        (r"\bmol\s*/\s*l\b", "mol/l"),
        (r"\bmolar\b", "m"),
        (r"\bmmol\s*/\s*l\b", "mm"),
        (r"\bumol\s*/\s*l\b", "um"),
        (r"\bmg\s*/\s*l\b", "mg/l"),
        (r"\bg\s*/\s*l\b", "g/l"),
        (r"\bml\s*/\s*l\b", "ml/l"),
        (r"\bwt\s*\.?\s*%", "wt%"),
        (r"\bvol\s*\.?\s*%", "vol%"),
        (r"\bkj\s*mol\s*[-^]?1\b", "kj/mol"),
        (r"\bkcal\s*mol\s*[-^]?1\b", "kcal/mol"),
        (r"\bev\b", "ev"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    return s


@lru_cache(maxsize=300_000)
def _normalize_text_cached(raw: str) -> Optional[str]:
    s = _basic_cleanup(raw)
    if s is None:
        return None

    s = _replace_common_unit_text(s)
    s = re.sub(r"\s*([%/=(),;:])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;,:")
    return TEXT_SYNONYMS.get(s, s) or None


def normalize_text(v: Any) -> Optional[str]:
    if _is_nullish(v):
        return None
    return _normalize_text_cached(str(v))


def _strip_parenthetical(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def _extract_abbreviation(s: str) -> Optional[str]:
    m = re.search(r"\(([a-z0-9+\-]+)\)\s*$", s, flags=re.I)
    return m.group(1).lower() if m else None


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _safe_token_contains(short: str, long: str, min_len: int = 4) -> bool:
    short = short.strip().lower()
    if len(short) < min_len:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", long.lower()))


def _canonical_inhibitor(v: Any) -> Optional[str]:
    s = normalize_text(v)
    if s is None:
        return None

    s = TEXT_SYNONYMS.get(s, s)
    if s in _INHIBITOR_CANONICAL:
        return _INHIBITOR_CANONICAL[s]

    stripped = _strip_parenthetical(s)
    if stripped in _INHIBITOR_CANONICAL:
        return _INHIBITOR_CANONICAL[stripped]

    return stripped or s


def inhibitor_name_match(a: Any, b: Any, relaxed: bool = True) -> bool:
    na, nb = normalize_text(a), normalize_text(b)

    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    if na == nb:
        return True

    ca, cb = _canonical_inhibitor(na), _canonical_inhibitor(nb)
    if ca and cb and ca == cb:
        return True

    abbr_a, abbr_b = _extract_abbreviation(na), _extract_abbreviation(nb)
    if abbr_a and abbr_a == nb:
        return True
    if abbr_b and abbr_b == na:
        return True
    if abbr_a and abbr_b and abbr_a == abbr_b:
        return True

    sa, sb = _strip_parenthetical(na), _strip_parenthetical(nb)
    if sa == sb:
        return True

    if not relaxed:
        return False

    if _safe_token_contains(sa, nb, min_len=5) or _safe_token_contains(sb, na, min_len=5):
        return True

    if min(len(sa), len(sb)) >= 6:
        if _seq_ratio(sa, sb) >= 0.88:
            return True
        if _token_jaccard(sa, sb) >= 0.80:
            return True

    return _alnum(sa) == _alnum(sb) and len(_alnum(sa)) >= 6


def _normalize_method(v: Any) -> Optional[str]:
    s = normalize_text(v)
    if s is None:
        return None

    s = TEXT_SYNONYMS.get(s, s)

    patterns = [
        (r"\beis\b|impedance", "electrochemical impedance spectroscopy"),
        (r"\bpdp\b|potentiodynamic|tafel", "potentiodynamic polarization"),
        (r"\bweight\s*loss\b|\bmass\s*loss\b|gravimetric", "weight loss"),
        (r"\blpr\b|linear polarization resistance", "linear polarization resistance"),
        (r"gasometric", "gasometric method"),
        (r"density functional theory|\bdft\b", "density functional theory"),
        (r"scanning vibrating electrode", "scanning vibrating electrode technique"),
    ]

    for pattern, repl in patterns:
        if re.search(pattern, s):
            return repl

    return s


def _normalize_medium(v: Any) -> Optional[str]:
    s = normalize_text(v)
    if s is None:
        return None

    s = TEXT_SYNONYMS.get(s, s)

    s = re.sub(
        r"\b[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*(?:m|mol/l|mm|um|mg/l|g/l|ppm|wt%|vol%|%)\b",
        "",
        s,
    )
    s = re.sub(r"\b(aqueous|solution|soln|medium|acidic)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;()")
    s = TEXT_SYNONYMS.get(s, s)

    return s or None


def _normalize_substrate(v: Any) -> Optional[str]:
    s = normalize_text(v)
    if s is None:
        return None

    s = TEXT_SYNONYMS.get(s, s)
    s = re.sub(r"\s+", " ", s).strip(" ,;()")
    return s or None


def text_match(a: Any, b: Any, field: str, relaxed: bool = True) -> bool:
    if field == "inhibitor_name":
        return inhibitor_name_match(a, b, relaxed=relaxed)

    if field == "test_method":
        na, nb = _normalize_method(a), _normalize_method(b)
    elif field == "corrosive_medium":
        na, nb = _normalize_medium(a), _normalize_medium(b)
    elif field == "substrate":
        na, nb = _normalize_substrate(a), _normalize_substrate(b)
    else:
        na, nb = normalize_text(a), normalize_text(b)

    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    if na == nb:
        return True

    abbr_a, abbr_b = _extract_abbreviation(na), _extract_abbreviation(nb)
    if abbr_a and abbr_a == nb:
        return True
    if abbr_b and abbr_b == na:
        return True

    if _strip_parenthetical(na) == _strip_parenthetical(nb):
        return True

    if not relaxed:
        return False

    if field == "test_method":
        return _seq_ratio(na, nb) >= 0.86 or _token_jaccard(na, nb) >= 0.75

    if field == "corrosive_medium":
        aa, bb = _alnum(na), _alnum(nb)
        if aa == bb:
            return True
        return _safe_token_contains(na, nb, min_len=3) or _safe_token_contains(nb, na, min_len=3)

    if field == "substrate":
        if _alnum(na) == _alnum(nb):
            return True
        return _seq_ratio(na, nb) >= 0.90 or _token_jaccard(na, nb) >= 0.80

    return _seq_ratio(na, nb) >= 0.90 or _token_jaccard(na, nb) >= 0.85


# ============================================================
# 5. 数值字段严格匹配
# ============================================================

def _decimal_from_str(x: str) -> Optional[Decimal]:
    try:
        return Decimal(x)
    except (InvalidOperation, ValueError):
        return None


def _all_numbers(s: str) -> List[Decimal]:
    vals: List[Decimal] = []
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
        d = _decimal_from_str(m.group(0))
        if d is not None:
            vals.append(d)
    return vals


def _dec_to_str(d: Decimal) -> str:
    # 稳健版：避免超长数字 / 科学计数法 / Decimal 精度不足导致 InvalidOperation
    if not d.is_finite():
        return str(d)

    try:
        from decimal import localcontext
        with localcontext() as ctx:
            ctx.prec = max(50, len(d.as_tuple().digits) + 20)
            q = d.quantize(Decimal("0.000000000001")).normalize()
    except Exception:
        # fallback：不强制量化，只做 normalize
        try:
            q = d.normalize()
        except Exception:
            return str(d)

    try:
        if q == q.to_integral_value():
            return format(q, "f").split(".")[0]
    except Exception:
        pass

    return format(q, "f")


@dataclass(frozen=True)
class NumericValue:
    kind: str
    unit: str
    values: Tuple[str, ...]

    def as_key(self) -> str:
        return f"{self.kind}|{self.unit}|{'~'.join(self.values)}"


def _detect_range(s: str) -> bool:
    if "~" in s or re.search(r"\bto\b", s):
        return True
    return bool(re.search(r"(?<!^)(?<![eE])\s+-\s+", s))


def _normalize_numeric_values(
    values: List[Decimal],
    multiplier: Decimal = Decimal("1"),
    offset: Decimal = Decimal("0"),
) -> Tuple[str, ...]:
    return tuple(_dec_to_str(v * multiplier + offset) for v in values)


def _parse_temperature(v: Any) -> Optional[NumericValue]:
    s = normalize_text(v)
    if s is None:
        return None

    if s in {"room temperature", "rt"}:
        return NumericValue("scalar", "c", ("25",))

    nums = _all_numbers(s)
    if not nums:
        return None

    values = nums[:2] if _detect_range(s) and len(nums) >= 2 else nums[:1]
    kind = "range" if len(values) == 2 else "scalar"

    if re.search(r"\bk\b", s) and "kj" not in s:
        vals = _normalize_numeric_values(values, offset=Decimal("-273.15"))
        return NumericValue(kind, "c", vals)

    if re.search(r"\bc\b", s):
        return NumericValue(kind, "c", _normalize_numeric_values(values))

    return NumericValue(kind, "raw_temperature", _normalize_numeric_values(values))


def _parse_duration(v: Any) -> Optional[NumericValue]:
    s = normalize_text(v)
    if s is None:
        return None

    nums = _all_numbers(s)
    if not nums:
        return NumericValue("text", "duration_text", (_alnum(s) or s,))

    values = nums[:2] if _detect_range(s) and len(nums) >= 2 else nums[:1]
    kind = "range" if len(values) == 2 else "scalar"

    if re.search(r"\bd\b", s):
        return NumericValue(kind, "h", _normalize_numeric_values(values, multiplier=Decimal("24")))

    if re.search(r"\bh\b", s):
        return NumericValue(kind, "h", _normalize_numeric_values(values))

    if re.search(r"\bmin\b", s):
        return NumericValue(
            kind,
            "h",
            _normalize_numeric_values(values, multiplier=Decimal("0.01666666666666666666666666667")),
        )

    if re.search(r"\bs\b", s):
        return NumericValue(
            kind,
            "h",
            _normalize_numeric_values(values, multiplier=Decimal("0.0002777777777777777777777777778")),
        )

    return NumericValue(kind, "raw_duration", _normalize_numeric_values(values))


def _parse_percent(v: Any) -> Optional[NumericValue]:
    s = normalize_text(v)
    if s is None:
        return None

    has_wrong_unit = bool(
        re.search(
            r"\b(?:mg/l|g/l|ppm|mol/l|ml/l|wt%|vol%|kj/mol|kcal/mol|ev|h|min|d|s|c|k)\b",
            s,
        )
    )
    if has_wrong_unit and "%" not in s:
        return None

    nums = _all_numbers(s)
    if not nums:
        return None

    values = nums[:2] if _detect_range(s) and len(nums) >= 2 else nums[:1]
    kind = "range" if len(values) == 2 else "scalar"

    return NumericValue(kind, "%", _normalize_numeric_values(values))


def _parse_concentration(v: Any) -> Optional[NumericValue]:
    s = normalize_text(v)
    if s is None:
        return None

    if "saturated" in s:
        return NumericValue("text", "concentration_text", ("saturated",))

    nums = _all_numbers(s)
    if not nums:
        return NumericValue("text", "concentration_text", (_alnum(s) or s,))

    values = nums[:2] if _detect_range(s) and len(nums) >= 2 else nums[:1]
    kind = "range" if len(values) == 2 else "scalar"

    if re.search(r"\bum\b", s):
        return NumericValue(
            kind,
            "mol/l",
            _normalize_numeric_values(values, multiplier=Decimal("0.000001")),
        )

    if re.search(r"\bmm\b", s):
        return NumericValue(
            kind,
            "mol/l",
            _normalize_numeric_values(values, multiplier=Decimal("0.001")),
        )

    if re.search(r"\bmol/l\b", s) or re.search(r"(?<![a-z])m(?![a-z])", s):
        return NumericValue(kind, "mol/l", _normalize_numeric_values(values))

    if "g/l" in s and "mg/l" not in s:
        return NumericValue(
            kind,
            "mg/l",
            _normalize_numeric_values(values, multiplier=Decimal("1000")),
        )

    if "mg/l" in s or "ppm" in s:
        return NumericValue(kind, "mg/l", _normalize_numeric_values(values))

    if "wt%" in s:
        return NumericValue(kind, "wt%", _normalize_numeric_values(values))

    if "vol%" in s:
        return NumericValue(kind, "vol%", _normalize_numeric_values(values))

    if "%" in s:
        return NumericValue(kind, "%", _normalize_numeric_values(values))

    if "ml/l" in s:
        return NumericValue(kind, "ml/l", _normalize_numeric_values(values))

    return NumericValue(kind, "raw_concentration", _normalize_numeric_values(values))


def _parse_energy(v: Any, field: str) -> Optional[NumericValue]:
    s = normalize_text(v)
    if s is None:
        return None

    nums = _all_numbers(s)
    if not nums:
        return None

    values = nums[:2] if _detect_range(s) and len(nums) >= 2 else nums[:1]
    kind = "range" if len(values) == 2 else "scalar"

    if field in {"homo", "lumo"}:
        return NumericValue(kind, "ev", _normalize_numeric_values(values))

    if "kcal" in s:
        return NumericValue(
            kind,
            "kj/mol",
            _normalize_numeric_values(values, multiplier=Decimal("4.184")),
        )

    if "kj" in s:
        return NumericValue(kind, "kj/mol", _normalize_numeric_values(values))

    if "ev" in s:
        return NumericValue(kind, "ev", _normalize_numeric_values(values))

    return NumericValue(kind, f"raw_{field}", _normalize_numeric_values(values))


def parse_numeric(v: Any, field: str) -> Optional[NumericValue]:
    if field == "temperature":
        return _parse_temperature(v)
    if field == "immersion_time":
        return _parse_duration(v)
    if field == "inhibition_efficiency":
        return _parse_percent(v)
    if field in {"medium_concentration", "inhibitor_concentration"}:
        return _parse_concentration(v)
    if field in {"homo", "lumo", "delta_g_ads", "adsorption_energy"}:
        return _parse_energy(v, field)
    return None


def numeric_match(a: Any, b: Any, field: str) -> bool:
    pa, pb = parse_numeric(a, field), parse_numeric(b, field)

    if pa is None and pb is None:
        return normalize_text(a) == normalize_text(b)

    if pa is None or pb is None:
        return False

    return pa == pb


# ============================================================
# 6. 字段匹配
# ============================================================

def values_match(a: Any, b: Any, field: str, relaxed: bool = True) -> bool:
    # 数值/枚举型字段：先把「未定量」等语义空值归一化为 None，
    # 避免将 LLM 输出的 "not quantified" 误判为幻觉或错误。
    if field in NUMERIC_FIELDS:
        a = normalize_empty(a)
        b = normalize_empty(b)

    na, nb = normalize_text(a), normalize_text(b)

    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False

    if field in NUMERIC_FIELDS:
        return numeric_match(a, b, field)

    if field in TEXT_FIELDS:
        return text_match(a, b, field, relaxed=relaxed)

    if na == nb:
        return True

    return relaxed and _seq_ratio(na, nb) >= 0.92


def canonical_for_key(v: Any, field: str) -> Optional[str]:
    if is_empty_value(v, field):
        return None

    if field in NUMERIC_FIELDS:
        v = normalize_empty(v)  # 先归一化语义空值
        if v is None:
            return None
        nv = parse_numeric(v, field)
        return nv.as_key() if nv is not None else normalize_text(v)

    if field == "inhibitor_name":
        return _canonical_inhibitor(v)

    if field == "test_method":
        return _normalize_method(v)

    if field == "corrosive_medium":
        return _normalize_medium(v)

    if field == "substrate":
        return _normalize_substrate(v)

    return normalize_text(v)


# ============================================================
# 7. 记录匹配与指标
# ============================================================

def record_similarity(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    matched_weight = 0.0
    comparable_weight = 0.0

    for field in EVAL_FIELDS:
        pv, gv = pred.get(field), gold.get(field)

        if is_empty_value(pv, field) and is_empty_value(gv, field):
            continue

        weight = FIELD_WEIGHTS.get(field, 1.0)
        comparable_weight += weight

        if values_match(pv, gv, field=field, relaxed=True):
            matched_weight += weight

    return matched_weight / comparable_weight if comparable_weight else 0.0


def match_records(
    pred_records: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
) -> Tuple[int, List[Tuple[int, int, float]]]:
    if not pred_records or not gold_records:
        return 0, []

    n_pred, n_gold = len(pred_records), len(gold_records)
    scores = [[0.0 for _ in range(n_gold)] for _ in range(n_pred)]

    for i, pred in enumerate(pred_records):
        for j, gold in enumerate(gold_records):
            scores[i][j] = record_similarity(pred, gold)

    matches: List[Tuple[int, int, float]] = []

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        cost_matrix: List[List[float]] = []
        for i in range(n_pred):
            row = []
            for j in range(n_gold):
                score = scores[i][j]
                if score < RECORD_MATCH_THRESHOLD:
                    score = 0.0
                row.append(-score)
            cost_matrix.append(row)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            score = scores[int(i)][int(j)]
            if score >= RECORD_MATCH_THRESHOLD:
                matches.append((int(i), int(j), float(score)))

    except Exception:
        candidates: List[Tuple[int, int, float]] = []
        for i in range(n_pred):
            for j in range(n_gold):
                score = scores[i][j]
                if score >= RECORD_MATCH_THRESHOLD:
                    candidates.append((i, j, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        used_pred, used_gold = set(), set()
        for i, j, score in candidates:
            if i in used_pred or j in used_gold:
                continue
            used_pred.add(i)
            used_gold.add(j)
            matches.append((i, j, score))

    return len(matches), matches


def greedy_match_records(
    pred_records: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
) -> Tuple[int, int, List[Tuple[int, int, float]]]:
    """
    旧接口兼容包装。
    旧代码可能期望返回：
      exact_tp, partial_tp, matches

    新版已经删除 exact 指标，所以这里让 exact_tp = partial_tp，
    避免旧脚本拿到 0 后误判。
    """
    partial_tp, matches = match_records(pred_records, gold_records)
    return partial_tp, partial_tp, matches


def prf(tp: int, pred_n: int, gold_n: int) -> Tuple[float, float, float]:
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gold_n if gold_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


_prf = prf


# ============================================================
# 8. 数据加载
# ============================================================

def _dedup_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not DEDUP_NORMALIZED_RECORDS:
        return records

    seen = set()
    deduped = []

    for row in records:
        key = tuple(canonical_for_key(row.get(f), f) for f in EVAL_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def load_gold(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    gold_map: Dict[str, List[Dict[str, Any]]] = {}
    duplicate_ids: Counter[str] = Counter()

    for obj in read_jsonl(path):
        pid = safe_paper_id(obj)
        if not pid:
            continue

        duplicate_ids[pid] += 1
        gold_map.setdefault(pid, []).extend(extract_records(obj))

    for pid, records in list(gold_map.items()):
        gold_map[pid] = _dedup_records(records)

    dup_paper_count = sum(1 for n in duplicate_ids.values() if n > 1)
    if dup_paper_count:
        print(f"[warn] gold 中有 {dup_paper_count} 个重复 paper_id，已合并 records。")

    return gold_map


def load_predictions(path: str | Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter[str]]:
    pred_map: Dict[str, List[Dict[str, Any]]] = {}
    duplicate_ids: Counter[str] = Counter()

    for obj in read_jsonl(path):
        pid = safe_paper_id(obj)
        if not pid:
            continue

        duplicate_ids[pid] += 1
        pred_map.setdefault(pid, []).extend(extract_records(obj))

    for pid, records in list(pred_map.items()):
        pred_map[pid] = _dedup_records(records)

    return pred_map, duplicate_ids


# ============================================================
# 9. 文档长度读取
# ============================================================

def _candidate_doc_paths(docs_dir: Path, paper_id: str) -> List[Path]:
    exts = [".md", ".txt", ".json", ".jsonl"]
    names = [paper_id]

    safe_name = re.sub(r"[^\w.\-]+", "_", paper_id)
    if safe_name != paper_id:
        names.append(safe_name)

    candidates: List[Path] = []
    for name in names:
        for ext in exts:
            candidates.append(docs_dir / f"{name}{ext}")

    for ext in exts:
        candidates.extend(docs_dir.glob(f"**/{paper_id}{ext}"))
        if safe_name != paper_id:
            candidates.extend(docs_dir.glob(f"**/{safe_name}{ext}"))

    return candidates


def _read_doc_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def build_doc_length_map(docs_dir: Optional[str | Path], paper_ids: Iterable[str]) -> Dict[str, int]:
    if not docs_dir:
        return {}

    d = Path(docs_dir)
    if not d.exists() or not d.is_dir():
        print(f"[warn] docs_dir 不存在或不是目录，跳过文档长度分桶: {docs_dir}")
        return {}

    out: Dict[str, int] = {}

    for pid in paper_ids:
        found_text = ""
        for p in _candidate_doc_paths(d, pid):
            if p.exists() and p.is_file():
                found_text = _read_doc_text(p)
                if found_text:
                    break

        if found_text:
            out[pid] = len(found_text)

    missing = len(set(paper_ids)) - len(out)
    if missing:
        print(f"[warn] 有 {missing} 篇论文没有找到对应文档，文档长度分桶会跳过这些论文。")

    return out


def _bucket_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f"{int(lo)}+"
    if hi - lo == 1:
        return str(int(lo))
    return f"{int(lo)}~{int(hi - 1)}"


def _assign_bucket(value: float, bins: List[float]) -> Optional[str]:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            return _bucket_label(lo, hi)
    return None


# ============================================================
# 10. 核心评估
# ============================================================

def evaluate_strategy(
    strategy: str,
    pred_file: str | Path,
    gold_map: Dict[str, List[Dict[str, Any]]],
    failed_file: Optional[str | Path] = None,
    doc_length_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    pred_map, duplicate_pred_ids = load_predictions(pred_file)

    failed_ids = set()
    if failed_file and Path(failed_file).exists():
        for obj in read_jsonl(failed_file):
            pid = safe_paper_id(obj)
            if pid:
                failed_ids.add(pid)

    gold_ids = set(gold_map)
    pred_ids = set(pred_map)
    extra_pred_ids = sorted(pred_ids - gold_ids)

    paper_ids = sorted(
        gold_ids | (set(extra_pred_ids) if COUNT_EXTRA_PRED_PAPERS_AS_FP else set())
    )

    doc_length_map = doc_length_map or {}

    field_stats: Dict[str, Dict[str, int]] = {
        field: {"tp": 0, "pred": 0, "gold": 0} for field in EVAL_FIELDS
    }

    null_stats: Dict[str, Dict[str, int]] = {
        field: {"gold_null": 0, "correct_null": 0} for field in EVAL_FIELDS
    }

    per_paper_rows: List[Dict[str, Any]] = []
    per_paper_field_rows: List[Dict[str, Any]] = []
    mismatch_examples: List[Dict[str, Any]] = []
    matched_pair_examples: List[Dict[str, Any]] = []

    rec_tp_total = 0
    rec_pred_total = 0
    rec_gold_total = 0
    null_correct = 0
    null_total = 0

    for pid in paper_ids:
        gold_records = gold_map.get(pid, [])
        pred_records = pred_map.get(pid, [])

        rec_gold_total += len(gold_records)
        rec_pred_total += len(pred_records)

        record_tp, matches = match_records(pred_records, gold_records)
        rec_tp_total += record_tp

        matched_pred_idx = {pi for pi, _, _ in matches}
        matched_gold_idx = {gi for _, gi, _ in matches}

        paper_precision, paper_recall, paper_f1 = prf(
            record_tp, len(pred_records), len(gold_records)
        )

        doc_len = doc_length_map.get(pid)
        record_count_bin = _assign_bucket(len(gold_records), RECORD_COUNT_BINS)
        doc_length_bin = _assign_bucket(doc_len, DOC_LENGTH_BINS) if doc_len is not None else None

        per_paper_rows.append(
            {
                "strategy": strategy,
                "paper_id": pid,
                "gold_record_count": len(gold_records),
                "pred_record_count": len(pred_records),
                "record_tp": record_tp,
                "precision": paper_precision,
                "recall": paper_recall,
                "F1": paper_f1,
                "has_prediction": int(pid in pred_map),
                "is_failed": int(pid in failed_ids),
                "is_extra_prediction": int(pid not in gold_map),
                "doc_length": doc_len if doc_len is not None else "",
                "record_count_bin": record_count_bin or "",
                "doc_length_bin": doc_length_bin or "",
            }
        )

        if len(matched_pair_examples) < 100:
            for pi, gi, sim in matches[: max(0, 100 - len(matched_pair_examples))]:
                matched_pair_examples.append(
                    {
                        "paper_id": pid,
                        "similarity": sim,
                        "pred_record": pred_records[pi],
                        "gold_record": gold_records[gi],
                    }
                )

        for field in EVAL_FIELDS:
            field_tp = 0
            field_pred_n = 0
            field_gold_n = 0

            for pi, gi, sim in matches:
                pv = pred_records[pi].get(field)
                gv = gold_records[gi].get(field)

                if not is_empty_value(pv, field):
                    field_pred_n += 1
                if not is_empty_value(gv, field):
                    field_gold_n += 1

                if (not is_empty_value(pv, field)) and (not is_empty_value(gv, field)) and values_match(pv, gv, field=field, relaxed=True):
                    field_tp += 1
                elif (not is_empty_value(pv, field) or not is_empty_value(gv, field)) and len(mismatch_examples) < 300:
                    mismatch_examples.append(
                        {
                            "paper_id": pid,
                            "field": field,
                            "record_similarity": sim,
                            "pred": pv,
                            "gold": gv,
                            "pred_canonical": canonical_for_key(pv, field),
                            "gold_canonical": canonical_for_key(gv, field),
                        }
                    )

            for pi, rec in enumerate(pred_records):
                if pi not in matched_pred_idx and not is_empty_value(rec.get(field), field):
                    field_pred_n += 1

            for gi, rec in enumerate(gold_records):
                if gi not in matched_gold_idx and not is_empty_value(rec.get(field), field):
                    field_gold_n += 1

            field_stats[field]["tp"] += field_tp
            field_stats[field]["pred"] += field_pred_n
            field_stats[field]["gold"] += field_gold_n

            fp = field_pred_n - field_tp
            fn = field_gold_n - field_tp
            p, r, f1 = prf(field_tp, field_pred_n, field_gold_n)

            per_paper_field_rows.append(
                {
                    "strategy": strategy,
                    "paper_id": pid,
                    "field": field,
                    "tp": field_tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_count": field_pred_n,
                    "gold_count": field_gold_n,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "gold_record_count": len(gold_records),
                    "doc_length": doc_len if doc_len is not None else "",
                    "record_count_bin": record_count_bin or "",
                    "doc_length_bin": doc_length_bin or "",
                }
            )

        paper_null_total = 0
        paper_null_correct = 0
        for pi, gi, _ in matches:
            for field in EVAL_FIELDS:
                gv_null = is_empty_value(gold_records[gi].get(field), field)
                pv_null = is_empty_value(pred_records[pi].get(field), field)

                if gv_null:
                    null_total += 1
                    paper_null_total += 1
                    null_stats[field]["gold_null"] += 1

                    if pv_null:
                        null_correct += 1
                        paper_null_correct += 1
                        null_stats[field]["correct_null"] += 1
        per_paper_rows[-1]["gold_null_count_matched"] = paper_null_total
        per_paper_rows[-1]["correct_null_count"] = paper_null_correct
        per_paper_rows[-1]["null_accuracy"] = (
            paper_null_correct / paper_null_total if paper_null_total else ""
        )

    total_field_tp = sum(st["tp"] for st in field_stats.values())
    total_field_pred = sum(st["pred"] for st in field_stats.values())
    total_field_gold = sum(st["gold"] for st in field_stats.values())
    field_micro_p, field_micro_r, field_micro_f1 = prf(
        total_field_tp, total_field_pred, total_field_gold
    )

    per_field_rows: List[Dict[str, Any]] = []
    macro_f1_all: List[float] = []
    macro_f1_supported: List[float] = []

    for field in EVAL_FIELDS:
        st = field_stats[field]
        tp = st["tp"]
        pred_n = st["pred"]
        gold_n = st["gold"]
        fp = pred_n - tp
        fn = gold_n - tp
        p, r, f1 = prf(tp, pred_n, gold_n)

        macro_f1_all.append(f1)
        if pred_n > 0 or gold_n > 0:
            macro_f1_supported.append(f1)

        ns = null_stats[field]
        null_acc = ns["correct_null"] / ns["gold_null"] if ns["gold_null"] else ""

        per_field_rows.append(
            {
                "strategy": strategy,
                "field": field,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred_count": pred_n,
                "gold_count": gold_n,
                "precision": p,
                "recall": r,
                "f1": f1,
                "gold_null_count_matched": ns["gold_null"],
                "correct_null_count": ns["correct_null"],
                "null_accuracy": null_acc,
            }
        )

    record_precision, record_recall, record_f1 = prf(rec_tp_total, rec_pred_total, rec_gold_total)

    summary = {
        "strategy": strategy,
        "gold_paper_count": len(gold_map),
        "predicted_paper_count": sum(1 for pid in gold_map if pid in pred_map),
        "extra_pred_paper_count": len(extra_pred_ids),
        "duplicate_pred_paper_count": sum(1 for n in duplicate_pred_ids.values() if n > 1),
        "failed_paper_count": len(failed_ids),
        "record_pred_total": rec_pred_total,
        "record_gold_total": rec_gold_total,
        "record_tp": rec_tp_total,
        "record_precision": record_precision,
        "record_recall": record_recall,
        "record_f1": record_f1,
        # precision / recall / F1 均指字段级（field micro），与 PPT 指标一致
        "precision": field_micro_p,
        "recall": field_micro_r,
        "F1": field_micro_f1,
        "field_tp_total": total_field_tp,
        "field_pred_total": total_field_pred,
        "field_gold_total": total_field_gold,
        "field_micro_precision": field_micro_p,
        "field_micro_recall": field_micro_r,
        "field_micro_f1": field_micro_f1,
        "field_macro_f1": sum(macro_f1_all) / len(macro_f1_all) if macro_f1_all else 0.0,
        "field_supported_macro_f1": (
            sum(macro_f1_supported) / len(macro_f1_supported) if macro_f1_supported else 0.0
        ),
        "null_accuracy": null_correct / null_total if null_total else 0.0,
        "null_correct": null_correct,
        "null_total": null_total,
        "docs_with_length_count": sum(1 for row in per_paper_rows if row.get("doc_length") != ""),
    }

    return {
        "summary": summary,
        "per_field": per_field_rows,
        "per_paper": per_paper_rows,
        "per_paper_field": per_paper_field_rows,
        "mismatch_examples": mismatch_examples,
        "matched_pair_examples": matched_pair_examples,
        "extra_pred_paper_ids": extra_pred_ids,
    }


# ============================================================
# 11. 分桶分析
# ============================================================

def _aggregate_record_rows(
    rows: List[Dict[str, Any]],
    bucket_key: str,
    bucket_order: List[str],
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    strategy = rows[0].get("strategy", "")
    stats = {
        label: {"tp": 0, "pred": 0, "gold": 0, "count": 0}
        for label in bucket_order
    }

    for row in rows:
        label = row.get(bucket_key)
        if not label:
            continue
        if label not in stats:
            stats[label] = {"tp": 0, "pred": 0, "gold": 0, "count": 0}

        st = stats[label]
        st["tp"] += int(row.get("record_tp", 0))
        st["pred"] += int(row.get("pred_record_count", 0))
        st["gold"] += int(row.get("gold_record_count", 0))
        st["count"] += 1

    out: List[Dict[str, Any]] = []
    for label in bucket_order:
        st = stats.get(label)
        if not st:
            continue

        p, r, f1 = prf(st["tp"], st["pred"], st["gold"])
        out.append(
            {
                "strategy": strategy,
                "bin": label,
                "doc_count": st["count"],
                "record_tp": st["tp"],
                "record_pred_total": st["pred"],
                "record_gold_total": st["gold"],
                "precision": p,
                "recall": r,
                "F1": f1,
            }
        )

    return out


def _aggregate_field_rows(
    rows: List[Dict[str, Any]],
    bucket_key: str,
    bucket_order: List[str],
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    strategy = rows[0].get("strategy", "")
    stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    for label in bucket_order:
        for field in EVAL_FIELDS:
            stats[(label, field)] = {"tp": 0, "pred": 0, "gold": 0}

    for row in rows:
        label = row.get(bucket_key)
        if not label:
            continue

        field = row.get("field")
        if not field:
            continue

        key = (label, field)
        if key not in stats:
            stats[key] = {"tp": 0, "pred": 0, "gold": 0}

        stats[key]["tp"] += int(row.get("tp", 0))
        stats[key]["pred"] += int(row.get("pred_count", 0))
        stats[key]["gold"] += int(row.get("gold_count", 0))

    out: List[Dict[str, Any]] = []

    for label in bucket_order:
        for field in EVAL_FIELDS:
            st = stats.get((label, field))
            if not st:
                continue

            tp = st["tp"]
            pred_n = st["pred"]
            gold_n = st["gold"]
            fp = pred_n - tp
            fn = gold_n - tp
            p, r, f1 = prf(tp, pred_n, gold_n)

            out.append(
                {
                    "strategy": strategy,
                    "bin": label,
                    "field": field,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_count": pred_n,
                    "gold_count": gold_n,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
            )

    return out


def _bucket_order_from_bins(bins: List[float]) -> List[str]:
    return [_bucket_label(lo, hi) for lo, hi in zip(bins[:-1], bins[1:])]


def evaluate_by_record_count(per_paper_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = _bucket_order_from_bins(RECORD_COUNT_BINS)
    return _aggregate_record_rows(per_paper_rows, "record_count_bin", order)


def evaluate_by_record_count_field(
    per_paper_field_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    order = _bucket_order_from_bins(RECORD_COUNT_BINS)
    return _aggregate_field_rows(per_paper_field_rows, "record_count_bin", order)


def evaluate_by_doc_length(per_paper_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = _bucket_order_from_bins(DOC_LENGTH_BINS)
    return _aggregate_record_rows(per_paper_rows, "doc_length_bin", order)


def evaluate_by_doc_length_field(
    per_paper_field_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    order = _bucket_order_from_bins(DOC_LENGTH_BINS)
    return _aggregate_field_rows(per_paper_field_rows, "doc_length_bin", order)


# 旧函数名兼容
_evaluate_by_record_count = evaluate_by_record_count
_evaluate_by_doc_length = evaluate_by_doc_length
_bucketed_field_level = evaluate_by_record_count_field


# ============================================================
# 12. 输出
# ============================================================

SUMMARY_FIELDS = [
    "strategy",
    "gold_paper_count",
    "predicted_paper_count",
    "extra_pred_paper_count",
    "duplicate_pred_paper_count",
    "failed_paper_count",
    "record_pred_total",
    "record_gold_total",
    "record_tp",
    "record_precision",
    "record_recall",
    "record_f1",
    # 主指标（字段级 micro）
    "precision",
    "recall",
    "F1",
    "field_tp_total",
    "field_pred_total",
    "field_gold_total",
    "field_micro_precision",
    "field_micro_recall",
    "field_micro_f1",
    "field_macro_f1",
    "field_supported_macro_f1",
    "null_accuracy",
    "null_correct",
    "null_total",
    "docs_with_length_count",
]

FIELD_FIELDS = [
    "strategy",
    "field",
    "tp",
    "fp",
    "fn",
    "pred_count",
    "gold_count",
    "precision",
    "recall",
    "f1",
    "gold_null_count_matched",
    "correct_null_count",
    "null_accuracy",
]

PAPER_FIELDS = [
    "strategy",
    "paper_id",
    "gold_record_count",
    "pred_record_count",
    "record_tp",
    "precision",
    "recall",
    "F1",
    "has_prediction",
    "is_failed",
    "is_extra_prediction",
    "doc_length",
    "record_count_bin",
    "doc_length_bin",
    "gold_null_count_matched",
    "correct_null_count",
    "null_accuracy",
]

PAPER_FIELD_FIELDS = [
    "strategy",
    "paper_id",
    "field",
    "tp",
    "fp",
    "fn",
    "pred_count",
    "gold_count",
    "precision",
    "recall",
    "f1",
    "gold_record_count",
    "doc_length",
    "record_count_bin",
    "doc_length_bin",
]

BUCKET_FIELDS = [
    "strategy",
    "bin",
    "doc_count",
    "record_tp",
    "record_pred_total",
    "record_gold_total",
    "precision",
    "recall",
    "F1",
]

BUCKET_FIELD_FIELDS = [
    "strategy",
    "bin",
    "field",
    "tp",
    "fp",
    "fn",
    "pred_count",
    "gold_count",
    "precision",
    "recall",
    "f1",
]


def write_evaluation_outputs(result: Dict[str, Any], output_dir: str | Path) -> None:
    output = ensure_dir(output_dir)

    write_csv(output / "summary_metrics.csv", [result["summary"]], SUMMARY_FIELDS)
    write_csv(output / "per_field_metrics.csv", result["per_field"], FIELD_FIELDS)
    write_csv(output / "per_paper_metrics.csv", result["per_paper"], PAPER_FIELDS)
    write_csv(output / "per_paper_field_metrics.csv", result["per_paper_field"], PAPER_FIELD_FIELDS)

    record_count_rows = evaluate_by_record_count(result["per_paper"])
    record_count_field_rows = evaluate_by_record_count_field(result["per_paper_field"])
    write_csv(output / "per_record_count_metrics.csv", record_count_rows, BUCKET_FIELDS)
    write_csv(
        output / "per_record_count_field_metrics.csv",
        record_count_field_rows,
        BUCKET_FIELD_FIELDS,
    )

    if any(row.get("doc_length") != "" for row in result["per_paper"]):
        doc_length_rows = evaluate_by_doc_length(result["per_paper"])
        doc_length_field_rows = evaluate_by_doc_length_field(result["per_paper_field"])
        write_csv(output / "per_doc_length_metrics.csv", doc_length_rows, BUCKET_FIELDS)
        write_csv(
            output / "per_doc_length_field_metrics.csv",
            doc_length_field_rows,
            BUCKET_FIELD_FIELDS,
        )

    write_json(output / "evaluation_details.json", result)
    write_json(output / "mismatch_examples.json", result.get("mismatch_examples", []))
    write_json(output / "matched_pair_examples.json", result.get("matched_pair_examples", []))


# ============================================================
# 13. 扫描结果文件
# ============================================================

def discover_prediction_files(results_dir: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    items: List[Tuple[str, Path, Optional[Path]]] = []

    if config is not None and hasattr(config, "ALL_STRATEGIES"):
        for strategy in list(getattr(config, "ALL_STRATEGIES")):
            lower = str(strategy).lower()
            pred = results_dir / lower / f"{lower}_results.jsonl"
            if pred.exists():
                failed = results_dir / lower / f"{lower}_failed.jsonl"
                items.append((str(strategy), pred, failed if failed.exists() else None))

    # 用 lower() 做去重，兼容 Windows 大小写不敏感路径
    # （E5/E5_results.jsonl 和 e5/e5_results.jsonl 在 Windows 是同一文件）
    seen = {str(p).lower() for _, p, _ in items}

    for pred in results_dir.glob("**/*results.jsonl"):
        if str(pred).lower() in seen:
            continue

        name = pred.stem
        strategy = pred.parent.name or name.replace("_results", "")
        failed = pred.with_name(pred.name.replace("_results.jsonl", "_failed.jsonl"))
        items.append((strategy, pred, failed if failed.exists() else None))

    return items


# ============================================================
# 14. 主入口
# ============================================================

def run_single_prediction(
    gold_path: str | Path,
    pred_file: str | Path,
    output_dir: str | Path,
    strategy: str = "LLM",
    docs_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    gold_map = load_gold(gold_path)
    if not gold_map:
        raise SystemExit(f"没有从 gold 文件读到记录: {gold_path}")

    doc_length_map = build_doc_length_map(docs_dir, gold_map.keys()) if docs_dir else {}

    result = evaluate_strategy(
        strategy=strategy,
        pred_file=pred_file,
        gold_map=gold_map,
        doc_length_map=doc_length_map,
    )
    write_evaluation_outputs(result, output_dir)

    print(f"[evaluate] 完成: {strategy} -> {output_dir}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    return result


def run_results_dir(
    gold_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    docs_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    gold_map = load_gold(gold_path)
    if not gold_map:
        raise SystemExit(f"没有从 gold 文件读到记录: {gold_path}")

    results_path = Path(results_dir)
    discovered = discover_prediction_files(results_path)
    if not discovered:
        raise SystemExit(
            f"没有在 {results_path} 下找到 *results.jsonl；如果只有一个预测文件，请使用 --pred。"
        )

    doc_length_map = build_doc_length_map(docs_dir, gold_map.keys()) if docs_dir else {}

    all_summary: List[Dict[str, Any]] = []
    all_fields: List[Dict[str, Any]] = []
    all_papers: List[Dict[str, Any]] = []
    all_paper_fields: List[Dict[str, Any]] = []
    all_record_count_rows: List[Dict[str, Any]] = []
    all_record_count_field_rows: List[Dict[str, Any]] = []
    all_doc_length_rows: List[Dict[str, Any]] = []
    all_doc_length_field_rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for strategy, pred_file, failed_file in discovered:
        result = evaluate_strategy(
            strategy=strategy,
            pred_file=pred_file,
            gold_map=gold_map,
            failed_file=failed_file,
            doc_length_map=doc_length_map,
        )

        all_summary.append(result["summary"])
        all_fields.extend(result["per_field"])
        all_papers.extend(result["per_paper"])
        all_paper_fields.extend(result["per_paper_field"])

        all_record_count_rows.extend(evaluate_by_record_count(result["per_paper"]))
        all_record_count_field_rows.extend(evaluate_by_record_count_field(result["per_paper_field"]))

        if any(row.get("doc_length") != "" for row in result["per_paper"]):
            all_doc_length_rows.extend(evaluate_by_doc_length(result["per_paper"]))
            all_doc_length_field_rows.extend(evaluate_by_doc_length_field(result["per_paper_field"]))

        details[strategy] = result
        print(f"[evaluate] 完成: {strategy}")

    output = ensure_dir(output_dir)

    write_csv(output / "summary_metrics.csv", all_summary, SUMMARY_FIELDS)
    write_csv(output / "per_field_metrics.csv", all_fields, FIELD_FIELDS)
    write_csv(output / "per_paper_metrics.csv", all_papers, PAPER_FIELDS)
    write_csv(output / "per_paper_field_metrics.csv", all_paper_fields, PAPER_FIELD_FIELDS)

    write_csv(output / "per_record_count_metrics.csv", all_record_count_rows, BUCKET_FIELDS)
    write_csv(
        output / "per_record_count_field_metrics.csv",
        all_record_count_field_rows,
        BUCKET_FIELD_FIELDS,
    )

    if all_doc_length_rows:
        write_csv(output / "per_doc_length_metrics.csv", all_doc_length_rows, BUCKET_FIELDS)
        write_csv(
            output / "per_doc_length_field_metrics.csv",
            all_doc_length_field_rows,
            BUCKET_FIELD_FIELDS,
        )

    write_json(output / "evaluation_details.json", details)

    print(f"[evaluate] 全部完成 -> {output}")
    return details


def main(
    gold_path: Optional[str | Path] = None,
    pred_file: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    strategy: str = "LLM",
    docs_dir: Optional[str | Path] = None,
    _parse_cli: bool = False,
) -> Any:
    """
    支持两种用法：

    1. 命令行：
       python evaluate.py --gold train.jsonl --pred result.jsonl --output evaluation

    2. 项目内函数调用：
       import evaluate
       evaluate.main(
           gold_path="train.jsonl",
           pred_file="result.jsonl",
           output_dir="evaluation",
           strategy="LLM"
       )
    """
    if _parse_cli:
        parser = argparse.ArgumentParser(
            description="Evaluate corrosion-inhibition literature-mining extraction results."
        )
        parser.add_argument(
            "--gold",
            type=str,
            default=str(getattr(config, "GOLD_PATH", "train.jsonl") if config else "train.jsonl"),
            help="Gold JSONL path.",
        )
        parser.add_argument(
            "--pred",
            type=str,
            default=None,
            help="Single prediction JSONL path.",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="LLM",
            help="Strategy name when --pred is used.",
        )
        parser.add_argument(
            "--results",
            type=str,
            default=str(getattr(config, "RESULTS_DIR", "results") if config else "results"),
            help="Results directory when --pred is not used.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output directory.",
        )
        parser.add_argument(
            "--docs-dir",
            type=str,
            default=DEFAULT_DOCS_DIR or None,
            help="Optional document directory for document-length bucket analysis.",
        )
        args = parser.parse_args()

        gold_path = args.gold
        pred_file = args.pred
        results_dir = args.results
        output_dir = args.output
        strategy = args.strategy
        docs_dir = args.docs_dir

    if gold_path is None:
        gold_path = str(getattr(config, "GOLD_PATH", "train.jsonl") if config else "train.jsonl")

    if results_dir is None:
        results_dir = str(getattr(config, "RESULTS_DIR", "results") if config else "results")

    if output_dir is None:
        output_dir = Path(results_dir) / "evaluation"

    if docs_dir is None:
        docs_dir = DEFAULT_DOCS_DIR or None

    if pred_file:
        return run_single_prediction(
            gold_path=gold_path,
            pred_file=pred_file,
            output_dir=output_dir,
            strategy=strategy,
            docs_dir=docs_dir,
        )

    return run_results_dir(
        gold_path=gold_path,
        results_dir=results_dir,
        output_dir=output_dir,
        docs_dir=docs_dir,
    )


if __name__ == "__main__":
    main(_parse_cli=True)