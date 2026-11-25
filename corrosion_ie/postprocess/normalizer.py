from __future__ import annotations

import re
from typing import Any, Dict, List

from ..schemas import FIELD_NAMES

_MULTI_SEP = re.compile(r"\s*(?:,|;|/| and |\+|、)\s*", re.I)
_METHOD_SYNONYMS = {
    "eis": "electrochemical impedance spectroscopy",
    "pdp": "potentiodynamic polarization",
    "tafel": "potentiodynamic polarization",
    "weight loss method": "weight loss",
    "gravimetric": "weight loss",
}
_FIELD_ALIASES = {
    "inhibitor_name": ["inhibitor_name", "inhibitor", "inh", "compound", "extract", "name"],
    "substrate": ["substrate", "sub", "metal", "material", "steel_type"],
    "corrosive_medium": ["corrosive_medium", "medium", "med", "corrosion_medium", "solution"],
    "medium_concentration": ["medium_concentration", "med_c", "medium_conc", "acid_concentration", "solution_concentration"],
    "inhibitor_concentration": ["inhibitor_concentration", "inh_c", "concentration", "inhibitor_dosage", "dosage", "dose"],
    "temperature": ["temperature", "temp", "test_temperature"],
    "immersion_time": ["immersion_time", "time", "duration", "exposure_time"],
    "test_method": ["test_method", "method", "technique"],
    "inhibition_efficiency": ["inhibition_efficiency", "ie", "efficiency", "IE", "IE%", "eta", "η", "η%"],
    "homo": ["homo", "HOMO", "EHOMO", "E_HOMO"],
    "lumo": ["lumo", "LUMO", "ELUMO", "E_LUMO"],
    "delta_g_ads": ["delta_g_ads", "dg", "ΔG_ads", "ΔGads", "deltag_ads", "gibbs_free_energy_adsorption"],
    "adsorption_energy": ["adsorption_energy", "eads", "Eads", "E_ads", "binding_energy"],
}
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "not reported", "not specified", "-", "—", "--"}

# Characterization-only techniques must not become corrosion performance test_method.
_BAD_METHOD_RE = re.compile(
    r"\b(sem|xrd|ftir|atr\s*[-/]?\s*ir|atr\s*[-/]?\s*ftir|xps|afm|tem|edx|eds|raman|uv\s*[-/]?\s*vis|uv|ir spectroscopy)\b",
    re.I,
)
_PERFORMANCE_METHOD_RE = re.compile(
    r"\b(eis|impedance|electrochemical impedance|polarization|potentiodynamic|tafel|weight loss|gravimetric|lpr|linear polarization|corrosion rate|icorr|rct|charge transfer)\b",
    re.I,
)
_FIELD_LABEL_METHOD_RE = re.compile(
    r"^\s*(?:e[_\s-]*homo|homo(?:\s+energy)?|e[_\s-]*lumo|lumo(?:\s+energy)?|"
    r"adsorption\s+energy|binding\s+energy|interaction\s+energy|delta\s*g(?:_?ads)?|"
    r"gibbs\s+free\s+energy(?:\s+of\s+adsorption)?)\s*$",
    re.I,
)
_BAD_SUBSTRATE_RE = re.compile(
    r"\b(ag/agcl|sce|calomel|platinum|pt electrode|graphite|counter electrode|reference electrode|working electrode)\b",
    re.I,
)


def _pick(rec: Dict[str, Any], field: str) -> Any:
    for key in _FIELD_ALIASES.get(field, [field]):
        if key in rec:
            val = rec.get(key)
            if val is not None:
                return val
    return None


def clean_value(v: Any) -> Any:
    """Clean scalar values while preserving structured lists."""
    if v is None:
        return None
    if isinstance(v, list):
        cleaned = [clean_value(x) for x in v]
        cleaned = [x for x in cleaned if x is not None and x != []]
        return cleaned or None
    if isinstance(v, dict):
        # Some LLMs return {"value": ..., "evidence": ...} for a field.
        for key in ["value", "text", "answer", "content"]:
            if key in v:
                return clean_value(v.get(key))
        return None
    s = str(v).strip()
    if s.lower() in _NULL_STRINGS:
        return None
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n;,.|")
    return s or None


def _norm_for_key(v: Any) -> str:
    if isinstance(v, list):
        return "|".join(_norm_for_key(x) for x in v)
    return str(v or "").lower().strip()


def _normalize_field_evidence(value: Any) -> Dict[str, List[str]]:
    """Normalize model- or pipeline-produced field-level evidence links.

    Accepted shapes include ``{"field": "unit"}``, ``{"field": ["u1", "u2"]}``,
    and lists of ``{"field": ..., "unit_id": ...}``. Unknown field names are
    ignored so evaluation fields remain stable.
    """
    out: Dict[str, List[str]] = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        pairs = []
        for item in value:
            if isinstance(item, dict):
                field = item.get("field") or item.get("field_name")
                unit = item.get("unit_id") or item.get("evidence_unit_id")
                if field:
                    pairs.append((field, unit))
        items = pairs
    else:
        return out
    for field, units in items:
        field = str(field or "").strip()
        if field not in FIELD_NAMES:
            continue
        if isinstance(units, (list, tuple, set)):
            vals = [str(x).strip() for x in units if str(x or "").strip()]
        else:
            vals = [str(units).strip()] if str(units or "").strip() else []
        if vals:
            out[field] = list(dict.fromkeys(vals))
    return out


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {f: clean_value(_pick(record, f)) for f in FIELD_NAMES}

    # Preserve evidence-aware metadata outside evaluation fields. These keys are
    # ignored by the legacy evaluator but are retained for record-link diagnostics.
    out["record_id"] = clean_value(record.get("record_id") or record.get("id"))
    out["evidence_text"] = clean_value(record.get("evidence_text") or record.get("evidence"))
    out["evidence_unit_id"] = clean_value(record.get("evidence_unit_id") or record.get("unit_id") or record.get("chunk_id"))
    out["field_evidence"] = _normalize_field_evidence(record.get("field_evidence") or record.get("evidence_by_field"))
    flags = record.get("verifier_flags")
    if isinstance(flags, list):
        out["verifier_flags"] = [str(x) for x in flags if str(x or "").strip()]
    else:
        out["verifier_flags"] = []
    try:
        confidence = record.get("confidence")
        out["confidence"] = float(confidence) if confidence not in [None, ""] else None
    except Exception:
        out["confidence"] = None

    method = out.get("test_method")
    if isinstance(method, str) and not _looks_multi_method(method):
        low = method.lower()
        for k, v in _METHOD_SYNONYMS.items():
            if k in low:
                out["test_method"] = v
                break
    temp = out.get("temperature")
    if isinstance(temp, str) and temp.lower() in {"rt", "ambient", "ambient temperature"}:
        out["temperature"] = "room temperature"
    return sanitize_record(out)


def normalize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_record(r) for r in records if isinstance(r, dict)]


LEGACY_CORE_DEDUP_FIELDS = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "test_method",
    "inhibition_efficiency",
]

PERFORMANCE_WITH_TIME_DEDUP_FIELDS = [
    "inhibitor_name",
    "substrate",
    "corrosive_medium",
    "medium_concentration",
    "inhibitor_concentration",
    "temperature",
    "immersion_time",
    "test_method",
    "inhibition_efficiency",
]


FULL_RECORD_DEDUP_FIELDS = [
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


def dedup_fields(mode: str = "full_record") -> List[str]:
    """Resolve a named record de-duplication protocol.

    ``legacy_core`` reproduces the old Q1 implementation.  It intentionally
    ignores immersion time and inhibitor-level quantum/adsorption fields so
    those optional values cannot create duplicate performance records.

    ``performance_with_time`` adds immersion time while intentionally excluding
    inhibitor-level quantum/adsorption facts from the performance identity.

    ``full_record`` includes all schema fields for generic refactored workflows.
    """
    key = str(mode or "full_record").strip().lower()
    aliases = {
        "legacy": "legacy_core",
        "legacy_q1": "legacy_core",
        "core": "legacy_core",
        "full": "full_record",
        "all": "full_record",
        "performance_time": "performance_with_time",
    }
    key = aliases.get(key, key)
    if key == "legacy_core":
        return list(LEGACY_CORE_DEDUP_FIELDS)
    if key == "performance_with_time":
        return list(PERFORMANCE_WITH_TIME_DEDUP_FIELDS)
    if key == "full_record":
        return list(FULL_RECORD_DEDUP_FIELDS)
    raise ValueError(
        f"Unknown dedup_mode={mode!r}; expected 'legacy_core', 'performance_with_time' or 'full_record'"
    )


def dedup_records(records: List[Dict[str, Any]], mode: str = "full_record") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    mode_key = str(mode or "full_record").strip().lower()
    key_fields = dedup_fields("full_record" if mode_key == "record_linked" else mode_key)
    for r in records:
        record_id = _norm_for_key(r.get("record_id"))
        if mode_key == "record_linked" and record_id:
            key = ("record_id", record_id)
        else:
            key = tuple(_norm_for_key(r.get(f)) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_obvious_multivalue_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Legacy-Q1 compatible expansion for obvious table-row summaries.

    The old Q1 prompt demanded one record per concentration/method/IE row.
    When a small model still compresses a table into a single JSON object such as
    "300, 600 and 1200 ppm" + "29.5%, 53.7%, 64.5%", this function expands the
    aligned values back into separate records.  It is deliberately conservative:
    it only uses inhibitor concentration, test method, and inhibition efficiency.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        expanded = _expand_one_record(r)
        out.extend(sanitize_record(x) for x in expanded)
    return out




def _filter_performance_methods(method: str) -> str | None:
    """Remove characterization-only methods while preserving valid corrosion tests.

    Examples:
    - "ATR-IR" -> None
    - "XPS, PDP" -> "PDP"
    - "SEM / EIS / XPS" -> "EIS"
    - "EIS and PDP" -> "EIS, PDP"
    """
    if not isinstance(method, str):
        return method
    s = method.strip()
    if not s:
        return None

    has_bad = bool(_BAD_METHOD_RE.search(s))
    has_perf = bool(_PERFORMANCE_METHOD_RE.search(s))
    if has_bad and not has_perf:
        return None
    if not has_bad:
        return s

    # Split mixed method strings and retain only performance entries.  Avoid
    # matching "IR" inside "linear polarization" by relying on _BAD_METHOD_RE.
    parts = [p.strip() for p in _MULTI_SEP.split(s) if p.strip()]
    kept = []
    for part in parts:
        if _BAD_METHOD_RE.search(part) and not _PERFORMANCE_METHOD_RE.search(part):
            continue
        if _PERFORMANCE_METHOD_RE.search(part):
            kept.append(part)
    if kept:
        return ", ".join(kept)

    # Last resort: extract canonical performance tokens from the original string.
    tokens = []
    for pattern, label in [
        (r"\bEIS\b|electrochemical impedance", "EIS"),
        (r"\bPDP\b|potentiodynamic polarization|tafel", "PDP"),
        (r"\bLPR\b|linear polarization", "LPR"),
        (r"weight loss|gravimetric", "weight loss"),
    ]:
        if re.search(pattern, s, re.I):
            tokens.append(label)
    return ", ".join(tokens) if tokens else None


def sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply legacy-style field sanity rules after normalization/expansion."""
    r = dict(record)

    method = r.get("test_method")
    if isinstance(method, str):
        # Schema field labels are quantities, not experimental/computational
        # methods. The actual method must be DFT, MD, EIS, PDP, etc.
        if _FIELD_LABEL_METHOD_RE.fullmatch(method.strip()):
            r["test_method"] = None
            method = None
    if isinstance(method, str):
        # Characterization-only methods are not corrosion performance methods.
        # If a mixed string is emitted, e.g. "XPS, PDP", keep only the
        # corrosion-performance method instead of preserving the bad method.
        filtered = _filter_performance_methods(method)
        if filtered is None:
            r["test_method"] = None
        else:
            r["test_method"] = filtered
            if not _looks_multi_method(filtered):
                low = filtered.lower()
                for k, v in _METHOD_SYNONYMS.items():
                    if k in low:
                        r["test_method"] = v
                        break

    substrate = r.get("substrate")
    if isinstance(substrate, str) and _BAD_SUBSTRATE_RE.search(substrate):
        r["substrate"] = None

    eff = r.get("inhibition_efficiency")
    if isinstance(eff, str) and eff and not _looks_like_efficiency(eff):
        r["inhibition_efficiency"] = None

    # Do not confuse dipole moment with adsorption energy.
    ads = r.get("adsorption_energy")
    if isinstance(ads, str):
        low = ads.lower()
        if "debye" in low or re.search(r"\b(dipole|mu|moment)\b", low):
            r["adsorption_energy"] = None
        elif re.search(r"\b(gibbs|gads|g_ads|free energy of adsorption|δg|Δg)\b", low):
            r["adsorption_energy"] = None

    dg = r.get("delta_g_ads")
    if isinstance(dg, str):
        low = dg.lower()
        if re.search(r"\b(adsorption energy|binding energy|activation energy|homo|lumo|debye|dipole)\b", low):
            r["delta_g_ads"] = None

    temp = r.get("temperature")
    if isinstance(temp, str) and re.search(r"\b(drying|dried|oven|calcination|synthesis|heated|anneal)\b", temp, re.I):
        r["temperature"] = None

    return r


def _expand_one_record(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    concs = _value_list(r.get("inhibitor_concentration"), kind="concentration")
    effs = _value_list(r.get("inhibition_efficiency"), kind="efficiency")
    methods = _value_list(r.get("test_method"), kind="method")

    lc, le, lm = len(concs), len(effs), len(methods)
    n = max(lc, le, lm)
    if n <= 1:
        return [r]

    out: List[Dict[str, Any]] = []
    # Table shape: rows are concentrations, columns are methods, efficiencies flattened row-major.
    if lc > 1 and lm > 1 and le == lc * lm:
        k = 0
        for c in concs:
            for m in methods:
                nr = dict(r)
                nr["inhibitor_concentration"] = c
                nr["test_method"] = m
                nr["inhibition_efficiency"] = effs[k]
                out.append(nr)
                k += 1
        return out

    # Aligned concentration/efficiency rows with one method.
    if lc == le and lc > 1 and lm == 1:
        return [_clone_with(r, concs[i], methods[0], effs[i]) for i in range(lc)]

    # Aligned method/efficiency entries with one concentration.
    if lm == le and lm > 1 and lc == 1:
        return [_clone_with(r, concs[0], methods[i], effs[i]) for i in range(lm)]

    # All three lists aligned.
    if lc == le == lm and lc > 1:
        return [_clone_with(r, concs[i], methods[i], effs[i]) for i in range(lc)]

    # Last-value repeat fallback. This preserves recall while avoiding one field
    # containing multiple concentrations/methods/efficiencies.
    for i in range(n):
        nr = dict(r)
        nr["inhibitor_concentration"] = concs[i] if i < lc else concs[-1]
        nr["inhibition_efficiency"] = effs[i] if i < le else effs[-1]
        nr["test_method"] = methods[i] if i < lm else methods[-1]
        out.append(nr)
    return out


def _clone_with(r: Dict[str, Any], conc: Any, method: Any, eff: Any) -> Dict[str, Any]:
    nr = dict(r)
    nr["inhibitor_concentration"] = conc
    nr["test_method"] = method
    nr["inhibition_efficiency"] = eff
    return nr


def _value_list(v: Any, kind: str) -> List[Any]:
    if isinstance(v, list):
        vals = [clean_value(x) for x in v]
        return [x for x in vals if x is not None] or [None]
    if not isinstance(v, str):
        return [v]
    s = v.strip()
    if not s:
        return [v]
    if kind == "concentration":
        return _split_concentrations(s)
    if kind == "efficiency":
        return _split_efficiencies(s)
    if kind == "method":
        return _split_methods(s)
    return [v]


def _split_concentrations(s: str) -> List[str]:
    if not _looks_multi(s):
        return [s]
    # Keep units for compact strings like "300, 600 and 1200 ppm".
    unit_match = re.search(r"\b(ppm|mg\s*/\s*l|mg/l|g\s*/\s*l|g/l|mM|µM|uM|mol\s*/\s*l|mol/l|M|wt\.?%|%)\b", s, re.I)
    unit = unit_match.group(0).replace(" ", "") if unit_match else ""
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) >= 2 and unit:
        return [f"{n} {unit}" for n in nums]
    return _split_values(s)


def _split_efficiencies(s: str) -> List[str]:
    if not _looks_multi(s):
        return [s]
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) >= 2 and "%" in s:
        return [f"{n}%" for n in nums]
    return _split_values(s)


def _split_methods(s: str) -> List[str]:
    if not _looks_multi_method(s):
        return [s]
    # Preserve common multi-word methods and split only separators.
    parts = [p.strip() for p in _MULTI_SEP.split(s) if p.strip()]
    return parts or [s]


def _looks_multi(s: str) -> bool:
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    return len(nums) >= 2 and bool(re.search(r"[,;/]| and |、", s, re.I))


def _looks_multi_method(s: str) -> bool:
    low = s.lower()
    method_hits = re.findall(r"\b(eis|pdp|weight loss|polarization|potentiodynamic|tafel|lpr|gravimetric)\b", low)
    return len(set(method_hits)) >= 2 and bool(re.search(r"[,;/]| and |\+|、", s, re.I))


def _split_values(s: str) -> List[str]:
    parts = [p.strip() for p in _MULTI_SEP.split(s) if p.strip()]
    return parts or [s]


def _looks_like_efficiency(s: str) -> bool:
    low = s.lower()
    if re.search(r"\b(rct|icorr|ecorr|corrosion rate|ohm|ω|ma\s*cm|ua\s*cm|a\s*cm|mpy|mm/y|mmpy)\b", low):
        return False
    if "%" in s:
        return True
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) == 1:
        try:
            return 0 <= float(nums[0]) <= 100
        except Exception:
            return False
    return False
