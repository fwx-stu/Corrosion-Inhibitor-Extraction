from __future__ import annotations

"""Evidence-grounded sanity filter for corrosion extraction records.

The filter is intentionally conservative. It blocks high-cost hallucination
patterns seen in Q1, especially acid-inhibitor defaults such as HCl/1 M/ppm
when the evidence is actually a coating/rebar/concrete-pore-solution study.

Important design choice:
- Do not require inhibition_efficiency by default.
- The project schema also contains quantum / adsorption-only records, such as
  homo, lumo, delta_g_ads, and adsorption_energy.
- Requiring inhibition_efficiency globally would silently delete valid auxiliary-field
  records.
"""

import re
from typing import Any, Dict, Iterable, List, Tuple


def _norm(s: Any) -> str:
    s = str(s or "").lower()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("℃", " c").replace("°c", " c").replace("° c", " c")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _numbers(s: Any) -> List[str]:
    return re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(s or ""))


def _contains_number_unit(e: str, value: Any, units: Iterable[str]) -> bool:
    """Return true when a numeric value is supported with one of the units.

    Handles both spaced and compact forms:
    - "1 M" and "1M"
    - "100 ppm" and "100ppm"
    - "15 %" and "15%"
    """
    nums = _numbers(value)
    if not nums:
        return False

    unit_pat = "|".join(re.escape(u) for u in units)

    compact_units = [re.sub(r"\s+", "", str(u)) for u in units]
    compact_unit_pat = "|".join(re.escape(u) for u in compact_units if u)
    compact_e = re.sub(r"\s+", "", e)

    for n in nums[:3]:
        n2 = re.escape(n)

        if re.search(rf"(?<!\d){n2}\s*(?:{unit_pat})(?![a-z0-9])", e, re.I):
            return True

        if compact_unit_pat and re.search(
            rf"(?<!\d){n2}(?:{compact_unit_pat})(?![a-z0-9])",
            compact_e,
            re.I,
        ):
            return True

    return False


def _literal_supported(e: str, value: Any) -> bool:
    if value is None:
        return True

    v = _norm(value)
    if not v:
        return True

    if v in e:
        return True

    compact_v = re.sub(r"\s+", "", v)
    compact_e = re.sub(r"\s+", "", e)
    return len(compact_v) >= 3 and compact_v in compact_e


def _medium_supported(e: str, medium: Any) -> bool:
    if medium is None:
        return True

    m = _norm(medium)
    if not m:
        return True

    checks: List[Tuple[str, Tuple[str, ...]]] = [
        ("hcl", ("hcl", "hydrochloric acid")),
        ("h2so4", ("h2so4", "sulfuric acid", "sulphuric acid")),
        ("h3po4", ("h3po4", "phosphoric acid")),
        ("nacl", ("nacl", "sodium chloride", "chloride")),
        ("seawater", ("seawater", "sea water")),
        (
            "concrete",
            (
                "concrete pore",
                "pore solution",
                "cement mortar",
                "rebar",
                "reinforced concrete",
            ),
        ),
    ]

    for key, aliases in checks:
        if key in m:
            return any(alias in e for alias in aliases)

    words = [w for w in re.findall(r"[a-z0-9]+", m) if len(w) >= 4]
    if len(words) >= 2:
        return sum(1 for w in words if w in e) >= min(2, len(words))

    return _literal_supported(e, m)


def _concentration_supported(e: str, value: Any, field: str) -> bool:
    if value is None:
        return True

    v = _norm(value)
    if not v:
        return True

    if _literal_supported(e, v):
        return True

    # Coating formulations are not ppm dosages.
    # Accept support by formulation tokens plus percentage values instead of
    # requiring one exact string.
    if any(
        token in v
        for token in ["spi", "soy", "sorbitol", "coating", "extract", "polyol", "polymer"]
    ):
        needed: List[str] = []
        for token in ["spi", "soy", "sorbitol", "polyol", "coating"]:
            if token in v:
                needed.append(token)

        nums = _numbers(v)
        ok_nums = sum(
            1
            for n in nums[:4]
            if re.search(rf"(?<!\d){re.escape(n)}\s*%", e)
        )
        return all(token in e for token in needed) and ok_nums >= min(1, len(nums))

    if re.search(r"\bppm\b", v):
        return _contains_number_unit(e, v, ["ppm"])

    if re.search(r"\bmg\s*/?\s*l\b|mg\s*l", v):
        return _contains_number_unit(
            e,
            v,
            ["mg/l", "mg l", "mgl", "mg l-1", "mg/l-1"],
        )

    if re.search(r"\bmm\b|mmol", v):
        return _contains_number_unit(
            e,
            v,
            ["mm", "mmol", "mmol/l", "mmol l-1"],
        )

    if re.search(r"\bmol\s*/?\s*l\b|\bm\b", v) and field == "medium_concentration":
        return _contains_number_unit(
            e,
            v,
            ["m", "mol/l", "mol l-1", "mol l"],
        )

    if "%" in v:
        return _contains_number_unit(
            e,
            v,
            ["%", "wt%", "wt.%", "% wt", "%wt"],
        )

    # Bare numeric values are only accepted if the number appears near
    # concentration/dosage/environment words.
    nums = _numbers(v)
    if nums:
        n = re.escape(nums[0])
        return bool(
            re.search(
                rf"(?<!\d){n}(?!\d).{{0,40}}"
                r"(concentration|dosage|dose|wt|sorbitol|spi|acid|nacl|hcl)",
                e,
            )
        )

    return False


def _method_supported(e: str, value: Any) -> bool:
    if value is None:
        return True

    v = _norm(value)
    if not v:
        return True

    aliases = {
        "potentiodynamic polarization": [
            "potentiodynamic polarization",
            "polarization",
            "tafel",
            "pdp",
        ],
        "electrochemical impedance spectroscopy": [
            "electrochemical impedance",
            "impedance",
            "eis",
        ],
        "weight loss": [
            "weight loss",
            "gravimetric",
        ],
        "rapid macrocell corrosion": [
            "rapid macrocell",
            "macrocell",
        ],
        "linear polarization": [
            "linear polarization",
            "lpr",
        ],
    }

    for key, vals in aliases.items():
        if key in v:
            return any(alias in e for alias in vals)

    return _literal_supported(e, v)


def _efficiency_supported(e: str, value: Any) -> bool:
    if value is None:
        return True

    v = _norm(value)
    if not v:
        return True

    if _literal_supported(e, v):
        return True

    nums = _numbers(v)
    if not nums:
        return False

    has_perf_phrase = bool(
        re.search(
            r"(efficien|inhibition|reduce[d]? the corrosion current|"
            r"corrosion current densit|corrosion rate)",
            e,
        )
    )

    return has_perf_phrase and any(
        re.search(rf"(?<!\d){re.escape(n)}\s*%", e)
        for n in nums[:2]
    )


def _generic_unbacked_temperature(e: str, value: Any) -> bool:
    if value is None:
        return False

    v = _norm(value)
    if not v or _literal_supported(e, v):
        return False

    return bool(
        re.fullmatch(
            r"(25|298|298\.15)\s*(c|k)?|room temperature|ambient",
            v,
        )
    )


def _generic_unbacked_time(e: str, value: Any) -> bool:
    if value is None:
        return False

    v = _norm(value)
    if not v or _literal_supported(e, v):
        return False

    return bool(
        re.fullmatch(
            r"(24|6|12)\s*(h|hr|hrs|hour|hours)",
            v,
        )
    )


def filter_records_by_evidence(
    records: List[Dict[str, Any]],
    evidence_text: str,
    enabled: bool = True,
    return_report: bool = False,
    require_inhibition_efficiency: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop or null obvious unsupported values.

    Parameters
    ----------
    records:
        Parsed extraction records.

    evidence_text:
        Evidence pack text actually shown to the model.

    enabled:
        Config-controlled on/off switch.

    return_report:
        Preserve backward compatibility. If false, return only records.
        If true, return ``(records, report)``.

    require_inhibition_efficiency:
        False by default. The shared LLM postprocessor must not require IE,
        because quantum / adsorption-only records can be valid without IE.
        Set true only for deterministic performance-row candidate filtering.
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "input_count": len(records or []),
        "output_count": len(records or []),
        "dropped_count": 0,
        "nulled_fields": {},
        "drop_reasons": {},
        "dropped_records": [],
        "require_inhibition_efficiency": bool(require_inhibition_efficiency),
    }

    if not enabled:
        return (records, report) if return_report else records

    e = _norm(evidence_text)
    if not e:
        return (records, report) if return_report else records

    out: List[Dict[str, Any]] = []

    def bump(map_name: str, key: str) -> None:
        d = report.setdefault(map_name, {})
        d[key] = int(d.get(key, 0)) + 1

    for idx, rec in enumerate(records or []):
        r = dict(rec)
        fatal: List[str] = []
        nulled: List[str] = []

        med = r.get("corrosive_medium")
        if med and not _medium_supported(e, med):
            # Field-level correction: unsupported medium such as hallucinated HCl
            # should be nulled, not cause the whole record to be dropped.
            r["corrosive_medium"] = None
            nulled.append("corrosive_medium")
            bump("drop_reasons", "unsupported_corrosive_medium")

        inh_c = r.get("inhibitor_concentration")
        if inh_c and not _concentration_supported(e, inh_c, "inhibitor_concentration"):
            # Field-level correction: unsupported ppm series should be nulled,
            # not cause the whole record to be dropped.
            r["inhibitor_concentration"] = None
            nulled.append("inhibitor_concentration")
            bump("drop_reasons", "unsupported_inhibitor_concentration")

        med_c = r.get("medium_concentration")
        if med_c and not _concentration_supported(e, med_c, "medium_concentration"):
            r["medium_concentration"] = None
            nulled.append("medium_concentration")

        method = r.get("test_method")
        if method and not _method_supported(e, method):
            r["test_method"] = None
            nulled.append("test_method")

        eff = r.get("inhibition_efficiency")
        if eff and not _efficiency_supported(e, eff):
            # Do not drop a whole record only because the efficiency value
            # is hard to align textually. Dense tables often contain values
            # in compact or OCR-normalized forms. Keep the record and let
            # evaluation/debug decide. The filter should mainly block
            # unsupported experimental-system hallucinations such as HCl/1 M/ppm.
            pass

        if _generic_unbacked_temperature(e, r.get("temperature")):
            r["temperature"] = None
            nulled.append("temperature")

        if _generic_unbacked_time(e, r.get("immersion_time")):
            r["immersion_time"] = None
            nulled.append("immersion_time")

        for field in nulled:
            bump("nulled_fields", field)

        if fatal:
            report["dropped_count"] += 1
            for reason in fatal:
                bump("drop_reasons", reason)

            if len(report["dropped_records"]) < 50:
                report["dropped_records"].append(
                    {
                        "index": idx,
                        "reasons": fatal,
                        "record": rec,
                    }
                )
            continue

        if require_inhibition_efficiency and not r.get("inhibition_efficiency"):
            report["dropped_count"] += 1
            bump("drop_reasons", "missing_inhibition_efficiency_after_filter")

            if len(report["dropped_records"]) < 50:
                report["dropped_records"].append(
                    {
                        "index": idx,
                        "reasons": ["missing_inhibition_efficiency_after_filter"],
                        "record": rec,
                    }
                )
            continue

        out.append(r)

    report["output_count"] = len(out)
    return (out, report) if return_report else out


def filter_records_with_report(
    records: List[Dict[str, Any]],
    evidence_text: str,
    enabled: bool = True,
    require_inhibition_efficiency: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filtered, report = filter_records_by_evidence(
        records,
        evidence_text,
        enabled=enabled,
        return_report=True,
        require_inhibition_efficiency=require_inhibition_efficiency,
    )
    return filtered, report
