from __future__ import annotations

from pathlib import Path

from corrosion_ie.llm.clients import (
    _repeat_cycle_trim,
    _scan_complete_compact_rows,
    generation_from_model_cfg,
)
from corrosion_ie.postprocess.json_parser import parse_llm_json
from corrosion_ie.postprocess.normalizer import normalize_records
from corrosion_ie.strategies.common import prepare_prompt_envelope


def _row(name: str, method: str = "EIS") -> list[object]:
    return [
        name,
        "mild steel",
        "HCl",
        "1 M",
        "100 ppm",
        "298 K",
        "6 h",
        method,
        "91%",
        None,
        None,
        None,
        None,
    ]


def test_compact_parser_pads_omitted_trailing_nulls() -> None:
    raw = '[["A","steel","HCl","1 M","100 ppm","298 K","6 h","EIS","91%"]]'
    parsed, err = parse_llm_json(raw, "p1")
    assert err is None
    assert len(parsed["records"]) == 1
    record = parsed["records"][0]
    assert record["inhibition_efficiency"] == "91%"
    assert record["homo"] is None
    assert record["adsorption_energy"] is None


def test_truncated_twelve_column_rows_are_salvaged() -> None:
    raw = (
        '[["A","steel","NaCl","3.5 wt%","1 wt%",null,"24 h","EIS",'
        'null,null,null,null],\n["unfinished"'
    )
    parsed, err = parse_llm_json(raw, "p2")
    assert err is not None and "salvaged" in err
    assert len(parsed["records"]) == 1
    assert parsed["records"][0]["adsorption_energy"] is None


def test_repeat_guard_detects_suffix_cycle_and_keeps_one_copy() -> None:
    a = _row("A")
    b = _row("B")
    rows = [_row("U1"), _row("U2"), a, b, a, b]
    detected = _repeat_cycle_trim(rows, min_rows=6, max_cycle=4)
    assert detected is not None
    safe, cycle = detected
    assert cycle == 2
    assert safe == [_row("U1"), _row("U2"), a, b]


def test_repeat_guard_scans_complete_rows_from_incomplete_outer_array() -> None:
    text = "[" + ",".join(__import__("json").dumps(x) for x in [_row("A"), _row("B")]) + ',["cut"'
    rows = _scan_complete_compact_rows(text)
    assert rows == [_row("A"), _row("B")]


def test_field_label_is_not_kept_as_test_method() -> None:
    record = {
        "inhibitor_name": "A",
        "test_method": "Adsorption Energy",
        "inhibition_efficiency": "90%",
    }
    normalized = normalize_records([record])[0]
    assert normalized["test_method"] is None


def test_v5_prompts_have_one_evidence_slot_and_render() -> None:
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    for template in ["templates/q0_full", "templates/q1_rag"]:
        envelope = prepare_prompt_envelope(
            paper_id="p",
            prompts_dir=prompts_dir,
            prompt_variant="balanced",
            prompt_template=template,
        )
        system, user = envelope.render("ARTICLE")
        assert "ARTICLE" in user or "ARTICLE" in system
        assert "Cartesian product" in system
        assert "exactly 13" in user


def test_generation_config_loads_repeat_guard() -> None:
    cfg = {
        "generation": {
            "max_new_tokens": 100,
            "json_repeat_guard": True,
            "json_repeat_guard_min_rows": 7,
            "json_repeat_guard_max_cycle": 3,
            "json_repeat_guard_check_interval": 8,
        }
    }
    gen = generation_from_model_cfg(cfg, seed=1)
    assert gen.json_repeat_guard is True
    assert gen.json_repeat_guard_min_rows == 7
    assert gen.json_repeat_guard_max_cycle == 3
    assert gen.json_repeat_guard_check_interval == 8
