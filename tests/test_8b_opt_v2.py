from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from corrosion_ie.llm.clients import LLMClient
from corrosion_ie.postprocess.json_parser import parse_llm_json
from corrosion_ie.schemas import EvidenceUnit, FIELD_NAMES
from corrosion_ie.strategies.common import prepare_prompt_envelope
from corrosion_ie.strategies.table_normalization import (
    classify_reference_grid,
    html_table_to_grid,
    normalize_and_sanitize_table,
)
from corrosion_ie.strategies.token_packing import pack_evidence_for_prompt


class CharClient(LLMClient):
    token_count_is_exact = True

    def count_text_tokens(self, text: str) -> int:
        return len(text or "")

    def count_chat_tokens(self, system: str, user: str) -> int:
        return len(system or "") + len(user or "") + 8


def envelope():
    return prepare_prompt_envelope(
        paper_id="p",
        prompts_dir=ROOT / "prompts",
        prompt_variant="balanced",
        prompt_template="templates/q1_rag",
    )


def test_compact_output_complete_and_truncated_salvage():
    row1 = ["A", "steel", "HCl", "1 M", "100 ppm", "298 K", "6 h", "EIS", "91%", None, None, None, None]
    row2 = ["A", "steel", "HCl", "1 M", "200 ppm", "298 K", "6 h", "EIS", "94%", None, None, None, None]
    parsed, err = parse_llm_json(json.dumps([row1, row2]), "p")
    assert err is None
    assert len(parsed["records"]) == 2
    assert parsed["records"][0]["inhibitor_name"] == "A"
    assert list(parsed["records"][0]) == FIELD_NAMES

    truncated = "[" + json.dumps(row1) + "," + json.dumps(row2)[:-8]
    salvaged, err = parse_llm_json(truncated, "p")
    assert len(salvaged["records"]) == 1
    assert err and "salvaged 1 compact rows" in err


def test_rowspan_and_colspan_are_expanded_to_explicit_logical_rows():
    html = """
    <table>
      <tr><th rowspan='2'>Inhibitor</th><th colspan='2'>303 K</th></tr>
      <tr><th>Concentration</th><th>IE%</th></tr>
      <tr><td rowspan='2'>Compound A</td><td>100 ppm</td><td>90</td></tr>
      <tr><td>200 ppm</td><td>94</td></tr>
    </table>
    """
    grid = html_table_to_grid(html, caption="Table 3 performance")
    assert grid is not None
    assert grid.headers == ["Inhibitor", "303 K / Concentration", "303 K / IE%"]
    assert grid.rows[0] == ["Compound A", "100 ppm", "90"]
    assert grid.rows[1] == ["Compound A", "200 ppm", "94"]
    assert grid.diagnostics["rowspan_count"] == 2
    assert grid.diagnostics["colspan_count"] == 1


def test_mixed_literature_table_keeps_all_current_work_continuation_rows():
    html = """
    <table>
      <tr><th>Inhibitor</th><th>IE%</th><th>Refs.</th></tr>
      <tr><td>Old A</td><td>80</td><td>[1]</td></tr>
      <tr><td>Old B</td><td>82</td><td></td></tr>
      <tr><td>New A</td><td>91</td><td>Our work</td></tr>
      <tr><td>New B</td><td>92</td><td></td></tr>
      <tr><td>New C</td><td>93</td><td></td></tr>
    </table>
    """
    ev = EvidenceUnit("p", "t", "table", html, "performance", caption="Comparison with literature data")
    normalized, info = normalize_and_sanitize_table(ev)
    assert info["kind"] == "mixed"
    assert info["current_study_row_indices"] == [2, 3, 4]
    assert normalized is not None
    assert "Old A" not in normalized.text
    assert "Old B" not in normalized.text
    assert all(name in normalized.text for name in ["New A", "New B", "New C"])


def test_reference_caption_variants_are_recognized():
    captions = [
        "Comparison of the inhibition effectiveness of various inhibitors",
        "Comparison with different plants as a green corrosion inhibitor",
        "Literature reviews on the corrosion inhibition behavior of similar derivatives",
        "Inhibition efficiency values according to the literature",
    ]
    html = "<table><tr><th>Inhibitor</th><th>IE%</th><th>Reference</th></tr><tr><td>X</td><td>90</td><td>[12]</td></tr></table>"
    for caption in captions:
        grid = html_table_to_grid(html, caption=caption)
        assert grid is not None
        assert classify_reference_grid(grid)["kind"] == "pure"


def test_group_coverage_falls_back_to_smaller_candidate_when_top_one_does_not_fit():
    client = CharClient()
    env = envelope()
    fixed = client.count_chat_tokens(*env.render("[Target Paper ID]: p\n"))
    too_large = EvidenceUnit(
        "p", "q_big", "chunk", "HOMO LUMO " + "X" * 2000, "quantum", score=100
    )
    fitting = EvidenceUnit(
        "p", "q_small", "chunk", "DFT table reports HOMO -5.2 eV and LUMO -1.1 eV.", "quantum", score=10
    )
    condition = EvidenceUnit(
        "p", "cond", "chunk", "Mild steel in 1 M HCl for 6 h.", "conditions", score=20
    )
    packed = pack_evidence_for_prompt(
        paper_id="p",
        evidence=[too_large, fitting, condition],
        envelope=env,
        client=client,
        max_input_tokens=fixed + 600,
        max_items=3,
        max_window_chunks=0,
        safety_margin_tokens=10,
        table_aware=True,
        condition_anchor_quota=1,
    )
    ids = [ev.unit_id for ev in packed.evidence]
    assert "q_big" not in ids
    assert "q_small" in ids
    assert "quantum" in packed.diagnostics["packed_groups"]
    assert packed.diagnostics["group_coverage_complete"] is True


def test_formal_test_script_requires_explicit_confirmation_and_manifest():
    text = (ROOT / "scripts/run_q0q1q2_test50_balanced_opt.sh").read_text(encoding="utf-8")
    assert 'CONFIRM_FROZEN_TEST:=YES' not in text
    assert 'CONFIRM_FROZEN_TEST:-' in text
    assert 'PROTOCOL_MANIFEST' in text
    assert '--models' not in text and '"$@"' not in text
    runner = (ROOT / 'scripts/run_matrix.py').read_text(encoding='utf-8')
    assert 'formal test matrix cannot be overridden' in runner


def test_all_active_configs_have_strict_quality_gate_and_compact_protocol():
    for path in sorted((ROOT / "configs").glob("experiment_*_opt.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        gate = cfg["quality_gate"]
        assert gate["max_parse_error_rate"] <= 0.05
        assert gate["max_empty_prediction_rate"] <= 0.05
        assert gate["max_input_over_budget"] == 0
        assert gate["max_input_truncations"] == 0
        assert gate["max_output_truncation_rate"] <= 0.05
        assert cfg["run"]["fail_on_quality_gate"] is True
        assert cfg["output_protocol"] == "compact_rows_v1"


def test_real_pdf2075_reference_table_keeps_all_four_our_work_rows():
    text = (ROOT / "input/test50/md/pdf_2075.md").read_text(encoding="utf-8")
    match = re.search(r"<table\b[^>]*>.*?</table>", text, re.I | re.S)
    assert match
    ev = EvidenceUnit(
        "pdf_2075",
        "table1",
        "table",
        match.group(0),
        "performance",
        caption="Table 1 Literature reviews on similar pyrimidine derivatives studied before",
    )
    normalized, info = normalize_and_sanitize_table(ev)
    assert normalized is not None
    assert info["kind"] == "mixed"
    assert info["rows_kept"] == 4
    for code in ["HM-1221", "HM-1222", "HM-1223", "HM-1224"]:
        assert code in normalized.text
    assert "Compound I" not in normalized.text


def test_review_article_mode_keeps_structured_literature_tables():
    from corrosion_ie.strategies.table_normalization import detect_review_article

    review_text = "# Review title\n\n## Abstract\nThis review gives a comprehensive account of plant inhibitors."
    primary_text = "# Primary study\n\n## Abstract\nWe investigate inhibitor A.\n\nLiterature survey shows prior work."
    assert detect_review_article(review_text) is True
    assert detect_review_article(primary_text) is False

    html = "<table><tr><th>Inhibitor</th><th>IE%</th><th>Reference</th></tr><tr><td>X</td><td>90</td><td>[12]</td></tr></table>"
    ev = EvidenceUnit("p", "review_table", "table", html, "performance", caption="Literature review table")
    dropped, _ = normalize_and_sanitize_table(ev)
    kept, info = normalize_and_sanitize_table(ev, allow_reference_tables=True)
    assert dropped is None
    assert kept is not None and "X" in kept.text
    assert info["action"] == "keep_review_reference_table"


def test_real_pdf1230_is_detected_as_review_and_review_table_is_retained():
    from corrosion_ie.strategies.table_normalization import detect_review_article

    text = (ROOT / "input/test50/md/pdf_1230.md").read_text(encoding="utf-8")
    assert detect_review_article(text) is True
    match = re.search(r"<table\b[^>]*>.*?</table>", text, re.I | re.S)
    assert match
    ev = EvidenceUnit("pdf_1230", "table1", "table", match.group(0), "performance")
    kept, info = normalize_and_sanitize_table(ev, allow_reference_tables=True)
    assert kept is not None
    assert info["kind"] == "pure"
    assert info["action"] == "keep_review_reference_table"
    assert len(kept.text) > 100


def test_source_column_and_method_citations_do_not_create_false_reference_tables():
    anova = """
    <table><tr><th>Response</th><th>Source</th><th>F value</th></tr>
    <tr><td>Weight loss</td><td>Model</td><td>117.9</td></tr></table>
    """
    grid = html_table_to_grid(anova, caption="ANOVA results")
    assert grid is not None
    assert classify_reference_grid(grid)["kind"] == "none"

    adsorption = """
    <table><tr><th>Isotherm</th><th>Parameter</th><th>Value</th></tr>
    <tr><td>Langmuir [61]</td><td>Kads</td><td>3.6e5</td></tr>
    <tr><td>Temkin [62]</td><td>R2</td><td>0.99</td></tr></table>
    """
    grid = html_table_to_grid(adsorption, caption="Thermodynamic parameters")
    assert grid is not None
    assert classify_reference_grid(grid)["kind"] == "none"
