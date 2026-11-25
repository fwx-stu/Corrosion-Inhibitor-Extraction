from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from corrosion_ie.config import load_config
from corrosion_ie.llm.clients import GenerationConfig, InputTokenBudgetError, LLMClient
from corrosion_ie.runner import _relevant_protocol_files, iter_matrix, validate_run_directory
from corrosion_ie.schemas import EvidenceUnit
from corrosion_ie.strategies.common import fit_sequential_content, prepare_prompt_envelope
from corrosion_ie.strategies.factory import normalize_strategy_name
from corrosion_ie.strategies.token_packing import (
    fragment_table_unit,
    infer_target_group,
    pack_evidence_for_prompt,
    serialize_legacy_evidence,
)


class CharClient(LLMClient):
    token_count_is_exact = True

    def count_text_tokens(self, text: str) -> int:
        return len(text or "")

    def count_chat_tokens(self, system: str, user: str) -> int:
        return len(system or "") + len(user or "") + 8

    def generate(self, system: str, user: str, gen: GenerationConfig) -> str:
        tokens = self.count_chat_tokens(system, user)
        if gen.max_input_tokens and tokens > gen.max_input_tokens:
            raise InputTokenBudgetError("over budget")
        self.last_generation_metadata = {
            "input_tokens_before_truncation": tokens,
            "input_tokens_after_truncation": tokens,
            "truncated": False,
            "input_over_budget": False,
            "generated_tokens": 2,
            "reached_max_new_tokens": False,
            "finish_reason": "eos",
        }
        return "[]"


def make_envelope(template: str = "templates/q1_rag"):
    return prepare_prompt_envelope(
        paper_id="p",
        prompts_dir=ROOT / "prompts",
        prompt_variant="balanced",
        prompt_template=template,
    )


def test_active_matrix_is_q0_q1_q2_only():
    cfg = load_config(ROOT / "configs/experiment_q0q1q2_dev37_balanced_opt.yaml")
    jobs = list(iter_matrix(cfg))
    assert len(jobs) == 9
    assert {job[1] for job in jobs} == {
        "q0_safe_full",
        "q1_budgeted_rag",
        "q2_table_aware_rag",
    }
    assert {job[2] for job in jobs} == {"balanced"}


def test_legacy_strategy_aliases_resolve_to_safe_active_names():
    assert normalize_strategy_name("q0_full") == "q0_safe_full"
    assert normalize_strategy_name("q1_naive_rag") == "q1_budgeted_rag"
    assert normalize_strategy_name("q2") == "q2_table_aware_rag"
    assert normalize_strategy_name("q6") == "q6"


def test_prompt_envelope_protects_output_suffix():
    env = make_envelope("templates/q0_full")
    system, user = env.render("ARTICLE")
    assert "ARTICLE" in user
    assert "OUTPUT JSON ARRAY ONLY:" in user
    assert user.index("ARTICLE") < user.index("OUTPUT JSON ARRAY ONLY:")


def test_safe_full_truncates_only_document_content():
    client = CharClient()
    env = make_envelope("templates/q0_full")
    empty_system, empty_user = env.render("")
    fixed = client.count_chat_tokens(empty_system, empty_user)
    text = "A" * 2000
    packed, diag = fit_sequential_content(
        text,
        envelope=env,
        client=client,
        max_input_tokens=fixed + 250,
        safety_margin_tokens=10,
    )
    system, user = env.render(packed)
    assert len(packed) < len(text)
    assert packed == text[: len(packed)]
    assert "OUTPUT JSON ARRAY ONLY:" in user
    assert client.count_chat_tokens(system, user) <= fixed + 240
    assert diag["content_truncated"] is True


def test_budgeted_rag_caps_windows_and_never_exceeds_chat_budget():
    client = CharClient()
    env = make_envelope()
    evidence = [
        EvidenceUnit("p", f"direct{i}", "chunk", "D" * 120, "performance", score=10 - i)
        for i in range(5)
    ] + [
        EvidenceUnit("p", f"window{i}", "chunk", "W" * 80, "window", score=1)
        for i in range(10)
    ]
    fixed = client.count_chat_tokens(*env.render(serialize_legacy_evidence("p", [])))
    packed = pack_evidence_for_prompt(
        paper_id="p",
        evidence=evidence,
        envelope=env,
        client=client,
        max_input_tokens=fixed + 650,
        max_items=6,
        max_window_chunks=2,
        safety_margin_tokens=10,
        table_aware=False,
    )
    assert len(packed.evidence) <= 6
    assert sum(ev.query_group == "window" for ev in packed.evidence) <= 2
    assert packed.diagnostics["dropped_window_cap"] == 8
    assert client.count_chat_tokens(*packed.messages) <= fixed + 640
    assert "OUTPUT JSON ARRAY ONLY:" in packed.messages[1]


def test_table_aware_prefers_condition_and_target_table_over_reference_table():
    client = CharClient()
    env = make_envelope()
    condition = EvidenceUnit(
        "p", "cond", "chunk", "Mild steel was immersed in 1 M HCl for 6 h.", "conditions", score=1
    )
    target = EvidenceUnit(
        "p",
        "target",
        "table",
        "| Concentration | IE% |\n|---|---|\n|100 ppm|91%|",
        "table_extra",
        score=0.5,
        caption="Table 3. Weight loss inhibition efficiency",
    )
    reference = EvidenceUnit(
        "p",
        "reference",
        "table",
        "| Inhibitor | IE% |\n|---|---|\n|Other|99%|",
        "performance",
        score=99,
        caption="Comparison with previously reported inhibitors in the literature",
    )
    packed = pack_evidence_for_prompt(
        paper_id="p",
        evidence=[reference, target, condition],
        envelope=env,
        client=client,
        max_input_tokens=100000,
        max_items=2,
        max_window_chunks=0,
        table_aware=True,
        condition_anchor_quota=1,
    )
    ids = [ev.unit_id for ev in packed.evidence]
    assert "cond" in ids
    assert "target" in ids
    assert "reference" not in ids
    assert packed.diagnostics["reference_table_candidates"] == 1



def test_table_semantics_override_mismatched_retrieval_query_group():
    ev = EvidenceUnit(
        "p",
        "quantum_table",
        "table",
        "| Compound | EHOMO | ELUMO |\n|---|---|---|\n|A|-5.2|-1.1|",
        "performance",
        caption="Table 8. Quantum chemical parameters",
    )
    assert infer_target_group(ev) == "quantum"


def test_chunkcap_ablation_configs_are_isolated_and_consistent():
    for cap in (40, 70):
        cfg = load_config(ROOT / f"configs/experiment_q1q2_chunkcap{cap}_dev37_7b_opt.yaml")
        assert cfg.models == ["qwen2_5_7b"]
        assert cfg.strategies == ["q1_budgeted_rag", "q2_table_aware_rag"]
        assert cfg.retrieval["retrieval_top_k"] == 120
        assert cfg.retrieval["rerank_top_n"] == 80
        assert cfg.retrieval["retrieval_candidate_max_chunks"] == 80
        assert cfg.retrieval["max_final_chunks"] == cap


def test_large_html_table_is_split_on_rows_with_repeated_header_and_caption():
    client = CharClient()
    rows = "".join(f"<tr><td>{i}</td><td>{90+i}%</td></tr>" for i in range(20))
    ev = EvidenceUnit(
        "p",
        "table1",
        "table",
        f"<table><tr><th>Concentration</th><th>IE%</th></tr>{rows}</table>",
        "performance",
        caption="Table 1 performance",
    )
    parts = fragment_table_unit(ev, client=client, max_fragment_tokens=250)
    assert len(parts) > 1
    for part in parts:
        assert "Table 1 performance" in part.text
        assert "| Concentration | IE% |" in part.text
        assert part.metadata["parent_unit_id"] == "table1"
        assert "|" in part.text


def test_protocol_fingerprint_sources_include_new_packer_and_exclude_q6():
    for strategy in ["q0_safe_full", "q1_budgeted_rag", "q2_table_aware_rag"]:
        sources, prompts = _relevant_protocol_files(ROOT, strategy, "balanced")
        names = {path.name for path in sources}
        assert "token_packing.py" in names
        assert "q6_evidence_aware.py" not in names
        assert all(path.exists() for path in sources + prompts)


def test_run_validation_can_enforce_quality_gate(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{
        "paper_id": "p",
        "parse_error": "bad json",
        "result": {"records": []},
        "metadata": {"generation": {"truncated": False, "input_over_budget": False}},
    }]
    (run / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (run / "run_meta.json").write_text(json.dumps({
        "docs_total": 1,
        "docs_ok": 1,
        "docs_failed": 0,
        "quality_gate_pass": False,
    }))
    structural, _ = validate_run_directory(run, enforce_quality=False)
    strict, details = validate_run_directory(run, enforce_quality=True)
    assert structural is True
    assert strict is False
    assert details["quality_counts"]["parse_error"] == 1


def test_active_source_tree_contains_no_q6_strategy():
    active = ROOT / "corrosion_ie" / "strategies"
    assert not (active / "q6_evidence_aware.py").exists()
