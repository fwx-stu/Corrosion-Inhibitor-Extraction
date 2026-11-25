from __future__ import annotations

import json
from pathlib import Path

from corrosion_ie.config import load_config
from corrosion_ie.eval.aggregate import discover_run_dirs as discover_aggregate_run_dirs
from corrosion_ie.llm.clients import LLMClient
from corrosion_ie.runner import _relevant_protocol_files, iter_matrix
from corrosion_ie.schemas import EvidenceUnit
from corrosion_ie.strategies.common import prepare_prompt_envelope
from corrosion_ie.strategies.factory import normalize_strategy_name
from corrosion_ie.strategies.non_rag_compression import (
    _load_token_budget,
    _section_aware_indices,
    _uniform_spread_indices,
)
from corrosion_ie.strategies.token_packing import pack_evidence_for_prompt
from scripts.run_matrix import merge_matrix_manifest
from scripts.statistical_analysis import STRATEGY_PAIRS, discover_conditions


ROOT = Path(__file__).resolve().parents[1]


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


def test_reviewer_control_configs_expand_only_new_jobs():
    expected = {
        "experiment_reviewer_controls_dev37_smoke_v1.yaml": 4,
        "experiment_reviewer_controls_test50_v1_gpu_3b7b.yaml": 8,
        "experiment_reviewer_controls_test50_v1_cpu_14b.yaml": 4,
    }
    allowed = {
        "b0_uniform_token_matched",
        "b1_section_aware_token_matched",
        "a1_q1_table_budget",
        "a2_q1_condition_anchors",
    }
    for name, job_count in expected.items():
        cfg = load_config(ROOT / "configs" / name)
        jobs = list(iter_matrix(cfg))
        assert len(jobs) == job_count
        assert {job[1] for job in jobs} == allowed
        assert {job[2] for job in jobs} == {"balanced"}


def test_new_strategy_aliases_and_fingerprint_sources_exist():
    aliases = {
        "uniform_token_matched": "b0_uniform_token_matched",
        "section_aware": "b1_section_aware_token_matched",
        "q1_table_only": "a1_q1_table_budget",
        "q1_anchor_only": "a2_q1_condition_anchors",
    }
    for alias, canonical in aliases.items():
        assert normalize_strategy_name(alias) == canonical
        sources, prompts = _relevant_protocol_files(ROOT, canonical, "balanced")
        assert all(path.is_file() for path in sources + prompts)


def test_q2_default_remains_identical_to_explicit_combined_mode():
    evidence = [
        EvidenceUnit(
            "p", "cond", "chunk",
            "Mild steel was immersed in 1 M HCl for 6 h.",
            "conditions", score=1,
        ),
        EvidenceUnit(
            "p", "target", "table",
            "| Concentration | IE% |\n|---|---|\n|100 ppm|91%|",
            "performance", score=0.5,
            caption="Table 3. Weight loss inhibition efficiency",
        ),
    ]
    kwargs = dict(
        paper_id="p",
        evidence=evidence,
        envelope=envelope(),
        client=CharClient(),
        max_input_tokens=100000,
        max_items=2,
        max_window_chunks=0,
        table_aware=True,
        condition_anchor_quota=1,
    )
    legacy_call = pack_evidence_for_prompt(**kwargs)
    explicit_call = pack_evidence_for_prompt(**kwargs, condition_aware=True)
    assert [item.unit_id for item in legacy_call.evidence] == [
        item.unit_id for item in explicit_call.evidence
    ]
    assert legacy_call.text == explicit_call.text
    assert legacy_call.diagnostics["packing_mode"] == "table_aware_select_then_present_v4"


def test_table_only_and_anchor_only_isolate_the_intended_component():
    condition = EvidenceUnit(
        "p", "cond", "chunk",
        "Mild steel was immersed in 1 M HCl for 6 h.",
        "conditions", score=100,
    )
    table = EvidenceUnit(
        "p", "target", "table",
        "| Concentration | IE% |\n|---|---|\n|100 ppm|91%|",
        "performance", score=1,
        caption="Table 3. Weight loss inhibition efficiency",
    )
    common = dict(
        paper_id="p",
        evidence=[condition, table],
        envelope=envelope(),
        client=CharClient(),
        max_input_tokens=100000,
        max_items=1,
        max_window_chunks=0,
        condition_anchor_quota=1,
    )
    table_only = pack_evidence_for_prompt(
        **common, table_aware=True, condition_aware=False,
    )
    anchor_only = pack_evidence_for_prompt(
        **common, table_aware=False, condition_aware=True,
    )
    assert [item.unit_id for item in table_only.evidence] == ["target"]
    assert table_only.diagnostics["packing_mode"] == "table_only_select_then_present_v1"
    assert table_only.diagnostics["condition_anchors_enabled"] is False
    assert [item.unit_id for item in anchor_only.evidence] == ["cond"]
    assert anchor_only.diagnostics["packing_mode"] == "condition_anchor_select_then_present_v1"
    assert anchor_only.diagnostics["table_aware_enabled"] is False


def test_non_rag_orders_are_deterministic_and_cover_every_chunk():
    uniform = _uniform_spread_indices(range(8))
    assert uniform[:2] == [0, 7]
    assert sorted(uniform) == list(range(8))
    chunks = [
        {"section": "Abstract"},
        {"section": "Experimental methods"},
        {"section": "Results and discussion"},
        {"content_type": "table"},
        {"section": "Conclusion"},
        {"section": "Other"},
    ]
    first = _section_aware_indices(chunks)
    assert first == _section_aware_indices(chunks)
    assert sorted(first) == list(range(len(chunks)))
    assert first[0] == 3  # fixed table-first section cycle, not semantic retrieval


def test_frozen_token_budget_manifest_covers_all_test_models_and_papers():
    path = ROOT / "frozen" / "token_budgets_test50_q1q2_v6.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["expected_paper_count"] == 50
    assert len(payload["paper_ids"]) == 50
    paper_id = payload["paper_ids"][0]
    for model in ("qwen2_5_3b", "qwen2_5_7b", "qwen2_5_14b"):
        for strategy in ("q1_budgeted_rag", "q2_table_aware_rag"):
            assert len(payload["budgets"][model][strategy]) == 50
    assert _load_token_budget(
        path,
        model_name="Qwen/Qwen2.5-14B-Instruct",
        reference_strategy="q1_budgeted_rag",
        paper_id=paper_id,
    ) == payload["budgets"]["qwen2_5_14b"]["q1_budgeted_rag"][paper_id]


def test_split_hardware_matrix_manifest_merges_without_duplicates(tmp_path: Path):
    path = tmp_path / "matrix_manifest.json"
    common = {
        "experiment_id": "reviewer-controls",
        "dataset_name": "test50",
        "dataset_role": "test",
        "expected_paper_count": 50,
    }
    first = {
        **common,
        "config_path": "gpu.yaml",
        "jobs": [{
            "model_key": "qwen2_5_3b", "strategy": "a1_q1_table_budget",
            "prompt_variant": "balanced", "seed": 1, "run_dir": "gpu-run",
        }],
    }
    path.write_text(json.dumps(first), encoding="utf-8")
    second = {
        **common,
        "config_path": "cpu.yaml",
        "jobs": [{
            "model_key": "qwen2_5_14b", "strategy": "a1_q1_table_budget",
            "prompt_variant": "balanced", "seed": 1, "run_dir": "cpu-run",
        }],
    }
    merged = merge_matrix_manifest(path, second)
    assert merged["jobs_expected"] == 2
    assert len(merged["jobs"]) == 2
    assert merged["config_paths"] == ["gpu.yaml", "cpu.yaml"]


def test_reviewer_comparisons_reuse_old_q1_q2_results():
    labels = {label for _, _, label in STRATEGY_PAIRS}
    assert {
        "Q1-B0_uniform",
        "Q2-B1_section",
        "A1_table-Q1",
        "A2_anchor-Q1",
        "Q2-A1_table",
        "Q2-A2_anchor",
    } <= labels


def test_moved_package_reconstructs_run_paths_from_manifest(tmp_path: Path):
    local_run = tmp_path / "qwen2_5_3b/a1_q1_table_budget/balanced/r1"
    manifest = {
        "jobs": [{
            "model_key": "qwen2_5_3b",
            "strategy": "a1_q1_table_budget",
            "prompt_variant": "balanced",
            "seed": 1,
            "run_dir": "/old/server/path/that/does/not/exist",
        }]
    }
    (tmp_path / "matrix_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert discover_aggregate_run_dirs(tmp_path) == [local_run]
    conditions = discover_conditions(tmp_path)
    assert len(conditions) == 1
    assert conditions[0].run_dir == local_run
