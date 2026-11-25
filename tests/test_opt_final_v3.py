from pathlib import Path
import os
import subprocess
import sys

from corrosion_ie.llm.clients import DryRunClient
from corrosion_ie.schemas import EvidenceUnit
from corrosion_ie.strategies.common import PromptEnvelope, _retreat_to_safe_document_boundary
from corrosion_ie.strategies.token_packing import pack_evidence_for_prompt

ROOT = Path(__file__).resolve().parents[1]


def env():
    return PromptEnvelope("S", "", "U", "Z", "user", "t", "balanced")


def ev(uid, score, doc, *, table=False, group="performance"):
    return EvidenceUnit(
        "p", uid, "table" if table else "chunk",
        "| Inhibitor | IE% |\n|---|---|\n|A|90|" if table else f"text {uid}",
        group, rank=doc, score=score,
        metadata={"selection_score": score, "direct_hit": True, "chunk_index_in_doc": doc},
    )


def test_q1_selects_by_relevance_before_document_presentation():
    packed = pack_evidence_for_prompt(
        paper_id="p", evidence=[ev("early", 0.1, 1), ev("late_table", 1.0, 99, table=True)],
        envelope=env(), client=DryRunClient(), max_input_tokens=10000,
        max_items=1, max_window_chunks=0, table_aware=False,
    )
    assert [x.unit_id for x in packed.evidence] == ["late_table"]
    assert packed.diagnostics["selection_order_unit_ids"] == ["late_table"]


def test_q1_reorders_only_after_selected_set_is_fixed():
    packed = pack_evidence_for_prompt(
        paper_id="p", evidence=[ev("late", 1.0, 99), ev("early", 0.8, 1), ev("noise", 0.1, 2)],
        envelope=env(), client=DryRunClient(), max_input_tokens=10000,
        max_items=2, max_window_chunks=0, table_aware=False,
    )
    assert packed.diagnostics["selection_order_unit_ids"] == ["late", "early"]
    assert [x.unit_id for x in packed.evidence] == ["early", "late"]


def test_q2_presents_conditions_then_target_tables():
    condition = ev("cond", 0.4, 5, group="conditions")
    condition.text = "Mild steel in 1 M HCl for 6 h."
    table = ev("perf_table", 0.8, 100, table=True, group="performance")
    prose = ev("prose", 1.0, 1, group="performance")
    packed = pack_evidence_for_prompt(
        paper_id="p", evidence=[prose, table, condition], envelope=env(),
        client=DryRunClient(), max_input_tokens=10000, max_items=3,
        max_window_chunks=0, table_aware=True, condition_anchor_quota=1,
    )
    assert [x.unit_id for x in packed.evidence][:2] == ["cond", "perf_table"]


def test_safe_full_retreats_only_from_partial_markdown_table():
    partial = "paragraph\n\n| A | B |\n|---|---|\n| 1 |"
    fixed, why = _retreat_to_safe_document_boundary(partial)
    assert "| A |" not in fixed
    assert "partial_markdown_table" in why
    complete = "paragraph\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    fixed2, _ = _retreat_to_safe_document_boundary(complete)
    assert "| 1 | 2 |" in fixed2


def test_formal_test_rejects_cli_matrix_override_before_model_loading(tmp_path):
    env_vars = os.environ.copy()
    env_vars["CONFIRM_FROZEN_TEST"] = "YES"
    env_vars["PROTOCOL_MANIFEST"] = str(tmp_path / "missing.json")
    proc = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_matrix.py"),
        "--config", str(ROOT / "configs/experiment_q0q1q2_test50_balanced_opt.yaml"),
        "--models", "qwen2_5_3b",
    ], cwd=ROOT, env=env_vars, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "formal test matrix cannot be overridden" in (proc.stdout + proc.stderr)


def test_q2_round_robins_fragments_across_source_tables():
    from corrosion_ie.strategies.token_packing import order_table_aware_rag
    def frag(uid, parent, score):
        item = ev(uid, score, 100, table=True, group="performance")
        item.metadata["parent_unit_id"] = parent
        return item
    ordered, _ = order_table_aware_rag([
        frag("a1", "A", 1.0), frag("a2", "A", 0.99), frag("a3", "A", 0.98),
        frag("b1", "B", 0.8), frag("c1", "C", 0.7),
    ], condition_anchor_quota=0)
    table_ids = [x.unit_id for x in ordered if x.unit_type == "table"]
    assert table_ids[:3] == ["a1", "b1", "c1"]


def test_formal_test_requires_protocol_manifest_after_matrix_checks(tmp_path):
    env_vars = os.environ.copy()
    env_vars["CONFIRM_FROZEN_TEST"] = "YES"
    env_vars.pop("PROTOCOL_MANIFEST", None)
    proc = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_matrix.py"),
        "--config", str(ROOT / "configs/experiment_q0q1q2_test50_balanced_opt.yaml"),
    ], cwd=ROOT, env=env_vars, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "PROTOCOL_MANIFEST is required" in (proc.stdout + proc.stderr)
