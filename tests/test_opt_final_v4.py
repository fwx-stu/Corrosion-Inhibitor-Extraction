from __future__ import annotations

from pathlib import Path
import re

from corrosion_ie.schemas import EvidenceUnit
from corrosion_ie.strategies.table_normalization import (
    html_tables_to_grids,
    markdown_table_to_grid,
    normalize_and_sanitize_tables,
)
from corrosion_ie.strategies.token_packing import (
    infer_target_group,
    is_low_value_table,
    order_table_aware_rag,
)

ROOT = Path(__file__).resolve().parents[1]


def _real_html_table(paper_id: str, label: str) -> str:
    text = (ROOT / f"input/test50/md/{paper_id}.md").read_text(encoding="utf-8")
    start = text.index(f"**{label}")
    match = re.search(r"<table\b[^>]*>.*?</table>", text[start:], re.I | re.S)
    assert match is not None
    return match.group(0)


def test_real_three_level_adsorption_header_is_fully_flattened():
    html = _real_html_table("pdf_1934", "Table 6")
    grids = html_tables_to_grids(html, caption="Table 6")
    assert len(grids) == 1
    grid = grids[0]
    assert grid.diagnostics["header_row_count"] == 3
    assert grid.rows[0][0] == "PTPEU"
    assert all("K_ads" not in row[0] for row in grid.rows)
    assert any("303K / ΔG°ads" in header for header in grid.headers)
    assert any("313K / K_ads" in header for header in grid.headers)
    assert any("323K / ΔG°ads" in header for header in grid.headers)


def test_real_three_level_thermodynamic_header_is_not_a_data_row():
    html = _real_html_table("pdf_1934", "Table 7")
    grid = html_tables_to_grids(html, caption="Table 7")[0]
    assert grid.diagnostics["header_row_count"] == 3
    assert grid.rows[0][0] == "303–323K"
    assert not any(row[0] == "T (K)" for row in grid.rows)
    assert any("PTPEU" in header and "Ea" in header for header in grid.headers)


def test_real_multi_isotherm_html_is_split_into_four_logical_tables():
    html = _real_html_table("pdf_1641", "Table 3")
    grids = html_tables_to_grids(html, caption="Table 3")
    assert len(grids) == 4
    labels = [str(grid.diagnostics.get("logical_subtable_label")) for grid in grids]
    assert labels == ["Langmuir's", "Freundlich's", "Temkin's", "El-Awady's"]
    for grid in grids:
        assert len(grid.rows) == 8
        assert grid.rows[0][0] == "EO rosemary"
        assert not any(row[0].lower() == "inhibitor" for row in grid.rows)
        assert "logical subtable:" in grid.caption

    ev = EvidenceUnit("pdf_1641", "table3", "table", html, "adsorption", caption="Table 3")
    normalized, info = normalize_and_sanitize_tables(ev)
    assert len(normalized) == 4
    assert len({item.unit_id for item in normalized}) == 4
    assert info["logical_table_count"] == 4


def test_table_semantics_cannot_be_promoted_by_retrieval_query_text():
    eds = EvidenceUnit(
        "p",
        "eds",
        "table",
        "| Element | Weight % |\n|---|---|\n| Fe | 95.2 |",
        query_group="performance",
        query="inhibition efficiency concentration IE%",
        caption="Table 4. EDS elemental composition",
    )
    assert is_low_value_table(eds) is True
    assert infer_target_group(eds) is None

    xps = EvidenceUnit(
        "p",
        "xps",
        "table",
        "| Peak | Binding energy |\n|---|---|\n| C 1s | 284.8 |",
        query_group="quantum",
        query="HOMO LUMO DFT quantum chemical parameters",
        caption="Table 8. XPS peak assignments",
    )
    assert is_low_value_table(xps) is True
    assert infer_target_group(xps) is None

    real_quantum = EvidenceUnit(
        "p",
        "quantum",
        "table",
        "| Compound | EHOMO | ELUMO |\n|---|---|---|\n| A | -5.2 | -1.1 |",
        query_group="performance",
        query="inhibition efficiency",
        caption="Table 10. Quantum chemical parameters",
    )
    assert is_low_value_table(real_quantum) is False
    assert infer_target_group(real_quantum) == "quantum"


def _table(uid: str, group: str, score: float) -> EvidenceUnit:
    if group == "performance":
        text = "| Inhibitor | Concentration | IE% |\n|---|---|---|\n| A | 100 ppm | 90 |"
    elif group == "quantum":
        text = "| Compound | HOMO | LUMO |\n|---|---|---|\n| A | -5.2 | -1.1 |"
    else:
        text = "| Compound | ΔGads |\n|---|---|\n| A | -32.0 |"
    return EvidenceUnit(
        "p", uid, "table", text, query_group="wrong", score=score,
        caption=f"Target {group} table", metadata={"selection_score": score},
    )


def test_q2_interleaves_target_groups_after_minimum_coverage():
    items = [
        _table("p1", "performance", 1.00), _table("p2", "performance", 0.99),
        _table("q1", "quantum", 0.90), _table("q2", "quantum", 0.89),
        _table("a1", "adsorption", 0.80), _table("a2", "adsorption", 0.79),
    ]
    ordered, diag = order_table_aware_rag(items, condition_anchor_quota=0)
    ids = [item.unit_id for item in ordered if item.unit_type == "table"]
    assert ids[:6] == ["p1", "q1", "a1", "p2", "q2", "a2"]
    assert diag["cross_group_round_robin"] is True


def test_formal_manifest_requires_and_hashes_resolved_local_artifacts(tmp_path, monkeypatch):
    from scripts.freeze_protocol import build_manifest

    for env_name, folder in [
        ("QWEN_3B_PATH", "qwen3b"),
        ("QWEN_7B_PATH", "qwen7b"),
        ("QWEN_14B_PATH", "qwen14b"),
    ]:
        model_dir = tmp_path / folder
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"fake-weight-" + folder.encode())
        monkeypatch.setenv(env_name, str(model_dir))

    embedding = tmp_path / "bge-m3"
    reranker = tmp_path / "bge-reranker"
    for path in (embedding, reranker):
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setenv("EMBEDDING_MODEL", str(embedding))
    monkeypatch.setenv("RERANKER_MODEL", str(reranker))
    for name in ("GEN_TEMPERATURE", "GEN_TOP_P", "GEN_MAX_NEW_TOKENS", "GEN_MAX_INPUT_TOKENS", "GEN_DO_SAMPLE"):
        monkeypatch.delenv(name, raising=False)

    manifest = build_manifest(ROOT / "configs/experiment_q0q1q2_test50_balanced_opt.yaml")
    assert manifest["schema"] == "8b-opt-protocol-manifest-v6"
    assert all(item["artifact"]["exists"] for item in manifest["models"].values())
    assert manifest["retrieval_models"]["embedding"]["exists"] is True
    assert manifest["retrieval_models"]["reranker"]["exists"] is True
    assert all("sha256" in weight for item in manifest["models"].values() for weight in item["artifact"]["weight_manifest"])


def test_report_consumes_wide_bootstrap_and_comparison_schemas(tmp_path):
    import csv
    import subprocess
    import sys

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(tmp_path / "aggregate_metrics.csv", [{
        "model_key": "qwen2_5_3b", "strategy": "q0_safe_full", "n_runs": 1,
        "field_micro_f1_mean": 0.5, "field_micro_f1_std": 0.0,
        "record_f1_mean": 0.4, "record_f1_std": 0.0, "null_accuracy_mean": 0.8,
    }])
    write_csv(tmp_path / "condition_metrics_with_bootstrap_ci.csv", [{
        "model_key": "qwen2_5_3b", "strategy": "q0_safe_full", "prompt_variant": "balanced",
        "repeat": "r1", "paper_count": 37,
        "field_micro_f1": 0.5, "field_micro_f1_ci_low": 0.4, "field_micro_f1_ci_high": 0.6,
        "record_f1": 0.4, "record_f1_ci_low": 0.3, "record_f1_ci_high": 0.5,
        "null_accuracy": 0.8, "null_accuracy_ci_low": 0.7, "null_accuracy_ci_high": 0.9,
    }])
    write_csv(tmp_path / "paired_statistical_comparisons.csv", [{
        "comparison_family": "strategy", "comparison": "Q1-Q0", "model_or_strategy": "qwen2_5_3b",
        "prompt_variant": "balanced", "repeat": "r1", "metric": "field_micro_f1",
        "value_low": 0.5, "value_high": 0.55, "difference_high_minus_low": 0.05,
        "ci_low": 0.01, "ci_high": 0.09, "p_value": 0.01, "p_holm": 0.03,
    }])
    subprocess.run([
        sys.executable, str(ROOT / "scripts/make_report.py"), "--run-root", str(tmp_path)
    ], check=True, cwd=ROOT)
    report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "field_micro_f1_ci_low" in report
    assert "difference_high_minus_low" in report
    assert "0.0500" in report


def _real_markdown_table(paper_id: str, first_row: str) -> str:
    split = "dev37" if (ROOT / f"input/dev37/md/{paper_id}.md").exists() else "test50"
    text = (ROOT / f"input/{split}/md/{paper_id}.md").read_text(encoding="utf-8")
    start = text.index(first_row)
    rows = []
    for line in text[start:].splitlines():
        if line.strip().startswith("|"):
            rows.append(line)
        elif rows:
            break
    return "\n".join(rows)


def test_real_markdown_lower_header_after_separator_is_promoted():
    grid = markdown_table_to_grid(
        _real_markdown_table("xml_2166", "| N° | Coded variables")
    )
    assert grid is not None
    assert grid.diagnostics["post_separator_header_rows_promoted"] == 1
    assert grid.diagnostics["header_row_count"] == 2
    assert grid.rows[0][0] == "1"
    assert "Coded variables / CRC (ppm)" in grid.headers
    assert not any(row[0] == "N°" for row in grid.rows)


def test_real_markdown_multi_isotherm_header_is_flattened():
    grid = markdown_table_to_grid(
        _real_markdown_table("xml_2101", "| Models | Langmuir isotherm")
    )
    assert grid is not None
    assert grid.diagnostics["post_separator_header_rows_promoted"] == 1
    assert grid.rows[0][0] == "ACP"
    assert "Langmuir isotherm / intercept" in grid.headers
    assert "El-Awady isotherm / R 2" in grid.headers
    assert not any(row[0].lower() == "inhibitor" for row in grid.rows)
