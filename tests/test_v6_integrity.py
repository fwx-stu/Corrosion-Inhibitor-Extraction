from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_current_evaluation(run_dir: Path, cfg, *, stale: bool = False) -> None:
    from corrosion_ie.eval.integrity import expected_evaluation_metadata, sha256_file

    run_dir.mkdir(parents=True, exist_ok=True)
    pred = run_dir / "predictions.jsonl"
    pred.write_text('{"paper_id":"p1","result":{"records":[]}}\n', encoding="utf-8")
    write_csv(run_dir / "evaluation/summary_metrics.csv", [{
        "record_precision": 0.5,
        "record_recall": 0.4,
        "record_f1": 0.4444,
        "field_micro_precision": 0.6,
        "field_micro_recall": 0.5,
        "field_micro_f1": 0.5455,
        "null_accuracy": 0.8,
        "pred_record_total": 1,
        "gold_record_total": 2,
    }])
    meta = {
        "prediction_sha256": "stale" if stale else sha256_file(pred),
        **expected_evaluation_metadata(cfg),
    }
    (run_dir / "evaluation/evaluation_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps({
        "docs_total": 1,
        "docs_ok": 1,
        "docs_failed": 0,
        "complete": True,
        "quality_gate_pass": True,
        "document_coverage": 1.0,
        "quality_metrics": {
            "parse_error_rate": 0.0,
            "empty_prediction_rate": 0.0,
            "output_truncation_rate": 0.0,
        },
    }), encoding="utf-8")


def test_evaluation_integrity_rejects_stale_prediction_hash(tmp_path):
    from corrosion_ie.config import load_config
    from corrosion_ie.eval.integrity import evaluation_integrity, expected_evaluation_metadata

    cfg = load_config(ROOT / "configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml")
    run_dir = tmp_path / "qwen2_5_3b/q0_safe_full/recall/r1"
    make_current_evaluation(run_dir, cfg, stale=True)
    valid, reason, _ = evaluation_integrity(run_dir, expected_evaluation_metadata(cfg))
    assert valid is False
    assert "prediction hash" in reason


def test_aggregate_excludes_stale_evaluation(tmp_path):
    from corrosion_ie.config import load_config
    from corrosion_ie.eval.aggregate import aggregate
    from corrosion_ie.eval.integrity import expected_evaluation_metadata

    cfg = load_config(ROOT / "configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml")
    valid_dir = tmp_path / "qwen2_5_3b/q0_safe_full/recall/r1"
    stale_dir = tmp_path / "qwen2_5_3b/q1_budgeted_rag/recall/r1"
    make_current_evaluation(valid_dir, cfg)
    make_current_evaluation(stale_dir, cfg, stale=True)
    (tmp_path / "matrix_manifest.json").write_text(json.dumps({"jobs": [
        {"run_dir": str(valid_dir)},
        {"run_dir": str(stale_dir)},
    ]}), encoding="utf-8")
    aggregate(tmp_path, expected_evaluation=expected_evaluation_metadata(cfg))
    with (tmp_path / "all_run_summary_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    exclusions = json.loads((tmp_path / "aggregation_exclusions.json").read_text(encoding="utf-8"))
    assert exclusions["excluded_runs"] == 1


def test_statistical_analysis_skips_missing_conditions_and_writes_prompt_mean(tmp_path):
    from corrosion_ie.config import load_config

    cfg_path = ROOT / "configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
    cfg = load_config(cfg_path)
    valid_dir = tmp_path / "qwen2_5_3b/q0_safe_full/recall/r1"
    missing_dir = tmp_path / "qwen2_5_3b/q1_budgeted_rag/recall/r1"
    make_current_evaluation(valid_dir, cfg)
    write_csv(valid_dir / "evaluation/per_paper_metrics.csv", [
        {"paper_id": "p1", "record_tp": 1, "pred_record_count": 1, "gold_record_count": 1, "correct_null_count": 1, "gold_null_count_matched": 1},
        {"paper_id": "p2", "record_tp": 0, "pred_record_count": 1, "gold_record_count": 1, "correct_null_count": 1, "gold_null_count_matched": 1},
    ])
    write_csv(valid_dir / "evaluation/per_paper_field_metrics.csv", [
        {"paper_id": "p1", "tp": 2, "pred_count": 2, "gold_count": 2},
        {"paper_id": "p2", "tp": 0, "pred_count": 1, "gold_count": 2},
    ])
    (tmp_path / "matrix_manifest.json").write_text(json.dumps({"jobs": [
        {"model_key": "qwen2_5_3b", "strategy": "q0_safe_full", "prompt_variant": "recall", "seed": 1, "run_dir": str(valid_dir)},
        {"model_key": "qwen2_5_3b", "strategy": "q1_budgeted_rag", "prompt_variant": "recall", "seed": 1, "run_dir": str(missing_dir)},
    ]}), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/statistical_analysis.py"),
        "--config", str(cfg_path), "--run-root", str(tmp_path),
        "--bootstrap-iterations", "20", "--permutation-iterations", "20",
    ], check=True, cwd=ROOT)
    with (tmp_path / "bootstrap_ci_prompt_mean.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {"record_f1", "field_micro_f1", "null_accuracy"}
    meta = json.loads((tmp_path / "statistical_analysis_meta.json").read_text(encoding="utf-8"))
    assert len(meta["conditions_skipped"]) == 1


def test_v6_static_preflight_counts_both_matrices():
    for config, jobs in [
        ("configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml", "27"),
        ("configs/experiment_q0q1q2_test50_balanced_opt_v6.yaml", "9"),
    ]:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/preflight_v6.py"),
            "--config", str(ROOT / config), "--expected-jobs", jobs, "--skip-models",
        ], check=True, cwd=ROOT)


def test_q0_q1_share_exhaustive_anchor_rules():
    q0 = (ROOT / "prompts/templates/q0_full.txt").read_text(encoding="utf-8")
    q1 = (ROOT / "prompts/templates/q1_rag.txt").read_text(encoding="utf-8")
    for phrase in [
        "Each distinct table body row is normally a separate anchor.",
        "Do not stop after extracting the first few valid rows.",
        "If a complete series is explicitly listed, extract the complete listed series",
    ]:
        assert phrase in q0
        assert phrase in q1
