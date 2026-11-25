from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_v6_config_is_tolerant_full27():
    cfg = yaml.safe_load(
        (ROOT / "configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml")
        .read_text(encoding="utf-8")
    )
    assert cfg["experiment_id"].endswith("final_v6")
    assert cfg["run"]["stop_on_error"] is False
    assert cfg["run"]["fail_on_quality_gate"] is False
    assert (
        len(cfg["models"])
        * len(cfg["strategies"])
        * len(cfg["prompt_variants"])
        * len(cfg["repeat_seeds"])
        == 27
    )


def test_v6_full27_script_exists_and_is_named_v6():
    script = ROOT / "scripts/run_v6_full27.sh"
    text = script.read_text(encoding="utf-8")
    assert script.exists()
    assert "experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml" in text
    assert "continuing" in text


def test_v6_matrix_and_evaluator_are_tolerant():
    matrix = (ROOT / "scripts/run_matrix.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts/evaluate_predictions.py").read_text(encoding="utf-8")
    assert "[matrix-job-error]" in matrix
    assert "matrix_failures.jsonl" in matrix
    assert "[evaluate-warn]" in evaluator
    assert "evaluation_failures.jsonl" in evaluator


def test_v6_composite_score_is_exposed():
    aggregate = (ROOT / "corrosion_ie/eval/aggregate.py").read_text(encoding="utf-8")
    report = (ROOT / "scripts/make_report.py").read_text(encoding="utf-8")
    assert '"composite_score"' in aggregate
    assert '"composite_score_mean"' in report
