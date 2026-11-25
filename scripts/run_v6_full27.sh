#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
RUN_ROOT="output/exp_v2/corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6"
STATUS_LOG="$RUN_ROOT/v6_pipeline_status.tsv"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
export PYTHONPATH=.

unset GEN_TEMPERATURE GEN_TOP_P GEN_MAX_NEW_TOKENS GEN_MAX_INPUT_TOKENS GEN_DO_SAMPLE

# Pre-inference checks are strict. A broken dataset, matrix, prompt tree, or
# local model path must be corrected before consuming GPU time.
python scripts/validate_dataset.py --config "$CONFIG" --write-fingerprint || exit $?
python scripts/preflight_v6.py --config "$CONFIG" --expected-jobs 27 || exit $?

mkdir -p "$RUN_ROOT"
printf "step\texit_code\n" > "$STATUS_LOG"

run_step() {
  local name="$1"
  shift
  echo
  echo "[v6-step] $name"
  "$@"
  local rc=$?
  printf "%s\t%s\n" "$name" "$rc" >> "$STATUS_LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "[v6-warning] step=$name exit_code=$rc; continuing"
  fi
  return 0
}

echo "[v6] config=$CONFIG"
echo "[v6] run_root=$RUN_ROOT"
echo "[v6] matrix=3 models × 3 strategies × 3 prompts = 27 jobs"
echo "[v6] document errors and quality-gate failures are retained as diagnostics and do not stop later jobs"

run_step matrix \
  python scripts/run_matrix.py --config "$CONFIG" --progress plain

# Post-inference audits are non-blocking. They expose partial or low-quality
# conditions without discarding the usable results from other conditions.
run_step validate_matrix \
  python scripts/validate_matrix.py --config "$CONFIG" --run-root "$RUN_ROOT" --expected-docs 37
run_step validate_outputs \
  python scripts/validate_outputs.py --run-root "$RUN_ROOT"
run_step evidence_consistency \
  python scripts/verify_evidence_consistency.py --run-root "$RUN_ROOT"
run_step truncation_summary \
  python scripts/summarize_truncation.py --run-root "$RUN_ROOT"
run_step evaluation \
  python scripts/evaluate_predictions.py --config "$CONFIG" --run-root "$RUN_ROOT" --only-missing
run_step aggregate \
  python scripts/aggregate_runs.py --config "$CONFIG" --run-root "$RUN_ROOT" --group-by model,strategy
run_step prompt_robustness \
  python scripts/summarize_prompt_robustness.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step statistics \
  python scripts/statistical_analysis.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step document_strata \
  python scripts/analyze_document_strata.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step prompt_plot \
  python scripts/plot_prompt_robustness.py --run-root "$RUN_ROOT"
run_step full_matrix_plot \
  python scripts/plot_full_matrix.py --run-root "$RUN_ROOT"
run_step report \
  python scripts/make_report.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step inspect \
  python scripts/inspect_v6_results.py --run-root "$RUN_ROOT"

echo
echo "[v6] pipeline status: $STATUS_LOG"
echo "[v6] main aggregate: $RUN_ROOT/aggregate_metrics.csv"
echo "[v6] final report: $RUN_ROOT/final_report.md"
exit 0
