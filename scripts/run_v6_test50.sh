#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/experiment_q0q1q2_test50_balanced_opt_v6.yaml"
DEV_CONFIG="configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
RUN_ROOT="output/exp_v2/corrosion_q0q1q2_test50_balanced_opt_final_v6"
STATUS_LOG="$RUN_ROOT/v6_pipeline_status.tsv"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
export PYTHONPATH=.
unset GEN_TEMPERATURE GEN_TOP_P GEN_MAX_NEW_TOKENS GEN_MAX_INPUT_TOKENS GEN_DO_SAMPLE

if [[ "${CONFIRM_FROZEN_TEST:-}" != "YES" ]]; then
  echo "[v6-test50-error] set CONFIRM_FROZEN_TEST=YES" >&2
  exit 2
fi
if [[ -z "${PROTOCOL_MANIFEST:-}" || ! -f "${PROTOCOL_MANIFEST}" ]]; then
  echo "[v6-test50-error] PROTOCOL_MANIFEST must point to frozen/test50_protocol_manifest_v6.json" >&2
  exit 2
fi

# Formal-test preflight is strict and protocol verification occurs again inside
# run_matrix.py before any inference job is created.
python scripts/validate_dataset.py --config "$CONFIG" || exit $?
python scripts/validate_split_isolation.py \
  --left-config "$DEV_CONFIG" \
  --right-config "$CONFIG" \
  --fail-on-near \
  --output frozen/dev37_test50_split_isolation_v6.json || exit $?
python scripts/preflight_v6.py --config "$CONFIG" --expected-jobs 9 || exit $?
python scripts/freeze_protocol.py --config "$CONFIG" --verify "$PROTOCOL_MANIFEST" || exit $?

mkdir -p "$RUN_ROOT"
printf "step\texit_code\n" > "$STATUS_LOG"

run_step() {
  local name="$1"
  shift
  echo
  echo "[v6-test50-step] $name"
  "$@"
  local rc=$?
  printf "%s\t%s\n" "$name" "$rc" >> "$STATUS_LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "[v6-test50-warning] step=$name exit_code=$rc; continuing"
  fi
  return 0
}

echo "[v6-test50] config=$CONFIG"
echo "[v6-test50] run_root=$RUN_ROOT"
echo "[v6-test50] matrix=3 models × 3 strategies × balanced = 9 jobs"

run_step matrix \
  python scripts/run_matrix.py --config "$CONFIG" --progress plain
run_step validate_matrix \
  python scripts/validate_matrix.py --config "$CONFIG" --run-root "$RUN_ROOT" --expected-docs 50
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
run_step statistics \
  python scripts/statistical_analysis.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step document_strata \
  python scripts/analyze_document_strata.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step balanced_plot \
  python scripts/plot_balanced_matrix.py --run-root "$RUN_ROOT"
run_step full_matrix_plot \
  python scripts/plot_full_matrix.py --run-root "$RUN_ROOT"
run_step report \
  python scripts/make_report.py --config "$CONFIG" --run-root "$RUN_ROOT"
run_step inspect \
  python scripts/inspect_v6_results.py --run-root "$RUN_ROOT"

echo
echo "[v6-test50] pipeline status: $STATUS_LOG"
echo "[v6-test50] main aggregate: $RUN_ROOT/aggregate_metrics.csv"
echo "[v6-test50] final report: $RUN_ROOT/final_report.md"
exit 0
