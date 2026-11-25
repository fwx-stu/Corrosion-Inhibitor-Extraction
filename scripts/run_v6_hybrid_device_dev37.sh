#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
RUN_ROOT="output/exp_v2/corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6"
STATUS_FILE="$RUN_ROOT/v6_hybrid_device_status.tsv"

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset GEN_TEMPERATURE
unset GEN_TOP_P
unset GEN_MAX_NEW_TOKENS
unset GEN_MAX_INPUT_TOKENS
unset GEN_DO_SAMPLE

mkdir -p "$RUN_ROOT"
printf "step\treturn_code\tdevice\n" > "$STATUS_FILE"

run_matrix_group() {
    local name="$1"
    local device="$2"
    local models="$3"

    echo
    echo "======================================================================"
    echo "[v6-hybrid] $name"
    echo "[v6-hybrid] models=$models"
    echo "[v6-hybrid] Qwen device=cuda:${CUDA_VISIBLE_DEVICES}"
    echo "[v6-hybrid] embedding/reranker device=$device"
    echo "======================================================================"

    RETRIEVAL_DEVICE="$device" \
    python scripts/run_matrix.py \
      --config "$CONFIG" \
      --models "$models" \
      --strategies q0_safe_full,q1_budgeted_rag,q2_table_aware_rag \
      --prompts recall,balanced,precision \
      --seeds 1 \
      --progress plain

    local rc=$?
    printf "%s\t%s\t%s\n" "$name" "$rc" "$device" >> "$STATUS_FILE"

    if [[ "$rc" -ne 0 ]]; then
        echo "[v6-hybrid-warning] $name returned $rc; continuing."
    fi
}

run_step() {
    local name="$1"
    shift

    echo
    echo "[v6-post] $name"
    "$@"
    local rc=$?
    printf "%s\t%s\tpostprocess\n" "$name" "$rc" >> "$STATUS_FILE"

    if [[ "$rc" -ne 0 ]]; then
        echo "[v6-post-warning] $name returned $rc; continuing."
    fi
}

# 3B and 7B: Qwen + embedding + reranker all use GPU.
run_matrix_group \
  "matrix_3b_7b" \
  "cuda:0" \
  "qwen2_5_3b,qwen2_5_7b"

# 14B: Qwen remains on GPU; only embedding and reranker move to CPU.
run_matrix_group \
  "matrix_14b" \
  "cpu" \
  "qwen2_5_14b"

# Non-blocking audit, evaluation and aggregation.
run_step validate_matrix \
  python scripts/validate_matrix.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --expected-docs 37

run_step validate_outputs \
  python scripts/validate_outputs.py \
    --run-root "$RUN_ROOT"

run_step verify_evidence_consistency \
  python scripts/verify_evidence_consistency.py \
    --run-root "$RUN_ROOT"

run_step summarize_truncation \
  python scripts/summarize_truncation.py \
    --run-root "$RUN_ROOT"

run_step evaluate_predictions \
  python scripts/evaluate_predictions.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --only-missing

run_step aggregate_model_strategy \
  python scripts/aggregate_runs.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --group-by model,strategy

run_step aggregate_by_prompt \
  python scripts/aggregate_runs.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --group-by model,strategy,prompt

run_step prompt_robustness \
  python scripts/summarize_prompt_robustness.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT"

run_step statistical_analysis \
  python scripts/statistical_analysis.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT"

run_step document_strata \
  python scripts/analyze_document_strata.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT"

run_step plot_prompt_robustness \
  python scripts/plot_prompt_robustness.py \
    --run-root "$RUN_ROOT"

if [[ -f scripts/plot_full_matrix.py ]]; then
  run_step plot_full_matrix \
    python scripts/plot_full_matrix.py \
      --run-root "$RUN_ROOT"
fi

run_step make_report \
  python scripts/make_report.py \
    --run-root "$RUN_ROOT"

echo
echo "======================================================================"
echo "[v6-hybrid-done]"
echo "run_root: $RUN_ROOT"
echo "status:   $STATUS_FILE"
echo
echo "Device policy:"
echo "  3B / 7B: Qwen GPU; embedding and reranker GPU"
echo "  14B:     Qwen GPU; embedding and reranker CPU"
echo
echo "Do not use --force when resuming."
echo "Re-run this same script after an interruption."
echo "======================================================================"

exit 0
