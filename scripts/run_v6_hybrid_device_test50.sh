#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

BASE_CONFIG="$PROJECT_ROOT/configs/experiment_q0q1q2_test50_balanced_opt_v6.yaml"
GPU_CONFIG="$PROJECT_ROOT/configs/experiment_q0q1q2_test50_balanced_opt_v6_gpu_3b7b.yaml"
CPU_CONFIG="$PROJECT_ROOT/configs/experiment_q0q1q2_test50_balanced_opt_v6_cpu_14b.yaml"
DEV_CONFIG="$PROJECT_ROOT/configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"

RUN_ROOT="$PROJECT_ROOT/output/exp_v2/corrosion_q0q1q2_test50_balanced_opt_final_v6"
GPU_MANIFEST="$PROJECT_ROOT/frozen/test50_protocol_manifest_v6_gpu_3b7b.json"
CPU_MANIFEST="$PROJECT_ROOT/frozen/test50_protocol_manifest_v6_cpu_14b.json"
STATUS_LOG="$RUN_ROOT/v6_hybrid_device_status.tsv"

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset GEN_TEMPERATURE GEN_TOP_P GEN_MAX_NEW_TOKENS GEN_MAX_INPUT_TOKENS GEN_DO_SAMPLE

for path in "$BASE_CONFIG" "$DEV_CONFIG"; do
  [[ -f "$path" ]] || {
    echo "[v6-test50-hybrid-error] missing: $path" >&2
    exit 2
  }
done

mkdir -p "$PROJECT_ROOT/frozen" "$PROJECT_ROOT/environment_lock" "$RUN_ROOT"

python - "$BASE_CONFIG" "$GPU_CONFIG" "$CPU_CONFIG" <<'PY'
from pathlib import Path
import copy
import sys
import yaml

base_path, gpu_path, cpu_path = map(Path, sys.argv[1:4])
base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

gpu = copy.deepcopy(base)
gpu["protocol_version"] = "8b-opt-final-v6-test50-balanced-gpu-retrieval-3b7b"
gpu["models"] = ["qwen2_5_3b", "qwen2_5_7b"]
gpu["retrieval"]["device"] = "cuda:0"

cpu = copy.deepcopy(base)
cpu["protocol_version"] = "8b-opt-final-v6-test50-balanced-cpu-retrieval-14b"
cpu["models"] = ["qwen2_5_14b"]
cpu["retrieval"]["device"] = "cpu"

gpu_path.write_text(yaml.safe_dump(gpu, allow_unicode=True, sort_keys=False), encoding="utf-8")
cpu_path.write_text(yaml.safe_dump(cpu, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"[write] {gpu_path}")
print(f"[write] {cpu_path}")
PY

python scripts/validate_dataset.py --config "$BASE_CONFIG" --write-fingerprint || exit $?
python scripts/validate_split_isolation.py \
  --left-config "$DEV_CONFIG" \
  --right-config "$BASE_CONFIG" \
  --fail-on-near \
  --output "$PROJECT_ROOT/frozen/dev37_test50_split_isolation_v6.json" || exit $?

RETRIEVAL_DEVICE=cuda:0 \
python scripts/preflight_v6.py --config "$GPU_CONFIG" --expected-jobs 6 || exit $?

RETRIEVAL_DEVICE=cpu \
python scripts/preflight_v6.py --config "$CPU_CONFIG" --expected-jobs 3 || exit $?

if [[ ! -f "$GPU_MANIFEST" || ! -f "$CPU_MANIFEST" ]]; then
  echo "[v6-test50-hybrid] creating frozen protocol manifests"
  python scripts/capture_environment.py --output-dir "$PROJECT_ROOT/environment_lock" || exit $?

  RETRIEVAL_DEVICE=cuda:0 \
  python scripts/freeze_protocol.py --config "$GPU_CONFIG" --write "$GPU_MANIFEST" || exit $?

  RETRIEVAL_DEVICE=cpu \
  python scripts/freeze_protocol.py --config "$CPU_CONFIG" --write "$CPU_MANIFEST" || exit $?
else
  echo "[v6-test50-hybrid] existing manifests found; verification only"
fi

RETRIEVAL_DEVICE=cuda:0 \
python scripts/freeze_protocol.py --config "$GPU_CONFIG" --verify "$GPU_MANIFEST" || exit $?

RETRIEVAL_DEVICE=cpu \
python scripts/freeze_protocol.py --config "$CPU_CONFIG" --verify "$CPU_MANIFEST" || exit $?

printf "step\treturn_code\tdevice\n" > "$STATUS_LOG"

run_matrix_stage() {
  local name="$1"
  local device="$2"
  local config="$3"
  local manifest="$4"

  echo
  echo "======================================================================"
  echo "[v6-test50-hybrid] $name"
  echo "[v6-test50-hybrid] Qwen device=cuda:${CUDA_VISIBLE_DEVICES}"
  echo "[v6-test50-hybrid] embedding/reranker device=$device"
  echo "======================================================================"

  CONFIRM_FROZEN_TEST=YES \
  PROTOCOL_MANIFEST="$manifest" \
  RETRIEVAL_DEVICE="$device" \
  python scripts/run_matrix.py --config "$config" --progress plain

  local rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$device" >> "$STATUS_LOG"
  [[ "$rc" -eq 0 ]] || echo "[v6-test50-hybrid-warning] $name returned $rc; continuing."
}

run_step() {
  local name="$1"
  shift
  echo
  echo "[v6-test50-post] $name"
  "$@"
  local rc=$?
  printf "%s\t%s\tpostprocess\n" "$name" "$rc" >> "$STATUS_LOG"
  [[ "$rc" -eq 0 ]] || echo "[v6-test50-post-warning] $name returned $rc; continuing."
  return "$rc"
}

run_matrix_stage matrix_3b_7b cuda:0 "$GPU_CONFIG" "$GPU_MANIFEST"
run_matrix_stage matrix_14b cpu "$CPU_CONFIG" "$CPU_MANIFEST"

# Restore the canonical combined 9-job manifest. Dry-run does not invoke a model.
RETRIEVAL_DEVICE=cuda:0 \
python scripts/run_matrix.py --config "$BASE_CONFIG" --dry-run --progress none || exit $?

run_step validate_matrix \
  python scripts/validate_matrix.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" \
    --expected-docs 50 || true

run_step validate_outputs \
  python scripts/validate_outputs.py --run-root "$RUN_ROOT" || true

run_step verify_evidence_consistency \
  python scripts/verify_evidence_consistency.py --run-root "$RUN_ROOT" || true

run_step summarize_truncation \
  python scripts/summarize_truncation.py --run-root "$RUN_ROOT" || true

# Absolute RUN_ROOT is required because manifest run directories are absolute.
run_step evaluate_predictions \
  python scripts/evaluate_predictions.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" \
    --only-missing
EVAL_RC=$?
if [[ "$EVAL_RC" -ne 0 ]]; then
  echo "[v6-test50-error] evaluation failed; not producing aggregate statistics."
  exit "$EVAL_RC"
fi

run_step aggregate \
  python scripts/aggregate_runs.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" \
    --group-by model,strategy || exit $?

run_step statistical_analysis \
  python scripts/statistical_analysis.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" || exit $?

run_step document_strata \
  python scripts/analyze_document_strata.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" || true

run_step plot_balanced_matrix \
  python scripts/plot_balanced_matrix.py --run-root "$RUN_ROOT" || true

run_step plot_full_matrix \
  python scripts/plot_full_matrix.py --run-root "$RUN_ROOT" || true

run_step make_report \
  python scripts/make_report.py \
    --config "$BASE_CONFIG" \
    --run-root "$RUN_ROOT" || exit $?

run_step inspect \
  python scripts/inspect_v6_results.py --run-root "$RUN_ROOT" || exit $?

echo
echo "======================================================================"
echo "[v6-test50-hybrid-done]"
echo "run_root: $RUN_ROOT"
echo "status:   $STATUS_LOG"
echo "Resume by running this same script again. Do not use --force."
echo "======================================================================"
