#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="${1:-${WORKSPACE}/8b（opt-final-v6）}"
cd "$PROJECT_ROOT" || exit 2

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate small_rag
fi

export PYTHONPATH=.
unset GEN_TEMPERATURE GEN_TOP_P GEN_MAX_NEW_TOKENS GEN_MAX_INPUT_TOKENS GEN_DO_SAMPLE

CONFIG="$PROJECT_ROOT/configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
RUN_ROOT="$PROJECT_ROOT/output/exp_v2/corrosion_q0q1q2_dev37_prompt_robustness_opt_final_v6"
STATUS="$RUN_ROOT/v6_postprocess_repair_status.tsv"

mkdir -p "$RUN_ROOT"
printf "step\treturn_code\n" > "$STATUS"

record_step() {
  local name="$1"
  shift
  echo
  echo "[repair-dev37] $name"
  "$@"
  local rc=$?
  printf "%s\t%s\n" "$name" "$rc" >> "$STATUS"
  return "$rc"
}

echo "[repair-dev37] Restoring the canonical 27-job manifest only; no inference will run."
RETRIEVAL_DEVICE=cuda:0 \
python scripts/run_matrix.py \
  --config "$CONFIG" \
  --dry-run \
  --progress none || exit $?

python - "$RUN_ROOT" <<'PY' || exit $?
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "matrix_manifest.json").read_text(encoding="utf-8"))
jobs = manifest.get("jobs", [])
complete = 0
missing = []
for job in jobs:
    run_dir = Path(job["run_dir"])
    meta_path = run_dir / "run_meta.json"
    pred_path = run_dir / "predictions.jsonl"
    if not meta_path.exists() or not pred_path.exists():
        missing.append(str(run_dir))
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("complete") is True
        and int(meta.get("docs_ok", 0)) == 37
        and int(meta.get("docs_failed", 0)) == 0
    ):
        complete += 1

print(f"[repair-dev37-check] jobs={len(jobs)} complete={complete}")
if len(jobs) != 27:
    raise SystemExit(f"expected 27 manifest jobs, got {len(jobs)}")
if complete != 27:
    print("[repair-dev37-check] incomplete/missing run directories:")
    for item in missing[:20]:
        print("  ", item)
    raise SystemExit(f"expected 27 structurally complete runs, got {complete}")
PY

# These audits remain non-blocking.
record_step validate_matrix \
  python scripts/validate_matrix.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --expected-docs 37 || true

record_step validate_outputs \
  python scripts/validate_outputs.py \
    --run-root "$RUN_ROOT" || true

record_step verify_evidence_consistency \
  python scripts/verify_evidence_consistency.py \
    --run-root "$RUN_ROOT" || true

record_step summarize_truncation \
  python scripts/summarize_truncation.py \
    --run-root "$RUN_ROOT" || true

# Evaluation is required. Absolute RUN_ROOT avoids the previous Path.relative_to failure.
record_step evaluate_predictions \
  python scripts/evaluate_predictions.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --only-missing
EVAL_RC=$?
if [[ "$EVAL_RC" -ne 0 ]]; then
  echo "[repair-dev37-error] Evaluation failed. See terminal output and:"
  echo "  $RUN_ROOT/evaluation_failures.jsonl"
  exit "$EVAL_RC"
fi

record_step aggregate \
  python scripts/aggregate_runs.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --group-by model,strategy || exit $?

record_step prompt_robustness \
  python scripts/summarize_prompt_robustness.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" || exit $?

record_step statistical_analysis \
  python scripts/statistical_analysis.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" || exit $?

record_step document_strata \
  python scripts/analyze_document_strata.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" || true

record_step plot_prompt_robustness \
  python scripts/plot_prompt_robustness.py \
    --run-root "$RUN_ROOT" || true

record_step plot_full_matrix \
  python scripts/plot_full_matrix.py \
    --run-root "$RUN_ROOT" || true

record_step make_report \
  python scripts/make_report.py \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" || exit $?

record_step inspect \
  python scripts/inspect_v6_results.py \
    --run-root "$RUN_ROOT" || exit $?

echo
echo "[repair-dev37-done]"
echo "run_root: $RUN_ROOT"
echo "status:   $STATUS"
