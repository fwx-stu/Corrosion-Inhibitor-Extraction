#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/experiment_q0q1q2_test50_balanced_opt_v6.yaml"
DEV_CONFIG="configs/experiment_q0q1q2_dev37_prompt_robustness_opt_v6.yaml"
MANIFEST="frozen/test50_protocol_manifest_v6.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
export PYTHONPATH=.
unset GEN_TEMPERATURE GEN_TOP_P GEN_MAX_NEW_TOKENS GEN_MAX_INPUT_TOKENS GEN_DO_SAMPLE

mkdir -p frozen environment_lock
python scripts/validate_dataset.py --config "$CONFIG" --write-fingerprint
python scripts/validate_split_isolation.py \
  --left-config "$DEV_CONFIG" \
  --right-config "$CONFIG" \
  --fail-on-near \
  --output frozen/dev37_test50_split_isolation_v6.json
python scripts/preflight_v6.py --config "$CONFIG" --expected-jobs 9
python scripts/capture_environment.py --output-dir environment_lock
python scripts/freeze_protocol.py --config "$CONFIG" --write "$MANIFEST"
python scripts/freeze_protocol.py --config "$CONFIG" --verify "$MANIFEST"

echo "[v6-test50-protocol-ready] $MANIFEST"
echo "Next: CONFIRM_FROZEN_TEST=YES PROTOCOL_MANIFEST=\"$PWD/$MANIFEST\" bash scripts/run_v6_test50.sh"
