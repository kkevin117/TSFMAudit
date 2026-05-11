#!/usr/bin/env bash
set -e

REPO_ROOT=$(dirname "$(dirname "$0")")
DATA_ROOT=${DATA_ROOT:-/path/to/data}
GIFTEVAL_ROOT=${GIFTEVAL_ROOT:-$DATA_ROOT/GIFT-Eval}
GIFTEVAL_PRETRAIN_ROOT=${GIFTEVAL_PRETRAIN_ROOT:-$DATA_ROOT/GIFT-Eval-Pretrain}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
CONFIG=${CONFIG:-configs/main_audit.yaml}

python "$REPO_ROOT/sft_audit_eval.py" \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  --gifteval_root "$GIFTEVAL_ROOT" \
  --gifteval_pretrain_root "$GIFTEVAL_PRETRAIN_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --results_root "$OUTPUT_DIR" \
  --out_csv "$OUTPUT_DIR/sft_audit_eval_table.csv" \
  "$@"
