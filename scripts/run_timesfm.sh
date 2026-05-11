#!/usr/bin/env bash
set -e

REPO_ROOT=$(dirname "$(dirname "$0")")
DATA_ROOT=${DATA_ROOT:-/path/to/data}
GIFTEVAL_ROOT=${GIFTEVAL_ROOT:-$DATA_ROOT/GIFT-Eval}
GIFTEVAL_PRETRAIN_ROOT=${GIFTEVAL_PRETRAIN_ROOT:-$DATA_ROOT/GIFT-Eval-Pretrain}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
CONFIG=${CONFIG:-configs/timesfm.yaml}

python "$REPO_ROOT/sft_timesfm.py" \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  --gifteval_root "$GIFTEVAL_ROOT" \
  --gifteval_pretrain_root "$GIFTEVAL_PRETRAIN_ROOT" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
