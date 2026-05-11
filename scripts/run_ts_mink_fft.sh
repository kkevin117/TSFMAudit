#!/usr/bin/env bash
set -e

REPO_ROOT=$(dirname "$(dirname "$0")")
DATA_ROOT=${DATA_ROOT:-/path/to/data}
GIFTEVAL_ROOT=${GIFTEVAL_ROOT:-$DATA_ROOT/GIFT-Eval}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}

python "$REPO_ROOT/ts_mink_fft.py" \
  --data_root "$DATA_ROOT" \
  --gifteval_root "$GIFTEVAL_ROOT" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
