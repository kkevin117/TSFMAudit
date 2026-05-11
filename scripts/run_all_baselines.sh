#!/usr/bin/env bash
set -e

REPO_ROOT=$(dirname "$(dirname "$0")")
DATA_ROOT=${DATA_ROOT:-/path/to/data}
GIFTEVAL_ROOT=${GIFTEVAL_ROOT:-$DATA_ROOT/GIFT-Eval}
GIFTEVAL_PRETRAIN_ROOT=${GIFTEVAL_PRETRAIN_ROOT:-$DATA_ROOT/GIFT-Eval-Pretrain}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoints}
RUN_MODEL_BASELINES=${RUN_MODEL_BASELINES:-0}

if [ "$RUN_MODEL_BASELINES" = "1" ]; then
  bash "$REPO_ROOT/scripts/run_chronos.sh"
  bash "$REPO_ROOT/scripts/run_kairos.sh"
  bash "$REPO_ROOT/scripts/run_moirai1.sh"
  bash "$REPO_ROOT/scripts/run_moirai2.sh"
  bash "$REPO_ROOT/scripts/run_timesfm.sh"
  bash "$REPO_ROOT/scripts/run_tirex.sh"
  bash "$REPO_ROOT/scripts/run_visionts.sh"
else
  echo "Skipping model baseline execution. Set RUN_MODEL_BASELINES=1 to run all model scripts."
fi

python "$REPO_ROOT/baseline_audit_eval.py" \
  --data_root "$DATA_ROOT" \
  --gifteval_root "$GIFTEVAL_ROOT" \
  --gifteval_pretrain_root "$GIFTEVAL_PRETRAIN_ROOT" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --results_root "$OUTPUT_DIR" \
  --out_csv "$OUTPUT_DIR/baseline_audit_eval_table.csv" \
  "$@"
