#!/usr/bin/env bash
set -e

OUTPUT_DIR=${OUTPUT_DIR:-outputs}

if [ -f "$OUTPUT_DIR/sft_audit_eval_table.csv" ]; then
  echo "Main audit table: $OUTPUT_DIR/sft_audit_eval_table.csv"
fi

if [ -f "$OUTPUT_DIR/baseline_audit_eval_table.csv" ]; then
  echo "Baseline audit table: $OUTPUT_DIR/baseline_audit_eval_table.csv"
fi

find "$OUTPUT_DIR" -maxdepth 3 -type f \( -name 'summary.csv' -o -name 'epoch_metrics.csv' -o -name '*audit_eval_table*.csv' \) 2>/dev/null | sort || true
