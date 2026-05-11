#!/usr/bin/env bash
set -e

DATA_ROOT=${DATA_ROOT:-/path/to/data}
GIFTEVAL_ROOT=${GIFTEVAL_ROOT:-$DATA_ROOT/GIFT-Eval}
GIFTEVAL_PRETRAIN_ROOT=${GIFTEVAL_PRETRAIN_ROOT:-$DATA_ROOT/GIFT-Eval-Pretrain}

printf 'DATA_ROOT=%s\n' "$DATA_ROOT"
printf 'GIFTEVAL_ROOT=%s\n' "$GIFTEVAL_ROOT"
printf 'GIFTEVAL_PRETRAIN_ROOT=%s\n' "$GIFTEVAL_PRETRAIN_ROOT"

if [ ! -d "$GIFTEVAL_ROOT" ]; then
  echo "Missing GIFT-Eval directory: $GIFTEVAL_ROOT"
fi

if [ ! -d "$GIFTEVAL_PRETRAIN_ROOT" ]; then
  echo "Missing GIFT-Eval-Pretrain directory: $GIFTEVAL_PRETRAIN_ROOT"
fi

echo "Download datasets separately and arrange them as described in README.md."
