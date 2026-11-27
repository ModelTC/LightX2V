#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PLATFORM=cambricon_mlu MLU_VISIBLE_DEVICES=0 DTYPE=BF16
if [[ "${1:-}" == "--help" ]]; then
  echo "Run one complete frozen SDXL dataset through PR #1754 on one MLU590 card."
  echo "Usage: MLU_PHYSICAL_CARD=0 PYTHON_BIN=/path/to/torch_mlu/python bash $0 --paths configs/mlu/paths.local.json --dataset geneval|parti|coco --output /absolute/new-run-dir [--count 64]"
  exit 0
fi
export MLU_PHYSICAL_CARD="${MLU_PHYSICAL_CARD:?Set the assigned host MLU card number}"
exec "${PYTHON_BIN}" "$EVAL_DIR/evaluate.py" campaign-mlu --backend lightx2v "$@"
