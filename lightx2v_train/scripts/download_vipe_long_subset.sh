#!/bin/bash
# Download a small subset of ViPE long dataset for DFD finetuning.
# Uses HF mirror by default for better connectivity in CN.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAR_DIR="${TAR_DIR:-/data/nvme7/wangshankun/train_data/vipe_long/long}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/nvme7/wangshankun/train_data/vipe_long/extracted}"
NUM_SHARDS="${NUM_SHARDS:-5}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${TAR_DIR}"

export HF_ENDPOINT
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  shard="$(printf '%06d' "${i}")"
  if [[ -f "${TAR_DIR}/${shard}.tar" ]]; then
    echo "Skip existing ${TAR_DIR}/${shard}.tar"
    continue
  fi
  hf download --repo-type dataset csusupergear/vipe_fliter_complete "long/${shard}.tar" --local-dir "$(dirname "${TAR_DIR}")"
done

python3 "${SCRIPT_DIR}/prepare_vipe_long_subset.py" \
  --tar-dir "${TAR_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-shards "${NUM_SHARDS}"

echo "Done. Dataset ready at ${OUTPUT_DIR}"
