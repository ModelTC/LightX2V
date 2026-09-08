#!/usr/bin/env bash

set -euo pipefail

# Encode the original H3 teacher; the historical cache paths remain unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/nvme6/gushiqiao/datasets/minimax_h3_teacher_api_5s_768p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/nvme6/gushiqiao/datasets/minimax_h3_turbo_vae_latents_5s_768p}"
STORAGE_DTYPE="${STORAGE_DTYPE:-fp16}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DECODE_WORKERS="${DECODE_WORKERS:-2}"
DECODE_PREFETCH="${DECODE_PREFETCH:-4}"
CPU_LIST="${CPU_LIST:-0-47,96-143}"

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.vae-latent-cache.lock"
if ! flock -n 9; then
    echo "Another VAE latent cache job is already writing to ${OUTPUT_ROOT}." >&2
    exit 1
fi

cd "${TRAIN_ROOT}"
export PYTHONPATH="${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for task in fl2av l2av; do
    input="${SOURCE_ROOT}/${task}/landscape/metadata.jsonl"
    output="${OUTPUT_ROOT}/${task}/landscape"

    echo "[vae-cache] task=${task} input=${input} output=${output}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc-per-node="${NUM_PROCESSES}" \
        tools/cache_minimax_h3_vae_latents.py \
        --input "${input}" \
        --output-dir "${output}" \
        --model-path "${MODEL_PATH}" \
        --num-frames 124 \
        --fps 24 \
        --posterior mode \
        --storage-dtype "${STORAGE_DTYPE}" \
        --batch-size "${BATCH_SIZE}" \
        --decode-workers "${DECODE_WORKERS}" \
        --decode-prefetch "${DECODE_PREFETCH}"
done

echo "[vae-cache] completed"
wc -l \
    "${OUTPUT_ROOT}/fl2av/landscape/metadata.jsonl" \
    "${OUTPUT_ROOT}/l2av/landscape/metadata.jsonl"
