#!/bin/bash
#
# Start Qwen Image I2I 3-way disagg services (Encoder, Transformer, Decoder).
#
# Request order: Decoder first -> Transformer -> Encoder. Result is saved on Decoder node.
#
# Default layout (Encoder + Decoder on GPU_ED, Transformer on GPU_T):
#   GPU_ED: Encoder (port 8012) + Decoder (port 8014)
#   GPU_T : Transformer (port 8013)
#
# Override GPUs via environment variables:
#   GPU_ENCODER=4 GPU_TRANSFORMER=5 GPU_DECODER=6 ./start_qwen_i2i_disagg_3way.sh

set -e

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
model_path=${QWEN_IMAGE_MODEL_PATH:-/data/nvme1/models/qwen-image-edit-release-251130}

GPU_ENCODER=${GPU_ENCODER:-4}
GPU_TRANSFORMER=${GPU_TRANSFORMER:-5}
GPU_DECODER=${GPU_DECODER:-4}

source "${lightx2v_path}/scripts/base/base.sh"

cleanup() {
    echo "Stopping all Qwen I2I 3-way disagg services..."
    kill "${PID_DECODER}" "${PID_TRANSFORMER}" "${PID_ENCODER}" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# 1. Start Decoder first
echo "[1/3] Starting Decoder service on GPU=${GPU_DECODER}, port=8014 ..."
CUDA_VISIBLE_DEVICES=${GPU_DECODER} python -m lightx2v.server \
    --model_cls qwen_image \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/qwen/qwen_image_i2i_disagg_decode.json" \
    --host 0.0.0.0 \
    --port 8014 &
PID_DECODER=$!

# 2. Start Transformer
echo "[2/3] Starting Transformer service on GPU=${GPU_TRANSFORMER}, port=8013 ..."
CUDA_VISIBLE_DEVICES=${GPU_TRANSFORMER} python -m lightx2v.server \
    --model_cls qwen_image \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/qwen/qwen_image_i2i_disagg_transformer.json" \
    --host 0.0.0.0 \
    --port 8013 &
PID_TRANSFORMER=$!

# 3. Start Encoder last
echo "[3/3] Starting Encoder service on GPU=${GPU_ENCODER}, port=8012 ..."
CUDA_VISIBLE_DEVICES=${GPU_ENCODER} python -m lightx2v.server \
    --model_cls qwen_image \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/qwen/qwen_image_i2i_disagg_encoder.json" \
    --host 0.0.0.0 \
    --port 8012 &
PID_ENCODER=$!

echo ""
echo "All Qwen I2I 3-way services started:"
echo "  Encoder     PID=${PID_ENCODER}     GPU=${GPU_ENCODER}     port=8012"
echo "  Transformer PID=${PID_TRANSFORMER} GPU=${GPU_TRANSFORMER} port=8013"
echo "  Decoder     PID=${PID_DECODER}     GPU=${GPU_DECODER}     port=8014"
echo ""
echo "Press Ctrl+C to stop all services."

wait
