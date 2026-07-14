#!/bin/bash
#
# Start all three I2V disagg services (Encoder, Transformer, Decoder).
#
# Default layout (Encoder + Decoder co-located on GPU_ED, Transformer on GPU_T):
#   GPU_ED: Encoder (port 8004)  + Decoder (port 8007)
#   GPU_T : Transformer (port 8005)
#
# Override GPUs via environment variables:
#   GPU_ENCODER=4 GPU_TRANSFORMER=5 GPU_DECODER=6 ./start_wan_i2v_disagg_all.sh

set -e

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
model_path=${WAN_I2V_MODEL_PATH:-/data/nvme0/models/Wan-AI/Wan2.1-I2V-14B-480P}

GPU_ENCODER=${GPU_ENCODER:-4}
GPU_TRANSFORMER=${GPU_TRANSFORMER:-5}
GPU_DECODER=${GPU_DECODER:-4}   # Default: share GPU with Encoder

source "${lightx2v_path}/scripts/base/base.sh"

cleanup() {
    echo "Stopping all I2V disagg services..."
    kill "${PID_DECODER}" "${PID_TRANSFORMER}" "${PID_ENCODER}" 2>/dev/null || true
    wait
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# 1. Start Decoder first
echo "[1/3] Starting Decoder service on GPU=${GPU_DECODER}, port=8007 ..."
CUDA_VISIBLE_DEVICES=${GPU_DECODER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_i2v_disagg_decode.json" \
    --host 0.0.0.0 \
    --port 8007 &
PID_DECODER=$!

# 2. Start Transformer
echo "[2/3] Starting Transformer service on GPU=${GPU_TRANSFORMER}, port=8005 ..."
CUDA_VISIBLE_DEVICES=${GPU_TRANSFORMER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_i2v_disagg_transformer.json" \
    --host 0.0.0.0 \
    --port 8005 &
PID_TRANSFORMER=$!

# 3. Start Encoder last
echo "[3/3] Starting Encoder service on GPU=${GPU_ENCODER}, port=8004 ..."
CUDA_VISIBLE_DEVICES=${GPU_ENCODER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_i2v_disagg_encoder.json" \
    --host 0.0.0.0 \
    --port 8004 &
PID_ENCODER=$!

echo ""
echo "All services started:"
echo "  Encoder     PID=${PID_ENCODER}     GPU=${GPU_ENCODER}     port=8004"
echo "  Transformer PID=${PID_TRANSFORMER} GPU=${GPU_TRANSFORMER} port=8005"
echo "  Decoder     PID=${PID_DECODER}     GPU=${GPU_DECODER}     port=8007"
echo ""
echo "Press Ctrl+C to stop all services."

wait
