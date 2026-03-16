#!/bin/bash
#
# Start all three T2V disagg services (Encoder, Transformer, Decoder).
#
# Default layout (Encoder + Decoder co-located on GPU_ED, Transformer on GPU_T):
#   GPU_ED: Encoder (port 8002)  + Decoder (port 8006)
#   GPU_T : Transformer (port 8003)
#
# Override GPUs via environment variables:
#   GPU_ENCODER=4 GPU_TRANSFORMER=5 GPU_DECODER=6 ./start_wan_t2v_disagg_all.sh

set -e

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
model_path=${WAN_T2V_MODEL_PATH:-/data/nvme0/models/Wan-AI/Wan2.1-T2V-14B}

GPU_ENCODER=${GPU_ENCODER:-1}
GPU_TRANSFORMER=${GPU_TRANSFORMER:-6}
GPU_DECODER=${GPU_DECODER:-1}   # Default: share GPU with Encoder

source "${lightx2v_path}/scripts/base/base.sh"

cleanup() {
    echo "Stopping all T2V disagg services..."
    kill "${PID_DECODER}" "${PID_TRANSFORMER}" "${PID_ENCODER}" 2>/dev/null || true
    wait
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# 1. Start Decoder first (sets up Phase2 receiver, waits for latents)
echo "[1/3] Starting Decoder service on GPU=${GPU_DECODER}, port=8006 ..."
CUDA_VISIBLE_DEVICES=${GPU_DECODER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_t2v_disagg_decode.json" \
    --host 0.0.0.0 \
    --port 8006 &
PID_DECODER=$!

# 2. Start Transformer (Phase1 receiver + Phase2 sender)
echo "[2/3] Starting Transformer service on GPU=${GPU_TRANSFORMER}, port=8003 ..."
CUDA_VISIBLE_DEVICES=${GPU_TRANSFORMER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_t2v_disagg_transformer.json" \
    --host 0.0.0.0 \
    --port 8003 &
PID_TRANSFORMER=$!

# 3. Start Encoder last (Phase1 sender, must start after Transformer is ready)
echo "[3/3] Starting Encoder service on GPU=${GPU_ENCODER}, port=8002 ..."
CUDA_VISIBLE_DEVICES=${GPU_ENCODER} python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/disagg/wan/wan_t2v_disagg_encoder.json" \
    --host 0.0.0.0 \
    --port 8002 &
PID_ENCODER=$!

echo ""
echo "All services started:"
echo "  Encoder     PID=${PID_ENCODER}     GPU=${GPU_ENCODER}     port=8002"
echo "  Transformer PID=${PID_TRANSFORMER} GPU=${GPU_TRANSFORMER} port=8003"
echo "  Decoder     PID=${PID_DECODER}     GPU=${GPU_DECODER}     port=8006"
echo ""
echo "Press Ctrl+C to stop all services."

wait
