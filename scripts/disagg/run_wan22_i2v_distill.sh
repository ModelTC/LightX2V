#!/bin/bash

# set path firstly
lightx2v_path=/root/zht/LightX2V
model_path=${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Keep flashinfer enabled while ensuring nvcc uses a supported host compiler.
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -allow-unsupported-compiler"
else
    export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
fi

# Pin disagg RDMA and Mooncake to one NIC to avoid cross-NIC session mismatch.
export RDMA_IFACE=${RDMA_IFACE:-erdma_0}
export MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME:-eth0}
if [[ -z "${MOONCAKE_LOCAL_HOSTNAME:-}" ]]; then
    _mc_ip=$(ip -4 -o addr show dev "${MOONCAKE_DEVICE_NAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1)
    if [[ -n "${_mc_ip}" ]]; then
        export MOONCAKE_LOCAL_HOSTNAME="${_mc_ip}"
    fi
fi
echo "RDMA_IFACE=${RDMA_IFACE} MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME} MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-unset}"

# Enable simulated network jitter by default for this test script.
# Set NETWORK_LATENCY=0 before running to disable it.
# export NETWORK_LATENCY=${NETWORK_LATENCY:-1}
# echo "NETWORK_LATENCY=${NETWORK_LATENCY}"

controller_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_controller.json
encoder_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_encoder.json
transformer_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_transformer.json
decoder_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_decoder.json

seed=42
request_count=30
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds."
negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path=${lightx2v_path}/save_results/wan22_i2v_distill_disagg.mp4
output_files=()
for ((i=0; i<request_count; i++)); do
    output_files+=("${save_result_path%.mp4}${i}.mp4")
done

# Remove old outputs so wait loop reflects current run status.
rm -f "${output_files[@]}"

cleanup() {
    local pids=("${encoder_pid:-}" "${transformer_pid:-}" "${decoder_pid:-}" "${controller_pid:-}")
    for pid in "${pids[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT INT TERM

if [[ ! -f "${controller_cfg}" ]]; then
    echo "Controller config not found: ${controller_cfg}"
    exit 1
fi

# These are kept for manual split-service debug if needed.
if [[ ! -f "${encoder_cfg}" || ! -f "${transformer_cfg}" || ! -f "${decoder_cfg}" ]]; then
    echo "One or more disagg stage configs are missing under configs/disagg"
    exit 1
fi

python -m lightx2v.disagg.examples.run_service \
    --service controller \
    --model_cls wan2.2_moe \
    --task i2v \
    --model_path ${model_path} \
    --config_json ${controller_cfg} \
    --seed ${seed} \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path ${save_result_path} \
    > ${lightx2v_path}/save_results/disagg_wan22_i2v_distill_controller.log 2>&1 &
controller_pid=$!

echo "Waiting for output videos: ${output_files[*]}"
wait_seconds=0
max_wait_seconds=$((200 * request_count))

while true; do
    all_generated=1
    for file in "${output_files[@]}"; do
        if [[ ! -f "${file}" ]]; then
            all_generated=0
            break
        fi
    done

    if (( all_generated )); then
        echo "All ${request_count} output videos are generated."
        break
    fi

    if (( wait_seconds >= max_wait_seconds )); then
        echo "Timeout waiting for output videos after ${max_wait_seconds}s"
        exit 1
    fi

    sleep 5
    wait_seconds=$((wait_seconds + 5))
done

sleep 60
