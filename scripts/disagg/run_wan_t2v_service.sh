#!/bin/bash

# set path firstly
lightx2v_path=/root/zht/LightX2V
model_path=/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B

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

controller_cfg=${lightx2v_path}/configs/disagg/wan_t2v_disagg_controller.json
encoder_cfg=${lightx2v_path}/configs/disagg/wan_t2v_disagg_encoder.json
transformer_cfg=${lightx2v_path}/configs/disagg/wan_t2v_disagg_transformer.json
decoder_cfg=${lightx2v_path}/configs/disagg/wan_t2v_disagg_decoder.json

seed=42
prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path=${lightx2v_path}/save_results/test_disagg.mp4
save_result_path_1=${save_result_path%.mp4}1.mp4
save_result_path_2=${save_result_path%.mp4}2.mp4

# Remove old outputs so wait loop reflects current run status.
rm -f "${save_result_path_1}" "${save_result_path_2}"

cleanup() {
	local pids=("${encoder_pid:-}" "${transformer_pid:-}" "${decoder_pid:-}" "${controller_pid:-}")
	for pid in "${pids[@]}"; do
		if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
			kill "${pid}" 2>/dev/null || true
		fi
	done
}

trap cleanup EXIT INT TERM

wait_for_port() {
	local host="$1"
	local port="$2"
	local timeout_secs="${3:-30}"
	local waited=0

	while true; do
		if (echo > /dev/tcp/${host}/${port}) >/dev/null 2>&1; then
			echo "Port ready: ${host}:${port}"
			return 0
		fi

		if (( waited >= timeout_secs )); then
			echo "Timeout waiting for port ${host}:${port} after ${timeout_secs}s"
			return 1
		fi

		sleep 1
		waited=$((waited + 1))
	done
}

rdma_request_port=5566
rdma_phase1_port=5567
rdma_phase2_port=5568

python -m lightx2v.disagg.examples.run_service \
	--service controller \
	--model_cls wan2.1 \
	--task t2v \
	--model_path ${model_path} \
	--config_json ${controller_cfg} \
	--seed ${seed} \
	--prompt "${prompt}" \
	--negative_prompt "${negative_prompt}" \
	--save_result_path ${save_result_path} \
	> ${lightx2v_path}/save_results/disagg_controller.log 2>&1 &
controller_pid=$!

wait_for_port 127.0.0.1 ${rdma_request_port} 60
wait_for_port 127.0.0.1 ${rdma_phase1_port} 60
wait_for_port 127.0.0.1 ${rdma_phase2_port} 60

CUDA_VISIBLE_DEVICES=0 python -m lightx2v.disagg.examples.run_service \
	--service encoder \
	--model_cls wan2.1 \
	--task t2v \
	--model_path ${model_path} \
	--config_json ${encoder_cfg} \
	--seed ${seed} \
	--prompt "${prompt}" \
	--negative_prompt "${negative_prompt}" \
	--save_result_path ${save_result_path} \
	> ${lightx2v_path}/save_results/disagg_encoder.log 2>&1 &
encoder_pid=$!

CUDA_VISIBLE_DEVICES=1 python -m lightx2v.disagg.examples.run_service \
	--service transformer \
	--model_cls wan2.1 \
	--task t2v \
	--model_path ${model_path} \
	--config_json ${transformer_cfg} \
	--seed ${seed} \
	--prompt "${prompt}" \
	--negative_prompt "${negative_prompt}" \
	--save_result_path ${save_result_path} \
	> ${lightx2v_path}/save_results/disagg_transformer.log 2>&1 &
transformer_pid=$!

CUDA_VISIBLE_DEVICES=2 python -m lightx2v.disagg.examples.run_service \
	--service decoder \
	--model_cls wan2.1 \
	--task t2v \
	--model_path ${model_path} \
	--config_json ${decoder_cfg} \
	--seed ${seed} \
	--prompt "${prompt}" \
	--negative_prompt "${negative_prompt}" \
	--save_result_path ${save_result_path} \
	> ${lightx2v_path}/save_results/disagg_decoder.log 2>&1 &
decoder_pid=$!

# Give background services time to flush and finish queued requests.

echo "Waiting for output videos: ${save_result_path_1}, ${save_result_path_2}"
wait_seconds=0
max_wait_seconds=1200

while true; do
	if [[ -f "${save_result_path_1}" && -f "${save_result_path_2}" ]]; then
		echo "Both output videos are generated."
		break
	fi

	if (( wait_seconds >= max_wait_seconds )); then
		echo "Timeout waiting for output videos after ${max_wait_seconds}s"
		exit 1
	fi

	sleep 5
	wait_seconds=$((wait_seconds + 5))
done
