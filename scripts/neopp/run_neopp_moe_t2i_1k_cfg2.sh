#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/neopp_moe

# Set the KV files and offsets below to match a captured LightLLM request.
export CUDA_VISIBLE_DEVICES=0,1
source "${lightx2v_path}/scripts/base/base.sh"
mkdir -p "${lightx2v_path}/save_results"

torchrun --standalone --nproc_per_node=2 "${lightx2v_path}/examples/neopp/replay_kv.py" \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/neopp/neopp_moe_parallel_cfg.json" \
    --task t2i \
    --cond_kv "/path/to/neopp_moe_kv/cond.pt" \
    --uncond_kv "/path/to/neopp_moe_kv/uncond.pt" \
    --index_offset_cond 325 \
    --index_offset_uncond 9 \
    --seed 200 \
    --size 1024 1024 \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_neopp_moe_t2i_1k_cfg2.png"
