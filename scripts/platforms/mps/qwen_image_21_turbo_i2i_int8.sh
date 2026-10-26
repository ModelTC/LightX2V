#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-2.1-Turbo-INT8
export LIGHTX2V_METALLIB_DIR=${model_path}/apple_silicon_metal

export PLATFORM=mps

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mps/qwen_image_21_turbo_int8.json" \
    --image_path "${model_path}/assets/turbo-boat-input.png" \
    --prompt "Create a photorealistic photograph of the motor yacht from the input sketch, sailing on open sea, captured in full side profile with the camera at water level. Preserve the hull, mast, wheelhouse, windows, portholes, lifeboats, and railings from the sketch. Paint the hull black, the superstructures and mast white, and the funnel yellow. Place the entire yacht on a calm deep-blue sea under a clear sky with soft clouds, lit by sunlight from the front left." \
    --size 1024 1024 \
    --seed 20261009 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_turbo_i2i_int8.png"
