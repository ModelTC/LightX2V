#!/bin/bash

lightx2v_path=/Path/To/LightX2V
model_path=/Path/To/Model

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/qwen_image_21/qwen_image_21.json" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --prompt "Transform the cat in the reference image into a photorealistic cream-colored dog with a natural canine muzzle, nose, and ears. Preserve the original close-up selfie composition, head position, body pose, and extended front leg. Keep the amber sunglasses and their reflections, fitting them naturally to the dog's face, and retain the collar and wet fur appearance. Preserve the turquoise water, yellow kayak, distant beach, cliffs, vegetation, and sky. Match the original sunlight, warm backlighting, shadows, shallow depth of field, and perspective. Change only the animal's species, with realistic canine anatomy and detailed fur that blends seamlessly into the scene." \
    --resolution 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_i2i.png"
