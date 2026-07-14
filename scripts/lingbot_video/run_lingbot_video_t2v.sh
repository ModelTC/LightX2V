#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme5/gushiqiao/codes/debug/lingbot-x2v/LightX2V
model_path=/data/nvme5/gushiqiao/models/lingbot-video-moe-30b-a3b

export CUDA_VISIBLE_DEVICES=3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=grouped_mm

prompt=$(cat <<'EOF'
{"comprehensive_description":{"scene_content_description":"A young woman with long, wavy brown hair is standing in a bright, modern apartment living room. She is wearing a stylish, oversized cream-colored knit cardigan over a white tank top, paired with high-waisted, wide-leg beige trousers. She holds a small, structured tan leather handbag in her left hand. The background features a neutral-toned interior with a beige sofa, a potted plant, and large windows that let in soft, natural light, creating a warm and inviting atmosphere. The woman is smiling and looking directly at the camera, showcasing her outfit with a confident and friendly demeanor.","camera_movement_description":"The camera is positioned at eye level and remains essentially stationary throughout the video, maintaining a medium shot that captures the subject from the waist up. There is a very shallow depth of field, keeping the woman in sharp focus while the background remains softly blurred."},"camera_info":{"color":"Warm","frame_size":"Medium","shot_type_angle":"Eye level","lens_size":"Medium","composition":"Center","lighting":"Soft light","lighting_type":"Daylight"},"world_knowledge":[],"prominent_elements":[{"name":"young woman","description":"A woman with long, wavy brown hair and a friendly expression, modeling a fashion outfit.","actions":[{"timestamp":"[0.0s - 0.5s]","action":"stands still, smiling at the camera"},{"timestamp":"[0.5s - 2.0s]","action":"shifts her weight and turns her body slightly to the right"},{"timestamp":"[2.0s - 3.5s]","action":"adjusts the collar of her cardigan with her right hand"},{"timestamp":"[3.5s - 5.0s]","action":"returns to a neutral pose, smiling at the camera"}],"location":"center of the frame","relative_size":"dominant","shape_and_color":"slender build; wearing cream, white, and beige","texture":"soft knit cardigan, smooth fabric trousers","appearance_details":"long wavy brown hair, gold hoop earrings, tan leather handbag","relationship":"the main subject of the video, standing in front of a blurred apartment background","orientation":"upright, facing the camera","pose":"standing, shifting weight, and adjusting clothing","expression":"smiling and confident","clothing":"oversized cream knit cardigan, white tank top, high-waisted wide-leg beige trousers","gender":"female","skin_tone_and_texture":"fair skin with a smooth texture"},{"name":"tan handbag","description":"A small, structured leather handbag with a top handle.","actions":[{"timestamp":"[0.0s - 5.0s]","action":"held steady in the woman's left hand"}],"location":"held in the woman's left hand, lower center of the frame","relative_size":"small","shape_and_color":"rectangular, tan or light brown","texture":"smooth leather","appearance_details":"structured shape with a top handle","relationship":"held by the woman as an accessory","orientation":"upright","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":""}]}
EOF
)

negative_prompt=$(cat <<'EOF'
{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}
EOF
)

python -m lightx2v.infer \
--model_cls lingbot_video \
--task t2v \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/lingbot_video/lingbot_video_t2v.json \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_video_t2v.mp4 \
--seed 42
