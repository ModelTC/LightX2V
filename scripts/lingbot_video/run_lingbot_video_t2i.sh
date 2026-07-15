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
{"comprehensive_description":"A clear glass bottle filled with water sits on a light-colored wooden table, acting as a lens to focus sunlight. The bottle is positioned slightly to the left of the center, with its base resting on the table. Bright, direct sunlight from the upper right creates a brilliant, starburst-like lens flare on the table's surface, casting a warm, golden glow. The background is a soft, out-of-focus blur of green foliage and bright light, suggesting an outdoor garden or patio setting. The overall atmosphere is warm, serene, and summery, emphasizing the interplay between light and water.","camera_info":{"color":"Warm","frame_size":"Extreme Close Up","shot_type_angle":"Low angle","lens_size":"Long Lens","composition":"Left heavy","lighting":"Hard light","lighting_type":"Daylight"},"world_knowledge":[],"prominent_elements":[{"name":"glass bottle","description":"A clear, cylindrical glass bottle with a narrow neck and a flat base, partially filled with water.","location":"center-left","relative_size":"medium","shape_and_color":"Cylindrical with a tapered neck; transparent and clear","texture":"smooth, glossy","appearance_details":"The bottle contains clear water that reflects the surrounding light. The glass surface shows subtle highlights and reflections from the sun.","relationship":"Sits on the wooden table and acts as the focal point for the sunlight, creating a lens flare.","orientation":"upright","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"wooden table","description":"A flat surface made of light-colored wood with visible grain and horizontal planks.","location":"spanning the bottom half of the frame","relative_size":"large","shape_and_color":"Rectangular planks; light tan and beige","texture":"matte, grainy","appearance_details":"The surface is illuminated by a bright, star-shaped lens flare caused by the bottle focusing the sunlight.","relationship":"Provides the base upon which the bottle rests.","orientation":"horizontal","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"lens flare","description":"A bright, star-shaped burst of light created by sunlight passing through the water in the bottle.","location":"center, on the table surface","relative_size":"medium","shape_and_color":"Star-shaped; bright white and golden yellow","texture":"ethereal, glowing","appearance_details":"The flare has multiple sharp rays extending outwards from the point where the bottle meets the table.","relationship":"Originates from the interaction of sunlight and the water inside the glass bottle.","orientation":"radiating","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"blurred foliage","description":"A soft-focus background consisting of green leaves and bright light spots.","location":"top half of the frame","relative_size":"large","shape_and_color":"Amorphous shapes; various shades of green and bright white","texture":"soft, blurry","appearance_details":"The background is heavily out of focus, creating a bokeh effect that emphasizes the sharpness of the bottle in the foreground.","relationship":"Provides a natural, outdoor context for the scene.","orientation":"upright","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":"","is_cluster":true,"number_of_objects":"numerous"}]}
EOF
)

negative_prompt=$(cat <<'EOF'
{"universal_negative":{"visual_quality":["low quality","worst quality","blurry","pixelated","jpeg artifacts","low resolution","underexposed","overexposed","invisible subject","subject hidden in darkness"],"artistic_style":["painting","illustration","drawing","cartoon","3d render","cgi","sketch","digital art"],"composition_and_content":["text","watermark","signature","logo","pillarboxed","side bars","portrait image in landscape frame"],"material_and_structure":["plastic-like glass","unrealistic texture","deformed bottle","distorted reflections"]}}
EOF
)

python -m lightx2v.infer \
--model_cls lingbot_video \
--task t2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/lingbot_video/lingbot_video_t2i.json \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_video_t2i.png \
--seed 42
