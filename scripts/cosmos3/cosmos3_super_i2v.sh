#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme5/gushiqiao/codes/LightX2V
model_path=/data/nvme5/gushiqiao/models/Cosmos3-Super-Image2Video
image_path=${model_path}/assets/example_first_frame.png

export CUDA_VISIBLE_DEVICES=7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

prompt='{"temporal_caption": "A fallen waffle cone lies on rough sunlit asphalt in an extreme low close-up, with a rounded scoop of vanilla and chocolate ice cream pressed against the road, a small melted puddle already spreading beneath it, and dry autumn leaves scattered around in warm late-afternoon light. The viewpoint stays near ground level and begins a slow, smooth arc around the cone from left to right, keeping the melting scoop dominant while the background street and leaves remain softly blurred. As the sun warms the ice cream, the glossy edges soften first, thin rivulets of vanilla and chocolate slide down the curved scoop, and the existing puddle widens into the cracks and pebbled texture of the asphalt. The waffle cone remains mostly rigid but grows slightly damp at the rim touching the ice cream, while the scoop loses its rounded shape, slumps lower, and exposes more of the cone’s open mouth. The moving viewpoint continues its gentle orbit, revealing the chocolate side thinning into streaks and the vanilla side collapsing into pale liquid that creeps outward under gravity. By the end, most of the ice cream has flattened into a shallow glossy stain that drains into small road fissures and spreads out of the immediate area, leaving the cone lying in place with only thin cream-colored and brown traces clinging to the asphalt in the warm light.", "duration": "7s", "fps": 24.0, "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}'
negative_prompt="The video captures a series of frames showing macroblocking artifacts, chromatic aberration, high-frequency noise, and rolling shutter distortion. It includes static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, bit-depth compression artifacts, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, hard cut, visual noise, and flickering. It features moiré patterns, edge halos, and temporal aliasing. Furthermore, the content defies common sense, generating illogical scenarios, nonsensical entities, absurd character behaviors, and conceptual paradoxes that violate basic human reasoning and everyday reality. The video looks like a surreal or glitchy hallucination. Overall, the video is of poor quality."

python -m lightx2v.infer \
--model_cls cosmos3 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_super_i2v.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--negative_prompt "${negative_prompt}" \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_i2v.mp4
