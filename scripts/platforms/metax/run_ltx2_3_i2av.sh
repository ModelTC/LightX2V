
#!/bin/bash

# set path and first
# The model_pathpoints to LTX-2, while the config includes the weights for LTX-2.3
lightx2v_path=
model_path=
AUDIO_PATH=
IMAGE_PATH=
IMAGE_STRENGTH=1.0

# export CUDA_VISIBLE_DEVICES=0
export PLATFORM="metax_cuda"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MACA_PATH=${MACA_PATH:-/opt/maca-3.3.0}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls ltx2 \
  --task i2av \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/platforms/metax/ltx_2_3.json \
  --image_path "${IMAGE_PATH}" \
  --image_strength "${IMAGE_STRENGTH}" \
  --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
  --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_i2av.mp4" \
