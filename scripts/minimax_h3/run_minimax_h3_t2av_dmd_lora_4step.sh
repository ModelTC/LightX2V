#!/bin/bash
set -e

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme6/gushiqiao/codes/latest/LightX2V}
model_path=${MINIMAX_H3_MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}
python_bin=${PYTHON_BIN:-python}

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

PYTHONPATH="${lightx2v_path}" "${python_bin}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_dmd_lora_4step.json" \
  --prompt "integrated_multimodal_description: Shot 1 Live-action sports documentary style, a handheld ringside close shot frames two female boxers wearing headguards and gloves during controlled sparring in a small neighborhood gym. The camera tracks left at fast speed as the boxer in blue slips a jab, answers with a rapid body-head combination against the pads, and pivots away while sweat droplets catch the overhead light in brief slow motion. Her coach with a firm, energetic voice S1 calls: English Move, breathe, reset! overall_soundscape: Gloves thud against padded targets, shoes squeak on canvas, the round timer beeps once, and both athletes breathe sharply. A jump rope and muffled training activity continue in the background. non_diegetic_music: A tight electronic beat at a fast tempo with clipped bass pulses, fading under the final breath." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_dmd_lora_4step_5s.mp4" \
  --seed 42
