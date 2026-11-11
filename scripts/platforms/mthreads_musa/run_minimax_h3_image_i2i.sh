#!/bin/bash
set -eo pipefail

# Default configs require a pre-generated AdaLN cache.
# Run tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh first, using the same model,
# platform, config JSON and adaln_cache_dir as inference.

lightx2v_path=/workspace/LightX2V
model_path=/workspace/MiniMax-H3-Image
image_path=/workspace/LightX2V/assets/inputs/imgs/woman.jpeg

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$MUSA_VISIBLE_DEVICES}"

source "$lightx2v_path/scripts/base/base.sh"
prompt='The dark, warmly lit podcast studio photograph of two hosts behind a wooden desk in <Picture 1> is <Subject 1>. <Subject 1> is mostly preserved: the large "EP. 47" title in the top right corner, the dark textured wall, the wooden shelves with plants, a brass globe, a lit candle, books and the "DISCIPLINE > MOTIVATION" chalkboard, the framed "COMPOUND CONSISTENCY CONFIDENCE" poster, the glowing floor lamp on the left and the table lamp on the right, the two black microphones on their stands, the black "INVEST IN YOURSELF" and "PLAN BUILD WEALTH" mugs, the carved wooden "MONEY TALKS" desk sign, the dark wooden desk, both hosts'"'"' faces, smiles and clothing, the man'"'"'s wristwatch and the woman'"'"'s hoop earrings all stay exactly the same. Only the headphones and the hosts'"'"' poses change. The over-ear headphones and their cable are removed from the man on the left, and his short, tousled dark brown curly hair is filled in naturally over the top and sides of his head where the headband had flattened it, leaving no dent or mark. The two hosts lean toward each other and bump fists directly above the "MONEY TALKS" sign: one fist from each host meets with the knuckles touching, both elbows lifted off the desk, so the fist bump sits exactly at the horizontal centre of the frame, midway between the two microphones. The target image is a square, cinematic photograph of the podcast set: the man in the black crew-neck sweater on the left and the woman with long wavy brown hair in the cream knit sweater on the right lean in and smile at each other, their forearms raised so the touching fists hover just above the top edge of the "MONEY TALKS" sign between the two microphones, while the warm lamp light, the dark shelves behind them and the "EP. 47" title in the top right keep the original moody lighting and colors.'

# Multiple references use the existing comma-separated image_path input, in Picture order.
python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task i2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/platforms/mthreads_musa/minimax_h3_image.json" \
  --image_path "$image_path" \
  --prompt "$prompt" \
  --size 4096 4096 \
  --seed 1234 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_i2i_musa.png" \
  "$@"
