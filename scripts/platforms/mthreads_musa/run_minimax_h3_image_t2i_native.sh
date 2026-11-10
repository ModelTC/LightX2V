#!/bin/bash
set -eo pipefail

lightx2v_path=/workspace/LightX2V
model_path=/workspace/MiniMax-H3-Image

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$MUSA_VISIBLE_DEVICES}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

source "$lightx2v_path/scripts/base/base.sh"
prompt='The image is a photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a heavily weathered, dark brown wooden table with deep visible grain and scratches. The composition is a top-down view looking directly at the table surface. In the center sits a large woven jute placemat supporting a freshly baked rustic sourdough loaf with a deeply scored, golden-brown crust and a dusting of flour. To the upper left, a hammered copper bowl overflows with a colorful mix of fresh seasonal vegetables, including purple kale, bright orange carrots, red beets, and vibrant green salad leaves. In the upper right, a dark wooden cheese board holds neatly arranged artisanal cheeses: a wedge of bloomy rind brie with a creamy yellow interior, a wedge of aged cheddar with a golden rind, and a log of goat cheese sprinkled with cracked peppercorns and rosemary sprigs. The lower left contains a cluster of heirloom tomatoes, small cherry tomatoes on the vine, scattered toasted walnuts, and a small ceramic bowl of green olives. In the lower right, a clear glass wine glass filled with dark red wine sits beside a vintage brass candle holder with a lit white taper candle. A smaller wooden cutting board holding sliced apples and a wedge of brie overlaps the center of the table. The warm natural sunlight originates from the left, creating a realistic interplay of dappled shadows, crisp highlights on the polished glass and glossy tomatoes, and rich drop shadows that emphasize the rough wood grain and textured vegetables.'

python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task t2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/platforms/mthreads_musa/minimax_h3_image_native.json" \
  --prompt "$prompt" \
  --size 3552 4736 \
  --seed 42 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_t2i_native_musa.png" \
  "$@"
