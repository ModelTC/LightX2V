#!/bin/bash

# AdaLN cache setup:
# The default inference JSON configs enable "use_adaln_cache": true.
# Generate the cache before inference; skip this step if a matching cache exists.
# 1. Set lightx2v_path, model_path, --config_json, and platform/device settings in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh.
# 2. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh
# 3. Run inference after cache generation completes.
# Cache generation and inference must use the same model weights, timesteps,
# platform, and adaln_cache_dir. Point --config_json to the inference JSON.
# One GPU generates both T2I and I2I cache profiles, reusable across TP/SP configs.
# The tool will not overwrite an existing cache directory; move it aside to rebuild.

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES=7

source "$lightx2v_path/scripts/base/base.sh"
prompt='The image is a photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a heavily weathered, dark brown wooden table with deep visible grain and scratches. The composition is a top-down view looking directly at the table surface. In the center sits a large woven jute placemat supporting a freshly baked rustic sourdough loaf with a deeply scored, golden-brown crust and a dusting of flour. To the upper left, a hammered copper bowl overflows with a colorful mix of fresh seasonal vegetables, including purple kale, bright orange carrots, red beets, and vibrant green salad leaves. In the upper right, a dark wooden cheese board holds neatly arranged artisanal cheeses: a wedge of bloomy rind brie with a creamy yellow interior, a wedge of aged cheddar with a golden rind, and a log of goat cheese sprinkled with cracked peppercorns and rosemary sprigs. The lower left contains a cluster of heirloom tomatoes, small cherry tomatoes on the vine, scattered toasted walnuts, and a small ceramic bowl of green olives. In the lower right, a clear glass wine glass filled with dark red wine sits beside a vintage brass candle holder with a lit white taper candle. A smaller wooden cutting board holding sliced apples and a wedge of brie overlaps the center of the table. The warm natural sunlight originates from the left, creating a realistic interplay of dappled shadows, crisp highlights on the polished glass and glossy tomatoes, and rich drop shadows that emphasize the rough wood grain and textured vegetables.'

python -m lightx2v.infer \
  --model_cls minimax_h3_image \
  --task t2i \
  --model_path "$model_path" \
  --config_json "$lightx2v_path/configs/minimax_h3_image/minimax_h3_image.json" \
  --prompt "$prompt" \
  --size 4096 4096 \
  --seed 42 \
  --save_result_path "$lightx2v_path/save_results/minimax_h3_image_t2i.png"
