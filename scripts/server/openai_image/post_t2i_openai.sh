#!/bin/bash

lightx2v_path=/data/nvme1/yongyang/nb/LightX2V

export PYTHONPATH="${lightx2v_path}"

python "${lightx2v_path}/scripts/server/openai_image/test_openai_images_client.py" \
--base_url "http://127.0.0.1:8000/v1" \
--api_key "dummy-key" \
--model "gpt-image-1" \
--mode generate \
--prompt "a beautiful sunset over the ocean" \
--size "1024x1024" \
--response_format "b64_json" \
--output_dir "${lightx2v_path}/save_results/qwen_image_t2i_openai"
