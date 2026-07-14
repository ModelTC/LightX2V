#!/bin/bash

lightx2v_path=/root/yongyang3/LightX2V
base_url=http://127.0.0.1:8000/v1
output_dir=${lightx2v_path}/save_results/hidream_o1_image_openai_test

export PYTHONPATH="${lightx2v_path}"

python "${lightx2v_path}/scripts/hidream_o1_image/test_openai_images_client.py" \
--base_url "${base_url}" \
--api_key "dummy-key" \
--model "gpt-image-1" \
--mode edit \
--prompt "remove the earphones" \
--image "/root/test.jpg" \
--seed 42 \
--size "2048x2048" \
--response_format "b64_json" \
--output_dir "${output_dir}" \
--output_prefix "hidream_o1_image_openai" \
--i2i_denoise_strength 0.9
