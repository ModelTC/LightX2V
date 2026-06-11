#!/bin/bash

lightx2v_path=/root/yongyang3/LightX2V
test_json=${TEST_JSON:-/root/test.json}
port=${PORT:-8000}
server_url=${SERVER_URL:-http://127.0.0.1:${port}}

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"

python "${lightx2v_path}/scripts/hidream_o1_image/post_async_t2i_and_wait.py" \
--url "${server_url}" \
--prompt_json "${test_json}" \
--negative_prompt "" \
--infer_steps 28 \
--seed 42 \
--aspect_ratio 1:1 \
--target_shape 2048 2048 \
--timeout_seconds 1200 \
--poll_interval 2.0 \
--output_dir "${lightx2v_path}/save_results/hidream_o1_image_test_json" \
--output_prefix hidream_o1_image_test_json
