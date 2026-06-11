#!/bin/bash

lightx2v_path=/root/yongyang3/LightX2V
port=8000
server_url=http://127.0.0.1:${port}

prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room."
negative_prompt=""
infer_steps=28
seed=32
aspect_ratio=1:1
target_height=2048
target_width=2048

# Keep this relative so /v1/tasks/{task_id}/result can download it from the server output dir.
server_save_result_path=hidream_o1_image_t2i_dev_2604_request.png
output=${lightx2v_path}/save_results/hidream_o1_image_t2i_dev_2604_request.png
timeout_seconds=1200
poll_interval=2.0

export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"

python "${lightx2v_path}/scripts/server/post_async_t2i_and_wait.py" \
--url "${server_url}" \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--infer_steps "${infer_steps}" \
--seed "${seed}" \
--aspect_ratio "${aspect_ratio}" \
--target_shape "${target_height}" "${target_width}" \
--save_result_path "${server_save_result_path}" \
--timeout_seconds "${timeout_seconds}" \
--poll_interval "${poll_interval}" \
--output "${output}"
