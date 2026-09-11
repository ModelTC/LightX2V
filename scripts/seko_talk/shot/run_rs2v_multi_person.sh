#!/bin/bash
lightx2v_path=

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export PROFILING_DEBUG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

python -m lightx2v.shot_runner.rs2v_infer \
--config_json ${lightx2v_path}/configs/seko_talk/shot/rs2v/main.json \
--prompt  "The video features a man and a woman standing by a bench in the park, their expressions tense and voices raised as they argue. The man gestures with both hands, his arms swinging slightly as if to emphasize each heated word, while the woman stands with her hands on her waist, her brows furrowed in frustration. The background is a wide expanse of sunlit grass, the golden light contrasting with the sharp energy of their quarrel. Their voices seem to clash in the air, and the rhythm of their hand movements and body postures interweaves with the rising tension, creating a vivid scene of confrontation." \
--image_path ${lightx2v_path}/assets/inputs/audio/multi_person/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/multi_person \
--save_result_path ${lightx2v_path}/save_results/output_seko_talk_shot_rs2v_multi_person.mp4 \
--video_duration 20
