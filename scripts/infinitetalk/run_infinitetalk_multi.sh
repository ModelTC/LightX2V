#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/InfiniteTalk

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls infinitetalk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/infinitetalk/h100/infinitetalk_mutli_distilled.json \
--prompt  "The video features a man and a woman standing by a bench in the park, their expressions tense and voices raised as they argue. The man gestures with both hands, his arms swinging slightly as if to emphasize each heated word, while the woman stands with her hands on her waist, her brows furrowed in frustration. The background is a wide expanse of sunlit grass, the golden light contrasting with the sharp energy of their quarrel. Their voices seem to clash in the air, and the rhythm of their hand movements and body postures interweaves with the rising tension, creating a vivid scene of confrontation." \
--image_path ${lightx2v_path}/assets/inputs/audio/multi_person/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/multi_person/p1.mp3,${lightx2v_path}/assets/inputs/audio/multi_person/p2.mp3 \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_multi_distilled.mp4 \
--seed 42
