#!/bin/bash

# set path and first
lightx2v_path=/path/to/LightX2V
model_path=Lightricks/LTX-2

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2 \
--task i2av \
--image_path "${lightx2v_path}/assets/inputs/imgs/frame_1.png,${lightx2v_path}/assets/inputs/imgs/frame_2.png,${lightx2v_path}/assets/inputs/imgs/frame_3.png,${lightx2v_path}/assets/inputs/imgs/frame_4.png" \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2.json \
--prompt "A motorcyclist wearing a full black leather racing suit and helmet rides a Yamaha R1 on a dry asphalt mountain road curve at afternoon golden hour in a continuous four-stage action shot: stage one before corner entry with the bike upright, front wheel nearly straight, rider slightly forward and brake light glowing; stage two at apex transition with clear turn-in, medium lean angle, early rear slip and controlled counter-steer, body posture becoming more aggressive with visible dust and tire smoke; stage three exits the sharp right turn with the bike progressively returning upright, rider sitting up slightly, throttle opening aggressively, front wheel lifting subtly from acceleration, rear tire gripping and propelling forward while smoke dissipates, emphasizing exit drift acceleration, camera already moving toward a rear-three-quarter chase; stage four is a dedicated side-on beat: the same rider and bike held in a stable lateral profile view, camera tracking parallel to the road on the outside of the curve at roughly bike height, wheels, fairing, helmet, and lean line clearly readable against the mountain and asphalt, as if cutting from chase to a classic side tracking shot. Between stage three and four, interpolate the viewpoint smoothly: continue the exit energy and speed, then ease the camera from rear-side chase into pure side profile without a hard cut—match direction of travel, horizon, and lighting so the handoff feels like one continuous take. Keep rider identity and bike appearance consistent across all stages, with strong temporal continuity, dynamic motion blur, and photorealistic detail. Emphasize realistic synchronized audio: deep engine rumble at approach, sharp downshift blips and brief backfire pops entering the turn, tire scrub and short skid noise at apex with wind rush, then a rising high-RPM engine roar and accelerating exhaust on exit, steady wind and tire noise under the side-on pass, no background music." \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_i2av_keyframes2.mp4 \
--image_strength 1.0,0.7,0.7,0.7 \
--image_frame_idx 0,120,240,360 \
--target_video_length 361
