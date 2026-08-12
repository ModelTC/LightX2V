#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export SOL_ATTN_STRICT=1

prompt='integrated_multimodal_description: [Shot 1] Live-action wildlife cinematography, a low-angle medium-wide tracking shot follows a red fox moving purposefully through a dense, snow-covered pine forest at dawn. The camera tracks backward at moderate speed, keeping the fox’s face and amber eyes sharply focused as its paws plunge into fresh powder and scatter fine snow crystals toward the lens. Its thick red-and-white winter coat ripples naturally in the cold wind while visible breath streams from its muzzle. Pale golden sunbeams flicker rapidly across its body as it passes between dark tree trunks. The fox suddenly hears a distant cracking branch, raises its ears, turns sharply to the right, and accelerates into a sprint.
[Shot 2] At 00:05.200, the camera cuts to a fast lateral tracking shot moving parallel to the sprinting fox. It weaves between closely spaced pine trunks, bounds over exposed roots, and ducks beneath a snow-laden branch. Its paws strike the ground in a rapid rhythm, throwing broad sprays of powder behind it. The disturbed branch snaps upward and releases a cascading curtain of snow as the camera passes through the falling crystals. The fox races down a short slope, briefly loses its footing in deep powder, recovers immediately, and launches toward a fallen log.
[Shot 3] At 00:10.300, the shot cuts to a low frontal angle on the opposite side of the log as the fox leaps directly across the frame in brief slow motion, individual snow crystals suspended around its outstretched body. As it lands, the camera arcs left with large amplitude at fast speed and transitions back to normal motion, following the fox into an open forest clearing. A small flock of ravens bursts from the nearby trees and crosses the pale sky while wind drives loose snow through shafts of golden light. The fox slows near the center of the clearing, turns its head toward the distant mountains, then runs into the luminous morning mist as the camera rises rapidly above the treetops to reveal the vast frozen forest.
overall_soundscape: Rapid paws crunch through deep snow, branches scrape against fur, frozen wood cracks, and cascading powder lands in soft layered impacts. The fox breathes faster during the sprint while wind rush intensifies through the trees; raven wings beat overhead and several sharp calls echo across the clearing.
non_diegetic_music: Repeating low-string ostinatos and deep hand-drum pulses gradually accelerate during the chase. A rising French-horn phrase and rapid high strings peak as the fox leaps over the log, then expand into sustained orchestral chords as the camera rises above the forest.'


python -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_bf16_4step_sol_attn_single_gpu_offload.json \
--prompt "$prompt" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_sol_attn_4step.mp4 \
--seed 0
