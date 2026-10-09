#!/bin/bash
# set path firstly
lightx2v_path=/data/wangshankun/LightX2V
model_path=/data/wangshankun/models/h3/MiniMax-H3

# Calibrate with the SAME Sol-Attn settings before inference:
# bash scripts/minimax_h3/run_minimax_h3_ref2av_sol_dpcache.sh calibrate
# bash scripts/minimax_h3/run_minimax_h3_ref2av_sol_dpcache.sh
case "${1:-infer}" in
  infer)
    default_config="${lightx2v_path}/configs/minimax_h3/decache/minimax_h3_ref2av_sp_k14_sol.json"
    default_output="${lightx2v_path}/save_results/dpcache_h3/k14_sol/dpcache.mp4"
    ;;
  calibrate)
    default_config="${lightx2v_path}/configs/minimax_h3/decache/minimax_h3_ref2av_sp_k14_sol_calibration.json"
    default_output="${lightx2v_path}/save_results/dpcache_h3/k14_sol/calibration.mp4"
    ;;
  *) echo "Usage: $0 [infer|calibrate]" >&2; exit 2 ;;
esac
cd "${lightx2v_path}" || exit 1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant ref2av \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${CONFIG_JSON:-${default_config}}" \
  --prompt "subject_definitions:\n<Subject 1> is the character in <Picture 1>, a stylized girl with purple and pink tentacle-like hair tied with rope, red eyes, sharp teeth, pointy ears, wearing a transparent plastic jacket over a black crop top, dark shorts, ropes around her waist and legs, and featuring a metallic robotic right arm from the elbow down. <Subject 2> is the UI style and elements in <Picture 2>, featuring neon green paint splatters, a \"MINIMAX\" player profile in the top-left, and a vertical list of menu buttons on the right side with graffiti-like graphics on a purple background.\n\nsummary:\n[reference generation] The target video features <Subject 1> interacting with a game menu styled after <Subject 2>. The camera zooms in as she customizes her robotic right and left arms via the holographic UI. After confirming the configuration, a loading bar appears, and the scene dynamically transitions from a solid purple background into a dense cyberpunk city, following <Subject 1> in a third-person perspective as she walks down a neon-lit street.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the target video uses the exact character design, retaining her tentacle hair, facial features, transparent jacket, and robotic right arm, while expanding on her left arm customization. <Subject 2> (appears in [Shot 1]): fully_preserved - the target video replicates the UI style, preserving the \"MINIMAX\" profile, neon green splatters, menu button layout, and overall graphic aesthetic.\n\ndetailed_description:\nThe target video is in a highly stylized, 3D animated video game aesthetic with vibrant neon colors and realistic textures.\n\n[Shot 1] The shot begins with a high-angle, top-down view of <Subject 1> sitting cross-legged on a highly saturated, bright purple floor. <Subject 1>, a stylized girl with pink and purple tentacle-like hair tied with rope, sharp teeth, red eyes, and a transparent plastic jacket, looks directly up at the camera. A holographic game menu UI, styled exactly like <Subject 2>, overlays the scene. On the top-left, the player profile \"MINIMAX\" is visible with neon green splatters. On the right, a vertical menu displays options: \"START NEW GAME,\" a brightly highlighted \"CONTINUE,\" \"SETTINGS,\" and \"EXIT GAME.\" A digital cursor moves and clicks on \"CONTINUE.\" The camera immediately executes a smooth, continuous zoom down toward <Subject 1>'s metallic robotic right arm. As the camera pushes in, a new UI panel slides in from the right, labeled \"RIGHT ARM EQUIPMENT.\" The UI cursor highlights \"PHANTOM Grip\" before sliding down to select \"CHRONOS CLAW.\" In response, <Subject 1>'s robotic right hand mechanically reconfigures; the metallic fingers separate and lock into new, sharp claw-like knuckles as a bright cyan LED light flashes from within the joints. The camera then smoothly orbits around <Subject 1> to her left side. A new UI grid labeled \"ARMAMENT CUSTOMIZATION\" slides into view, displaying components for the hand, forearm, elbow, and upper arm. The selection rapidly cycles through the parts. <Subject 1>'s left arm is dynamically disassembled piece by piece: the forearm plates detach and fall away, new dark armor slides in, the elbow joint is entirely replaced, and the hand mechanically reconfigures into a cybernetic limb. During this rapid swap, glowing internal wires and moving pistons are briefly exposed. The camera then pulls back out to a medium shot. A \"CONFIRM CONFIG\" button flashes on the screen and is clicked. Instantly, all UI panels shrink inward and disappear. <Subject 1> uncrosses her legs, shifts her posture, and sits back in a relaxed position with one knee raised. She effortlessly lifts her newly configured robotic hands, performing a subtle, fluid movement to showcase the upgraded joints. At the bottom of the screen, a digital \"LOADING\" bar appears, rapidly filling from 0% to 100%. As the bar fills, the bright purple environment rapidly darkens, with deep shadows creeping in from the edges of the frame while a warm golden light begins to bleed into the scene.\n\n[Shot 2] At 00:10.000, the shot cuts to <Subject 1> standing up as a completely new environment loads in around her. The scene seamlessly transitions into a dense, gritty cyberpunk slum. Flickering neon signs illuminate the dark, narrow alleys, and the wet, rain-slicked asphalt reflects vibrant pink and cyan lights. A bustling crowd of diverse characters surges through the street, while futuristic motorcycles weave past them. Above, a chaotic, intricate web of electrical wires stretches across the gap between towering, stacked megabuildings that reach into a hazy, futuristic skyline. The camera stabilizes into a classic third-person video game perspective, positioned directly behind <Subject 1> to show her back and the glowing details of her new robotic arms. Game HUD elements smoothly fade into the frame: a circular minimap appears in the top-right corner, and a health and ammo counter illuminates the bottom-left. A glowing yellow quest marker materializes in the distance. <Subject 1> takes a confident step forward, walking down the crowded cyberpunk street as the video ends.\n\noverall_soundscape:\nA high-tech, futuristic video game soundscape featuring digital UI clicks, mechanical whirs and clanks of robotic armor reconfiguring, and transitioning into a bustling cyberpunk city ambience with distant engine rumbles, crowd murmurs, and buzzing neon signs.\n\nnon_diegetic_music:\nA fast-paced, synth-heavy cyberpunk electronic track with a driving bassline that plays underneath the menu interactions and swells dynamically as the scene transitions into the bustling city." \
  --image_path "${lightx2v_path}/assets/figs/01.png, ${lightx2v_path}/assets/figs/02.png" \
  --save_result_path "${SAVE_RESULT_PATH:-${default_output}}" \
  --seed 42
