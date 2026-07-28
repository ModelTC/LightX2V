#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/lingbot-video-moe-30b-a3b

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=grouped_mm

prompt=$(cat <<'EOF'
{
  "caption": {
    "comprehensive_description": {
      "scene_content_description": "The video presents a first-person perspective of a workspace, likely a desk, viewed from a fixed, slightly elevated angle. The environment is well-lit with neutral, even lighting, creating a clear and focused atmosphere. The desk surface is a light grey color. On the left side of the desk, there is a black game controller with a glowing blue light bar, resting on a black mousepad. In the center of the desk lies a pair of black over-ear headphones. To the right, there is an open, empty black box with a white interior. In the foreground, two robotic arms are visible. The left robotic arm, featuring a black and silver body with a two-fingered gripper, remains completely stationary throughout the video. The right robotic arm, similar in design, is the active subject. It begins by moving forward and to the left, positioning its gripper over the game controller. It then grasps the controller, lifts it off the desk, and transports it to the right, moving it over the open box. Finally, the right robotic arm lowers the controller into the box and releases it. The headphones and the left robotic arm remain undisturbed during this entire sequence.",
      "camera_movement_description": ""
    },
    "camera_info": {
      "color": "Cyan",
      "frame_size": "Wide",
      "shot_type_angle": "High angle",
      "lens_size": "Ultra Wide / Fisheye",
      "composition": "Balanced",
      "lighting": "Soft light",
      "lighting_type": "Artificial light"
    },
    "world_knowledge": [],
    "prominent_elements": [
      {
        "name": "right robotic arm",
        "description": "A mechanical arm with a black and silver body, equipped with a two-fingered gripper and visible wiring.",
        "actions": [
          {
            "timestamp": "[0.0s - 1.5s]",
            "action": "Moves forward and to the left towards the game controller."
          },
          {
            "timestamp": "[1.5s - 2.5s]",
            "action": "Grasps the game controller and lifts it upward."
          },
          {
            "timestamp": "[2.5s - 4.0s]",
            "action": "Moves to the right, carrying the game controller over the open box."
          },
          {
            "timestamp": "[4.0s - 5.0s]",
            "action": "Lowers the game controller into the box and releases it."
          }
        ],
        "location": "Originates from the bottom right, moves to the center, then to the right.",
        "relative_size": "large",
        "shape_and_color": "Cylindrical and angular segments, black and silver.",
        "texture": "Metallic and matte plastic.",
        "appearance_details": "Visible joints, wiring, and a two-fingered gripper mechanism.",
        "relationship": "Interacts directly with the game controller.",
        "orientation": "Extends forward and slightly upward from the bottom right.",
        "pose": "",
        "expression": "",
        "clothing": "",
        "is_cluster": false,
        "number_of_objects": ""
      },
      {
        "name": "left robotic arm",
        "description": "A mechanical arm with a black and silver body, equipped with a two-fingered gripper.",
        "actions": [
          {
            "timestamp": "[0.0s - 5.0s]",
            "action": "Remains stationary."
          }
        ],
        "location": "Bottom left corner of the frame.",
        "relative_size": "large",
        "shape_and_color": "Cylindrical and angular segments, black and silver.",
        "texture": "Metallic and matte plastic.",
        "appearance_details": "Visible joints and a two-fingered gripper mechanism.",
        "relationship": "Positioned opposite the right robotic arm, not interacting with other objects.",
        "orientation": "Extends forward and slightly upward from the bottom left.",
        "pose": "",
        "expression": "",
        "clothing": "",
        "is_cluster": false,
        "number_of_objects": ""
      },
      {
        "name": "game controller",
        "description": "A standard video game controller with joysticks, buttons, and a glowing light bar.",
        "actions": [
          {
            "timestamp": "[0.0s - 1.5s]",
            "action": "Rests stationary on the desk."
          },
          {
            "timestamp": "[1.5s - 2.5s]",
            "action": "Is grasped and lifted upward by the right robotic arm."
          },
          {
            "timestamp": "[2.5s - 4.0s]",
            "action": "Is moved to the right by the right robotic arm."
          },
          {
            "timestamp": "[4.0s - 5.0s]",
            "action": "Is lowered into the open box and released."
          }
        ],
        "location": "Initially on the left side of the desk, moved to the right side inside the box.",
        "relative_size": "medium",
        "shape_and_color": "Contoured shape, black with a blue light bar.",
        "texture": "Matte plastic.",
        "appearance_details": "Two joysticks, directional pad, action buttons, and a glowing blue light bar in the center.",
        "relationship": "Initially on the desk, then grasped and moved by the right robotic arm, finally placed in the open box.",
        "orientation": "Horizontal on the desk, then tilted while being carried.",
        "pose": "",
        "expression": "",
        "clothing": "",
        "is_cluster": false,
        "number_of_objects": ""
      },
      {
        "name": "headphones",
        "description": "A pair of over-ear headphones with a headband and ear cups.",
        "actions": [
          {
            "timestamp": "[0.0s - 5.0s]",
            "action": "Remains stationary on the desk."
          }
        ],
        "location": "Center of the desk.",
        "relative_size": "medium",
        "shape_and_color": "Curved headband with circular ear cups, black.",
        "texture": "Matte plastic and soft ear cushions.",
        "appearance_details": "Visible ear cushions and a headband.",
        "relationship": "Rests on the desk, untouched by the robotic arms.",
        "orientation": "Lying flat on the desk.",
        "pose": "",
        "expression": "",
        "clothing": "",
        "is_cluster": false,
        "number_of_objects": ""
      },
      {
        "name": "open box",
        "description": "A rectangular box with the lid open, revealing a white interior.",
        "actions": [
          {
            "timestamp": "[0.0s - 5.0s]",
            "action": "Remains stationary on the desk."
          }
        ],
        "location": "Right side of the desk.",
        "relative_size": "medium",
        "shape_and_color": "Rectangular, black exterior with a white interior.",
        "texture": "Smooth cardboard or plastic.",
        "appearance_details": "Open lid, empty interior.",
        "relationship": "Serves as the receptacle for the game controller.",
        "orientation": "Horizontal on the desk.",
        "pose": "",
        "expression": "",
        "clothing": "",
        "is_cluster": false,
        "number_of_objects": ""
      }
    ]
  },
  "duration": 5
}
EOF
)

negative_prompt=$(cat <<'EOF'
{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}
EOF
)

python -m lightx2v.infer \
--model_cls lingbot_video \
--task t2v \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/lingbot_video/lingbot_video_lora.json \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_video_t2v.mp4 \
--seed 42
