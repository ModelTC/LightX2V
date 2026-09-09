import json
import os

# Paths
LIGHTX2V_PATH = "/path/to/LightX2V"
CONFIG_PATH = os.path.join(LIGHTX2V_PATH, "configs/worldplay/worldplay_ar_i2v_480p.json")
MODEL_PATH = "/path/to/HunyuanVideo-1.5"
ACTION_CKPT = "/path/to/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors"
IMAGE_PATH = "/path/to/HY-WorldPlay/assets/img/test.png"
OUTPUT_PATH = os.path.join(LIGHTX2V_PATH, "save_results/HY-WorldPlay")

# Input parameters
PROMPT = "A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere."
SEED = 1
POSE = "d-31"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config

    # Load config from JSON
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)

    # Add runtime paths
    config_dict["model_path"] = MODEL_PATH
    config_dict["action_ckpt"] = ACTION_CKPT
    config_dict["transformer_model_path"] = os.path.join(MODEL_PATH, "transformer/480p_i2v")

    runner = build_runner(build_startup_config(config_dict))

    # Prepare input info
    input_data = {
        "seed": SEED,
        "prompt": PROMPT,
        "image_path": IMAGE_PATH,
        "save_result_path": os.path.join(OUTPUT_PATH, "worldplay_ar_test.mp4"),
        "return_result_tensor": False,
        "pose": POSE,
    }

    input_info = runner.prepare_request(input_data)
    result = runner.run_request(input_info)

    return result


if __name__ == "__main__":
    main()
