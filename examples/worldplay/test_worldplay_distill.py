import json
import os

# Paths
CONFIG_PATH = "/workspace/LightX2V/configs/worldplay/worldplay_distill_i2v_480p.json"
MODEL_PATH = "/data/nvme1/models/hunyuan/hf_cache/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
ACTION_CKPT = "/data/nvme1/models/hunyuan/HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors"
IMAGE_PATH = "/workspace/HY-WorldPlay/assets/img/test.png"
OUTPUT_PATH = "/workspace/LightX2V/save_results/HY-WorldPlay/"

# Input parameters
PROMPT = "A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape."
SEED = 1
POSE = "w-31"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def main():
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.lockable_dict import LockableDict
    from lightx2v.utils.registry_factory import RUNNER_REGISTER

    # Load config from JSON
    with open(CONFIG_PATH, "r") as f:
        config_dict = json.load(f)

    # Add runtime paths
    config_dict["model_path"] = MODEL_PATH
    config_dict["action_ckpt"] = ACTION_CKPT
    config_dict["transformer_model_path"] = os.path.join(MODEL_PATH, "transformer/480p_i2v")

    config = LockableDict(config_dict)

    runner = RUNNER_REGISTER[config["model_cls"]](config)

    runner.init_modules()

    # Prepare input info
    input_data = {
        "seed": SEED,
        "prompt": PROMPT,
        "prompt_enhanced": "",
        "negative_prompt": "",
        "image_path": IMAGE_PATH,
        "save_result_path": os.path.join(OUTPUT_PATH, "worldplay_distill_test.mp4"),
        "return_result_tensor": False,
        "pose": POSE,
    }

    input_info = init_empty_input_info("i2v")
    update_input_info_from_dict(input_info, input_data)

    result = runner.run_pipeline(input_info)
    return result


if __name__ == "__main__":
    main()
