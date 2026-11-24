import os
from pathlib import Path

import torch
import yaml

from lightx2v.models.networks.three_models.ops import bind_modules

from .motion import MotionFlowMatching
from .text import HYTextWeights


def load_hymotion(config):
    root = Path(config["model_path"])
    checkpoint_root = root / "HY-Motion-1.0" if (root / "HY-Motion-1.0/config.yml").exists() else root
    spec = yaml.safe_load((checkpoint_root / "config.yml").read_text())
    args = spec["train_pipeline_args"]
    args["network_module"] = spec["network_module"]
    args["network_module_args"] = spec["network_module_args"]
    stats = Path(config.get("motion_stats_path", str(Path(__file__).parents[1] / "utils/assets/stats")))
    args["mean_std_dir"] = str(stats)
    weights = MotionFlowMatching(**args)
    state = torch.load(checkpoint_root / "latest.ckpt", map_location="cpu", weights_only=False)["model_state_dict"]
    missing, unexpected = weights.load_state_dict(state, strict=False)
    # Native checkpoint saves mean/std and all inference parameters. Reject any
    # mismatch instead of silently continuing with random initial weights.
    geometry_buffers = {"body_model.v_template", "body_model.j_template", "body_model.skin_weights", "body_model.skin_indices", "body_model.parents"}
    if set(missing) != geometry_buffers or unexpected:
        raise ValueError(f"HY-Motion checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    weights = weights.eval().float().cuda()
    counts = bind_modules(weights.motion_transformer)
    weights.requires_grad_(False)
    text = HYTextWeights(
        config.get("qwen_model_path", os.getenv("QWEN_MODEL_PATH", str(root / "Qwen3-8B"))), config.get("clip_model_path", os.getenv("CLIP_MODEL_PATH", str(root / "clip-vit-large-patch14")))
    )
    return weights, text, counts
