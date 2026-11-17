import os

from lightx2v.models.runners.config import read_config
from lightx2v.models.runners.h3.config import VAE_CONFIG, prepare_h3_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_cls = deployment_config["model_cls"]
    causal = model_cls == "minimax_h3_causal"
    if causal:
        if deployment_config.get("task") not in {"refa2v", "refa2av"}:
            raise ValueError("MiniMax-H3 causal requires task='refa2v' or 'refa2av'")
        if not deployment_config.get("dit_original_ckpt"):
            raise ValueError("MiniMax-H3 causal inference requires dit_original_ckpt pointing to the causal checkpoint")
        transformer_subfolder = "transformer_ref"
    else:
        model_variant = deployment_config.get("model_variant")
        if model_variant not in ("fl2av", "ref2av"):
            raise ValueError("MiniMax-H3 requires model_variant='fl2av' or 'ref2av'; set --model-variant when starting the model")
        if model_cls == "taomate_h3" and model_variant != "fl2av":
            raise ValueError("TaoMate-H3 Stage3 requires the FL2VA base partition")
        transformer_subfolder = "transformer_ref" if model_variant == "ref2av" else "transformer"
    transformer_path = os.path.join(deployment_config["model_path"], transformer_subfolder)
    path = os.path.join(transformer_path, "config.json")
    config = {"patch_size": (1, 2, 2)} | read_config(path)
    if not causal:
        config["dit_original_ckpt"] = transformer_path
    config.setdefault("fps", 24)
    config.setdefault("video_flow_shift", 12.0)
    config.setdefault("audio_flow_shift", 3.0)
    config.setdefault("audio_sampling_rate", 32000)
    config.update(VAE_CONFIG)
    config = merge_config(config, deployment_config)
    prepare_h3_config(config)
    config.pop("model_variant" if config["model_cls"] == "minimax_h3_causal" else "task", None)
    return config
