import os
from contextlib import nullcontext

import torch
from loguru import logger

from lightx2v_train.model_zoo.native.wan.fastwam import FastWAM
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype


def _resolve_local_path(path, name, directory=False):
    if not path:
        raise ValueError(f"model.{name} must be an explicit local path.")
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    exists = os.path.isdir(resolved) if directory else os.path.isfile(resolved)
    if not exists:
        expected = "directory" if directory else "file"
        raise FileNotFoundError(f"model.{name} {expected} does not exist: {resolved}")
    return resolved


@MODEL_REGISTER("wan_fastwam")
class WanFastWAMModel:
    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self.running_dtype = get_running_dtype(self.model_config.get("running_dtype", "bf16"))
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.module = None

    def load_components(self, transformer_only=False, reference_model=None):
        if transformer_only:
            raise ValueError("wan_fastwam does not support transformer_only loading.")

        model_path = _resolve_local_path(self.model_config.get("model_path"), "model_path", directory=True)
        configured_action_dit_path = self.model_config.get("action_dit_pretrained_path")
        action_dit_pretrained_path = _resolve_local_path(configured_action_dit_path, "action_dit_pretrained_path") if configured_action_dit_path else None
        load_text_encoder = bool(self.model_config.get("load_text_encoder", False))

        video_scheduler = self.model_config.get("video_scheduler", {})
        action_scheduler = self.model_config.get("action_scheduler", {})
        loss_config = self.model_config.get("loss", {})
        self.module = FastWAM.from_wan22_pretrained(
            model_path=model_path,
            device=str(self.device),
            torch_dtype=self.running_dtype,
            tokenizer_max_len=int(self.model_config.get("tokenizer_max_len", 128)),
            load_text_encoder=load_text_encoder,
            proprio_dim=self.model_config.get("proprio_dim", 8),
            video_dit_config=dict(self.model_config.get("video_dit_config", {})),
            action_dit_config=dict(self.model_config.get("action_dit_config", {})),
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=bool(self.model_config.get("skip_dit_load_from_pretrain", False)),
            mot_checkpoint_mixed_attn=bool(self.model_config.get("mot_checkpoint_mixed_attn", False)),
            video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
            video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
            action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
            action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
            action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
            loss_lambda_video=float(loss_config.get("lambda_video", 1.0)),
            loss_lambda_action=float(loss_config.get("lambda_action", 1.0)),
        )

        checkpoint_path = self.model_config.get("checkpoint_path")
        if checkpoint_path:
            logger.info("[model] loading FastWAM checkpoint {}", checkpoint_path)
            self.unwrap_module().load_checkpoint(checkpoint_path)

    def unwrap_module(self):
        if self.module is None:
            raise RuntimeError("FastWAM module has not been loaded.")
        return self.module

    def autocast_context(self):
        if self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.running_dtype)
        return nullcontext()

    def set_dit_only_trainable(self):
        module = self.unwrap_module()
        module.eval()
        module.requires_grad_(False)
        module.dit.train()
        module.dit.requires_grad_(True)
        proprio_encoder = getattr(module, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def trainable_parameters(self):
        module = self.unwrap_module()
        params = list(module.dit.parameters())
        proprio_encoder = getattr(module, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return (param for param in params if param.requires_grad)

    def save_checkpoint(self, path, step=None):
        self.unwrap_module().save_checkpoint(path, step=step)

    def load_checkpoint(self, path):
        return self.unwrap_module().load_checkpoint(path)

    def log_model_structure(self):
        module = self.unwrap_module()
        logger.info("[model] class=WanFastWAMModel fastwam_class={}", module.__class__.__name__)
        logger.info("[model] trainable root={}", module.dit.__class__.__name__)
