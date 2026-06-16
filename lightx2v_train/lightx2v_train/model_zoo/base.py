import os

import torch
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.peft_utils import get_adapter_name
from loguru import logger
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import is_fsdp2_module
from lightx2v_train.utils.utils import get_running_dtype


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.vae = None

    def load_components(self):
        raise NotImplementedError

    def denoiser_module(self):
        raise NotImplementedError(f"{self.__class__.__name__} must define denoiser_module().")

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.denoiser_module().add_adapter(lora_config)

    def set_lora_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(False)
        denoiser.train()
        for name, param in denoiser.named_parameters():
            param.requires_grad = "lora" in name

    def set_full_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(True)
        denoiser.train()

    def trainable_parameters(self):
        return (p for p in self.denoiser_module().parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "enable_gradient_checkpointing"):
            denoiser.enable_gradient_checkpointing()

    def set_denoiser_eval(self):
        self.denoiser_module().eval()

    def is_fsdp2_wrapped(self):
        return is_fsdp2_module(self.denoiser_module())

    def fsdp2_state_module(self):
        return self.denoiser_module()

    def set_fsdp2_gradient_sync(self, enabled):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "set_requires_gradient_sync"):
            denoiser.set_requires_gradient_sync(enabled)
        if hasattr(denoiser, "set_is_last_backward"):
            denoiser.set_is_last_backward(enabled)

    def fsdp2_shard_plan(self, fsdp_config):
        raise NotImplementedError(f"{self.__class__.__name__} must define fsdp2_shard_plan().")

    def log_model_structure(self):
        logger.info("[model] class={}", self.__class__.__name__)
        text_encoder = getattr(getattr(self, "text_pipeline", None), "text_encoder", None)
        if text_encoder is not None:
            logger.info("[model] text_encoder structure:\n{}", text_encoder)
        if self.vae is not None:
            logger.info("[model] vae structure:\n{}", self.vae)
        logger.info("[model] denoiser structure:\n{}", self.denoiser_module())

    def encode_to_latent(self, sample):
        raise NotImplementedError

    def encode_condition(self, sample):
        raise NotImplementedError

    def prepare_denoiser_input(self, noisy_latent, sample, condition):
        raise NotImplementedError

    def denoise(self, denoiser_input, timesteps, condition):
        raise NotImplementedError

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        raise NotImplementedError

    def prepare_infer_latents(self, height, width, generator=None):
        raise NotImplementedError

    def decode_latent(self, latent):
        raise NotImplementedError

    def assemble_pipeline(self, scheduler=None):
        raise NotImplementedError

    def get_pipeline_infer_kwargs(self, infer_config):
        """Return kwargs to pass to pipeline.__call__. Override to adapt model-specific parameter names."""
        return {
            "height": infer_config.get("height", 1024),
            "width": infer_config.get("width", 1024),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 4.0),
        }

    def get_pipeline_sample_kwargs(self, sample):
        """Return per-sample kwargs to pass to pipeline.__call__ during native inference."""
        return {}

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        denoiser = self.denoiser_module()
        if adapter_name is None:
            adapter_name = get_adapter_name(denoiser)
        denoiser.load_lora_adapter(lora_path, adapter_name=adapter_name)
        self._infer_lora_adapter_name = adapter_name

    def unload_lora_for_infer(self):
        adapter_name = getattr(self, "_infer_lora_adapter_name", None)
        if adapter_name is not None:
            self.denoiser_module().delete_adapters(adapter_name)
            self._infer_lora_adapter_name = None

    def save_lora_weights(self, save_dir):
        peft_state_dict = self._get_lora_state_dict_for_save()
        if not is_main_process():
            return

        lora_state_dict = convert_state_dict_to_diffusers(peft_state_dict)
        if hasattr(self.pipeline_cls, "save_lora_weights"):
            self.pipeline_cls.save_lora_weights(save_dir, lora_state_dict, safe_serialization=True)
        else:
            save_file(lora_state_dict, f"{save_dir}/pytorch_lora_weights.safetensors")

    def _get_lora_state_dict_for_save(self):
        denoiser = self.denoiser_module()
        if not is_fsdp2_module(denoiser):
            return get_peft_model_state_dict(denoiser)

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=True,
            strict=False,
        )
        state_dict, _ = get_state_dict(denoiser, (), options=options)
        if not is_main_process():
            return {}
        return get_peft_model_state_dict(denoiser, state_dict=state_dict)

    def load_lora_weights_for_resume(self, lora_path):
        raw = load_file(os.path.join(lora_path, "pytorch_lora_weights.safetensors"))
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        incompatible = set_peft_model_state_dict(self.denoiser_module(), peft_state_dict)
        if incompatible and incompatible.unexpected_keys:
            logger.warning("Unexpected keys when resuming LoRA: {}", incompatible.unexpected_keys)

    def save_full_model(self, save_dir):
        self.denoiser_module().save_pretrained(f"{save_dir}/transformer", safe_serialization=True)
