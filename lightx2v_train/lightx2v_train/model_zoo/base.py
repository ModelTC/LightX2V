import os

import torch
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.peft_utils import get_adapter_name
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from lightx2v_train.utils.utils import get_running_dtype


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])
        self.device = torch.device("cuda")
        self.transformer = None
        self.vae = None

    def load_components(self):
        raise NotImplementedError

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.transformer.add_adapter(lora_config)

    def set_lora_trainable(self):
        self.transformer.requires_grad_(False)
        self.transformer.train()
        for name, param in self.transformer.named_parameters():
            param.requires_grad = "lora" in name

    def set_full_trainable(self):
        self.transformer.requires_grad_(True)
        self.transformer.train()

    def trainable_parameters(self):
        return (p for p in self.transformer.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

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

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        if adapter_name is None:
            adapter_name = get_adapter_name(self.transformer)
        self.transformer.load_lora_adapter(lora_path, adapter_name=adapter_name)
        self._infer_lora_adapter_name = adapter_name

    def unload_lora_for_infer(self):
        adapter_name = getattr(self, "_infer_lora_adapter_name", None)
        if adapter_name is not None:
            self.transformer.delete_adapters(adapter_name)
            self._infer_lora_adapter_name = None

    def save_lora_weights(self, save_dir):
        lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.transformer))
        if hasattr(self.pipeline_cls, "save_lora_weights"):
            self.pipeline_cls.save_lora_weights(save_dir, lora_state_dict, safe_serialization=True)
        else:
            save_file(lora_state_dict, f"{save_dir}/pytorch_lora_weights.safetensors")

    def load_lora_weights_for_resume(self, lora_path):
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        raw = load_file(os.path.join(lora_path, "pytorch_lora_weights.safetensors"))
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        incompatible = set_peft_model_state_dict(self.transformer, peft_state_dict)
        if incompatible and incompatible.unexpected_keys:
            print(f"Warning: unexpected keys when resuming LoRA: {incompatible.unexpected_keys}")

    def save_full_model(self, save_dir):
        self.transformer.save_pretrained(f"{save_dir}/transformer", safe_serialization=True)
