import copy

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


class DenoiserInput:
    def __init__(self, hidden_states, extra):
        self.hidden_states = hidden_states
        self.extra = extra


class BaseModel:
    """Framework-level model object.

    Subclasses wrap the real diffusers pipeline/model components and expose only
    the small surface that training algorithms need.
    """

    def __init__(self, config):
        self.config = config
        self.dtype = self.get_dtype(config["model"]["dtype"])
        self.device = torch.device("cuda")
        self.pipeline = None
        self.scheduler = None
        self.scheduler_copy = None
        self.transformer = None
        self.vae = None

    def get_dtype(self, name):
        if name == "bf16":
            return torch.bfloat16
        elif name == "fp16":
            return torch.float16
        elif name == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Invalid dtype: {name}")

    def load_components(self):
        raise NotImplementedError

    def build_pipeline(self):
        raise NotImplementedError

    def generate(self, **kwargs):
        lora_path = kwargs.pop("lora_path", None)
        pipe = self.build_pipeline()
        if lora_path is not None:
            pipe.load_lora_weights(lora_path)
        pipe.to(self.device)
        return pipe(**kwargs)

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

    def init_training_scheduler(self):
        self.scheduler_copy = copy.deepcopy(self.scheduler)

    def sample_timesteps(self, batch_size, latents_device):
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.scheduler_copy.config.num_train_timesteps).long()
        return self.scheduler_copy.timesteps[indices].to(device=latents_device)

    def get_sigmas(self, timesteps, n_dim, dtype):
        sigmas = self.scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(self, latents, noise, sigmas):
        return (1.0 - sigmas) * latents + sigmas * noise

    def loss_weighting(self, sigmas):
        return compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

    def build_target(self, latents, noise):
        return noise - latents

    def encode_media(self, batch):
        raise NotImplementedError

    def encode_conditions(self, batch):
        raise NotImplementedError

    def prepare_denoiser_input(self, noisy_latents, batch, conditions):
        raise NotImplementedError

    def denoise(self, denoiser_input, timesteps, conditions):
        raise NotImplementedError

    def unpack_prediction(self, prediction, denoiser_input):
        raise NotImplementedError

    def save_lora_weights(self, save_dir):
        lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.transformer))
        if hasattr(self.pipeline_cls, "save_lora_weights"):
            self.pipeline_cls.save_lora_weights(save_dir, lora_state_dict, safe_serialization=True)
        else:
            save_file(lora_state_dict, f"{save_dir}/pytorch_lora_weights.safetensors")

    def save_full_model(self, save_dir):
        self.transformer.save_pretrained(f"{save_dir}/transformer", safe_serialization=True)
