from pathlib import Path

import torch
from transformers import CLIPTokenizer

from lightx2v.models.networks.three_models.clip import CLIPTextInfer
from lightx2v.models.schedulers.sdxl import SDXLEulerScheduler

from .infer import SDXLUNetInfer, SDXLVAEInfer
from .weights import SDXLWeights


class SDXLModel:
    def __init__(self, config):
        self.config = config
        self.weights = SDXLWeights(config["model_path"])
        root = Path(config["model_path"])
        self.tokenizers = [CLIPTokenizer.from_pretrained(root / name, local_files_only=True) for name in ("tokenizer", "tokenizer_2")]
        self.text_infers = [CLIPTextInfer(self.weights.configs[name]) for name in ("text_encoder", "text_encoder_2")]
        self.unet_infer = SDXLUNetInfer(self.weights.configs["unet"])
        self.vae_infer = SDXLVAEInfer()

    def encode(self, text):
        hidden, pooled = [], None
        for tokenizer, infer, w in zip(self.tokenizers, self.text_infers, (self.weights.text_encoder, self.weights.text_encoder_2)):
            ids = tokenizer([text], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.cuda()
            feature, pooled = infer.infer(w, ids)
            hidden.append(feature)
        return torch.cat(hidden, dim=-1), pooled

    def generate(self, prompt, negative_prompt, seed, size, steps, cfg_scale, return_latents=False):
        height, width = size
        if height % 64 or width % 64 or min(height, width) < 128:
            raise ValueError("SDXL height and width must be >=128 and divisible by 64")
        positive, pooled = self.encode(prompt)
        do_cfg = cfg_scale > 1
        if do_cfg and negative_prompt:
            negative, negative_pooled = self.encode(negative_prompt)
        else:
            negative, negative_pooled = torch.zeros_like(positive), torch.zeros_like(pooled)
        context = torch.cat((negative, positive)) if do_cfg else positive
        pooled = torch.cat((negative_pooled, pooled)) if do_cfg else pooled
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device="cuda", dtype=torch.float16).repeat(2 if do_cfg else 1, 1)
        schedule = SDXLEulerScheduler(self.weights.scheduler_config, steps)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn((1, 4, height // 8, width // 8), device="cuda", dtype=torch.float16, generator=generator)
        latents = latents * schedule.init_noise_sigma
        for i, t in enumerate(schedule.timesteps):
            model_input = schedule.scale_model_input(torch.cat((latents, latents)) if do_cfg else latents, i)
            noise = self.unet_infer.infer(self.weights.unet, model_input, t, context, pooled, time_ids)
            if do_cfg:
                base, conditioned = noise.chunk(2)
                noise = base + cfg_scale * (conditioned - base)
            latents = schedule.step(noise, i, latents)
        if return_latents:
            return latents
        decoded = self.vae_infer.infer(self.weights.vae, latents.float() / self.weights.configs["vae"]["scaling_factor"])
        return (decoded / 2 + 0.5).clamp(0, 1)
