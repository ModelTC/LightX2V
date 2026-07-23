import json
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from loguru import logger
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0
PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class LingBotVideoDenoiserInput:
    hidden_states: torch.Tensor


@MODEL_REGISTER("lingbot_video")
class LingBotVideoModel(BaseModel):
    pipeline_cls = None
    vae_scale_factor_temporal = 4
    vae_scale_factor_spatial = 8

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        self.model_path = os.path.abspath(os.path.expanduser(str(model_config["pretrained_model_name_or_path"])))
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", "bf16"))
        self.text_encoder_dtype = get_running_dtype(model_config.get("text_encoder_dtype", "bf16"))
        self.vae_dtype = get_running_dtype(model_config.get("vae_dtype", "fp32"))
        self.load_transformer = bool(model_config.get("load_transformer", True))
        self.load_text_encoder = bool(model_config.get("load_text_encoder", True))
        self.load_vae = bool(model_config.get("load_vae", False))
        self.text_encoder_cpu = bool(model_config.get("text_encoder_cpu", True))
        self.max_sequence_length = int(model_config.get("max_sequence_length", TOKEN_LENGTH))
        self.hidden_state_skip_layer = int(model_config.get("hidden_state_skip_layer", HIDDEN_STATE_SKIP_LAYER))
        self.prompt_template = model_config.get("prompt_template", PROMPT_TEMPLATE)
        self._crop_start = None

        self.text_encoder = None
        self.processor = None
        self.vae = None
        if transformer_only:
            self.load_text_encoder = False
            self.load_vae = False
            if reference_model is not None:
                self.text_encoder = reference_model.text_encoder
                self.processor = reference_model.processor
                self.vae = reference_model.vae
                self._crop_start = reference_model._crop_start

        self.transformer = self._load_transformer() if self.load_transformer else None
        if self.load_text_encoder:
            self.text_encoder, self.processor = self._load_text_components()
        if self.load_vae:
            self.vae = self._load_vae()

    def _load_transformer(self):
        try:
            from .native.lingbot_video import LingBotVideoTransformer3DModel
        except ImportError as exc:
            raise ImportError("LingBot-Video training requires the diffusers APIs used by LingBotVideoTransformer3DModel.") from exc

        transformer = LingBotVideoTransformer3DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.transformer_param_dtype,
            local_files_only=True,
        )
        return transformer.to(self.device)

    def _load_text_components(self):
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError as exc:
            raise ImportError("LingBot-Video prompt encoding requires transformers with Qwen3-VL support.") from exc

        text_path = os.path.join(self.model_path, "text_encoder")
        processor_path = os.path.join(self.model_path, "processor")
        text_device = torch.device("cpu") if self.text_encoder_cpu else self.device
        load_kwargs = {
            "local_files_only": True,
            "attn_implementation": "sdpa",
        }
        try:
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                text_path,
                dtype=self.text_encoder_dtype,
                **load_kwargs,
            )
        except TypeError:
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                text_path,
                torch_dtype=self.text_encoder_dtype,
                **load_kwargs,
            )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        text_encoder.to(text_device)
        processor = Qwen3VLProcessor.from_pretrained(processor_path, local_files_only=True)
        return text_encoder, processor

    def _load_vae(self):
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:
            raise ImportError("LingBot-Video VAE loading requires diffusers.AutoencoderKLWan.") from exc

        vae = AutoencoderKLWan.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
            local_files_only=True,
        )
        vae.requires_grad_(False)
        vae.eval()
        return vae.to(self.device)

    def denoiser_module(self):
        if self.transformer is None:
            raise RuntimeError("LingBot-Video transformer is not loaded. Set model.load_transformer=true.")
        return self.transformer

    def transformer_forward_context(self):
        if self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.running_dtype)
        return nullcontext()

    def add_lora(self, rank, alpha, target_modules):
        if not target_modules:
            raise ValueError("LingBot-Video DMD LoRA requires training.student.lora.target_modules and training.fake.lora.target_modules.")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self._inject_lora(lora_config)

    def _inject_lora(self, lora_config, adapter_name="default"):
        try:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer, adapter_name=adapter_name)
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def _lora_config_for_infer(self):
        lora_config = self.config.get("inference", {}).get("lora_config", {})
        rank = int(lora_config.get("rank", 64))
        alpha = int(lora_config.get("alpha", rank))
        target_modules = lora_config.get("target_modules")
        if not target_modules:
            raise ValueError("LingBot-Video inference.lora_config.target_modules must be set when loading a LoRA.")
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        adapter_name = adapter_name or "default"
        if not hasattr(self.denoiser_module(), "peft_config"):
            self._inject_lora(self._lora_config_for_infer(), adapter_name=adapter_name)

        weight_path = lora_path
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_lora_weights.safetensors")
        raw = load_file(weight_path)
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        incompatible = set_peft_model_state_dict(self.denoiser_module(), peft_state_dict, adapter_name=adapter_name)
        if incompatible and incompatible.unexpected_keys:
            logger.warning("Unexpected keys when loading LingBot-Video LoRA: {}", incompatible.unexpected_keys)
        self._infer_lora_adapter_name = adapter_name

    def unload_lora_for_infer(self):
        adapter_name = getattr(self, "_infer_lora_adapter_name", None)
        if adapter_name is None:
            return

        denoiser = self.denoiser_module()
        if hasattr(denoiser, "delete_adapter"):
            denoiser.delete_adapter(adapter_name)
        elif hasattr(denoiser, "delete_adapters"):
            denoiser.delete_adapters(adapter_name)
        else:
            self.transformer = self._load_transformer()
        self._infer_lora_adapter_name = None

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config.get(
            "reshard_after_forward",
            {"root_reshard": False, "block_reshard": True},
        )
        return [
            {
                "modules": self.transformer.blocks,
                "reshard_after_forward": reshard_config.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config.get("root_reshard", False),
            },
        ]

    def _compute_crop_start(self):
        if self._crop_start is not None:
            return self._crop_start
        marker = "<|USER_INPUT_MARKER|>"
        marked = self.prompt_template.format(marker)
        marker_pos = marked.find(marker)
        if marker_pos < 0:
            self._crop_start = 0
            return self._crop_start
        prefix = self.processor(
            text=marked[:marker_pos],
            images=None,
            videos=None,
            return_tensors="pt",
        )
        self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    @torch.no_grad()
    def encode_prompt_condition(self, prompt):
        if self.text_encoder is None or self.processor is None:
            raise RuntimeError("LingBot-Video text encoder is not loaded. Use cached conditions or set model.load_text_encoder=true.")
        prompts = [prompt] if isinstance(prompt, (str, dict)) else list(prompt)
        prompts = [self._normalize_prompt(text) for text in prompts]
        texts = [self.prompt_template.format(text) for text in prompts]
        inputs = self.processor(
            text=texts,
            images=None,
            videos=None,
            do_resize=False,
            truncation=True,
            max_length=self.max_sequence_length,
            padding="longest",
            return_tensors="pt",
        )
        text_device = torch.device("cpu") if self.text_encoder_cpu else self.device
        inputs = inputs.to(text_device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=True,
        )
        prompt_embed = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embed = prompt_embed[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]
        if prompt_embed.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embed = prompt_embed[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]
        return {
            "prompt_embed": prompt_embed.to(device=self.device, dtype=self.running_dtype),
            "prompt_attention_mask": prompt_mask.to(device=self.device),
        }

    @staticmethod
    def _normalize_prompt(prompt):
        if isinstance(prompt, str):
            return prompt

        prompt_value = prompt

        if isinstance(prompt_value, dict):
            if "caption" in prompt_value:
                prompt_value = prompt_value["caption"]
            else:
                runtime_keys = {
                    "duration",
                    "fps",
                    "height",
                    "width",
                    "num_frames",
                    "resolution",
                    "ratio",
                }
                prompt_value = {key: value for key, value in prompt_value.items() if key not in runtime_keys}
        if isinstance(prompt_value, (dict, list)):
            return json.dumps(prompt_value, ensure_ascii=False, separators=(",", ":"))
        return str(prompt_value)

    def prepare_text_condition(self, condition):
        if not isinstance(condition, dict):
            raise TypeError(f"LingBot-Video cached condition must be a dict, got {type(condition)!r}.")
        if "prompt_embed" not in condition or "prompt_attention_mask" not in condition:
            raise KeyError("LingBot-Video cached condition requires prompt_embed and prompt_attention_mask.")
        prompt_embed = condition["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
        prompt_mask = condition["prompt_attention_mask"].to(device=self.device)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
        return {"prompt_embed": prompt_embed, "prompt_attention_mask": prompt_mask}

    def encode_condition(self, sample):
        conditioning = sample["conditioning"]
        positive = conditioning.get("positive")
        if positive is not None:
            return self.prepare_text_condition(positive)
        return self.encode_prompt_condition(conditioning.get("prompt", ""))

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        return LingBotVideoDenoiserInput(hidden_states=noisy_latent)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        hidden_states = denoiser_input.hidden_states.to(device=self.device, dtype=self.running_dtype)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.unsqueeze(0)
        sigma = timestep_or_sigma.to(device=self.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(hidden_states.shape[0])
        elif sigma.shape[0] == 1 and hidden_states.shape[0] > 1:
            sigma = sigma.expand(hidden_states.shape[0])
        timestep = sigma * 1000.0
        prompt_embed = condition["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
        prompt_mask = condition["prompt_attention_mask"].to(device=self.device)
        with self.transformer_forward_context():
            return self.transformer(
                hidden_states,
                timestep,
                prompt_embed,
                encoder_attention_mask=prompt_mask,
                return_dict=False,
            )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return prediction

    def cfg_on_denoiser_output(self):
        return False

    def _latent_channels(self):
        return int(self.transformer.config.in_channels)

    def dmd_latent_shape(self, batch_size, height, width):
        num_frames = int(self.config.get("inference", {}).get("num_frames", 9))
        if num_frames != 1 and (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            raise ValueError(f"LingBot-Video num_frames must be 1 or 4n+1, got {num_frames}.")
        if int(height) % 16 != 0 or int(width) % 16 != 0:
            raise ValueError(f"LingBot-Video height and width must be multiples of 16, got {height}x{width}.")
        return (
            int(batch_size),
            self._latent_channels(),
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

    def prepare_infer_latents(self, height, width, generator=None):
        shape = self.dmd_latent_shape(1, height, width)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def _vae_latent_to_dit(self, latents):
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std_inv = (1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32)).view(1, -1, 1, 1, 1)
        return (latents.float() - mean) * std_inv

    def _dit_latent_to_vae(self, latents):
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return latents.float() * std + mean

    @torch.no_grad()
    def encode_to_latent(self, sample):
        latent = sample["inputs"].get("latents")
        if latent is not None:
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            return latent.to(device=self.device, dtype=self.running_dtype)
        if self.vae is None:
            raise RuntimeError("LingBot-Video VAE is not loaded. Use cached latents or set model.load_vae=true.")
        video = sample["inputs"].get("video")
        if video is None:
            raise KeyError("LingBot-Video encode_to_latent expects inputs.video or inputs.latents.")
        video = video.to(device=self.device, dtype=torch.float32)
        encoded = self.vae.encode(video)
        latents = encoded.latent_dist.sample() if hasattr(encoded, "latent_dist") else encoded[0]
        return self._vae_latent_to_dit(latents).to(dtype=self.running_dtype)

    @torch.no_grad()
    def decode_latent(self, latent):
        if self.vae is None:
            raise RuntimeError("LingBot-Video VAE is not loaded. Set model.load_vae=true for decoding.")
        vae_latent = self._dit_latent_to_vae(latent).to(device=self.device, dtype=torch.float32)
        if vae_latent.ndim == 5:
            vae_latent = vae_latent.contiguous(memory_format=torch.channels_last_3d)
        decoded = self.vae.decode(vae_latent)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = (frames.float().clamp(-1, 1) + 1.0) / 2.0
        frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy()
        return [video for video in frames]

    def assemble_pipeline(self, scheduler=None):
        raise NotImplementedError("LingBot-Video training currently supports DMD only; native training inference is disabled.")

    def get_pipeline_infer_kwargs(self, infer_config):
        return {
            "height": infer_config.get("default_height", 480),
            "width": infer_config.get("default_width", 832),
            "num_frames": infer_config.get("num_frames", 81),
            "num_inference_steps": infer_config.get("num_inference_steps", 40),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 3.0),
            "shift": infer_config.get("shift", 3.0),
        }

    def log_model_structure(self):
        logger.info("[model] class={}", self.__class__.__name__)
        logger.info("[model] LingBot-Video transformer blocks={} experts_per_block={}", len(self.transformer.blocks), self.transformer.config.num_experts)
