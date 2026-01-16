import inspect
import json
import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate shift for timestep scheduling based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """Create random tensors on the desired device with the desired dtype."""
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout) for i in range(batch_size)]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 256,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None):
    """Prepare position IDs for LongCat.

    Args:
        modality_id: 0 for text, 1 for image
        type: "text" or "image"
        start: Starting position (row, col)
        num_token: Number of tokens (for text)
        height, width: Image dimensions (for image)

    Returns:
        Position IDs tensor of shape [L, 3] where columns are [modality_id, row_pos, col_pos]
    """
    if type == "text":
        assert num_token is not None
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == "image":
        assert height is not None and width is not None
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(f'Unknown type {type}, only support "text" or "image".')
    return pos_ids


def get_1d_rotary_pos_embed(dim, pos, theta=10000, repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float64):
    """Get 1D rotary position embeddings.

    Args:
        dim: Embedding dimension
        pos: Position indices [L]
        theta: Base frequency
        repeat_interleave_real: Whether to interleave real/imag
        use_real: Return real (cos, sin) instead of complex
        freqs_dtype: Dtype for frequency computation

    Returns:
        (cos, sin) tuple if use_real else complex freqs
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos, freqs)  # Don't cast pos to freqs_dtype - match diffusers

    if use_real:
        # Match diffusers: always return float32
        cos = freqs.cos().repeat_interleave(2, dim=-1).float() if repeat_interleave_real else torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        sin = freqs.sin().repeat_interleave(2, dim=-1).float() if repeat_interleave_real else torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return cos, sin
    else:
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


class LongCatImagePosEmbed(nn.Module):
    """Position embedding for LongCat Image model.

    Uses 3-axis rotary embeddings: [modality, row, col]
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = [16, 56, 56]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings from position IDs.

        Args:
            ids: Position IDs tensor [L, 3]

        Returns:
            (freqs_cos, freqs_sin) tuple, each of shape [L, sum(axes_dim)]
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class LongCatImageScheduler(BaseScheduler):
    """Scheduler for LongCat Image model.

    Handles:
    - Latent preparation and packing
    - Timestep scheduling with flow matching
    - Position embedding computation
    - CFG (classifier-free guidance) support
    - CFG renormalization
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        scheduler_path = config.get("scheduler_path", os.path.join(config["model_path"], "scheduler"))
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)

        with open(os.path.join(config["model_path"], "scheduler", "scheduler_config.json"), "r") as f:
            self.scheduler_config = json.load(f)

        self.dtype = GET_DTYPE()
        self.sample_guide_scale = self.config.get("sample_guide_scale", 4.0)
        self.enable_cfg_renorm = self.config.get("enable_cfg_renorm", True)
        self.cfg_renorm_min = self.config.get("cfg_renorm_min", 0.0)

        # Position embedding
        axes_dims_rope = config.get("axes_dims_rope", [16, 56, 56])
        self.pos_embed = LongCatImagePosEmbed(theta=10000, axes_dim=axes_dims_rope)

        # VAE parameters
        self.vae_scale_factor = config.get("vae_scale_factor", 8)
        self.patch_size = config.get("patch_size", 1)
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.in_channels = config.get("transformer_in_channels", config.get("in_channels", 64))

        # Sequence parallel
        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        """Pack latents from [B, C, H, W] to [B, (H//2)*(W//2), C*4] for transformer input.

        Uses 2x2 packing like Flux: combines 2x2 spatial patches into channel dimension.
        This increases channels from 16 to 64 and reduces spatial dims by half.
        """
        # [B, C, H, W] -> [B, C, H//2, 2, W//2, 2]
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        # -> [B, H//2, W//2, C, 2, 2]
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        # -> [B, (H//2)*(W//2), C*4]
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    def prepare_latents(self, input_info):
        """Prepare random latents for denoising."""
        self.input_info = input_info
        shape = input_info.target_shape
        # target_shape is already in latent space: (B, C, H, W)
        # where C=16 (VAE latent channels), H and W are latent dimensions
        vae_latent_channels = shape[1]  # 16
        latent_height = shape[-2]
        latent_width = shape[-1]

        # Generate random latents with VAE channel count (16)
        latent_shape = (1, vae_latent_channels, latent_height, latent_width)
        latents = randn_tensor(latent_shape, generator=self.generator, device=AI_DEVICE, dtype=self.dtype)

        # Pack latents for transformer: [B, 16, H, W] -> [B, (H//2)*(W//2), 64]
        latents = self._pack_latents(latents, 1, vae_latent_channels, latent_height, latent_width)

        self.latents = latents
        # Store packed spatial dimensions (half of latent dims due to 2x2 packing)
        self.latent_height = latent_height // 2
        self.latent_width = latent_width // 2
        self.noise_pred = None

    def set_timesteps(self):
        """Set timesteps for the scheduler."""
        sigmas = np.linspace(1.0, 1 / self.config["infer_steps"], self.config["infer_steps"])
        image_seq_len = self.latents.shape[1]

        mu = calculate_shift(
            image_seq_len,
            self.scheduler_config.get("base_image_seq_len", 256),
            self.scheduler_config.get("max_image_seq_len", 4096),
            self.scheduler_config.get("base_shift", 0.5),
            self.scheduler_config.get("max_shift", 1.15),
        )

        num_inference_steps = self.config["infer_steps"]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            AI_DEVICE,
            sigmas=sigmas,
            mu=mu,
        )

        self.timesteps = timesteps
        self.infer_steps = num_inference_steps

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self.num_warmup_steps = num_warmup_steps

    def prepare(self, input_info):
        """Prepare scheduler for inference."""
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(input_info.seed)
        self.prepare_latents(input_info)
        self.set_timesteps()

        # Compute position IDs and rotary embeddings
        # Following diffusers LongCatImagePipeline:
        # - Text: modality_id=0, start=(0, 0)
        # - Image: modality_id=1, start=(tokenizer_max_length, tokenizer_max_length)
        txt_seq_len = input_info.txt_seq_lens[0]
        tokenizer_max_length = txt_seq_len  # 512

        txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=txt_seq_len)
        img_ids = prepare_pos_ids(modality_id=1, type="image", start=(tokenizer_max_length, tokenizer_max_length), height=self.latent_height, width=self.latent_width)

        # Concatenate [txt, img] position IDs
        # Note: pos_embed expects float32 ids for accurate rope computation
        ids = torch.cat([txt_ids, img_ids], dim=0).to(AI_DEVICE, dtype=torch.float32)
        freqs_cos, freqs_sin = self.pos_embed(ids)

        # Convert to flashinfer format if needed
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            # pos_embed returns interleaved format: [c0, c0, c1, c1, ...]
            # flashinfer needs: [c0, c1, ..., s0, s1, ...]
            cos_half = freqs_cos[:, ::2].contiguous()  # [L, D/2]
            sin_half = freqs_sin[:, ::2].contiguous()  # [L, D/2]
            self.image_rotary_emb = torch.cat([cos_half, sin_half], dim=-1)  # [L, D]
        else:
            self.image_rotary_emb = (freqs_cos, freqs_sin)

        # Handle CFG: prepare negative embeddings rotary
        if self.config.get("enable_cfg", True):
            neg_txt_seq_len = input_info.txt_seq_lens[1] if len(input_info.txt_seq_lens) > 1 else txt_seq_len
            neg_txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=neg_txt_seq_len)
            neg_ids = torch.cat([neg_txt_ids, img_ids], dim=0).to(AI_DEVICE, dtype=torch.float32)
            neg_freqs_cos, neg_freqs_sin = self.pos_embed(neg_ids)

            if self.config.get("rope_type", "flashinfer") == "flashinfer":
                neg_cos_half = neg_freqs_cos[:, ::2].contiguous()
                neg_sin_half = neg_freqs_sin[:, ::2].contiguous()
                self.negative_image_rotary_emb = torch.cat([neg_cos_half, neg_sin_half], dim=-1)
            else:
                self.negative_image_rotary_emb = (neg_freqs_cos, neg_freqs_sin)

    def step_pre(self, step_index):
        """Prepare for a single denoising step."""
        super().step_pre(step_index)

        # Compute timestep embedding input
        # Note: scheduler.timesteps are already in 0-1000 range
        # (diffusers pipeline divides by 1000, then transformer multiplies by 1000)
        timestep_input = torch.tensor([self.timesteps[self.step_index]], device=AI_DEVICE, dtype=self.dtype)
        self.timesteps_proj = get_timestep_embedding(timestep_input).to(self.dtype)

    def step_post(self):
        """Process after model forward to update latents."""
        t = self.timesteps[self.step_index]
        latents = self.scheduler.step(self.noise_pred, t, self.latents, return_dict=False)[0]
        self.latents = latents

    def prepare_i2i(self, input_info, input_image, vae):
        """Prepare scheduler for I2I (image editing) inference.

        Args:
            input_info: Input information
            input_image: Input image tensor [B, C, H, W] (preprocessed)
            vae: VAE model for encoding
        """
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(input_info.seed)
        self.vae = vae
        self.prepare_latents(input_info)
        self.set_timesteps()

        # Encode input image to latents
        self.input_image_latents = self._encode_image(input_image)

        # Compute position IDs and rotary embeddings
        txt_seq_len = input_info.txt_seq_lens[0]
        tokenizer_max_length = txt_seq_len  # 512

        # Text: modality_id=0
        txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=txt_seq_len)

        # Output image: modality_id=1
        output_img_ids = prepare_pos_ids(modality_id=1, type="image", start=(tokenizer_max_length, tokenizer_max_length), height=self.latent_height, width=self.latent_width)

        # Input image: modality_id=2
        input_img_ids = prepare_pos_ids(modality_id=2, type="image", start=(tokenizer_max_length, tokenizer_max_length), height=self.latent_height, width=self.latent_width)

        # Combined image IDs: [output_img, input_img]
        combined_img_ids = torch.cat([output_img_ids, input_img_ids], dim=0)

        # Concatenate [txt, output_img, input_img] position IDs
        ids = torch.cat([txt_ids, combined_img_ids], dim=0).to(AI_DEVICE, dtype=torch.float32)
        freqs_cos, freqs_sin = self.pos_embed(ids)

        # Convert to flashinfer format if needed
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            cos_half = freqs_cos[:, ::2].contiguous()
            sin_half = freqs_sin[:, ::2].contiguous()
            self.image_rotary_emb = torch.cat([cos_half, sin_half], dim=-1)
        else:
            self.image_rotary_emb = (freqs_cos, freqs_sin)

        # Store output sequence length for later truncation
        self.output_seq_len = self.latents.shape[1]

        # Handle CFG: prepare negative embeddings rotary
        if self.config.get("enable_cfg", True):
            neg_txt_seq_len = input_info.txt_seq_lens[1] if len(input_info.txt_seq_lens) > 1 else txt_seq_len
            neg_txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=neg_txt_seq_len)
            neg_ids = torch.cat([neg_txt_ids, combined_img_ids], dim=0).to(AI_DEVICE, dtype=torch.float32)
            neg_freqs_cos, neg_freqs_sin = self.pos_embed(neg_ids)

            if self.config.get("rope_type", "flashinfer") == "flashinfer":
                neg_cos_half = neg_freqs_cos[:, ::2].contiguous()
                neg_sin_half = neg_freqs_sin[:, ::2].contiguous()
                self.negative_image_rotary_emb = torch.cat([neg_cos_half, neg_sin_half], dim=-1)
            else:
                self.negative_image_rotary_emb = (neg_freqs_cos, neg_freqs_sin)

    def _encode_image(self, image):
        """Encode input image using VAE.

        Args:
            image: Input image tensor [B, C, H, W]

        Returns:
            Packed latents [B, L, 64]
        """
        with torch.no_grad():
            # Encode image
            latents = self.vae.model.encode(image.to(self.vae.model.dtype)).latent_dist.mode()

            # Apply scaling: (latents - shift_factor) * scaling_factor
            latents = (latents - self.vae.shift_factor) * self.vae.scaling_factor

            # Pack latents
            batch_size = latents.shape[0]
            num_channels = latents.shape[1]
            height = latents.shape[2]
            width = latents.shape[3]
            latents = self._pack_latents(latents, batch_size, num_channels, height, width)

        return latents.to(self.dtype)

    def apply_cfg(self, noise_pred_cond, noise_pred_uncond):
        """Apply classifier-free guidance with optional renormalization.

        Args:
            noise_pred_cond: Conditional prediction
            noise_pred_uncond: Unconditional prediction

        Returns:
            Combined noise prediction
        """
        noise_pred = noise_pred_uncond + self.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

        if self.enable_cfg_renorm:
            # CFG renormalization: rescale to match conditional prediction norm
            noise_pred_cond_norm = torch.linalg.vector_norm(noise_pred_cond, dim=-1, keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, dim=-1, keepdim=True)
            renorm_factor = noise_pred_cond_norm / noise_pred_norm.clamp(min=1e-8)
            renorm_factor = renorm_factor.clamp(min=self.cfg_renorm_min, max=1.0)
            noise_pred = noise_pred * renorm_factor

        return noise_pred
