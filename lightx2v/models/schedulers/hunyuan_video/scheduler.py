import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn import functional as F

from lightx2v.models.schedulers.scheduler import BaseScheduler

from .posemb_layers import get_nd_rotary_pos_embed


class HunyuanVideo15Scheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config.get("run_device", "cuda"))
        self.reverse = True
        self.num_train_timesteps = 1000
        self.sample_shift = self.config["sample_shift"]
        self.reorg_token = False
        self.keep_latents_dtype_in_scheduler = True
        self.sample_guide_scale = self.config["sample_guide_scale"]
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.bfloat16)
        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)
        self.multitask_mask = self.get_task_mask(self.config["task"], latent_shape[-3])
        self.cond_latents_concat, self.mask_concat = self._prepare_cond_latents_and_mask(self.config["task"], image_encoder_output["cond_latents"], self.latents, self.multitask_mask, self.reorg_token)
        self.cos_sin = self.prepare_cos_sin((latent_shape[1], latent_shape[2], latent_shape[3]))

    def prepare_latents(self, seed, latent_shape, dtype=torch.bfloat16):
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.latents = torch.randn(
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def set_timesteps(self, num_inference_steps, device, shift):
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas, shift)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(dtype=torch.float32, device=device)

    def sd3_time_shift(self, t: torch.Tensor, shift):
        return (shift * t) / (1 + (shift - 1) * t)

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _prepare_cond_latents_and_mask(self, task_type, cond_latents, latents, multitask_mask, reorg_token):
        """
        Prepare multitask mask training logic.

        Args:
            task_type: Type of task ("i2v" or "t2v")
            cond_latents: Conditional latents tensor
            latents: Main latents tensor
            multitask_mask: Multitask mask tensor
            reorg_token: Whether to reorganize tokens

        Returns:
            tuple: (latents_concat, mask_concat) - may contain None values
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == "i2v":
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            if reorg_token:
                latents_concat = torch.zeros(latents.shape[0], latents.shape[1] // 2, latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
            else:
                latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = self.merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

        return latents_concat, mask_concat

    def merge_tensor_by_mask(self, tensor_1, tensor_2, mask, dim):
        assert tensor_1.shape == tensor_2.shape
        # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
        masked_indices = torch.nonzero(mask).squeeze(1)
        tmp = tensor_1.clone()
        if dim == 0:
            tmp[masked_indices] = tensor_2[masked_indices]
        elif dim == 1:
            tmp[:, masked_indices] = tensor_2[:, masked_indices]
        elif dim == 2:
            tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
        return tmp

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.latents = sample + model_output * dt

    def prepare_cos_sin(self, rope_sizes):
        target_ndim = 3
        head_dim = self.config["hidden_size"] // self.config["heads_num"]
        rope_dim_list = self.config["rope_dim_list"]
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(rope_dim_list, rope_sizes, theta=self.config["rope_theta"], use_real=True, theta_rescale_factor=1, device=self.device)
        cos_half = freqs_cos[:, ::2].contiguous()
        sin_half = freqs_sin[:, ::2].contiguous()
        cos_sin = torch.cat([cos_half, sin_half], dim=-1)
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            seqlen = cos_sin.shape[0]
            padding_size = (world_size - (seqlen % world_size)) % world_size
            if padding_size > 0:
                cos_sin = F.pad(cos_sin, (0, 0, 0, padding_size))
            cos_sin = torch.chunk(cos_sin, world_size, dim=0)[cur_rank]
        return cos_sin


class HunyuanVideo15SRScheduler(HunyuanVideo15Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.noise_scale = 0.7

    def prepare(self, seed, latent_shape, lq_latents, upsampler, image_encoder_output=None):
        dtype = lq_latents.dtype
        device = lq_latents.device
        self.prepare_latents(seed, latent_shape, lq_latents, dtype=dtype)
        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)
        self.cos_sin = self.prepare_cos_sin((latent_shape[1], latent_shape[2], latent_shape[3]))

        tgt_shape = latent_shape[-2:]
        bsz = lq_latents.shape[0]
        lq_latents = rearrange(lq_latents, "b c f h w -> (b f) c h w")
        lq_latents = F.interpolate(lq_latents, size=tgt_shape, mode="bilinear", align_corners=False)
        lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)

        lq_latents = upsampler(lq_latents.to(dtype=torch.float32, device=device))
        lq_latents = lq_latents.to(dtype=dtype)

        lq_latents = self.add_noise_to_lq(lq_latents, self.noise_scale)

        condition = self.get_condition(lq_latents, image_encoder_output["cond_latents"], self.config["task"])
        c = lq_latents.shape[1]

        zero_condition = condition.clone()
        zero_condition[:, c + 1 : 2 * c + 1] = torch.zeros_like(lq_latents)
        zero_condition[:, 2 * c + 1] = 0

        self.condition = condition
        self.zero_condition = zero_condition

    def prepare_latents(self, seed, latent_shape, lq_latents, dtype=torch.bfloat16):
        self.generator = torch.Generator(device=lq_latents.device).manual_seed(seed)
        self.latents = torch.randn(
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=lq_latents.device,
            generator=self.generator,
        )

    def get_condition(self, lq_latents, img_cond, task):
        """
        latents: shape (b c f h w)
        """
        b, c, f, h, w = self.latents.shape
        cond = torch.zeros([b, c * 2 + 2, f, h, w], device=lq_latents.device, dtype=lq_latents.dtype)

        cond[:, c + 1 : 2 * c + 1] = lq_latents
        cond[:, 2 * c + 1] = 1
        if "t2v" in task:
            return cond
        elif "i2v" in task:
            cond[:, :c, :1] = img_cond
            cond[:, c + 1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task: {task}")

    def add_noise_to_lq(self, lq_latents, strength=0.7):
        def expand_dims(tensor: torch.Tensor, ndim: int):
            shape = tensor.shape + (1,) * (ndim - tensor.ndim)
            return tensor.reshape(shape)

        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0], device=lq_latents.device) * strength
        t = expand_dims(timestep, lq_latents.ndim)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise
