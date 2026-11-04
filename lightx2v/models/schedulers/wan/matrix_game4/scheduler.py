"""Matrix-Game-4 chunk scheduling with the released flow/RNG contract."""

import torch

from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


def validate_steps(steps):
    values = tuple(int(value) for value in steps)
    if not values or values[0] != 1000 or any(not 0 < value <= 1000 for value in values) or any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("Matrix-Game-4 timesteps must decrease strictly from 1000")
    return values


class WanMatrixGame4Scheduler(WanSFScheduler):
    def __init__(self, config):
        ar = config["ar_config"]
        if ar.get("num_frame_per_chunk", 1) != 1:
            raise ValueError("Matrix-Game-4 uses one latent frame per chunk")
        self.steps = validate_steps(ar.get("denoising_step_list", (1000, 500)))
        self.first_steps = validate_steps(ar.get("first_denoising_step_list", self.steps))
        self.sink_mode = ar.get("sink_mode", "kv_only")
        if self.sink_mode not in ("kv_only", "first_frame"):
            raise ValueError("Matrix-Game-4 sink_mode must be kv_only or first_frame")
        reference_frames = int(self.sink_mode == "kv_only")
        if ar.get("ref_num_frames", reference_frames) != reference_frames:
            raise ValueError("Matrix-Game-4 ref_num_frames must be 1 for kv_only and 0 for first_frame")
        ar["ref_num_frames"] = reference_frames
        if config.get("sample_shift", 5.0) != 5.0:
            raise ValueError("Released Matrix-Game-4 checkpoints require sample_shift=5")
        ar["num_frame_per_chunk"] = 1
        ar["denoising_step_list"] = list(self.steps)
        super().__init__(config)
        self.clear()

    def clear(self):
        super().clear()
        self.control_latents = None
        self.control_latents_input = None
        self.reference_latents = None
        self.initial_generator = None
        self.racing_generator = None
        self.renoise_generator = None
        self.is_reference_prefill = False
        self._prepared_segment = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        controls = image_encoder_output["control_latents"]
        if tuple(controls.shape) != tuple(latent_shape):
            raise ValueError("Matrix-Game-4 control latents must match the output latent shape")
        self.control_latents = controls.to(device=AI_DEVICE, dtype=self.dtype)
        reference = image_encoder_output.get("reference_latents")
        if reference is not None and tuple(reference.shape) != (latent_shape[0], 1, *latent_shape[2:]):
            raise ValueError("Matrix-Game-4 reference must contain one latent frame")
        self.reference_latents = None if reference is None else reference.to(device=AI_DEVICE, dtype=self.dtype)
        self.latents = torch.empty(latent_shape, device=AI_DEVICE, dtype=self.dtype)
        self.noise_pred = torch.zeros_like(self.latents)
        if self.sink_mode == "first_frame" and self.reference_latents is not None:
            self.latents[:, :1].copy_(self.reference_latents)
        self.initial_generator = torch.Generator("cpu").manual_seed(seed + 1)
        self.racing_generator = torch.Generator("cpu").manual_seed(seed)
        self.renoise_generator = torch.Generator(AI_DEVICE).manual_seed(seed)
        sigma = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
        sigma = 5.0 * sigma / (1 + 4.0 * sigma)
        self.sigmas_sf = sigma.to(AI_DEVICE)
        self.timesteps_sf = (sigma * 1000).to(AI_DEVICE)
        timesteps = torch.cat((sigma * 1000, sigma.new_zeros(1)))
        self._normal_timesteps = timesteps[[1000 - value for value in self.steps]].tolist()
        self._first_timesteps = timesteps[[1000 - value for value in self.first_steps]].tolist()
        self._prepared_segment = None
        self.stream_output = None
        self.is_reference_prefill = False

    def prepare_segment(self, seg_index):
        reference_first = self.sink_mode == "first_frame" and self.reference_latents is not None and seg_index == 0
        offset = int(self.reference_latents is not None and self.sink_mode == "kv_only")
        use_first = seg_index == 0 if self.sink_mode == "first_frame" else seg_index + offset < 2
        self.denoising_step_list = self._first_timesteps if use_first else self._normal_timesteps
        self.infer_steps = 0 if reference_first else len(self.denoising_step_list)
        if reference_first:
            self.stream_output = self.latents[:, :1]
        self.seg_index = seg_index

    def prepare_reference(self):
        if self.reference_latents is None or self.sink_mode != "kv_only":
            raise ValueError("Only a KV-only reference can be prefetched outside the output sequence")
        self.seg_index, self.step_index = 0, 0
        self.is_rerun = self.is_reference_prefill = True
        self.latents_input = self.reference_latents
        self.control_latents_input = None
        self.timestep_input = torch.zeros((1, 1), device=AI_DEVICE, dtype=torch.float32)

    def step_pre(self, seg_index, step_index, is_rerun=False):
        if not is_rerun and self._prepared_segment != seg_index:
            self.prepare_segment(seg_index)
            generator = self.racing_generator if self.sink_mode == "first_frame" and seg_index > 0 else self.initial_generator
            shape = (1, self.latents.shape[0], *self.latents.shape[2:])
            noise = torch.randn(shape, generator=generator, dtype=torch.float32)
            self.latents[:, seg_index : seg_index + 1].copy_(noise.transpose(0, 1).to(device=AI_DEVICE, dtype=self.dtype))
            self._prepared_segment = seg_index
        super().step_pre(seg_index, step_index, is_rerun)
        self.is_reference_prefill = is_rerun and self.sink_mode == "first_frame" and seg_index == 0 and self.reference_latents is not None
        self.control_latents_input = None if is_rerun else self.control_latents[:, seg_index : seg_index + 1]

    def _step_post_denoise(self, seg_start, seg_end):
        flow = self.noise_pred[:, seg_start:seg_end].transpose(0, 1)
        current = self.latents_input.transpose(0, 1)
        timestep = self.timestep_input.flatten().to(device=current.device, dtype=torch.float64)
        sigma = self._sigma_at_timestep(timestep).reshape(-1, 1, 1, 1).double()
        x0 = (current.double() - sigma * flow.double()).to(flow.dtype)
        if self.step_index + 1 < self.infer_steps:
            next_timestep = current.new_tensor([self.denoising_step_list[self.step_index + 1]], dtype=torch.float32)
            sigma_next = self._sigma_at_timestep(next_timestep).reshape(-1, 1, 1, 1)
            noise = torch.randn(x0.shape, generator=self.renoise_generator, device=current.device, dtype=torch.float32)
            x0 = ((1 - sigma_next) * x0.float() + sigma_next * noise).to(self.dtype)
        self.latents[:, seg_start:seg_end].copy_(x0.transpose(0, 1))
        if self.step_index + 1 == self.infer_steps:
            self.stream_output = self.latents[:, seg_start:seg_end]
