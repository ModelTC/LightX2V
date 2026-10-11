import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class EulerScheduler(WanScheduler):
    def __init__(self, config):
        self.config = config
        self.latents = None
        self.generator = None
        self.step_index = 0
        self.flag_df = False
        self.transformer_infer = None
        self.infer_condition = True  # cfg status
        self.keep_latents_dtype_in_scheduler = False
        self.sample_shift = self.config["sample_shift"]
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.patch_size = (1, 2, 2)
        self.shift = 1
        self.num_train_timesteps = 1000
        self.disable_corrector = []
        self.solver_order = 2
        self.noise_pred = None
        self.sample_guide_scale = self.config["sample_guide_scale"]
        self.head_size = self.config["dim"] // self.config["num_heads"]
        self._audio_t_emb_cache = {}
        self._audio_t_emb_cache_sig = None

    def set_audio_adapter(self, audio_adapter):
        self.audio_adapter = audio_adapter
        self._audio_t_emb_cache.clear()

    def step_pre(self, step_index):
        super().step_pre(step_index)
        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to(AI_DEVICE)
        self.audio_adapter_t_emb = self.audio_adapter.time_embedding(self.timestep_input).unflatten(1, (3, -1))
        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to("cpu")

    def prepare(self, seed, latent_shape, infer_steps=None, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        if infer_steps is not None:
            self.infer_steps = infer_steps
        else:
            self.infer_steps = self.config["infer_steps"]

        timesteps = np.linspace(self.num_train_timesteps, 0, self.infer_steps + 1, dtype=np.float32)

        self.timesteps_ori = torch.from_numpy(timesteps).to(dtype=torch.float32, device=AI_DEVICE)

        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)

        self.timesteps = self.sigmas * self.num_train_timesteps

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma = self.sigmas[self.step_index].to(sample.device, sample.dtype)
        sigma_next = self.sigmas[self.step_index + 1].to(sample.device, sample.dtype)
        x_t_next = sample + (sigma_next - sigma) * model_output
        self.latents = x_t_next

    def reset(self, seed, latent_shape):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)


class WanAudioARScheduler(EulerScheduler):
    def _get_timesteps(self, num_steps, max_steps: int = 1000):
        return np.linspace(max_steps, 0, num_steps + 1, dtype=np.float32)

    def set_shift(self, shift: float = 1.0):
        self.sigmas = self.timesteps_ori / self.num_train_timesteps
        self.sigmas = shift / (shift + (1 / self.sigmas - 1))
        self.timesteps = self.sigmas * self.num_train_timesteps
        self._shift = shift
        return self

    def set_timesteps(self, num_inference_steps: int, device=None):
        cache_sig = (int(num_inference_steps), float(self._shift), str(device or AI_DEVICE))
        if self._audio_t_emb_cache_sig != cache_sig:
            self._audio_t_emb_cache.clear()
            self._audio_t_emb_cache_sig = cache_sig
        timesteps = self._get_timesteps(num_steps=num_inference_steps, max_steps=self.num_train_timesteps)
        self.timesteps_ori = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device or AI_DEVICE)
        self.set_shift(self._shift)
        self._step_index = None
        self._begin_index = None
        return self

    @property
    def source_step_index(self):
        return getattr(self, "_step_index", None)

    def _init_step_index(self, timestep):
        timestep = torch.as_tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
        indices = (self.timesteps == timestep).nonzero()
        if indices.numel() == 0:
            indices = torch.argmin((self.timesteps - timestep).abs()).reshape(1, 1)
        self._step_index = int(indices.flatten()[0].item())

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError("Passing integer indices as timesteps is not supported. Pass one of the scheduler timesteps instead.")
        if self.source_step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        sigma = self.sigmas[self.source_step_index].to(sample.device, sample.dtype)
        sigma_next = self.sigmas[self.source_step_index + 1].to(sample.device, sample.dtype)
        x0 = sample - model_output * sigma
        x_t_next = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return x_t_next, x0

    def prepare(self, seed, latent_shape, infer_steps=None, image_encoder_output=None):
        self.generator = torch.Generator("cpu").manual_seed(seed)
        self.latents = torch.randn(
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=GET_DTYPE(),
            device="cpu",
            generator=self.generator,
        )

        self.infer_steps = infer_steps if infer_steps is not None else self.config["infer_steps"]
        self._shift = self.sample_shift
        self.set_timesteps(self.infer_steps, device=AI_DEVICE)
        self.noise = self.latents
        self.noise_pred = torch.zeros_like(self.latents)
        self.chunk_size = int(self.config.get("ar_config", {}).get("num_frame_per_chunk", 1))
        self.num_chunks = self.latents.shape[1] // self.chunk_size

    def step_pre_ref(self, step_index, ref_frames):
        self.step_index = step_index
        self.seg_index = 0
        self.timestep_input = torch.full((1, ref_frames), self.timesteps[step_index], dtype=torch.float32, device=AI_DEVICE)
        self._set_audio_t_emb()

    def step_pre(self, segment_idx, step_index, xt):
        self.step_index = step_index
        self.seg_index = segment_idx
        self.latents = xt
        frames = xt.shape[1]
        self.timestep_input = torch.full((1, frames), self.timesteps[step_index], dtype=torch.float32, device=AI_DEVICE)
        self._set_audio_t_emb()

    def step_post(self, xt):
        timestep = self.timesteps[self.step_index]
        x_t_next, _ = self.step(self.noise_pred, timestep, xt)
        return x_t_next.to(xt.dtype)

    def _set_audio_t_emb(self):
        cache_key = (int(self.step_index), tuple(self.timestep_input.shape), self.timestep_input.device.type, str(self.timestep_input.device))
        cached = self._audio_t_emb_cache.get(cache_key)
        if cached is not None:
            self.audio_adapter_t_emb = cached
            return

        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to(AI_DEVICE)
        self.audio_adapter_t_emb = self.audio_adapter.time_embedding(self.timestep_input.flatten()).unflatten(1, (3, -1))
        if self.audio_adapter.cpu_offload:
            self.audio_adapter.time_embedding.to("cpu")
        self._audio_t_emb_cache[cache_key] = self.audio_adapter_t_emb
