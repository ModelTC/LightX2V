import torch

from lightx2v.models.networks.minimax_h3.packing import audio_latent_num_frames, build_row_timesteps, patchify_video_latents, video_latent_num_frames
from lightx2v.models.networks.minimax_h3.packing_ref2av import build_ref2av_packed_sequence
from lightx2v.models.networks.minimax_h3_causal.streaming import MiniMaxH3StreamingPlan
from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler, _layout_to_device
from lightx2v_platform.base.global_var import AI_DEVICE


class MiniMaxH3CausalScheduler(MiniMaxH3Scheduler):
    """Euler denoising of one video chunk with fixed, clean stereo audio."""

    def __init__(self, config):
        super().__init__(config)
        if self.step_update != "training_euler":
            raise ValueError("Causal H3 requires h3_step_update='training_euler'")
        self.cache = None
        # Copy the actual device schedule once, preserving its FP32 rounding
        # without synchronizing on a CUDA scalar in every denoising step.
        self.video_timesteps_cpu = self.video_timesteps.cpu().tolist()
        self.prepared_inputs = None

    def prepare(self, seed, num_frames, height, width, text_token_tags, *, condition_video_latents, references, **kwargs):
        if num_frames < 22 or (num_frames - 5) % 17 or height % 32 or width % 32:
            raise ValueError("Causal H3 requires 17*n+5 frames (n >= 1) and canvas dimensions divisible by 32")
        self.num_frames, self.height, self.width = num_frames, height, width
        self.num_latent_frames = video_latent_num_frames(num_frames)
        self.latent_height, self.latent_width = height // 16, width // 16
        self.num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = tuple(self.config["patch_size"])
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(seed))
        condition = []
        for clean in condition_video_latents:
            clean = clean.to(AI_DEVICE)
            noise = torch.randn(clean.shape, generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
            condition.append(0.999 * patchify_video_latents(clean, patch_size) + 0.001 * patchify_video_latents(noise, patch_size))
        self.condition_video = torch.cat(condition)
        video_noise = torch.randn((1, 24, self.num_latent_frames, self.latent_height, self.latent_width), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
        self.video_noise = patchify_video_latents(video_noise, patch_size).cpu()
        # Consume the same random draws as the upstream joint state constructor.
        torch.randn((1, 2 * self.num_audio_latents, 32), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
        full_layout = build_ref2av_packed_sequence(text_token_tags.cpu(), references, self.num_latent_frames, self.latent_height, self.latent_width, self.num_audio_latents, patch_size)
        self.plan = MiniMaxH3StreamingPlan(full_layout, self.num_latent_frames, self.num_audio_latents, (self.latent_height // 2) * (self.latent_width // 2), self.config["ar_config"])
        self.num_condition_video_rows = full_layout.num_condition_video_rows
        self.num_condition_audio_rows = 0
        self.prepare_condition()

    def prepare_condition(self):
        self.prepared_inputs = None
        self.condition_phase = True
        self.layout_cpu = self.plan.condition_layout
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        self.key_position_ids = self.layout.position_ids
        self.video_latents = self.condition_video
        self.audio_latents = self.condition_video.new_empty((0, 32))
        self.cache_start = self.evicted_rows = 0

    def prepare_chunk(self, index, clean_audio):
        self.prepared_inputs = None
        self.condition_phase = False
        self.chunk_index = index
        self.layout_cpu = self.plan.chunk_layout(index)
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        prefix = self.plan.prefix_positions(index)
        self.key_position_ids = torch.cat((prefix, self.layout_cpu.position_ids)).to(AI_DEVICE)
        self.cache_start = prefix.shape[0]
        previous_end = self.plan.condition_layout.sequence_length if index == 0 else self.plan.prefix_positions(index - 1).shape[0] + self.plan.chunks[index - 1].positions.shape[0]
        self.evicted_rows = previous_end - self.cache_start
        self.video_latents = self.video_noise[self.plan.chunks[index].video_rows].to(AI_DEVICE)
        self.audio_latents = clean_audio.to(AI_DEVICE)

    def step_pre(self, step_index):
        self.step_index = int(step_index)
        self.cache.current_step = self.step_index
        unique, inverse = build_row_timesteps(self.layout_cpu, self.video_timesteps_cpu[self.step_index], 1.0, condition_video_timestep=0.999)
        self.unique_timesteps_cpu = unique
        self.unique_timesteps = unique.to(AI_DEVICE)
        self.timestep_indices = inverse.to(AI_DEVICE)

    def step_post(self):
        self.video_latents = self._step(self.video_latents, self.video_noise_pred, self.video_timesteps[self.step_index], self.video_sigmas, self.step_index, self.step_update)
        # Audio remains exactly the causal encoder output for every step.

    def clear(self):
        super().clear()
        for name in ("condition_video", "video_noise", "plan", "key_position_ids", "cache", "prepared_inputs"):
            setattr(self, name, None)
