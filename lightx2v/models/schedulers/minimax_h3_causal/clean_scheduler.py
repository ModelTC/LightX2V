"""Joint AV renoise with the v0.1.0 CUDA random stream and clean KV lifecycle."""

import torch

from lightx2v.models.networks.minimax_h3.packing import KEYFRAME_NOISE_AUG, audio_latent_num_frames, build_row_timesteps, patchify_video_latents, video_latent_num_frames
from lightx2v.models.networks.minimax_h3_causal.clean.kv_cache import MiniMaxH3CleanKVCache
from lightx2v.models.networks.minimax_h3_causal.clean.streaming import GLOBAL_KEY, REFERENCE_KEY, CleanAttentionPolicy, build_clean_streaming_plan_from_geometry, chunk_key
from lightx2v.models.schedulers.minimax_h3.scheduler import _layout_to_device
from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


def build_renoise_schedule(config) -> tuple[torch.Tensor, torch.Tensor]:
    """Share the exact v0.1 schedule between inference and AdaLN cache building."""
    if config.get("scheduler_type") != "renoise":
        raise ValueError("H3 v0.1.0 requires scheduler_type='renoise'")
    if config["infer_steps"] < 1:
        raise ValueError("infer_steps must be positive")
    # Construct on CUDA, as in Zoe; CPU schedules can round differently.
    base = torch.linspace(1.0, 0.0, config["infer_steps"] + 1, device=AI_DEVICE)
    video = config["video_flow_shift"] * base / (1 + (config["video_flow_shift"] - 1) * base)
    audio = config["audio_flow_shift"] * base / (1 + (config["audio_flow_shift"] - 1) * base)
    return video, audio


class MiniMaxH3CleanScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.generate_audio = config["task"] == "refa2av"
        self.video_sigmas, self.audio_sigmas = build_renoise_schedule(config)
        self.policy = CleanAttentionPolicy(**config["ar_config"])
        self.video_timesteps_cpu = (1 - self.video_sigmas).cpu().tolist()
        self.audio_timesteps_cpu = (1 - self.audio_sigmas).cpu().tolist()
        self.cache = None
        self.prepared_rope = None
        self.audio_embedding = None
        self.live_stream = False
        self.stream_start_idx = 0

    def prepare(self, seed, num_frames, height, width, text_token_tags, *, condition_video_latents, condition_audio_latents, references, segment_starts, live_stream=False, stream_start_idx=0):
        self.live_stream = bool(live_stream)
        self.stream_start_idx = int(stream_start_idx)
        if self.live_stream and self.stream_start_idx < 0:
            raise ValueError("Live H3 requires a non-negative stream_start_idx")
        self.num_frames = num_frames
        self.num_latent_frames = video_latent_num_frames(num_frames)
        self.latent_height, self.latent_width = height // 16, width // 16
        self.num_audio_latents = audio_latent_num_frames(num_frames)
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(seed))
        condition = []
        for clean in condition_video_latents:
            clean = clean.to(AI_DEVICE)
            noise = torch.randn(clean.shape, generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
            condition.append(KEYFRAME_NOISE_AUG * patchify_video_latents(clean) + (1.0 - KEYFRAME_NOISE_AUG) * patchify_video_latents(noise))
        self.condition_video = torch.cat(condition)
        self.condition_audio = (
            torch.cat([value.transpose(1, 2).reshape(-1, 32) for value in condition_audio_latents]).to(AI_DEVICE) if condition_audio_latents else self.condition_video.new_empty((0, 32))
        )
        if self.live_stream:
            # An unbounded session draws one media chunk at a time.
            self.video_noise = self.audio_noise = None
        else:
            video_noise = torch.randn((1, 24, self.num_latent_frames, self.latent_height, self.latent_width), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
            self.video_noise = patchify_video_latents(video_noise).cpu()
            self.audio_noise = torch.randn((1, 2 * self.num_audio_latents, 32), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)[0].cpu()
        self.plan = build_clean_streaming_plan_from_geometry(
            text_token_tags=tuple(tags.cpu() for tags in text_token_tags),
            references=[reference for reference in references if reference.kind == "image"],
            num_reference_audio_latents=self.condition_audio.shape[0] // 2,
            num_latent_frames=self.num_latent_frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            num_audio_latents=self.num_audio_latents,
            segment_starts=tuple(segment_starts),
            policy=self.policy,
            live_stream=self.live_stream,
            stream_start_idx=self.stream_start_idx,
        )
        self.cache = MiniMaxH3CleanKVCache(num_blocks=self.config["num_layers"], plan=self.plan)

    def prepare_condition(self, entry_key):
        self.prepared_rope = None
        self.audio_embedding = None
        self.condition_phase = True
        if entry_key == GLOBAL_KEY:
            layout = self.plan.global_layout()
            prefix = ()
        elif entry_key == REFERENCE_KEY:
            layout = self.plan.reference_layout()
            prefix = (GLOBAL_KEY,)
        else:
            layout = self.plan.prompt_layout(entry_key[1])
            prefix = self.plan.fixed_keys()
        self.cache.prepare(entry_key, prefix, persist=True)
        self.video_latents = self.condition_video if entry_key == REFERENCE_KEY else self.condition_video[:0]
        self.audio_latents = self.condition_audio if entry_key == REFERENCE_KEY else self.condition_audio[:0]
        self.layout_cpu = layout
        self.layout = _layout_to_device(layout, AI_DEVICE)
        self.key_position_ids = torch.cat((self.cache.condition_prefix_position_ids(), layout.position_ids)).to(AI_DEVICE)
        self.step_pre(0, clean=True)

    def append_motion(self, chunk_index, token_tags):
        self.plan = self.plan.append_motion(chunk_index, token_tags)
        self.cache.plan = self.plan

    def prepare_live_chunk(self, index, audio_latents=None, *, clean=False):
        """Prepare one epoch-local chunk without resetting its absolute audio phase."""
        if not self.live_stream:
            raise RuntimeError("prepare_live_chunk requires live_stream=True")
        self.prepare_chunk(index, clean=clean, audio_latents=audio_latents)

    def prepare_chunk(self, index, *, clean=False, audio_latents=None):
        # Positions stay fixed across denoising steps, but clean history uses
        # a different prefix and motion updates can change its length.
        self.prepared_rope = None
        self.condition_phase = False
        self.chunk_index = index
        if not clean:
            self.audio_embedding = None
            video, audio = self.plan.chunk_slices(index)
            if self.live_stream:
                frame_start, frame_end, _, _ = self.plan.chunk_spec(index)
                noise = torch.randn((1, 24, frame_end - frame_start, self.latent_height, self.latent_width), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
                self.video_latents = patchify_video_latents(noise)
                # Draw joint stereo audio for RefA2AV; RefA2V consumes the same
                # random draw but conditions on the supplied clean audio.
                chunk_audio_noise = torch.randn((2 * (audio.stop - audio.start), 32), generator=self.generator, device=AI_DEVICE, dtype=torch.float32)
            else:
                self.video_latents = self.video_noise[video].to(AI_DEVICE)
            if self.generate_audio:
                self.audio_latents = (
                    chunk_audio_noise
                    if self.live_stream
                    else torch.cat((self.audio_noise[audio], self.audio_noise[self.num_audio_latents + audio.start : self.num_audio_latents + audio.stop])).to(AI_DEVICE)
                )
            else:
                if audio_latents is None or audio_latents.shape != (2 * (audio.stop - audio.start), 32):
                    raise ValueError("RefA2V requires this chunk's clean stereo audio latents")
                self.audio_latents = audio_latents.to(AI_DEVICE)
        prefix = self.plan.clean_prefix_keys(index) if clean else self.plan.noisy_prefix_keys(index)
        self.cache.prepare(chunk_key(index) if clean else ("noisy", index), prefix, persist=clean)
        self.layout_cpu = self.plan.chunk_layout(index)
        self.layout = _layout_to_device(self.layout_cpu, AI_DEVICE)
        self.key_position_ids = torch.cat((self.cache.prefix_position_ids(index), self.layout_cpu.position_ids)).to(AI_DEVICE)
        if clean:
            self.step_pre(0, clean=True)

    def step_pre(self, step_index, *, clean=False):
        self.step_index = step_index
        # Clean forwards only commit KV; their velocity predictions are unused.
        self.clean_phase = clean
        video_t = 1.0 if clean else self.video_timesteps_cpu[step_index]
        audio_t = 1.0 if clean or not self.generate_audio else self.audio_timesteps_cpu[step_index]
        unique, indices = build_row_timesteps(self.layout_cpu, video_t, audio_t, condition_video_timestep=KEYFRAME_NOISE_AUG, condition_audio_timestep=1.0)
        self.unique_timesteps_cpu = unique
        self.unique_timesteps = unique.to(AI_DEVICE)
        self.timestep_indices = indices.to(AI_DEVICE)

    def step_post(self):
        # Draw both modalities even for sigma_next=0, preserving the next chunk's RNG.
        video_noise = torch.randn(self.video_latents.shape, generator=self.generator, device=AI_DEVICE)
        audio_noise = torch.randn(self.audio_latents.shape, generator=self.generator, device=AI_DEVICE)
        states = [(self.video_latents, self.video_noise_pred, video_noise, self.video_sigmas)]
        if self.generate_audio:
            states.append((self.audio_latents, self.audio_noise_pred, audio_noise, self.audio_sigmas))
        updated = []
        for state, prediction, noise, sigmas in states:
            # Keep sigma non-scalar so BF16 predictions promote to FP32 like Zoe.
            sigma = sigmas[self.step_index : self.step_index + 1].view(1, 1)
            next_sigma = sigmas[self.step_index + 1 : self.step_index + 2].view(1, 1)
            clean = state + (-sigma) * (-prediction)
            updated.append((1 - next_sigma) * clean + next_sigma * noise)
        self.video_latents = updated[0]
        if self.generate_audio:
            self.audio_latents = updated[1]

    def clear(self):
        self.live_stream = False
        self.stream_start_idx = 0
        if self.cache is not None:
            self.cache.clear()
        for name in (
            "cache",
            "plan",
            "condition_video",
            "condition_audio",
            "video_noise",
            "audio_noise",
            "video_latents",
            "audio_latents",
            "video_noise_pred",
            "audio_noise_pred",
            "layout",
            "layout_cpu",
            "key_position_ids",
            "prepared_rope",
            "audio_embedding",
        ):
            setattr(self, name, None)
