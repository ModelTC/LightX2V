import math
from functools import lru_cache

import torch
import torch.distributed as dist

from .infer import AvatarPostInfer, AvatarPreInfer, AvatarTransformerInfer
from .weights import AvatarWeights


def _make_step_modulation_cache(scheduler, pre_infer, weights):
    # Keep the first and continuation segment layouts for one denoising schedule.
    @lru_cache(maxsize=2 * scheduler.infer_steps)
    def step_modulations(step, frames, num_cond_latents):
        # Project the original frame layout before keeping one row per distinct timestep.
        timesteps = scheduler.timesteps.new_zeros(frames, dtype=torch.bfloat16)
        timesteps[num_cond_latents:] = scheduler.timesteps[step]
        time = pre_infer.embed_timesteps(timesteps)
        noisy_time = time[num_cond_latents:]
        video_rows = time.new_tensor([0, frames - 1] if num_cond_latents else [0], dtype=torch.long)
        modulations = []
        for block in weights.blocks:
            video = block.modulation.apply(time).index_select(0, video_rows)
            audio = block.audio_modulation.apply(noisy_time)[-1:].clone()
            modulations.append((video, audio))
        return modulations, weights.final_modulation.apply(noisy_time)

    return step_modulations


class LongCatVideoAvatarModel:
    def __init__(self, config, device):
        self.weights = AvatarWeights(config, device)
        self.cache_condition_kv = config.get("cache_condition_kv", False)
        self.group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None
        size = dist.get_world_size(self.group) if self.group is not None else 1
        rows = next(n for n in range(math.isqrt(size), 0, -1) if size % n == 0)
        self.spatial_split = (rows, size // rows)
        self.rank = dist.get_rank(self.group) if self.group is not None else 0
        self.pre_infer = AvatarPreInfer(config, self.weights.pre)
        self.transformer_infer = AvatarTransformerInfer(config, self.group, self.spatial_split)
        self.post_infer = AvatarPostInfer(config, self.weights, self.group, self.spatial_split)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.step_modulations = _make_step_modulation_cache(scheduler, self.pre_infer, self.weights)

    def prepare_condition(self, projected_text, audio, speaker_masks, rope=None):
        return self.pre_infer.prepare(projected_text, audio, speaker_masks, self.scheduler.latents.shape, self.scheduler.num_cond_latents, self.spatial_split, self.rank, rope)

    def infer(self, condition):
        scheduler = self.scheduler
        frames, height, width = condition["rope"][0].shape[:3]
        num_cond_latents = condition["num_cond_latents"]
        noisy_frames = frames - num_cond_latents
        strengths = scheduler.lora_strengths
        unmerged_strengths = self.weights.select_lora(strengths)
        layer_caches = condition.pop("condition_kv", None)
        start = num_cond_latents if layer_caches is not None else 0
        next_step = scheduler.step_index + 1
        keep_kv = self.cache_condition_kv and num_cond_latents > 0 and next_step < scheduler.infer_steps and scheduler.lora_schedule[next_step] == strengths
        step_condition = condition
        if start:
            step_condition = dict(condition, num_cond_latents=0, rope=tuple(part[start:] for part in condition["rope"]))
        elif keep_kv:
            layer_caches = [{} for _ in self.weights.blocks]
        modulations, final = self.step_modulations(scheduler.step_index, frames - start, step_condition["num_cond_latents"])
        tokens = self.pre_infer.infer(scheduler.latents[:, :, start:], condition["spatial_slices"])
        tokens = self.transformer_infer.infer(tokens, modulations, step_condition, self.weights.blocks, unmerged_strengths, layer_caches)
        scheduler.noise_pred = self.post_infer.infer(tokens[-noisy_frames:], final, (noisy_frames, height, width))
        if keep_kv:
            condition["condition_kv"] = layer_caches
