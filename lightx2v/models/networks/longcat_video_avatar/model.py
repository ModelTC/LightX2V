import torch.distributed as dist

from .infer import AvatarPostInfer, AvatarPreInfer, AvatarTransformerInfer
from .weights import AvatarWeights


class LongCatVideoAvatarModel:
    def __init__(self, config, device):
        self.weights = AvatarWeights(config, device)
        self.group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None
        self.width_shards = dist.get_world_size(self.group) if self.group is not None else 1
        self.rank = dist.get_rank(self.group) if self.group is not None else 0
        self.pre_infer = AvatarPreInfer(config, self.weights.pre)
        self.transformer_infer = AvatarTransformerInfer(config, self.group)
        self.post_infer = AvatarPostInfer(config, self.weights, self.group)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def prepare_condition(self, projected_text, audio, speaker_masks, rope=None):
        return self.pre_infer.prepare(projected_text, audio, speaker_masks, self.scheduler.latents.shape, self.scheduler.num_cond_latents, self.width_shards, self.rank, rope)

    def infer(self, condition):
        scheduler = self.scheduler
        tokens, time = self.pre_infer.infer(scheduler.latents, scheduler.frame_timesteps, condition["width_slice"])
        for block in self.weights.blocks:
            tokens = self.transformer_infer.infer(tokens, time, condition, block, scheduler.lora_strengths)
        scheduler.noise_pred = self.post_infer.infer(tokens, time, condition["grid"])
