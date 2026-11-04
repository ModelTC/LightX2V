import torch.distributed as dist

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer


class MiniMaxH3CausalPostInfer(MiniMaxH3PostInfer):
    def __init__(self, config):
        super().__init__(config)
        self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None

    def infer(self, weights, hidden_states, pre_infer_out):
        # Condition prefill only populates KV inside the transformer blocks.
        if self.scheduler.condition_phase:
            return MiniMaxH3VelocityOutput(video=None, audio=None)

        hidden_states = self.norm_out(weights, hidden_states, pre_infer_out)
        video = weights.proj_out.apply(hidden_states)
        if self.seq_p_group is not None:
            gathered = video.new_empty((video.shape[0] * dist.get_world_size(self.seq_p_group), video.shape[1]))
            dist.all_gather_into_tensor(gathered, video.contiguous(), group=self.seq_p_group)
            video = gathered[: pre_infer_out.valid_sequence_length]
        video = video.index_select(0, pre_infer_out.video_indices)
        # Causal sampling keeps the input audio latents fixed at every step.
        return MiniMaxH3VelocityOutput(video=video, audio=None)
