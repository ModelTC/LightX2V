import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3PostInfer:
    def __init__(self, config):
        self.config = config
        self.h3_adaln_curve = bool(config.get("h3_adaln_curve", False))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, pre_infer_out):
        adaln_input = pre_infer_out.temb if self.h3_adaln_curve else F.silu(pre_infer_out.temb).to(GET_DTYPE())
        shift, scale = weights.norm_out_linear.apply(adaln_input).chunk(2, dim=-1)
        indices = pre_infer_out.timestep_indices
        hidden_states = weights.norm_out.apply(hidden_states)
        hidden_states = hidden_states * (1.0 + scale.index_select(0, indices))
        hidden_states = hidden_states + shift.index_select(0, indices)

        # Both released output heads are fp32 and run over all packed rows
        # before modality selection.
        hidden_states = hidden_states.float()
        video = weights.proj_out.apply(hidden_states).index_select(0, pre_infer_out.video_indices)
        audio = weights.audio_proj_out.apply(hidden_states).index_select(0, pre_infer_out.audio_indices)
        return MiniMaxH3VelocityOutput(video=video, audio=audio)
