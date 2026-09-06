import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import indexed_scale_shift_sglang
from lightx2v.models.networks.minimax_h3.infer.sglang_parity import tp_all_gather_last_dim
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3PostInfer:
    def __init__(self, config):
        self.config = config
        self.tp_group = None
        self.tp_size = 1
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_size = dist.get_world_size(self.tp_group)
        self.sglang_parity_ops = config.get("h3_sglang_parity_ops", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, pre_infer_out):
        modulation = pre_infer_out.norm_out_modulation
        if modulation is None:
            # ADALN CACHE SYNC: The offline builder persists this exact
            # norm_out.linear result. Mirror changes there and regenerate caches.
            if pre_infer_out.temb is None:
                raise RuntimeError("MiniMax-H3 final-norm modulation is missing")
            modulation = weights.norm_out_linear.apply(F.silu(pre_infer_out.temb).to(GET_DTYPE()))
            if self.sglang_parity_ops:
                modulation = tp_all_gather_last_dim(modulation, self.tp_group, self.tp_size)
        shift, scale = modulation.chunk(2, dim=-1)
        indices = pre_infer_out.timestep_indices
        hidden_states = weights.norm_out.apply(hidden_states)
        if self.sglang_parity_ops:
            hidden_states = indexed_scale_shift_sglang(hidden_states, shift, scale, indices)
        else:
            hidden_states = hidden_states * (1.0 + scale.index_select(0, indices))
            hidden_states = hidden_states + shift.index_select(0, indices)

        # Both released output heads are fp32 and run over all packed rows
        # before modality selection.
        hidden_states = hidden_states.float()
        video = weights.proj_out.apply(hidden_states)
        audio = weights.audio_proj_out.apply(hidden_states)
        video = video.index_select(0, pre_infer_out.video_indices)
        audio = audio.index_select(0, pre_infer_out.audio_indices)
        if self.sglang_parity_ops:
            video = tp_all_gather_last_dim(video, self.tp_group, self.tp_size)
            audio = tp_all_gather_last_dim(audio, self.tp_group, self.tp_size)
        return MiniMaxH3VelocityOutput(video=video, audio=audio)
