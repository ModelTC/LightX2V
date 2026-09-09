import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import indexed_scale_shift_sglang
from lightx2v.models.networks.minimax_h3.infer.tensor_parallel import all_gather_last_dim
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3PostInfer:
    def __init__(self, config):
        self.config = config
        self.tp_group = None
        self.tp_size = 1
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_size = dist.get_world_size(self.tp_group)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _gather_tp_last_dim(self, tensor):
        return all_gather_last_dim(tensor, self.tp_group, self.tp_size)

    @staticmethod
    def _apply_modulation(hidden_states, shift, scale, indices):
        return indexed_scale_shift_sglang(hidden_states, shift, scale, indices)

    def infer(self, weights, hidden_states, pre_infer_out):
        modulation = pre_infer_out.norm_out_modulation
        if modulation is None:
            # ADALN CACHE SYNC: The offline builder persists this exact
            # norm_out.linear result. Mirror changes there and regenerate caches.
            if pre_infer_out.temb is None:
                raise RuntimeError("MiniMax-H3 final-norm modulation is missing")
            modulation = weights.norm_out_linear.apply(F.silu(pre_infer_out.temb).to(GET_DTYPE()))
            modulation = self._gather_tp_last_dim(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        indices = pre_infer_out.timestep_indices
        hidden_states = weights.norm_out.apply(hidden_states)
        hidden_states = self._apply_modulation(hidden_states, shift, scale, indices)

        hidden_states = hidden_states.float()
        video = weights.proj_out.apply(hidden_states)
        audio = weights.audio_proj_out.apply(hidden_states)
        video = video.index_select(0, pre_infer_out.video_indices)
        audio = audio.index_select(0, pre_infer_out.audio_indices)
        video = self._gather_tp_last_dim(video)
        audio = self._gather_tp_last_dim(audio)
        return MiniMaxH3VelocityOutput(video=video, audio=audio)
