import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.s2v.post_infer import WanS2VPostInfer
from lightx2v.models.networks.wan.infer.s2v.pre_infer import WanS2VPreInfer
from lightx2v.models.networks.wan.infer.s2v.transformer_infer import WanS2VTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.s2v.pre_weights import WanS2VPreWeights
from lightx2v.models.networks.wan.weights.s2v.transformer_weights import WanS2VTransformerWeights


class WanS2VModel(WanModel):
    pre_weight_class = WanS2VPreWeights
    transformer_weight_class = WanS2VTransformerWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type="wan2.2_s2v", lora_path=lora_path, lora_strength=lora_strength)
        self.sensitive_layer.update(
            {
                "cond_encoder",
                "trainable_cond_mask",
                "casual_audio_encoder",
                "frame_packer",
                "audio_injector",
            }
        )

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanS2VPreInfer
        self.post_infer_class = WanS2VPostInfer
        self.transformer_infer_class = WanS2VTransformerInfer

    def _seq_p_group(self):
        if not self.config.get("seq_parallel"):
            return None
        return self.config["device_mesh"].get_group(mesh_dim="seq_p")

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        """Chunk along sequence dim (dim=1), matching Wan2.2 S2V context parallel."""
        x = pre_infer_out.x
        if x.dim() == 2:
            x = x.unsqueeze(0)

        group = self._seq_p_group()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        chunks = torch.chunk(x, world_size, dim=1)
        sq_sizes = [c.shape[1] for c in chunks]
        sq_start = sum(sq_sizes[:rank])
        pre_infer_out.x = chunks[rank]

        global_original_seq_len = pre_infer_out.original_seq_len
        pre_infer_out.s2v_extra["global_original_seq_len"] = global_original_seq_len
        pre_infer_out.original_seq_len = max(0, global_original_seq_len - sq_start)

        if isinstance(pre_infer_out.embed0, (list, tuple)) and len(pre_infer_out.embed0) == 2:
            pre_infer_out.embed0 = [pre_infer_out.embed0[0], pre_infer_out.original_seq_len]

        freqs = pre_infer_out.freqs
        if freqs is not None:
            pre_infer_out.freqs = torch.chunk(freqs, world_size, dim=1)[rank]

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        """All-gather sequence shards (dim=1 for [1, L, C], else dim=0 for [L, C])."""
        group = self._seq_p_group()
        world_size = dist.get_world_size(group)

        if x.dim() == 2:
            x = x.unsqueeze(0)
            seq_dim = 1
        else:
            seq_dim = 1

        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        if self.config["seq_parallel"]:
            x = self._seq_parallel_post_process(x)
            global_seq_len = pre_infer_out.s2v_extra.get("global_original_seq_len", pre_infer_out.original_seq_len)
            x = self.transformer_infer.infer_non_blocks(self.transformer_weights, x, pre_infer_out.embed, global_seq_len)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred
