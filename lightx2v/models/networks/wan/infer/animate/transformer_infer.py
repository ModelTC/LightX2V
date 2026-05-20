import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class WanAnimateTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.has_post_adapter = True
        self.phases_num = 4

    def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if block_idx == 0:
                self.offload_manager.init_first_buffer(blocks, block_idx // 5)
            if block_idx < len(blocks) - 1:
                self.offload_manager.prefetch_weights(block_idx + 1, blocks, (block_idx + 1) // 5)

            with torch.cuda.stream(self.offload_manager.compute_stream):
                x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)
            self.offload_manager.swap_blocks()
        return x

    def infer_phases(self, block_idx, blocks, x, pre_infer_out, lazy=None):
        if lazy is None:
            lazy = self.lazy_load
        for phase_idx in range(self.phases_num):
            if block_idx == 0 and phase_idx == 0:
                if lazy:
                    obj_key = (block_idx, phase_idx)
                    phase = self.offload_manager.pin_memory_buffer.get(obj_key)
                    phase.to_cuda()
                    self.offload_manager.cuda_buffers[0] = (obj_key, phase)
                else:
                    self.offload_manager.init_first_buffer(blocks, block_idx // 5)
            is_last_phase = block_idx == len(blocks) - 1 and phase_idx == self.phases_num - 1
            if not is_last_phase:
                next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                next_phase_idx = (phase_idx + 1) % self.phases_num
                self.offload_manager.prefetch_phase(next_block_idx, next_phase_idx, blocks, (block_idx + 1) // 5)

            with torch.cuda.stream(self.offload_manager.compute_stream):
                x = self.infer_phase(phase_idx, self.offload_manager.cuda_buffers[phase_idx], x, pre_infer_out)

            self.offload_manager.swap_phases()

        return x

    @torch.no_grad()
    def infer_post_adapter(self, phase, x, pre_infer_out):
        if phase.is_empty() or phase.linear1_kv.weight is None:
            return x
        motion_vec = pre_infer_out.adapter_args["motion_vec"]
        t = motion_vec.shape[0]
        x_motion = phase.pre_norm_motion.apply(motion_vec)
        x_feat = phase.pre_norm_feat.apply(x)
        kv = phase.linear1_kv.apply(x_motion.view(-1, x_motion.shape[-1]))
        kv = kv.view(t, -1, kv.shape[-1])
        q = phase.linear1_q.apply(x_feat)
        k, v = rearrange(kv, "L N (K H D) -> K L N H D", K=2, H=self.config["num_heads"])
        q = rearrange(q, "S (H D) -> S H D", H=self.config["num_heads"])

        f, h, w = pre_infer_out.grid_sizes.tuple
        valid_len = f * h * w

        sp_size = 1
        sp_rank = 0
        seq_pad = 0
        if self.seq_p_group is not None:
            sp_size = dist.get_world_size(self.seq_p_group)
            sp_rank = dist.get_rank(self.seq_p_group)
            if sp_size > 1:
                seq_pad = (sp_size - (valid_len % sp_size)) % sp_size
                gathered_q = [torch.empty_like(q) for _ in range(sp_size)]
                dist.all_gather(gathered_q, q, group=self.seq_p_group)
                q = torch.cat(gathered_q, dim=0)
                # x was tail-padded for SP chunk; face attn only on real video tokens
                if q.shape[0] > valid_len:
                    q = q[:valid_len]

        tokens_per_step = q.shape[0] // t
        if q.shape[0] % t != 0:
            raise RuntimeError(f"face adapter: seq {q.shape[0]} not divisible by T={t} (tokens_per_step={tokens_per_step})")
        q = phase.q_norm.apply(q).view(t, tokens_per_step, q.shape[1], q.shape[2])
        k = phase.k_norm.apply(k)
        q_b = q.permute(0, 2, 1, 3).contiguous()
        k_b = k.permute(0, 2, 1, 3).contiguous()
        v_b = v.permute(0, 2, 1, 3).contiguous()
        attn_b = F.scaled_dot_product_attention(q_b, k_b, v_b)
        attn = attn_b.permute(0, 2, 1, 3).reshape(t * q.shape[1], -1)

        output = phase.linear2.apply(attn)
        if sp_size > 1:
            if seq_pad > 0:
                output = F.pad(output, (0, 0, 0, seq_pad))
            output = torch.chunk(output, sp_size, dim=0)[sp_rank]
        x = x.add_(output)
        return x
