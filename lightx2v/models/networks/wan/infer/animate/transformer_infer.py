import torch
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

    def infer_phases(self, block_idx, blocks, x, pre_infer_out, lazy):
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
        T = pre_infer_out.adapter_args["motion_vec"].shape[0]
        x_motion = phase.pre_norm_motion.apply(pre_infer_out.adapter_args["motion_vec"])
        x_feat = phase.pre_norm_feat.apply(x)
        kv = phase.linear1_kv.apply(x_motion.view(-1, x_motion.shape[-1]))
        kv = kv.view(T, -1, kv.shape[-1])
        q = phase.linear1_q.apply(x_feat)
        k, v = rearrange(kv, "L N (K H D) -> K L N H D", K=2, H=self.config["num_heads"])
        q = rearrange(q, "S (H D) -> S H D", H=self.config["num_heads"])

        q = phase.q_norm.apply(q).view(T, q.shape[0] // T, q.shape[1], q.shape[2])
        k = phase.k_norm.apply(k)
        attn = phase.adapter_attn.apply(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=q.shape[1],
            model_cls=self.config["model_cls"],
        )

        output = phase.linear2.apply(attn)
        x = x.add_(output)
        return x
