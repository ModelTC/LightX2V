import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class TBDubTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        # The reference/target RoPE layout is not one contiguous video grid.
        self._sol_global_morton_enabled = False

    def infer_audio_cross_attn(self, phase, target, audio):
        frames, audio_tokens, _ = audio.shape
        spatial_tokens = target.shape[0] // frames
        q = phase.audio_norm_q.apply(phase.audio_q.apply(phase.norm4.apply(target))).reshape(frames, spatial_tokens, self.num_heads, self.head_dim)
        context = audio.reshape(-1, audio.shape[-1])
        k = phase.audio_norm_k.apply(phase.audio_k.apply(context)).reshape(frames, audio_tokens, self.num_heads, self.head_dim)
        v = phase.audio_v.apply(context).reshape(frames, audio_tokens, self.num_heads, self.head_dim)
        output = phase.audio_attn.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.arange(frames + 1, device=q.device, dtype=torch.int32) * spatial_tokens,
            cu_seqlens_kv=torch.arange(frames + 1, device=q.device, dtype=torch.int32) * audio_tokens,
            max_seqlen_q=spatial_tokens,
            max_seqlen_kv=audio_tokens,
        )
        return phase.audio_o.apply(output.reshape(-1, self.num_heads * self.head_dim))

    def infer_block(self, block, x, pre_infer_out, self_attn_kwargs=None):
        self_phase, cross_phase, ffn_phase = block.compute_phases
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = self.pre_process(self_phase.modulation, pre_infer_out.embed0)
        y = self.infer_self_attn(self_phase, x, shift, scale, **(self_attn_kwargs or {}))
        x = self.post_process(x, y, gate, pre_infer_out)
        local_length = x.shape[0]
        if self.seq_parallel:
            gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(self.seq_p_group))]
            dist.all_gather(gathered, x.contiguous(), group=self.seq_p_group)
            full = torch.cat(gathered)[: pre_infer_out.valid_token_len]
        else:
            full = x
        target_tokens = pre_infer_out.adapter_args["target_tokens"]
        reference, target = full[:target_tokens], full[target_tokens : 2 * target_tokens]
        # The common Wan cross-attention returns its residual separately.
        self.cross_attn_cu_seqlens_q = torch.tensor([0, target_tokens], dtype=torch.int32)
        target, text_output = self.infer_cross_attn(cross_phase, target, pre_infer_out.context, torch.zeros_like(target), target.new_zeros(()))
        target = target + text_output
        target = target + self.infer_audio_cross_attn(cross_phase, target, pre_infer_out.adapter_args["audio_embedding"])
        full = torch.cat((reference, target))
        if self.seq_parallel:
            world_size = dist.get_world_size(self.seq_p_group)
            padding = local_length * world_size - full.shape[0]
            if padding:
                full = F.pad(full, (0, 0, 0, padding))
            x = full.chunk(world_size, dim=0)[dist.get_rank(self.seq_p_group)]
        else:
            x = full
        y = self.infer_ffn(ffn_phase, x, torch.zeros_like(x), ffn_shift, ffn_scale, ffn_gate)
        return self.post_process(x, y, ffn_gate, pre_infer_out)
