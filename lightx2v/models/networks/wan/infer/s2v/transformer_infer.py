import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.s2v.audio_inject import apply_audio_inject
from lightx2v.models.networks.wan.infer.s2v.rope import apply_precomputed_rope
from lightx2v.models.networks.wan.infer.s2v.wan_ops import (
    cross_attn_forward,
    mm_weight_autocast_nd,
    mm_weight_fp32_nd,
    s2v_self_attn_forward,
    segment_gate_bld,
    segment_modulate_bld,
    wan_layer_norm,
    wan_layer_norm_float,
    wan_rms_norm,
)
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanS2VTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.injected_block_id = {layer_idx: idx for idx, layer_idx in enumerate(config.get("audio_inject_layers", []))}

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        global_seq_len = pre_infer_out.s2v_extra.get("global_original_seq_len", pre_infer_out.original_seq_len)
        if self.config["seq_parallel"]:
            return x
        return self.infer_non_blocks(weights, x, pre_infer_out.embed, global_seq_len)

    def infer_non_blocks(self, weights, x, e, original_seq_len=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if original_seq_len is not None:
            x = x[:, :original_seq_len]
        modulation = weights.head_modulation.tensor
        with torch.amp.autocast(AI_DEVICE, dtype=torch.float32):
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            norm_x = wan_layer_norm(weights.norm, x).float()
            x = norm_x * (1 + e[1]) + e[0]
            x = mm_weight_fp32_nd(weights.head, x)
        if x.dim() == 3:
            x = x.squeeze(0)
        return x

    def _gather_along_seq(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        world_size = dist.get_world_size(self.seq_p_group)
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=self.seq_p_group)
        return torch.cat(gathered, dim=1)

    def _chunk_along_seq(self, x):
        rank = dist.get_rank(self.seq_p_group)
        return torch.chunk(x, dist.get_world_size(self.seq_p_group), dim=1)[rank]

    def _seg_idx(self, original_seq_len, seq_len):
        seg = min(max(0, int(original_seq_len)), seq_len)
        return [0, seg, seq_len]

    @torch.no_grad()
    def pre_process(self, modulation, embed0):
        e = embed0[0] if isinstance(embed0, (list, tuple)) else embed0
        modulation = modulation.tensor.unsqueeze(2)
        with torch.amp.autocast(str(AI_DEVICE), dtype=torch.float32):
            e = (modulation + e).chunk(6, dim=1)
        return [element.squeeze(1) for element in e]

    @torch.no_grad()
    def infer_without_offload(self, blocks, x, pre_infer_out):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_s2v_block(blocks[block_idx], x, pre_infer_out)
            if block_idx in self.injected_block_id:
                if self.config["seq_parallel"]:
                    x_full = self._gather_along_seq(x)
                    x_full = apply_audio_inject(blocks[block_idx], x_full, pre_infer_out, self.config)
                    x = self._chunk_along_seq(x_full)
                else:
                    x = apply_audio_inject(blocks[block_idx], x, pre_infer_out, self.config)
        return x

    def _s2v_self_attn(self, phase0, norm_x, seq_lens, freqs):
        if self.config["seq_parallel"]:
            b, s, n, d = norm_x.size(0), norm_x.size(1), self.num_heads, self.head_dim
            q = wan_rms_norm(phase0.self_attn_norm_q, mm_weight_autocast_nd(phase0.self_attn_q, norm_x)).view(b, s, n, d)
            k = wan_rms_norm(phase0.self_attn_norm_k, mm_weight_autocast_nd(phase0.self_attn_k, norm_x)).view(b, s, n, d)
            v = mm_weight_autocast_nd(phase0.self_attn_v, norm_x).view(b, s, n, d)
            q = apply_precomputed_rope(q, freqs)
            k = apply_precomputed_rope(k, freqs)
            if self.self_attn_cu_seqlens_qkv is None:
                self.self_attn_cu_seqlens_qkv = torch.tensor([0, s], dtype=torch.int32, device=norm_x.device)
            attn_out = (
                phase0.self_attn_1_parallel.apply(
                    q=q.squeeze(0).to(self.infer_dtype),
                    k=k.squeeze(0).to(self.infer_dtype),
                    v=v.squeeze(0).to(self.infer_dtype),
                    slice_qkv_len=s,
                    cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
                    attention_module=phase0.self_attn_1,
                    seq_p_group=self.seq_p_group,
                    use_fp8_comm=self.seq_p_fp8_comm,
                    use_fp4_comm=self.seq_p_fp4_comm,
                    use_tensor_fusion=self.seq_p_tensor_fusion,
                    enable_head_parallel=self.enable_head_parallel,
                    block_idx=self.block_idx,
                    scheduler=self.scheduler,
                )
                .float()
                .view(b, s, -1)
            )
            return mm_weight_autocast_nd(phase0.self_attn_o, attn_out)

        return s2v_self_attn_forward(phase0, norm_x, seq_lens, freqs, self.num_heads, self.head_dim, apply_precomputed_rope)

    @torch.no_grad()
    def infer_s2v_block(self, block, x, pre_infer_out):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        phase0, phase1, phase2 = block.compute_phases[0], block.compute_phases[1], block.compute_phases[2]
        e = self.pre_process(phase0.modulation, pre_infer_out.embed0)
        seg_idx = self._seg_idx(pre_infer_out.original_seq_len, x.size(1))
        seq_lens = torch.tensor([x.size(1)], dtype=torch.long, device=x.device)

        # norm1 + modulate (Wan 212-217)
        norm_x = wan_layer_norm_float(phase0.norm1, x)
        norm_x = segment_modulate_bld(norm_x, e[0], e[1], seg_idx)

        # self-attention (Wan 219-225)
        y = self._s2v_self_attn(phase0, norm_x, seq_lens, pre_infer_out.freqs)
        with torch.amp.autocast(AI_DEVICE, dtype=torch.float32):
            y = segment_gate_bld(y, e[2], seg_idx)
            x = x + y

        # cross-attention & ffn (Wan cross_attn_ffn 228-243)
        context = pre_infer_out.context
        ctx_lens = pre_infer_out.s2v_extra.get("context_lens")
        x = x + cross_attn_forward(phase1, wan_layer_norm(phase1.norm3, x), context, ctx_lens, self.num_heads, self.head_dim)

        norm2_x = wan_layer_norm_float(phase2.norm2, x)
        norm2_x = segment_modulate_bld(norm2_x, e[3], e[4], seg_idx)
        y = mm_weight_autocast_nd(phase2.ffn_0, norm2_x)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = mm_weight_autocast_nd(phase2.ffn_2, y)
        with torch.amp.autocast(AI_DEVICE, dtype=torch.float32):
            y = segment_gate_bld(y, e[5], seg_idx)
            x = x + y
        return x
