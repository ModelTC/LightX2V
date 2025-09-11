import torch
import torch.distributed as dist
from loguru import logger

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import calculate_n_query_tokens, get_qk_lens_audio_range
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class WanAudioTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_tokens = 32
        self.num_tokens_x4 = self.num_tokens * 4

    @torch.no_grad()
    def post_process(self, phase, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(phase, x, y, c_gate_msa, pre_infer_out)

        x = self.infer_adapter_ca(
            phase,
            x,
            pre_infer_out.grid_sizes.tensor,
            pre_infer_out.adapter_output["audio_encoder_output"],
            self.scheduler.audio_adapter_t_emb,
        )
        return x

    @torch.no_grad()
    def infer_adapter_ca(self, phase, hidden_states, grid_sizes, audio_encoder_output, t_emb):
        total_tokens = grid_sizes[0].prod()
        pre_frame_tokens = grid_sizes[0][1:].prod()
        n_tokens = total_tokens - pre_frame_tokens  # 去掉ref image的token数

        ori_dtype = hidden_states.dtype
        device = hidden_states.device
        n_tokens_per_rank = torch.tensor(hidden_states.size(0), dtype=torch.int32, device=device)

        if self.seq_p_group is not None:
            sp_size = dist.get_world_size(self.seq_p_group)
            sp_rank = dist.get_rank(self.seq_p_group)
        else:
            sp_size = 1
            sp_rank = 0

        n_query_tokens, hidden_states_aligned, hidden_states_tail = calculate_n_query_tokens(hidden_states, sp_rank, sp_size, n_tokens_per_rank, n_tokens)

        q_lens, k_lens, max_seqlen_q, max_seqlen_k, t0, t1 = get_qk_lens_audio_range(
            n_tokens_per_rank=n_tokens_per_rank, n_query_tokens=n_query_tokens, n_tokens_per_frame=pre_frame_tokens, sp_rank=sp_rank, num_tokens_x4=self.num_tokens_x4
        )

        # residual = ca_block(audio_encoder_output[:, t0:t1], hidden_states_aligned, t_emb, q_lens, k_lens, max_seqlen_q, max_seqlen_k) * weight
        audio_encoder_output = audio_encoder_output[:, t0:t1].reshape(-1, audio_encoder_output.size(-1))
        residual = self.perceiver_attention_ca(phase, audio_encoder_output, hidden_states_aligned, t_emb, q_lens, k_lens, max_seqlen_q, max_seqlen_k)

        residual = residual.to(ori_dtype)  # audio做了CrossAttention之后以Residual的方式注入
        if n_query_tokens == 0:
            residual = residual * 0.0
        hidden_states = torch.cat([hidden_states_aligned + residual, hidden_states_tail], dim=0)
        return hidden_states

    @torch.no_grad()
    def perceiver_attention_ca(self, phase, x, latents, t_emb, q_lens, k_lens, max_seqlen_q, max_seqlen_k):
        x = phase.norm_kv.apply(x)
        shift, scale, gate = (t_emb + phase.shift_scale_gate.tensor)[0].chunk(3, dim=0)
        norm_q = phase.norm_q.apply(latents)
        latents = norm_q * (1 + scale) + shift
        q = phase.to_q.apply(latents)
        k, v = phase.to_kv.apply(x).chunk(2, dim=-1)

        q = q.view(q.size(0), self.num_heads, self.head_dim)
        k = k.view(k.size(0), self.num_heads, self.head_dim)
        v = v.view(v.size(0), self.num_heads, self.head_dim)

        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
        )
        out = out.view(-1, self.num_heads * self.head_dim)
        return phase.to_out.apply(out) * gate
