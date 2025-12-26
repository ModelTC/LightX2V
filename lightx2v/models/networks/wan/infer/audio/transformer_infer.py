import torch
import torch.distributed as dist
from loguru import logger

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import align_hidden_states_and_mask, calculate_n_query_tokens, get_qk_lens_audio_range
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class WanAudioTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.has_post_adapter = True
        self.phases_num = 4

    @torch.no_grad()
    def reset_post_adapter_states(self):
        self.post_adapter_states_ready = False

    @torch.no_grad()
    def infer_post_adapter(self, phase, x, pre_infer_out):
        grid_sizes = pre_infer_out.grid_sizes.tensor
        audio_encoder_output = pre_infer_out.adapter_args["audio_encoder_output"]
        person_mask_latens = pre_infer_out.adapter_args["person_mask_latens"]
        total_tokens = grid_sizes[0].prod()
        pre_frame_tokens = grid_sizes[0][1:].prod()
        n_tokens = total_tokens - pre_frame_tokens  # 去掉ref image的token数

        ori_dtype = x.dtype
        device = x.device

        if self.seq_p_group is not None:
            sp_size = dist.get_world_size(self.seq_p_group)
            sp_rank = dist.get_rank(self.seq_p_group)
        else:
            sp_size = 1
            sp_rank = 0

        if not self.post_adapter_states_ready:
            n_tokens_per_rank = torch.tensor(x.size(0), dtype=torch.int32)
            self.n_query_tokens = calculate_n_query_tokens(sp_rank, sp_size, n_tokens_per_rank, n_tokens)
            self.q_lens, self.k_lens, self.max_seqlen_q, self.max_seqlen_k, self.t0, self.t1 = get_qk_lens_audio_range(
                n_tokens_per_rank=n_tokens_per_rank, n_query_tokens=self.n_query_tokens, n_tokens_per_frame=pre_frame_tokens, sp_rank=sp_rank, num_tokens_x4=128
            )
            self.perceiver_attn_cu_seqlens_q = torch.cat([self.q_lens.new_zeros([1]), self.q_lens]).cumsum(0, dtype=torch.int32).to(device, non_blocking=True)
            self.perceiver_attn_cu_seqlens_k = torch.cat([self.k_lens.new_zeros([1]), self.k_lens]).cumsum(0, dtype=torch.int32).to(device, non_blocking=True)
            self.post_adapter_states_ready = True

        hidden_states_aligned, hidden_states_tail, person_mask_aligned = align_hidden_states_and_mask(self.n_query_tokens, x, person_mask_latens)

        total_residual = None
        for i in range(audio_encoder_output.shape[0]):
            audio_encoder = audio_encoder_output[i]
            audio_encoder = audio_encoder[self.t0 : self.t1].reshape(-1, audio_encoder.size(-1))
            residual = self.perceiver_attention_ca(phase, audio_encoder, hidden_states_aligned, self.scheduler.audio_adapter_t_emb)

            residual = residual.to(ori_dtype)  # audio做了CrossAttention之后以Residual的方式注入
            if self.n_query_tokens == 0:
                residual = residual * 0.0
            if person_mask_aligned is not None:
                residual = residual * person_mask_aligned[i].unsqueeze(-1)

            if total_residual is None:
                total_residual = residual
            else:
                total_residual += residual

        x = torch.cat([hidden_states_aligned + total_residual, hidden_states_tail], dim=0)
        return x

    @torch.no_grad()
    def perceiver_attention_ca(self, phase, audio_encoder_output, latents, t_emb):
        audio_encoder_output = phase.norm_kv.apply(audio_encoder_output)
        shift, scale, gate = (t_emb + phase.shift_scale_gate.tensor)[0].chunk(3, dim=0)
        norm_q = phase.norm_q.apply(latents)
        latents = norm_q * (1 + scale) + shift
        q = phase.to_q.apply(latents)
        k, v = phase.to_kv.apply(audio_encoder_output).chunk(2, dim=-1)

        q = q.view(q.size(0), self.num_heads, self.head_dim)
        k = k.view(k.size(0), self.num_heads, self.head_dim)
        v = v.view(v.size(0), self.num_heads, self.head_dim)

        if "npu" in AI_DEVICE:
            out = ATTN_WEIGHT_REGISTER.get("npu_flash_attn")().apply(
                q=q, k=k, v=v, cu_seqlens_q=self.perceiver_attn_cu_seqlens_q, cu_seqlens_kv=self.perceiver_attn_cu_seqlens_k, max_seqlen_q=self.max_seqlen_q, max_seqlen_kv=self.max_seqlen_k
            )
        else:
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.perceiver_attn_cu_seqlens_q,
                cu_seqlens_k=self.perceiver_attn_cu_seqlens_k,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),
                deterministic=False,
            )
        out = out.view(-1, self.num_heads * self.head_dim)
        return phase.to_out.apply(out) * gate
