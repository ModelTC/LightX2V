import torch
import torch.distributed as dist

from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import calculate_n_query_tokens, get_qk_lens_audio_range
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class WanAudioTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_tokens = 32
        self.num_tokens_x4 = self.num_tokens * 4

    def set_audio_adapter(self, audio_adapter):
        self.audio_adapter = audio_adapter

    @torch.no_grad()
    def post_process(self, x, y, c_gate_msa, pre_infer_out, block_idx=0):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)

        x = self.modify_hidden_states(
            hidden_states=x.to(self.infer_dtype),
            grid_sizes=pre_infer_out.grid_sizes.tensor,
            ca_block=self.audio_adapter.ca[self.block_idx],
            audio_encoder_output=pre_infer_out.adapter_output["audio_encoder_output"],
            t_emb=self.scheduler.audio_adapter_t_emb,
            weight=1.0,
            seq_p_group=self.seq_p_group,
        )
        return x

    @torch.no_grad()
    def modify_hidden_states(self, hidden_states, grid_sizes, ca_block, audio_encoder_output, t_emb, weight, seq_p_group):
        """thw specify the latent_frame, latent_height, latenf_width after
        hidden_states is patchified.

        latent_frame does not include the reference images so that the
        audios and hidden_states are strictly aligned
        """
        if len(hidden_states.shape) == 2:  # 扩展batchsize dim
            hidden_states = hidden_states.unsqueeze(0)  # bs = 1

        total_tokens = grid_sizes[0].prod()
        pre_frame_tokens = grid_sizes[0][1:].prod()
        n_tokens = total_tokens - pre_frame_tokens  # 去掉ref image的token数

        ori_dtype = hidden_states.dtype
        device = hidden_states.device
        n_tokens_per_rank = torch.tensor(hidden_states.size(1), dtype=torch.int32, device=device)

        if seq_p_group is not None:
            sp_size = dist.get_world_size(seq_p_group)
            sp_rank = dist.get_rank(seq_p_group)
        else:
            sp_size = 1
            sp_rank = 0

        n_query_tokens, hidden_states_aligned, hidden_states_tail = calculate_n_query_tokens(hidden_states, sp_rank, sp_size, n_tokens_per_rank, n_tokens)

        q_lens, k_lens, max_seqlen_q, max_seqlen_k, t0, t1 = get_qk_lens_audio_range(
            n_tokens_per_rank=n_tokens_per_rank, n_query_tokens=n_query_tokens, n_tokens_per_frame=pre_frame_tokens, sp_rank=sp_rank, num_tokens_x4=self.num_tokens_x4
        )
        # ca_block:CrossAttention函数
        if self.audio_adapter.cpu_offload:
            ca_block.to("cuda")
        residual = ca_block(audio_encoder_output[:, t0:t1], hidden_states_aligned, t_emb, q_lens, k_lens, max_seqlen_q, max_seqlen_k) * weight
        if self.audio_adapter.cpu_offload:
            ca_block.to("cpu")
        residual = residual.to(ori_dtype)  # audio做了CrossAttention之后以Residual的方式注入
        if n_query_tokens == 0:
            residual = residual * 0.0
        hidden_states = torch.cat([hidden_states_aligned + residual, hidden_states_tail], dim=1)

        if len(hidden_states.shape) == 3:  #
            hidden_states = hidden_states.squeeze(0)  # bs = 1
        return hidden_states
