import torch
import torch.distributed as dist

from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import get_q_lens_audio_range
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.infer.utils import compute_freqs_audio, compute_freqs_audio_dist


class WanAudioTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_tokens = 32
        self.num_tokens_x4 = self.num_tokens * 4

    def set_audio_adapter(self, audio_adapter):
        self.audio_adapter = audio_adapter

    @torch.no_grad()
    def compute_freqs(self, q, grid_sizes, freqs):
        if self.config["seq_parallel"]:
            freqs_i = compute_freqs_audio_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs, self.seq_p_group)
        else:
            freqs_i = compute_freqs_audio(q.size(2) // 2, grid_sizes, freqs)
        return freqs_i

    @torch.no_grad()
    def post_process(self, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)

        x = self.modify_hidden_states(
            hidden_states=x,
            grid_sizes=pre_infer_out.grid_sizes,
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
        t, h, w = grid_sizes[0].tolist()
        n_tokens = t * h * w
        ori_dtype = hidden_states.dtype
        device = hidden_states.device
        bs, n_tokens_per_rank = hidden_states.shape[:2]

        if seq_p_group is not None:
            sp_size = dist.get_world_size(seq_p_group)
            sp_rank = dist.get_rank(seq_p_group)
        else:
            sp_size = 1
            sp_rank = 0

        tail_length = n_tokens_per_rank * sp_size - n_tokens
        n_unused_ranks = tail_length // n_tokens_per_rank
        if sp_rank > sp_size - n_unused_ranks - 1:
            n_query_tokens = 0
        elif sp_rank == sp_size - n_unused_ranks - 1:
            n_query_tokens = n_tokens_per_rank - tail_length % n_tokens_per_rank
        else:
            n_query_tokens = n_tokens_per_rank

        if n_query_tokens > 0:
            hidden_states_aligned = hidden_states[:, :n_query_tokens]
            hidden_states_tail = hidden_states[:, n_query_tokens:]
        else:
            # for ranks that should be excluded from cross-attn, fake cross-attn will be applied so that FSDP works.
            hidden_states_aligned = hidden_states[:, :1]
            hidden_states_tail = hidden_states[:, 1:]

        q_lens, t0, t1 = get_q_lens_audio_range(batchsize=bs, n_tokens_per_rank=n_tokens_per_rank, n_query_tokens=n_query_tokens, n_tokens_per_frame=h * w, sp_rank=sp_rank)
        q_lens = torch.tensor(q_lens, device=device, dtype=torch.int32)
        """
        processing audio features in sp_state can be moved outside.
        """
        audio_encoder_output = audio_encoder_output[:, t0:t1]
        k_lens = torch.tensor([self.num_tokens_x4] * (t1 - t0) * bs, device=device, dtype=torch.int32)
        assert q_lens.shape == k_lens.shape
        # ca_block:CrossAttention函数
        residual = ca_block(audio_encoder_output, hidden_states_aligned, t_emb, q_lens, k_lens) * weight

        residual = residual.to(ori_dtype)  # audio做了CrossAttention之后以Residual的方式注入
        if n_query_tokens == 0:
            residual = residual * 0.0
        hidden_states = torch.cat([hidden_states_aligned + residual, hidden_states_tail], dim=1)

        if len(hidden_states.shape) == 3:  #
            hidden_states = hidden_states.squeeze(0)  # bs = 1
        return hidden_states
