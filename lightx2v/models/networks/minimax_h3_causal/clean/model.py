"""Native DiT with selective clean-history KV and joint AV output."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head_qkv
from lightx2v.models.networks.minimax_h3.autoregressive_model import H3AutoregressiveModel
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3VelocityOutput
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3_causal.clean.fp8_communication import exchange_output_fp8, exchange_qkv_fp8
from lightx2v.models.networks.minimax_h3_causal.clean.module_io import MiniMaxH3CleanPreInferOutput


class MiniMaxH3CleanPreInfer(MiniMaxH3PreInfer):
    def project_audio(self, weights):
        if self.scheduler.generate_audio or self.scheduler.condition_phase:
            return super().project_audio(weights)
        # Driving audio stays fixed through denoising and clean commit.
        if self.scheduler.audio_embedding is None:
            self.scheduler.audio_embedding = super().project_audio(weights)
        return self.scheduler.audio_embedding

    def prepare_rope(self):
        if self.scheduler.prepared_rope is None:
            self.scheduler.prepared_rope = super().prepare_rope(), self._rotary_embedding(self.scheduler.key_position_ids)
        return self.scheduler.prepared_rope[0]

    def infer(self, weights, prompt_embeds):
        output = super().infer(weights, prompt_embeds)
        return MiniMaxH3CleanPreInferOutput(
            **vars(output),
            key_rotary_emb=self.scheduler.prepared_rope[1],
            valid_sequence_length=self.scheduler.layout.sequence_length,
        )


class MiniMaxH3CleanTransformerInfer(MiniMaxH3TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.fp8_communication = self.seq_p_group is not None and self.seq_p_quant_scheme is not None
        if self.fp8_communication and (self.seq_p_quant_scheme != "fp8" or self.seq_p_prepost_backend != "triton" or not self.seq_p_tensor_fusion):
            raise ValueError("CleanAR quantized communication requires FP8, Triton pre/post and seq_p_tensor_fusion=true")

    def _attention(self, weights, hidden_states, pre):
        q, k, v = self._prepare_qkv(weights, hidden_states, None)
        if self.seq_p_group is not None:
            if self.fp8_communication:
                # Fused QKV projections can return strided views.
                group = self.seq_p_group
                q, k, v = exchange_qkv_fp8(q.contiguous(), k.contiguous(), v.contiguous(), group.group_name, dist.get_world_size(group))
            else:
                q, k, v = all2all_seq2head_qkv(q, k, v, self.seq_p_group)
        padded_length = q.shape[0]
        length = pre.valid_sequence_length
        q, k, v = q[:length], k[:length], v[:length]
        # Store normalized, unrotated K. Segment changes move historical RoPE.
        q, k, v = self.scheduler.cache.get_attention_qkv(q, k, v, self.block_idx, rope=weights.rope, query_freqs=pre.rotary_emb, key_freqs=pre.key_rotary_emb)
        output = weights.calculate.apply(q=q, k=k, v=v, causal=False).reshape(length, q.shape[1], q.shape[2])
        if self.seq_p_group is not None:
            output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
            if self.fp8_communication:
                output = exchange_output_fp8(output.contiguous(), group.group_name, dist.get_world_size(group))
            else:
                output = all2all_head2seq(output, self.seq_p_group)
        return weights.to_out.apply(output.flatten(1).to(self.infer_dtype))


class MiniMaxH3CleanPostInfer(MiniMaxH3PostInfer):
    def __init__(self, config):
        super().__init__(config)
        self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None

    def infer(self, weights, hidden_states, pre):
        if self.scheduler.clean_phase:
            return MiniMaxH3VelocityOutput(video=None, audio=None)
        hidden_states = self.norm_out(weights, hidden_states, pre)
        outputs = []
        # Project the sequence shards before gather, matching Zoe's GEMM shapes.
        heads = [(weights.proj_out, pre.video_indices)]
        if self.scheduler.generate_audio:
            heads.append((weights.audio_proj_out, pre.audio_indices))
        for head, indices in heads:
            output = head.apply(hidden_states)
            if self.seq_p_group is not None:
                gathered = output.new_empty((output.shape[0] * dist.get_world_size(self.seq_p_group), output.shape[1]))
                dist.all_gather_into_tensor(gathered, output.contiguous(), group=self.seq_p_group)
                output = gathered[: pre.valid_sequence_length]
            outputs.append(output.index_select(0, indices))
        return MiniMaxH3VelocityOutput(video=outputs[0], audio=outputs[1] if self.scheduler.generate_audio else None)


class MiniMaxH3CleanModel(H3AutoregressiveModel, MiniMaxH3Model):
    def _init_infer_class(self):
        self.pre_infer_class = MiniMaxH3CleanPreInfer
        self.transformer_infer_class = MiniMaxH3CleanTransformerInfer
        self.post_infer_class = MiniMaxH3CleanPostInfer

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        pre = self.pre_infer.infer(self.pre_weight, inputs["text_encoder_output"]["prompt_embeds"])
        if self.seq_p_group is not None:
            pre = self._seq_parallel_pre_process(pre)
        hidden = self.transformer_infer.infer(self.transformer_weights, pre)
        return self.post_infer.infer(self.post_weight, hidden, pre)

    def reset_autoregressive_state(self):
        if self.scheduler.cache is not None:
            self.scheduler.cache.clear()
            self.scheduler.cache = None

    def reset_cuda_graph(self):
        # The clean-cache path does not capture CUDA Graphs.
        pass
