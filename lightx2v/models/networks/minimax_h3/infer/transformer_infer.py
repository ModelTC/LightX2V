import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.hidden_size = int(config.get("hidden_size", 5376))
        self.global_num_heads = int(config.get("num_attention_heads", 56))
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0
        self.num_heads = self.global_num_heads // self.tp_size
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.infer_dtype = GET_DTYPE()
        if config.get("seq_parallel", False):
            self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p")
            parallel = config.get("parallel", {})
            self.seq_p_prepost_backend = parallel.get("seq_p_prepost_backend", "torch")
            self.seq_p_a2a_backend = parallel.get("seq_p_a2a_backend", "torch")
            self.seq_p_quant_scheme = parallel.get("seq_p_quant_scheme")
            if self.seq_p_quant_scheme is None:
                self.seq_p_quant_scheme = "fp8" if parallel.get("seq_p_fp8_comm", False) else "fp4" if parallel.get("seq_p_fp4_comm", False) else None
            self.seq_p_tensor_fusion = parallel.get("seq_p_tensor_fusion", False)
            self.seq_p_head_parallel = parallel.get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
        self.infer_func = self.infer_without_offload
        self.init_compile(config)

    def _gather_tp_last_dim(self, tensor):
        if self.tp_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.tp_group)
        return torch.cat(gathered, dim=-1)

    def _attention(self, weights, hidden_states, pre_infer_out):
        q = weights.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        q = weights.norm_q.apply(q)
        k = weights.norm_k.apply(k)
        q, k = weights.rope.apply(
            q,
            k,
            pre_infer_out.rotary_emb,
            rotary_dim=pre_infer_out.rotary_emb[0].shape[-1],
        )
        sp_state = pre_infer_out.sequence_parallel_state
        if sp_state is None:
            seq_len = q.shape[0]
            cu_seqlens = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
            out = weights.calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_kv=seq_len,
                causal=False,
            )
        else:
            aux_length = sp_state.aux_length
            out, aux_out = weights.calculate_parallel.apply_new(
                q=q[aux_length:].contiguous(),
                k=k[aux_length:].contiguous(),
                v=v[aux_length:].contiguous(),
                aux_q=q[:aux_length].contiguous() if aux_length else None,
                aux_k=k[:aux_length].contiguous() if aux_length else None,
                aux_v=v[:aux_length].contiguous() if aux_length else None,
                attention_module=weights.calculate,
                seq_p_group=self.seq_p_group,
                prepost_backend=self.seq_p_prepost_backend,
                a2a_backend=self.seq_p_a2a_backend,
                quant_scheme=self.seq_p_quant_scheme,
                tensor_fusion=self.seq_p_tensor_fusion,
                head_parallel=self.seq_p_head_parallel,
                aux_first=True,
                attention_kwargs={"causal": False},
            )
            if aux_out is not None:
                out = torch.cat((aux_out, out), dim=0)
        return weights.to_out.apply(out.to(self.infer_dtype))

    @staticmethod
    def _ff(weights, hidden_states):
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        return weights.out_proj.apply(value * F.silu(gate))

    def infer_block(self, weights, hidden_states, pre_infer_out):
        # Activation is evaluated in fp32, then cast to the inference dtype
        # immediately before the (possibly quantized) AdaLN projection.
        modulation = weights.adaln.apply(F.silu(pre_infer_out.temb).to(self.infer_dtype))
        modulation = self._gather_tp_last_dim(modulation)
        modulation = modulation.view(-1, 6 * self.hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        indices = pre_infer_out.adaln_indices

        residual = hidden_states
        normed = weights.norm1.apply(hidden_states)
        normed = normed * (1.0 + scale_msa.index_select(0, indices))
        normed = normed + shift_msa.index_select(0, indices)
        hidden_states = residual + gate_msa.index_select(0, indices) * self._attention(weights.attn, normed, pre_infer_out)

        residual = hidden_states
        normed = weights.norm2.apply(hidden_states)
        normed = normed * (1.0 + scale_mlp.index_select(0, indices))
        normed = normed + shift_mlp.index_select(0, indices)
        hidden_states = residual + gate_mlp.index_select(0, indices) * self._ff(weights.ff, normed)
        return hidden_states

    def infer_without_offload(self, blocks, hidden_states, pre_infer_out):
        for block_index, block in enumerate(blocks):
            hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        return self.infer_func(block_weights.blocks, pre_infer_out.hidden_states, pre_infer_out)
