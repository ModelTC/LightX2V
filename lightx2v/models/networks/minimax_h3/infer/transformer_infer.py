import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.minimax_h3.adaln_cache import load_persistent_adaln_cache
from lightx2v.models.networks.minimax_h3.config import resolve_minimax_h3_sgl_alignment
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import (
    apply_mlp_sglang,
    apply_qk_norm_rope_sglang,
    indexed_gate_sglang,
    indexed_scale_shift_sglang,
)
from lightx2v.models.networks.minimax_h3.infer.sglang_parity import (
    build_sglang_rope_cache,
    project_merged_qkv,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


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
        self.sglang_parity_ops = resolve_minimax_h3_sgl_alignment(config).parity_ops
        if self.sglang_parity_ops and config.get("cpu_offload") and config.get("offload_granularity") == "block":
            raise NotImplementedError("SGLang parity ops do not support block CPU offload")
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
        self.use_adaln_cache = bool(config.get("use_adaln_cache", False))
        self._current_adaln_tables = None
        self._adaln_cache_hit = False
        if self.use_adaln_cache:
            self._adaln_cache, self._norm_out_cache = load_persistent_adaln_cache(config, self._cache_device())
        self.init_compile(config)

    @staticmethod
    def _cache_device():
        device_module = getattr(torch, AI_DEVICE)
        return torch.device(AI_DEVICE, device_module.current_device())

    def _gather_tp_last_dim(self, tensor):
        if self.tp_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.tp_group)
        return torch.cat(gathered, dim=-1)

    def _attention(self, weights, hidden_states, pre_infer_out):
        if self.sglang_parity_ops:
            q, k, v = project_merged_qkv(weights, hidden_states)
        else:
            q = weights.to_q.apply(hidden_states)
            k = weights.to_k.apply(hidden_states)
            v = weights.to_v.apply(hidden_states)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        if self.sglang_parity_ops:
            if pre_infer_out.sglang_rope_cache is None:
                pre_infer_out.sglang_rope_cache = build_sglang_rope_cache(pre_infer_out.rotary_emb, q.dtype)
            q, k = apply_qk_norm_rope_sglang(q, k, weights.norm_q, weights.norm_k, pre_infer_out.sglang_rope_cache)
        else:
            q = weights.norm_q.apply(q)
            k = weights.norm_k.apply(k)
            q, k = weights.rope.apply(
                q,
                k,
                pre_infer_out.rotary_emb,
                rotary_dim=pre_infer_out.rotary_emb[0].shape[-1],
            )
        sp_state = pre_infer_out.sequence_parallel_state
        attention_kwargs = {
            "causal": False,
            "scheduler": self.scheduler,
            "block_idx": self.block_idx,
            "softmax_scale": self.head_dim**-0.5,
        }
        if sp_state is None:
            used_seq_len = self.scheduler.layout.used_sequence_length
            out = weights.calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=pre_infer_out.cu_seqlens,
                cu_seqlens_kv=pre_infer_out.cu_seqlens,
                max_seqlen_q=used_seq_len,
                max_seqlen_kv=used_seq_len,
                **attention_kwargs,
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
                attention_kwargs=attention_kwargs,
            )
            if aux_out is not None:
                out = torch.cat((aux_out, out), dim=0)
        return weights.to_out.apply(out.to(self.infer_dtype))

    def _ff(self, weights, hidden_states):
        if self.sglang_parity_ops:
            return apply_mlp_sglang(weights, hidden_states)
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        return weights.out_proj.apply(value * F.silu(gate))

    def infer_block(self, weights, hidden_states, pre_infer_out, modulation=None):
        # Keep the Python cache lookup outside the compiled block.
        if modulation is None:
            modulation = self._compute_adaln_table(weights, pre_infer_out)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        indices = pre_infer_out.adaln_indices

        residual = hidden_states
        normed = weights.norm1.apply(hidden_states)
        if self.sglang_parity_ops:
            normed = indexed_scale_shift_sglang(normed, shift_msa, scale_msa, indices)
        else:
            normed = normed * (1.0 + scale_msa.index_select(0, indices))
            normed = normed + shift_msa.index_select(0, indices)
        attention_output = self._attention(weights.attn, normed, pre_infer_out)
        if self.sglang_parity_ops:
            hidden_states = indexed_gate_sglang(residual, gate_msa, attention_output, indices)
        else:
            hidden_states = residual + gate_msa.index_select(0, indices) * attention_output

        residual = hidden_states
        normed = weights.norm2.apply(hidden_states)
        if self.sglang_parity_ops:
            normed = indexed_scale_shift_sglang(normed, shift_mlp, scale_mlp, indices)
        else:
            normed = normed * (1.0 + scale_mlp.index_select(0, indices))
            normed = normed + shift_mlp.index_select(0, indices)
        ff_output = self._ff(weights.ff, normed)
        if self.sglang_parity_ops:
            return indexed_gate_sglang(residual, gate_mlp, ff_output, indices)
        return residual + gate_mlp.index_select(0, indices) * ff_output

    def _compute_adaln_table(self, weights, pre_infer_out):
        # ADALN CACHE SYNC: This projection is reproduced by the offline builder.
        # Keep the offline AdaLN cache builder and its tests aligned with changes
        # to activation placement, dtype, projection, gather, or reshape.
        # Activation is evaluated in fp32, then cast to the inference dtype
        # immediately before the (possibly quantized) AdaLN projection.
        if pre_infer_out.temb is None:
            raise RuntimeError("MiniMax-H3 timestep embedding is missing")
        modulation = weights.adaln.apply(F.silu(pre_infer_out.temb).to(self.infer_dtype))
        modulation = self._gather_tp_last_dim(modulation)
        return modulation.view(-1, 6 * self.hidden_size)

    def _clear_adaln_cache(self):
        # Online mode has no reusable cache; it only creates a temporary table
        # for the current block invocation in _compute_adaln_table().
        self._current_adaln_tables = None
        self._adaln_cache_hit = False

    def _prepare_adaln_cache(self, pre_infer_out):
        current_timesteps = tuple(self.scheduler.unique_timesteps_cpu.tolist())
        cached_tables = self._adaln_cache.get(current_timesteps)
        norm_out_modulation = self._norm_out_cache.get(current_timesteps)
        if cached_tables is not None and norm_out_modulation is not None:
            self._current_adaln_tables = cached_tables
            pre_infer_out.norm_out_modulation = norm_out_modulation
            self._adaln_cache_hit = True
        else:
            raise KeyError(f"Persistent MiniMax-H3 AdaLN cache has no entry for timesteps {current_timesteps}")

    def _get_cached_adaln(self, block_index):
        if self._adaln_cache_hit:
            return self._current_adaln_tables[block_index]
        raise RuntimeError("Persistent MiniMax-H3 AdaLN cache was not prepared")

    def run_block(self, block_idx, block, hidden_states, pre_infer_out):
        if self.use_adaln_cache:
            adaln_table = self._get_cached_adaln(block_idx)
            return super().run_block(block_idx, block, hidden_states, pre_infer_out, adaln_table)
        return super().run_block(block_idx, block, hidden_states, pre_infer_out)

    def infer_without_offload(self, blocks, hidden_states, pre_infer_out):
        for block_index, block in enumerate(blocks):
            self.block_idx = block_index
            hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        return self.infer_func(block_weights.blocks, pre_infer_out.hidden_states, pre_infer_out)
