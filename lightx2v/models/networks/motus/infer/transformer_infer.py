from functools import partial

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import fuse_scale_shift_kernel
from lightx2v.models.networks.wan.infer.utils import (
    apply_wan_rope_with_chunk,
    apply_wan_rope_with_flashinfer,
    apply_wan_rope_with_torch,
    apply_wan_rope_with_torch_naive,
)
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.registry_factory import ROPE_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from ..ops import RegistryAttention

torch_device_module = getattr(torch, AI_DEVICE)


def modulate(x, scale, shift):
    return x * (1 + scale.squeeze(2)) + shift.squeeze(2)


class MotusTransformerInfer(BaseTransformerInfer):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.self_joint_attn_type = config.get("self_joint_attn_type", "flash_attn2")
        self.cross_attn_type = config.get("cross_attn_type","flash_attn2")
        self.joint_self_attn = RegistryAttention(self.self_joint_attn_type)
        self.cross_attn = RegistryAttention(self.cross_attn_type)
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()
        self.modulate_func = fuse_scale_shift_kernel if config.get("modulate_type", "triton") == "triton" else modulate

        rope_funcs = {
            "flashinfer": apply_wan_rope_with_flashinfer,
            "torch": apply_wan_rope_with_torch,
            "torch_naive": apply_wan_rope_with_torch_naive,
        }
        rope_type = config.get("rope_type", "flashinfer")
        if rope_type in ROPE_REGISTER:
            rope_class = ROPE_REGISTER[rope_type]
            self.rope_instance = rope_class()

            def rope_wrapper(xq, xk, cos_sin_cache):
                return self.rope_instance.apply(xq, xk, cos_sin_cache)

            rope_func = rope_wrapper
        else:
            rope_func = rope_funcs.get(rope_type, apply_wan_rope_with_torch)
        if config.get("rope_chunk", False):
            rope_func = partial(
                apply_wan_rope_with_chunk,
                chunk_size=config.get("rope_chunk_size", 100),
                rope_func=rope_func,
            )
        self.apply_rope_func = rope_func

        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
            self.seq_p_tensor_fusion = self.config["parallel"].get("seq_p_tensor_fusion", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False
            self.seq_p_tensor_fusion = False

        self.infer_func = self.infer_without_offload
        self.cos_sin = None
        self.reset_infer_states()

    def reset_infer_states(self):
        self.joint_attn_cu_seqlens_q = None
        self.joint_attn_cu_seqlens_kv = None
        self.cross_attn_cu_seqlens_q = None
        self.cross_attn_cu_seqlens_kv = None

    def _maybe_empty_cache(self):
        if self.clean_cuda_cache:
            torch_device_module.empty_cache()

    def _get_wan_layer(self, layer_idx):
        return self.model.model.video_module.video_model.wan_model.blocks[layer_idx]

    def _get_action_block(self, layer_idx):
        return self.model.model.action_expert.blocks[layer_idx]

    def _get_und_block(self, layer_idx):
        return self.model.model.und_expert.blocks[layer_idx]

    def _get_cu_seqlens(self, cache_name, batch, seq_len, device, attn_type):
        cu_seqlens = getattr(self, cache_name)
        expected_total = batch * seq_len
        if cu_seqlens is None or int(cu_seqlens[-1].item()) != expected_total or cu_seqlens.device != device:
            tensor = torch.arange(0, expected_total + seq_len, seq_len, dtype=torch.int32)
            if attn_type in ["flash_attn2", "flash_attn3"]:
                cu_seqlens = tensor.to(device, non_blocking=True)
            else:
                cu_seqlens = tensor
            setattr(self, cache_name, cu_seqlens)
        return cu_seqlens

    def _modulate(self, x, scale, shift):
        out_dtype = x.dtype
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x = self.modulate_func(x.contiguous(), scale=scale, shift=shift)
        if x.dtype != out_dtype:
            x = x.to(out_dtype)
        return x

    def _apply_gate(self, x, y, gate):
        out_dtype = x.dtype
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = x.to(self.sensitive_layer_dtype) + y.to(self.sensitive_layer_dtype) * gate.squeeze(2)
            return out.to(out_dtype)
        out = x + y * gate.squeeze(2)
        return out if out.dtype == out_dtype else out.to(out_dtype)

    def _apply_norm(self, norm_module, x):
        if isinstance(norm_module, torch.nn.LayerNorm):
            weight = norm_module.weight.float() if norm_module.weight is not None else None
            bias = norm_module.bias.float() if norm_module.bias is not None else None
            out = F.layer_norm(x.float(), norm_module.normalized_shape, weight, bias, norm_module.eps)
            return out.to(x.dtype)
        return norm_module(x)

    def _get_module_dtype(self, module):
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                return tensor.dtype
        for tensor in module.parameters():
            if tensor.is_floating_point():
                return tensor.dtype
        for tensor in module.buffers():
            if tensor.is_floating_point():
                return tensor.dtype
        return None

    def _cast_for_module_input(self, x, module):
        target_dtype = self._get_module_dtype(module)
        if target_dtype is not None and x.dtype != target_dtype:
            return x.to(target_dtype)
        return x

    def _apply_video_rope(self, q, k, cos_sin_cache):
        if q.dim() != 4:
            raise ValueError("Motus video rope expects q/k with shape [B, L, H, D].")

        if q.shape[0] == 1:
            q_out, k_out = self.apply_rope_func(q.squeeze(0), k.squeeze(0), cos_sin_cache)
            return q_out.unsqueeze(0), k_out.unsqueeze(0)

        q_list = []
        k_list = []
        for batch_idx in range(q.shape[0]):
            q_i, k_i = self.apply_rope_func(q[batch_idx], k[batch_idx], cos_sin_cache)
            q_list.append(q_i)
            k_list.append(k_i)
        return torch.stack(q_list, dim=0), torch.stack(k_list, dim=0)

    def _infer_joint_attention(self, pre_infer_out, video_tokens, action_tokens, und_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx):
        model = self.model.model
        wan_layer = self._get_wan_layer(layer_idx)
        action_block = self._get_action_block(layer_idx)
        und_block = self._get_und_block(layer_idx)

        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        norm_video = self._modulate(wan_layer.norm1(video_tokens), v_mod[1], v_mod[0])
        norm_action = self._modulate(action_block.norm1(action_tokens), a_mod[1], a_mod[0])
        norm_und = und_block.norm1(und_tokens)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm_und = norm_und.to(self.infer_dtype)

        batch, video_len, video_dim = norm_video.shape
        action_len = norm_action.shape[1]
        und_len = norm_und.shape[1]
        num_heads = model.video_model.wan_model.num_heads
        head_dim = video_dim // num_heads

        video_q = wan_layer.self_attn.norm_q(wan_layer.self_attn.q(norm_video)).view(batch, video_len, num_heads, head_dim)
        video_k = wan_layer.self_attn.norm_k(wan_layer.self_attn.k(norm_video)).view(batch, video_len, num_heads, head_dim)
        video_v = wan_layer.self_attn.v(norm_video).view(batch, video_len, num_heads, head_dim)

        video_q, video_k = self._apply_video_rope(video_q, video_k, pre_infer_out.cos_sin)

        action_q, action_k, action_v = action_block.wan_action_qkv_mm(norm_action)
        action_q = action_block.wan_action_norm_q(action_q.flatten(-2)).view(batch, action_len, num_heads, head_dim)
        action_k = action_block.wan_action_norm_k(action_k.flatten(-2)).view(batch, action_len, num_heads, head_dim)

        und_q, und_k, und_v = und_block.wan_und_qkv_mm(norm_und)
        und_q = und_block.wan_und_norm_q(und_q.flatten(-2)).view(batch, und_len, num_heads, head_dim)
        und_k = und_block.wan_und_norm_k(und_k.flatten(-2)).view(batch, und_len, num_heads, head_dim)

        q_all = torch.cat([video_q, action_q, und_q], dim=1)
        k_all = torch.cat([video_k, action_k, und_k], dim=1)
        v_all = torch.cat([video_v, action_v, und_v], dim=1)
        total_len = q_all.shape[1]

        cu_seqlens = self._get_cu_seqlens("joint_attn_cu_seqlens_q", batch, total_len, q_all.device, self.self_joint_attn_type)
        attn_running_args = {"block_idx": layer_idx, "scheduler": self.scheduler}
        attn_out = self.joint_self_attn(
            q_all,
            k_all,
            v_all,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            **attn_running_args,
        )

        video_out = wan_layer.self_attn.o(attn_out[:, :video_len, :])
        action_out = action_block.wan_action_o(attn_out[:, video_len : video_len + action_len, :])
        und_out = und_block.wan_und_o(attn_out[:, video_len + action_len :, :])

        video_tokens = self._apply_gate(video_tokens, video_out, v_mod[2])
        action_tokens = self._apply_gate(action_tokens, action_out, a_mod[2])
        und_tokens = und_tokens + und_out

        if self.clean_cuda_cache:
            del norm_video, norm_action, norm_und
            del video_q, video_k, video_v
            del action_q, action_k, action_v
            del und_q, und_k, und_v
            del q_all, k_all, v_all, attn_out
            self._maybe_empty_cache()

        return video_tokens, action_tokens, und_tokens

    def _infer_cross_attention(self, video_tokens, processed_t5_context, layer_idx):
        wan_layer = self._get_wan_layer(layer_idx)
        norm_video = self._apply_norm(wan_layer.norm3, video_tokens)
        cross_attn_dtype = wan_layer.cross_attn.q.weight.dtype
        norm_video = norm_video.to(cross_attn_dtype)

        batch, q_len, dim = norm_video.shape
        ctx_len = processed_t5_context.shape[1]
        num_heads = wan_layer.cross_attn.num_heads
        head_dim = dim // num_heads

        q = wan_layer.cross_attn.norm_q(wan_layer.cross_attn.q(norm_video)).view(batch, q_len, num_heads, head_dim)
        context = processed_t5_context.to(wan_layer.cross_attn.k.weight.dtype)
        k = wan_layer.cross_attn.norm_k(wan_layer.cross_attn.k(context)).view(batch, ctx_len, num_heads, head_dim)
        v = wan_layer.cross_attn.v(context).view(batch, ctx_len, num_heads, head_dim)

        cu_seqlens_q = self._get_cu_seqlens("cross_attn_cu_seqlens_q", batch, q_len, q.device, self.cross_attn_type)
        cu_seqlens_kv = self._get_cu_seqlens("cross_attn_cu_seqlens_kv", batch, ctx_len, k.device, self.cross_attn_type)
        attn_out = self.cross_attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=q_len,
            max_seqlen_kv=ctx_len,
        )
        attn_out = wan_layer.cross_attn.o(attn_out)
        video_tokens = video_tokens + attn_out

        if self.clean_cuda_cache:
            del norm_video, q, k, v, attn_out, context
            self._maybe_empty_cache()

        return video_tokens

    def _infer_video_ffn(self, video_tokens, video_adaln_modulation, layer_idx):
        wan_layer = self._get_wan_layer(layer_idx)
        v_mod = video_adaln_modulation
        ffn_input = self._modulate(wan_layer.norm2(video_tokens), v_mod[4], v_mod[3])
        ffn_input = self._cast_for_module_input(ffn_input, wan_layer.ffn)
        ffn_out = wan_layer.ffn(ffn_input)
        return self._apply_gate(video_tokens, ffn_out, v_mod[5])

    def _infer_action_ffn(self, action_tokens, action_adaln_modulation, layer_idx):
        action_block = self._get_action_block(layer_idx)
        a_mod = action_adaln_modulation
        ffn_input = self._modulate(action_block.norm2(action_tokens), a_mod[4], a_mod[3])
        ffn_input = self._cast_for_module_input(ffn_input, action_block.ffn)
        ffn_out = action_block.ffn(ffn_input)
        return self._apply_gate(action_tokens, ffn_out, a_mod[5])

    def _infer_und_ffn(self, und_tokens, layer_idx):
        und_block = self._get_und_block(layer_idx)
        ffn_input = und_block.norm2(und_tokens)
        ffn_input = self._cast_for_module_input(ffn_input, und_block.ffn)
        return und_tokens + und_block.ffn(ffn_input)

    def _prepare_action_tokens(self, pre_infer_out, action_latents):
        model = self.model.model
        input_dtype = self._get_module_dtype(model.action_expert.input_encoder) or model.dtype
        state_tokens = pre_infer_out.state.unsqueeze(1).to(dtype=input_dtype)
        action_latents = action_latents.to(dtype=input_dtype)
        registers = model.action_expert.registers
        if registers is not None:
            registers = registers.expand(state_tokens.shape[0], -1, -1).to(dtype=input_dtype)
        action_tokens = model.action_expert.input_encoder(state_tokens, action_latents, registers)
        return state_tokens, action_tokens.to(model.dtype)

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        del weights
        self.cos_sin = pre_infer_out.cos_sin
        self.reset_infer_states()

        model = self.model.model
        processed_t5_context = pre_infer_out.context
        und_tokens = pre_infer_out.und_tokens.clone()
        timestep = self.scheduler.timestep_input[0]

        video_tokens = model.video_module.prepare_input(self.scheduler.video_latents.to(model.dtype))
        _, action_tokens = self._prepare_action_tokens(pre_infer_out, self.scheduler.action_latents)

        timestep_scaled = (timestep * 1000).expand(action_tokens.shape[0]).to(model.dtype)
        video_head_time_emb, video_adaln_params = model.video_module.get_time_embedding(timestep_scaled, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = model.action_module.get_time_embedding(timestep_scaled, action_tokens.shape[1])
        hidden_states = (video_tokens, action_tokens, und_tokens)
        infer_state = {
            "processed_t5_context": processed_t5_context,
            "video_head_time_emb": video_head_time_emb,
            "video_adaln_params": video_adaln_params,
            "action_head_time_emb": action_head_time_emb,
            "action_adaln_params": action_adaln_params,
        }
        hidden_states = self.infer_main_blocks(range(model.config.num_layers), hidden_states, pre_infer_out, infer_state)
        return self.infer_non_blocks(hidden_states, infer_state)

    def infer_main_blocks(self, blocks, hidden_states, pre_infer_out, infer_state):
        return self.infer_func(blocks, hidden_states, pre_infer_out, infer_state)

    def infer_without_offload(self, blocks, hidden_states, pre_infer_out, infer_state):
        for block in blocks:
            hidden_states = self.infer_block(block, hidden_states, pre_infer_out, infer_state)
        return hidden_states

    def infer_block(self, block, hidden_states, pre_infer_out, infer_state):
        video_tokens, action_tokens, und_tokens = hidden_states
        model = self.model.model
        video_adaln_modulation = model.video_module.compute_adaln_modulation(infer_state["video_adaln_params"], block)
        action_adaln_modulation = model.action_module.compute_adaln_modulation(infer_state["action_adaln_params"], block)

        video_tokens, action_tokens, und_tokens = self._infer_joint_attention(
            pre_infer_out,
            video_tokens,
            action_tokens,
            und_tokens,
            video_adaln_modulation,
            action_adaln_modulation,
            block,
        )
        video_tokens = self._infer_cross_attention(video_tokens, infer_state["processed_t5_context"], block)
        action_tokens = self._infer_action_ffn(action_tokens, action_adaln_modulation, block)
        und_tokens = self._infer_und_ffn(und_tokens, block)
        video_tokens = self._infer_video_ffn(video_tokens, video_adaln_modulation, block)
        return video_tokens, action_tokens, und_tokens

    def infer_non_blocks(self, hidden_states, infer_state):
        model = self.model.model
        video_tokens, action_tokens, _ = hidden_states
        video_velocity = model.video_module.apply_output_head(video_tokens, infer_state["video_head_time_emb"])
        action_pred_full = model.action_expert.decoder(action_tokens, infer_state["action_head_time_emb"])
        num_regs = model.action_expert.config.num_registers
        action_velocity = action_pred_full[:, 1:-num_regs, :] if num_regs > 0 else action_pred_full[:, 1:, :]
        return video_velocity, action_velocity
