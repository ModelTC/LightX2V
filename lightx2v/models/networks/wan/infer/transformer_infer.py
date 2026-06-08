from functools import partial

import torch
from loguru import logger

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import *
from lightx2v_platform.base.global_var import AI_DEVICE

from .triton_ops import fuse_scale_shift_kernel
from .utils import apply_wan_rope_with_chunk, apply_wan_rope_with_flashinfer, apply_wan_rope_with_torch, apply_wan_rope_with_torch_naive

try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_mxfp8_mm,
        cutlass_scaled_mxfp8_mm_residual_gate,
        scaled_mxfp8_gelu_quant,
        scaled_mxfp8_modulate_quant,
    )

    _WAN_MXFP8_FFN_IMPORT_ERROR = None
except Exception as exc:
    cutlass_scaled_mxfp8_mm, cutlass_scaled_mxfp8_mm_residual_gate = None, None
    scaled_mxfp8_gelu_quant = None
    scaled_mxfp8_modulate_quant = None
    _WAN_MXFP8_FFN_IMPORT_ERROR = exc

torch_device_module = getattr(torch, AI_DEVICE)


def modulate(x, scale, shift):
    return x * (1 + scale.squeeze()) + shift.squeeze()


class WanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.task = config["task"]
        self.blocks_num = config["num_layers"]
        self.phases_num = 3
        self.has_post_adapter = False
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = modulate
        rope_funcs = {
            "flashinfer": apply_wan_rope_with_flashinfer,
            "torch": apply_wan_rope_with_torch,
            "torch_naive": apply_wan_rope_with_torch_naive,
        }
        rope_type = self.config.get("rope_type", "flashinfer")
        # Try to get rope function from registry first (for platform-specific implementations)
        if rope_type in ROPE_REGISTER:
            rope_class = ROPE_REGISTER[rope_type]
            self.rope_instance = rope_class()

            # Create a wrapper function that matches the expected signature
            def rope_wrapper(xq, xk, cos_sin_cache):
                return self.rope_instance.apply(xq, xk, cos_sin_cache)

            rope_func = rope_wrapper
        else:
            # Fallback to hardcoded functions
            rope_func = rope_funcs.get(rope_type, apply_wan_rope_with_torch)
        if self.config.get("rope_chunk", False):
            self.apply_rope_func = partial(apply_wan_rope_with_chunk, chunk_size=self.config.get("rope_chunk_size", 100), rope_func=rope_func)
        else:
            self.apply_rope_func = rope_func
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.mxfp8_fuse_enable = self.config.get("mxfp8_fuse_enable", True)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

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

        self._mxfp8_fuse_available = self._probe_mxfp8_fuse_availability() if self.mxfp8_fuse_enable else False

    def _probe_mxfp8_fuse_availability(self):
        """Probe once whether MXFP8 fused ops can run on this device.

        Returns False (with a warning) if the kernel is unavailable or the GPU
        is not SM120/SM120a, so the inference falls back to the non-fused path.
        """
        if self.config.get("dit_quant_scheme", "Default") != "mxfp8":
            return False
        if not torch.cuda.is_available():
            logger.warning("MXFP8 fused ops require a CUDA device, falling back to non-fused path")
            return False
        if cutlass_scaled_mxfp8_mm is None or cutlass_scaled_mxfp8_mm_residual_gate is None or scaled_mxfp8_gelu_quant is None or scaled_mxfp8_modulate_quant is None:
            detail = f": {type(_WAN_MXFP8_FFN_IMPORT_ERROR).__name__}: {_WAN_MXFP8_FFN_IMPORT_ERROR}" if _WAN_MXFP8_FFN_IMPORT_ERROR is not None else ""
            logger.warning(f"MXFP8 fused ops unavailable, falling back to non-fused path{detail}")
            return False
        major, minor = torch.cuda.get_device_capability()
        if major != 12:
            logger.warning(f"MXFP8 fused ops require SM120/SM120a, got SM{major}.{minor}, falling back to non-fused path")
            return False
        return True

    def _use_mxfp8_quant_fuse(self):
        return self.mxfp8_fuse_enable and self._mxfp8_fuse_available

    def _ensure_mxfp8_quant_fuse_ready(self, phase, *tensors, module_names=(), required_module_attrs=("weight", "weight_scale", "alpha")):
        if not self._use_mxfp8_quant_fuse():
            return
        for tensor in tensors:
            if tensor is None:
                continue
            if not tensor.is_cuda:
                raise RuntimeError("mxfp8_quant_fuse expects CUDA activations")
        device_tensor = next((tensor for tensor in tensors if tensor is not None), None)
        if device_tensor is None:
            raise RuntimeError("mxfp8_quant_fuse requires at least one CUDA tensor for device validation")
        major, _minor = torch.cuda.get_device_capability(device_tensor.device)
        if major != 12:
            raise RuntimeError("mxfp8_quant_fuse is only enabled on SM120/SM120a GPUs")
        if cutlass_scaled_mxfp8_mm is None or cutlass_scaled_mxfp8_mm_residual_gate is None or scaled_mxfp8_gelu_quant is None or scaled_mxfp8_modulate_quant is None:
            detail = f": {type(_WAN_MXFP8_FFN_IMPORT_ERROR).__name__}: {_WAN_MXFP8_FFN_IMPORT_ERROR}" if _WAN_MXFP8_FFN_IMPORT_ERROR is not None else ""
            raise RuntimeError(f"mxfp8_quant_fuse requires lightx2v_kernel with MXFP8 fused quant ops{detail}")
        for name in module_names:
            module = getattr(phase, name)
            if getattr(module, "has_lora_branch", False) or getattr(module, "has_diff", False):
                raise RuntimeError(f"mxfp8_quant_fuse does not support active LoRA/diff on {name}")
            if not all(hasattr(module, attr) for attr in required_module_attrs):
                raise RuntimeError(f"mxfp8_quant_fuse expects {name} to be an MXFP8 quantized weight module")

    def _ensure_mxfp8_quant_ffn_ready(self, phase, norm2_out, residual, c_gate_msa=None, c_scale_msa=None, c_shift_msa=None):
        if not self._use_mxfp8_quant_fuse():
            return
        if (c_scale_msa is None) != (c_shift_msa is None):
            raise RuntimeError("MXFP8 FFN modulate-quant readiness requires both c_scale_msa and c_shift_msa")
        extra_tensors = []
        self._ensure_mxfp8_quant_fuse_ready(
            phase,
            norm2_out,
            residual,
            c_scale_msa,
            c_shift_msa,
            module_names=("ffn_0", "ffn_2"),
            required_module_attrs=("act_quant_func", "weight", "weight_scale", "alpha"),
        )
        if c_gate_msa is None:
            raise RuntimeError("mxfp8_quant_fuse requires c_gate_msa for residual-gate fusion")
        extra_tensors.append(c_gate_msa)
        if extra_tensors:
            self._ensure_mxfp8_quant_fuse_ready(phase, *extra_tensors)

    def _can_use_mxfp8_modulate_quant(self, norm2_out, c_scale_msa, c_shift_msa):
        if scaled_mxfp8_modulate_quant is None:
            return False
        if not self._use_mxfp8_quant_fuse():
            return False
        if self.sensitive_layer_dtype != self.infer_dtype:
            return False
        if norm2_out.dtype != torch.bfloat16 or c_scale_msa.dtype != torch.bfloat16 or c_shift_msa.dtype != torch.bfloat16:
            return False
        if not (norm2_out.is_cuda and c_scale_msa.is_cuda and c_shift_msa.is_cuda):
            return False
        if norm2_out.device != c_scale_msa.device or norm2_out.device != c_shift_msa.device:
            return False
        if norm2_out.dim() != 2 or not norm2_out.is_contiguous():
            return False
        hidden = norm2_out.shape[1]
        tokens = norm2_out.shape[0]
        valid_numel = (hidden, tokens * hidden)
        return c_scale_msa.numel() in valid_numel and c_shift_msa.numel() in valid_numel

    def _can_reuse_self_attn_mxfp8_quant(self, phase, norm1_out, scale_msa, shift_msa):
        if cutlass_scaled_mxfp8_mm is None:
            return False
        if not self._can_use_mxfp8_modulate_quant(norm1_out, scale_msa, shift_msa):
            return False
        for name in ("self_attn_q", "self_attn_k", "self_attn_v"):
            module = getattr(phase, name)
            if getattr(module, "has_lora_branch", False) or getattr(module, "has_diff", False):
                return False
            if not all(hasattr(module, attr) for attr in ("weight", "weight_scale", "alpha")):
                return False
        return True

    def _mxfp8_quant_bias(self, module):
        if hasattr(module, "_get_actual_bias"):
            return module._get_actual_bias()
        return module.bias if hasattr(module, "bias") else None

    def _mxfp8_apply(self, module, input_tensor):
        input_tensor_quant, input_tensor_scale = module.act_quant_func(input_tensor)
        return self._mxfp8_apply_quantized(module, input_tensor_quant, input_tensor_scale)

    def _mxfp8_apply_quantized(self, module, input_tensor_quant, input_tensor_scale):
        if module.alpha.device != module.weight.device:
            module.alpha = module.alpha.to(module.weight.device)
        return cutlass_scaled_mxfp8_mm(
            input_tensor_quant,
            module.weight,
            input_tensor_scale,
            module.weight_scale,
            alpha=module.alpha,
            bias=self._mxfp8_quant_bias(module),
        )

    def _mxfp8_apply_residual_gate(self, module, input_tensor, residual, gate):
        input_tensor_quant, input_tensor_scale = module.act_quant_func(input_tensor)
        return self._mxfp8_apply_residual_gate_quantized(module, input_tensor_quant, input_tensor_scale, residual, gate)

    def _mxfp8_apply_residual_gate_quantized(self, module, input_tensor_quant, input_tensor_scale, residual, gate):
        if module.alpha.device != module.weight.device:
            module.alpha = module.alpha.to(module.weight.device)
        return cutlass_scaled_mxfp8_mm_residual_gate(
            input_tensor_quant,
            module.weight,
            input_tensor_scale,
            module.weight_scale,
            alpha=module.alpha,
            residual=residual,
            gate=gate,
            bias=self._mxfp8_quant_bias(module),
        )

    def _infer_ffn_with_mxfp8_quant_fuse(self, phase, norm2_out, residual, c_gate_msa=None, c_scale_msa=None, c_shift_msa=None):
        """Run the fused MXFP8 FFN path and update residual in place.

        The fused residual-gate kernel writes the FFN contribution directly
        into ``residual``. Returning ``None`` signals ``post_process`` to skip
        the usual ``x + y * gate`` accumulation.
        """
        self._ensure_mxfp8_quant_ffn_ready(phase, norm2_out, residual, c_gate_msa, c_scale_msa, c_shift_msa)
        if c_scale_msa is not None and c_shift_msa is not None and self._can_use_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa):
            norm2_quant, norm2_scale = scaled_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa)
            y = self._mxfp8_apply_quantized(phase.ffn_0, norm2_quant, norm2_scale)
        else:
            norm2_quant = None
            norm2_scale = None
            y = self._mxfp8_apply(phase.ffn_0, norm2_out)
        y_quant, y_scale = scaled_mxfp8_gelu_quant(y)
        self._mxfp8_apply_residual_gate_quantized(phase.ffn_2, y_quant, y_scale, residual, c_gate_msa.squeeze())
        if self.clean_cuda_cache:
            del norm2_out
            del y, y_quant, y_scale
            if norm2_quant is not None:
                del norm2_quant, norm2_scale
            torch_device_module.empty_cache()
        return None

    @torch.no_grad()
    def reset_post_adapter_states(self):
        pass

    def reset_infer_states(self):
        self.self_attn_cu_seqlens_qkv = None
        self.cross_attn_cu_seqlens_q = None
        self.cross_attn_cu_seqlens_kv = None
        self.cross_attn_cu_seqlens_kv_img = None
        if self.has_post_adapter:
            self.reset_post_adapter_states()

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.cos_sin = pre_infer_out.cos_sin
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def infer_main_blocks(self, blocks, pre_infer_out):
        x = self.infer_func(blocks, pre_infer_out.x, pre_infer_out)
        return x

    def infer_non_blocks(self, weights, x, e):
        if e.dim() == 2:
            modulation = weights.head_modulation.tensor  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:  # For Diffustion forcing
            modulation = weights.head_modulation.tensor.unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]

        x = weights.norm.apply(x)

        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.infer_dtype)

        x = weights.head.apply(x)

        if self.clean_cuda_cache:
            del e
            torch_device_module.empty_cache()
        return x

    def infer_without_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out)
        return x

    def infer_block(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
        )
        x, attn_out = self.infer_cross_attn(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )
        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def pre_process(self, modulation, embed0):
        if embed0.dim() == 3 and embed0.shape[2] == 1:
            modulation = modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (modulation.tensor + embed0).chunk(6, dim=1)

        if self.clean_cuda_cache:
            del embed0
            torch_device_module.empty_cache()

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def infer_self_attn(self, phase, x, shift_msa, scale_msa):
        cos_sin = self.cos_sin
        norm1_quant = None
        norm1_scale = None
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        else:
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            if self._use_mxfp8_quant_fuse():
                self._ensure_mxfp8_quant_fuse_ready(
                    phase,
                    norm1_out,
                    scale_msa,
                    shift_msa,
                    module_names=("self_attn_q", "self_attn_k", "self_attn_v"),
                )
            if self._can_reuse_self_attn_mxfp8_quant(phase, norm1_out, scale_msa, shift_msa):
                norm1_quant, norm1_scale = scaled_mxfp8_modulate_quant(norm1_out, scale_msa, shift_msa)
            else:
                norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        if norm1_quant is not None:
            q = phase.self_attn_norm_q.apply(self._mxfp8_apply_quantized(phase.self_attn_q, norm1_quant, norm1_scale)).view(s, n, d)
            k = phase.self_attn_norm_k.apply(self._mxfp8_apply_quantized(phase.self_attn_k, norm1_quant, norm1_scale)).view(s, n, d)
            v = self._mxfp8_apply_quantized(phase.self_attn_v, norm1_quant, norm1_scale).view(s, n, d)
        else:
            q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
            k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
            v = phase.self_attn_v.apply(norm1_out).view(s, n, d)
        q, k = self.apply_rope_func(q, k, cos_sin)
        img_qkv_len = q.shape[0]
        if self.self_attn_cu_seqlens_qkv is None:
            self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)

        if self.clean_cuda_cache:
            del norm1_out, shift_msa, scale_msa
            if norm1_quant is not None:
                del norm1_quant, norm1_scale
            torch_device_module.empty_cache()

        attn_running_args = {
            "block_idx": self.block_idx,
            "scheduler": self.scheduler,
        }

        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=k,
                v=v,
                slice_qkv_len=img_qkv_len,
                cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                use_tensor_fusion=self.seq_p_tensor_fusion,
                enable_head_parallel=self.enable_head_parallel,
                **attn_running_args,
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.self_attn_cu_seqlens_qkv,
                cu_seqlens_kv=self.self_attn_cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
                **attn_running_args,
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch_device_module.empty_cache()

        return y

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        norm3_out = phase.norm3.apply(x)
        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
        v = phase.cross_attn_v.apply(context).view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)

            if self.cross_attn_cu_seqlens_kv_img is None:
                self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32)

            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch_device_module.empty_cache()

        mxfp8_modulate_scale = None
        mxfp8_modulate_shift = None
        if hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
        else:
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            if self._use_mxfp8_quant_fuse():
                self._ensure_mxfp8_quant_ffn_ready(phase, norm2_out, x, c_gate_msa, c_scale_msa, c_shift_msa)
            if self._can_use_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa):
                mxfp8_modulate_scale = c_scale_msa
                mxfp8_modulate_shift = c_shift_msa
            else:
                norm2_out = self.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)

        if self._use_mxfp8_quant_fuse():
            return self._infer_ffn_with_mxfp8_quant_fuse(
                phase,
                norm2_out,
                x,
                c_gate_msa,
                c_scale_msa=mxfp8_modulate_scale,
                c_shift_msa=mxfp8_modulate_shift,
            )

        y = phase.ffn_0.apply(norm2_out)
        if self.clean_cuda_cache:
            del norm2_out, x
            torch_device_module.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch_device_module.empty_cache()
        y = phase.ffn_2.apply(y)

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        if y is None:
            return x
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y.to(self.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x.add_(y * c_gate_msa.squeeze())

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch_device_module.empty_cache()
        return x
