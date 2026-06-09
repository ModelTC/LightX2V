import torch
from loguru import logger

from lightx2v_platform.base.global_var import AI_DEVICE

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


class WanMxfp8FuseMixin:
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
