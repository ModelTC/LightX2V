import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class Hunyuan3DFeedForwardWeights(WeightModule):
    """Diffusers FeedForward weights (gelu MLP)."""

    def __init__(self, prefix, mm_type):
        super().__init__()
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.net.0.proj.weight",
                f"{prefix}.net.0.proj.bias",
            ),
        )
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.net.2.weight",
                f"{prefix}.net.2.bias",
            ),
        )


class Hunyuan3DMLPWeights(WeightModule):
    def __init__(self, prefix, mm_type):
        super().__init__()
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.fc1.weight",
                f"{prefix}.fc1.bias",
            ),
        )
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.fc2.weight",
                f"{prefix}.fc2.bias",
            ),
        )


class Hunyuan3DMoEWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type):
        super().__init__()
        prefix = f"blocks.{block_idx}.moe"
        num_experts = config.get("num_experts", 8)
        moe_mm_type = "Default" if str(mm_type).startswith("fp8") else mm_type
        self.num_experts = num_experts
        self.moe_top_k = config.get("moe_top_k", 2)
        self.moe_backend = str(config.get("moe_backend", "pytorch")).strip().lower()
        if self.moe_backend not in ("pytorch", "flashinfer"):
            raise ValueError(f"Invalid Hunyuan3D moe_backend={self.moe_backend!r}, expected 'pytorch' or 'flashinfer'")
        fi_cfg = config.get("moe_flashinfer_setting") or {}
        if fi_cfg.get("autotune") and self.moe_backend != "flashinfer":
            raise ValueError("moe_flashinfer_setting.autotune=true requires moe_backend='flashinfer'")
        self.moe_flashinfer_tune_max_num_tokens = int(fi_cfg.get("tune_max_num_tokens", 8192))
        self.add_module(
            "gate",
            MM_WEIGHT_REGISTER[moe_mm_type](
                f"{prefix}.gate.weight",
                None,
            ),
        )
        self.add_module(
            "shared_experts",
            Hunyuan3DFeedForwardWeights(f"{prefix}.shared_experts", moe_mm_type),
        )
        experts = WeightModuleList(Hunyuan3DFeedForwardWeights(f"{prefix}.experts.{expert_idx}", moe_mm_type) for expert_idx in range(num_experts))
        self.add_module("experts", experts)

    def load(self, weight_dict):
        super().load(weight_dict)
        self._validate_no_fp8_moe_weights()
        if self.moe_backend == "flashinfer" and self.experts[0].fc1._get_actual_weight() is not None:
            self._build_flashinfer_weights()

    @staticmethod
    def _is_fp8_tensor(tensor):
        return tensor is not None and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    def _iter_moe_mm_weights(self):
        yield "gate", self.gate
        yield "shared_experts.fc1", self.shared_experts.fc1
        yield "shared_experts.fc2", self.shared_experts.fc2
        for expert_idx, expert in enumerate(self.experts):
            yield f"experts.{expert_idx}.fc1", expert.fc1
            yield f"experts.{expert_idx}.fc2", expert.fc2

    def _validate_no_fp8_moe_weights(self):
        for name, module in self._iter_moe_mm_weights():
            for attr in ("weight", "pin_weight", "weight_cuda_buffer"):
                if self._is_fp8_tensor(getattr(module, attr, None)):
                    raise ValueError(f"Hunyuan3D MoE FP8 is not supported yet, but {name}.{attr} is FP8. Regenerate the checkpoint so .moe. weights stay BF16.")

    def to_cuda(self, non_blocking=False):
        super().to_cuda(non_blocking=non_blocking)
        if self.moe_backend == "flashinfer":
            self._build_flashinfer_weights()

    def to_cpu(self, non_blocking=False):
        super().to_cpu(non_blocking=non_blocking)
        for attr in ("_fi_fc1_weight", "_fi_fc2_weight", "_fi_fc1_bias", "_fi_fc2_bias"):
            if hasattr(self, attr):
                delattr(self, attr)

    @staticmethod
    def _stack_optional_biases(biases):
        if all(bias is None for bias in biases):
            return None
        if any(bias is None for bias in biases):
            raise ValueError("FlashInfer Hunyuan3D MoE requires either all expert biases or no expert biases")
        return torch.stack([bias.contiguous() for bias in biases], dim=0)

    def _build_flashinfer_weights(self):
        fc1_list, fc2_list = [], []
        fc1_biases, fc2_biases = [], []
        for expert_w in self.experts:
            fc1_list.append(expert_w.fc1._get_actual_weight().t().contiguous())
            fc2_list.append(expert_w.fc2._get_actual_weight().t().contiguous())
            fc1_biases.append(expert_w.fc1._get_actual_bias())
            fc2_biases.append(expert_w.fc2._get_actual_bias())
        self._fi_fc1_weight = torch.stack(fc1_list, dim=0)
        self._fi_fc2_weight = torch.stack(fc2_list, dim=0)
        self._fi_fc1_bias = self._stack_optional_biases(fc1_biases)
        self._fi_fc2_bias = self._stack_optional_biases(fc2_biases)


class Hunyuan3DSelfAttentionWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type, ln_type, rms_norm_type, attn_type, qkv_bias):
        super().__init__()
        prefix = f"blocks.{block_idx}.attn1"
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.qk_norm = config.get("qk_norm", True)
        self.use_fused_qkv_attn = bool(config.get("use_fused_qkv_attn", False))

        self.add_module("to_q", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_q.weight", f"{prefix}.to_q.bias" if qkv_bias else None))
        self.add_module("to_k", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_k.weight", f"{prefix}.to_k.bias" if qkv_bias else None))
        self.add_module("to_v", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_v.weight", f"{prefix}.to_v.bias" if qkv_bias else None))
        if self.use_fused_qkv_attn:
            self.to_qkv = MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.to_qkv.weight",
                f"{prefix}.to_qkv.bias" if qkv_bias else None,
            )
        else:
            self.to_qkv = None
        if self.qk_norm:
            self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.q_norm.weight"))
            self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.k_norm.weight"))
        else:
            self.norm_q = None
            self.norm_k = None
        self.add_module("out_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.out_proj.weight", f"{prefix}.out_proj.bias"))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[attn_type]())

    def load(self, weight_dict):
        super().load(weight_dict)
        self._build_fused_qkv()

    def to_cuda(self, non_blocking=False):
        super().to_cuda(non_blocking=non_blocking)
        self._build_fused_qkv()

    def to_cpu(self, non_blocking=False):
        super().to_cpu(non_blocking=non_blocking)
        self._build_fused_qkv()

    @staticmethod
    def _output_dim(module):
        if hasattr(module, "weight_need_transpose"):
            return 1 if module.weight_need_transpose else 0
        return 1

    @staticmethod
    def _cat_optional_biases(biases):
        if all(bias is None for bias in biases):
            return None
        if any(bias is None for bias in biases):
            raise ValueError("Fused Hunyuan3D QKV requires either all QKV biases or no QKV biases")
        return torch.cat([bias.contiguous() for bias in biases], dim=0)

    def _cat_output_weights(self, modules, tensors):
        dim = self._output_dim(modules[0])
        if dim == 1:
            return torch.cat([tensor.t().contiguous() for tensor in tensors], dim=0).t()
        return torch.cat([tensor.contiguous() for tensor in tensors], dim=dim)

    def _cat_output_scales(self, scales):
        if scales[0].dim() == 1:
            dim = 0
        elif scales[0].shape[-1] == 1:
            dim = 0
        else:
            dim = scales[0].dim() - 1
        return torch.cat([scale.contiguous() for scale in scales], dim=dim)

    def _build_fused_qkv(self):
        if self.to_qkv is None:
            return
        modules = (self.to_q, self.to_k, self.to_v)
        weights = [module._get_actual_weight() for module in modules]
        if any(weight is None for weight in weights):
            return

        self.to_qkv.weight = self._cat_output_weights(modules, weights)
        biases = [module._get_actual_bias() for module in modules]
        self.to_qkv.bias = self._cat_optional_biases(biases)
        self.to_qkv.has_lora_branch = False

        if all(hasattr(module, "weight_scale") for module in modules):
            scales = [module.weight_scale for module in modules]
            if all(scale is not None for scale in scales):
                self.to_qkv.weight_scale = self._cat_output_scales(scales)

    @property
    def has_fused_qkv(self):
        if self.to_qkv is None or not hasattr(self.to_qkv, "weight"):
            return False
        return not any(getattr(module, "has_lora_branch", False) or getattr(module, "has_diff", False) for module in (self.to_q, self.to_k, self.to_v))


class Hunyuan3DCrossAttentionWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type, rms_norm_type, attn_type, qkv_bias):
        super().__init__()
        prefix = f"blocks.{block_idx}.attn2"
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.qk_norm = config.get("qk_norm", True)

        self.add_module("to_q", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_q.weight", f"{prefix}.to_q.bias" if qkv_bias else None))
        self.add_module("to_k", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_k.weight", f"{prefix}.to_k.bias" if qkv_bias else None))
        self.add_module("to_v", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_v.weight", f"{prefix}.to_v.bias" if qkv_bias else None))
        if self.qk_norm:
            self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.q_norm.weight"))
            self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.k_norm.weight"))
        else:
            self.norm_q = None
            self.norm_k = None
        self.add_module("out_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.out_proj.weight", f"{prefix}.out_proj.bias"))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[attn_type]())


class Hunyuan3DTransformerBlockWeights(WeightModule):
    """Weights for one HunYuanDiT block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.depth = config["depth"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.ln_type = config.get("ln_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")
        qkv_bias = config.get("qkv_bias", False)
        prefix = f"blocks.{block_idx}"

        self.add_module("norm1", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm1.weight", f"{prefix}.norm1.bias", eps=1e-6))
        self.add_module("norm2", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm2.weight", f"{prefix}.norm2.bias", eps=1e-6))
        self.add_module("norm3", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm3.weight", f"{prefix}.norm3.bias", eps=1e-6))
        self.add_module(
            "attn1",
            Hunyuan3DSelfAttentionWeights(config, block_idx, self.mm_type, self.ln_type, self.rms_norm_type, self.attn_type, qkv_bias),
        )
        self.add_module(
            "attn2",
            Hunyuan3DCrossAttentionWeights(config, block_idx, self.mm_type, self.rms_norm_type, self.attn_type, qkv_bias),
        )

        use_moe = self.depth - block_idx <= config.get("num_moe_layers", 6)
        if use_moe:
            self.add_module("moe", Hunyuan3DMoEWeights(config, block_idx, self.mm_type))
            self.mlp = None
        else:
            self.add_module("mlp", Hunyuan3DMLPWeights(f"{prefix}.mlp", self.mm_type))
            self.moe = None

        if block_idx > self.depth // 2:
            self.add_module("skip_norm", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.skip_norm.weight", f"{prefix}.skip_norm.bias", eps=1e-6))
            self.add_module(
                "skip_linear",
                MM_WEIGHT_REGISTER[self.mm_type](f"{prefix}.skip_linear.weight", f"{prefix}.skip_linear.bias"),
            )
        else:
            self.skip_norm = None
            self.skip_linear = None


class Hunyuan3DTransformerWeights(WeightModule):
    """Transformer weights for Hunyuan3D shape DiT."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth = config["depth"]
        blocks = WeightModuleList(Hunyuan3DTransformerBlockWeights(config, block_idx) for block_idx in range(self.depth))
        self.add_module("blocks", blocks)
