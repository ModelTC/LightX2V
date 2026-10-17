import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.mm.optimized_linear import make_projection_group
from lightx2v.common.ops.norm.l2_normalize import l2_normalize
from lightx2v.common.ops.rope.rotary_qk import apply_rotary_qk
from lightx2v.models.input_encoders.edenlive.proxy_cuda_graph import ProxyCUDAGraph
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV2D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class _ProxyAttention:
    """Dense batched attention; every input is [batch, tokens, heads, width]."""

    def __init__(self, attention_type):
        self.attention_type = attention_type
        if attention_type == "flash_attn2":
            try:
                from flash_attn import flash_attn_func
            except ImportError as error:
                raise ImportError("ProxyDINO flash_attn2 requires flash-attn") from error
            self.function = flash_attn_func
        elif attention_type == "sage_attn2":
            try:
                from sageattention import sageattn
            except ImportError as error:
                raise ImportError("ProxyDINO sage_attn2 requires sageattention") from error
            self.function = sageattn
        else:
            raise ValueError(f"Unsupported ProxyDINO attention_type: {attention_type!r}")

    def apply(self, q, k, v):
        if self.attention_type == "flash_attn2":
            return self.function(q, k, v, dropout_p=0.0, causal=False)
        return self.function(q, k, v, tensor_layout="NHD", is_causal=False)


class ProxyDINO(WeightModule):
    def __init__(
        self,
        model_path,
        resize_mode,
        use_cuda_graph=False,
        attention_type="torch_sdpa",
        *,
        quantized_ckpt=None,
    ):
        super().__init__()
        root = Path(model_path) / "vision-encoder"
        self.config = json.loads((root / "backbone/config.json").read_text())
        self.resize_mode = resize_mode
        self.infer_dtype = GET_DTYPE()
        self.quantized = quantized_ckpt is not None
        linear_type = "int8-convrot" if self.quantized else "Default"
        self.heads = self.config["num_attention_heads"]
        self.head_dim = self.config["hidden_size"] // self.heads
        self.prefix = 1 + self.config["num_register_tokens"]
        state = load_file(str(quantized_ckpt or root / "projection.safetensors"))
        state = {k.removeprefix("model.module.encoder."): v.to(AI_DEVICE) for k, v in state.items()}
        if self.quantized and (state["patch_projection.weight"].dtype != torch.int8 or "patch_projection.convrot_groupsize" not in state):
            raise ValueError("ProxyDINO quantized_ckpt must contain converted INT8 ConvRot weights")
        self.add_module("patch", CONV2D_WEIGHT_REGISTER["Default"]("backbone.embeddings.patch_embeddings.weight", "backbone.embeddings.patch_embeddings.bias", stride=self.config["patch_size"]))
        for name in ("cls_token", "register_tokens"):
            self.register_parameter(name, TENSOR_REGISTER["Default"]("backbone.embeddings." + name))
        self.add_module("final_norm", LN_WEIGHT_REGISTER["torch"]("backbone.norm.weight", "backbone.norm.bias", eps=self.config["layer_norm_eps"]))
        self.add_module("projection", MM_WEIGHT_REGISTER[linear_type]("patch_projection.weight", "patch_projection.bias"))
        self.blocks = []
        for i in range(self.config["num_hidden_layers"]):
            block = WeightModule()
            prefix = f"backbone.layer.{i}."
            for name in ("norm1", "norm2"):
                block.add_module(name, LN_WEIGHT_REGISTER["torch"](prefix + name + ".weight", prefix + name + ".bias", eps=self.config["layer_norm_eps"]))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = prefix + "attention." + name
                block.add_module(name, MM_WEIGHT_REGISTER[linear_type](key + ".weight", key + ".bias" if name != "k_proj" else None))
            for name in ("up_proj", "down_proj"):
                key = prefix + "mlp." + name
                block.add_module(name, MM_WEIGHT_REGISTER[linear_type](key + ".weight", key + ".bias"))
            for name in ("layer_scale1", "layer_scale2"):
                block.register_parameter(name, TENSOR_REGISTER["Default"](prefix + name + ".lambda1"))
            self.add_module(f"block_{i}", block)
            self.blocks.append(block)
        self.load(state)
        self.set_attention_type(attention_type)
        # Preserve the former CPU division and GPU normalization rounding exactly.
        values = torch.arange(256, dtype=torch.float32).div_(127.5).sub_(1)
        self.pixel_values = values.add_(1).mul_(0.5).to(AI_DEVICE)
        self._rope_cache = {}
        # PIL-based resize modes move data to the CPU and cannot be captured.
        self._cuda_graph = ProxyCUDAGraph(self._infer_eager) if use_cuda_graph and resize_mode == "bilinear" else None

    def set_attention_type(self, attention_type):
        """Change the dense backend, discarding graphs of the previous kernel."""
        if attention_type not in ("torch_sdpa", "flash_attn2", "sage_attn2"):
            raise ValueError(f"Unsupported ProxyDINO attention_type: {attention_type!r}")
        if getattr(self, "attention_type", None) == attention_type:
            return
        attention = ATTN_WEIGHT_REGISTER["torch_sdpa"]() if attention_type == "torch_sdpa" else _ProxyAttention(attention_type)
        graph = getattr(self, "_cuda_graph", None)
        if graph is not None:
            graph.clear()
        self.attention_type = attention_type
        self.attention = attention

    def load(self, state):
        graph = getattr(self, "_cuda_graph", None)
        if graph is not None:
            graph.clear()
        for block in getattr(self, "blocks", ()):
            block.qkv_group = None
            if getattr(block.k_proj, "bias_name", None) is None:
                block.k_proj.bias = None
        super().load(state)
        if hasattr(self, "infer_dtype"):
            self._prepare_loaded_weights()

    def _prepare_loaded_weights(self):
        # Floating weights, normalization, tokens and residuals share the model dtype.
        # Preserve converted INT8 weights and their FP32 quantization scales.
        for module in (self.patch, self.projection, *(m for b in self.blocks for m in (b.q_proj, b.k_proj, b.v_proj, b.o_proj, b.up_proj, b.down_proj))):
            if module.weight.dtype != torch.int8:
                module.weight = module.weight.to(self.infer_dtype)
            if getattr(module, "bias", None) is not None:
                module.bias = module.bias.to(self.infer_dtype)
        for parameter in (self.cls_token, self.register_tokens, *(p for b in self.blocks for p in (b.layer_scale1, b.layer_scale2))):
            parameter.tensor = parameter.tensor.to(self.infer_dtype)
        for norm in (self.final_norm, *(n for b in self.blocks for n in (b.norm1, b.norm2))):
            norm.weight = norm.weight.to(self.infer_dtype)
            norm.bias = norm.bias.to(self.infer_dtype)
        for block in self.blocks:
            block.qkv_group = None
            if self.quantized:
                # DINO omits K bias; an explicit zero enables shared-input
                # GEMMs without changing the effective linear projection.
                block.k_proj.bias = torch.zeros_like(block.q_proj.bias)
                block.qkv_group = make_projection_group((block.q_proj, block.k_proj, block.v_proj), packed=False)

    def preprocess(self, images):
        """Convert uint8 RGB to the original normalized DINO input."""
        images = self.pixel_values[images.int()].contiguous()
        size = (480, 832)
        if self.resize_mode == "bilinear":
            pixels = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
        else:
            pictures = images.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            resized = []
            for picture in pictures:
                image = Image.fromarray(picture)
                if self.resize_mode == "bicubic":
                    image = image.resize((size[1], size[0]), Image.Resampling.BICUBIC)
                    resized.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
                    continue
                width, height = image.size
                scale = max(size[0] / height, size[1] / width)
                image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
                left, top = (image.width - size[1]) // 2, (image.height - size[0]) // 2
                resized.append(torch.from_numpy(np.array(image.crop((left, top, left + size[1], top + size[0])))).permute(2, 0, 1))
            pixels = torch.stack(resized).to(device=images.device, dtype=torch.float32) / 255
        if not hasattr(self, "pixel_mean"):
            self.pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            self.pixel_std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return pixels.sub_(self.pixel_mean).div_(self.pixel_std)

    def _rope(self, height, width, device):
        key = (height, width, device)
        if key not in self._rope_cache:
            cy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
            cx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
            coordinates = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1) * 2 - 1
            frequencies = (1 / self.config["rope_theta"] ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)).to(device)
            angles = (2 * math.pi * coordinates[:, :, None] * frequencies).flatten(1).tile(2)[:, None]
            self._rope_cache[key] = (angles.cos(), angles.sin())
        return self._rope_cache[key]

    @torch.no_grad()
    def infer(self, images):
        if self._cuda_graph is not None:
            return self._cuda_graph(images)
        return self._infer_eager(images)

    @torch.no_grad()
    def _infer_eager(self, images):
        pixels = self.preprocess(images)
        height, width = (n // self.config["patch_size"] for n in pixels.shape[-2:])
        cos, sin = self._rope(height, width, pixels.device)
        batch = pixels.shape[0]
        patch = self.patch.apply(pixels.to(self.infer_dtype)).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.tensor.expand(batch, -1, -1), self.register_tokens.tensor.expand(batch, -1, -1), patch), dim=1)
        sequence, hidden = x.shape[1:]
        for block in self.blocks:
            normalized = block.norm1.apply(x).flatten(0, 1).to(self.infer_dtype)
            if block.qkv_group is None:
                q, k, v = (module.apply(normalized) for module in (block.q_proj, block.k_proj, block.v_proj))
            else:
                q, k, v = block.qkv_group.apply(normalized)
            q, k, v = (tensor.view(batch, sequence, self.heads, self.head_dim) for tensor in (q, k, v))
            q, k = apply_rotary_qk(q, k, cos, sin, self.prefix)
            attention = self.attention.apply(q=q, k=k, v=v).reshape(batch * sequence, hidden)
            x.addcmul_(block.o_proj.apply(attention).view_as(x), block.layer_scale1.tensor)
            normalized = block.norm2.apply(x).flatten(0, 1).to(self.infer_dtype)
            y = block.down_proj.apply(F.gelu(block.up_proj.apply(normalized), approximate="none"))
            x.addcmul_(y.view_as(x), block.layer_scale2.tensor)
        x = self.final_norm.apply(x)[:, self.prefix :].reshape(-1, hidden).to(self.infer_dtype)
        projected = self.projection.apply(x)
        features = l2_normalize(projected, out_dtype=self.infer_dtype)
        return features.view(batch, height, width, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
