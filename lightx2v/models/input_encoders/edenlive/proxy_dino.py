import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV2D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class ProxyDINO(WeightModule):
    def __init__(self, model_path, resize_mode):
        super().__init__()
        root = Path(model_path) / "vision-encoder"
        self.config = json.loads((root / "backbone/config.json").read_text())
        self.resize_mode = resize_mode
        self.heads = self.config["num_attention_heads"]
        self.head_dim = self.config["hidden_size"] // self.heads
        self.prefix = 1 + self.config["num_register_tokens"]
        state = load_file(str(root / "projection.safetensors"))
        state = {k.removeprefix("model.module.encoder."): v.to(AI_DEVICE) for k, v in state.items()}
        self.add_module("patch", CONV2D_WEIGHT_REGISTER["Default"]("backbone.embeddings.patch_embeddings.weight", "backbone.embeddings.patch_embeddings.bias", stride=self.config["patch_size"]))
        for name in ("cls_token", "register_tokens"):
            self.register_parameter(name, TENSOR_REGISTER["Default"]("backbone.embeddings." + name))
        self.add_module("final_norm", LN_WEIGHT_REGISTER["torch"]("backbone.norm.weight", "backbone.norm.bias", eps=self.config["layer_norm_eps"]))
        self.add_module("projection", MM_WEIGHT_REGISTER["Default"]("patch_projection.weight", "patch_projection.bias"))
        self.blocks = []
        for i in range(self.config["num_hidden_layers"]):
            block = WeightModule()
            prefix = f"backbone.layer.{i}."
            for name in ("norm1", "norm2"):
                block.add_module(name, LN_WEIGHT_REGISTER["torch"](prefix + name + ".weight", prefix + name + ".bias", eps=self.config["layer_norm_eps"]))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = prefix + "attention." + name
                block.add_module(name, MM_WEIGHT_REGISTER["Default"](key + ".weight", key + ".bias" if name != "k_proj" else None))
            for name in ("up_proj", "down_proj"):
                key = prefix + "mlp." + name
                block.add_module(name, MM_WEIGHT_REGISTER["Default"](key + ".weight", key + ".bias"))
            for name in ("layer_scale1", "layer_scale2"):
                block.register_parameter(name, TENSOR_REGISTER["Default"](prefix + name + ".lambda1"))
            self.add_module(f"block_{i}", block)
            self.blocks.append(block)
        self.load(state)
        # Match upstream autocast linear/conv while residuals and LayerScale stay FP32.
        for module in (self.patch, self.projection, *(m for b in self.blocks for m in (b.q_proj, b.k_proj, b.v_proj, b.o_proj, b.up_proj, b.down_proj))):
            module.weight = module.weight.to(torch.bfloat16)
            if module.bias is not None:
                module.bias = module.bias.to(torch.bfloat16)
        self.attention = ATTN_WEIGHT_REGISTER["torch_sdpa"]()
        # Preserve the former CPU division and GPU normalization rounding exactly.
        values = torch.arange(256, dtype=torch.float32).div_(127.5).sub_(1)
        self.pixel_values = values.add_(1).mul_(0.5).to(AI_DEVICE)

    def preprocess(self, images):
        """Convert uint8 RGB to the original normalized DINO input."""
        images = self.pixel_values[images.int()].contiguous()
        size = (480, 832)
        if self.resize_mode == "bilinear":
            pixels = F.interpolate(images.float(), size=size, mode="bilinear", align_corners=False, antialias=True)
        else:
            pictures = images.float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
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
        mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return pixels.sub_(mean).div_(std)

    @torch.no_grad()
    def infer(self, images):
        pixels = self.preprocess(images)
        height, width = (n // self.config["patch_size"] for n in pixels.shape[-2:])
        cy = (torch.arange(height, device=pixels.device).float() + 0.5) / height
        cx = (torch.arange(width, device=pixels.device).float() + 0.5) / width
        coordinates = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1) * 2 - 1
        frequencies = (1 / self.config["rope_theta"] ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)).to(pixels.device)
        angles = (2 * math.pi * coordinates[:, :, None] * frequencies).flatten(1).tile(2)[:, None]
        cos, sin = angles.cos(), angles.sin()
        batch = pixels.shape[0]
        patch = self.patch.apply(pixels.to(torch.bfloat16)).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.tensor.expand(batch, -1, -1), self.register_tokens.tensor.expand(batch, -1, -1), patch), dim=1)
        sequence, hidden = x.shape[1:]
        for block in self.blocks:
            normalized = block.norm1.apply(x).flatten(0, 1).to(torch.bfloat16)
            q = block.q_proj.apply(normalized).view(batch, sequence, self.heads, self.head_dim)
            k = block.k_proj.apply(normalized).view_as(q)
            v = block.v_proj.apply(normalized).view_as(q)
            for tensor in (q, k):
                patches = tensor[:, self.prefix :].float()
                first, second = patches.chunk(2, dim=-1)
                tensor[:, self.prefix :] = (patches * cos + torch.cat((-second, first), dim=-1) * sin).to(tensor.dtype)
            attention = self.attention.apply(q=q, k=k, v=v).reshape(batch * sequence, hidden)
            x.addcmul_(block.o_proj.apply(attention).view_as(x), block.layer_scale1.tensor)
            normalized = block.norm2.apply(x).flatten(0, 1).to(torch.bfloat16)
            y = block.down_proj.apply(F.gelu(block.up_proj.apply(normalized), approximate="none"))
            x.addcmul_(y.view_as(x), block.layer_scale2.tensor)
        x = self.final_norm.apply(x)[:, self.prefix :].reshape(-1, hidden).to(torch.bfloat16)
        features = F.normalize(self.projection.apply(x).float(), dim=-1)
        return features.view(batch, height, width, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
