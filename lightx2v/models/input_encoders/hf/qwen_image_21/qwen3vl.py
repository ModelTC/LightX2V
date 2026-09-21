"""Native Qwen3-VL conditioning for Qwen-Image-2.1.

Only tokenization/pixel preprocessing come from Transformers. All language and
vision layers execute through LightX2V weights and operators. The language
model's final RMSNorm and LM head are intentionally neither loaded nor run.
"""

import itertools
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import Qwen3VLProcessor

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER, EMBEDDING_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def rotate_half(x):
    first, second = x.chunk(2, -1)
    return torch.cat((-second, first), -1)


class Qwen3VLTextLayer(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        self.config = config
        prefix = f"model.language_model.layers.{index}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.self_attn.{name}.weight", bias_name=None))
        for name in ("gate_proj", "up_proj", "down_proj"):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.mlp.{name}.weight", bias_name=None))
        for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            self.add_module(name.split(".")[-1], RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.{name}.weight", eps=config["rms_norm_eps"]))
        self.add_module("attention", ATTN_WEIGHT_REGISTER["torch_sdpa"]())

    def forward(self, x, cos, sin):
        h = self.input_layernorm.apply(x)
        q = self.q_norm.apply(self.q_proj.apply(h).reshape(-1, self.config["num_attention_heads"], self.config["head_dim"]))
        k = self.k_norm.apply(self.k_proj.apply(h).reshape(-1, self.config["num_key_value_heads"], self.config["head_dim"]))
        v = self.v_proj.apply(h).reshape_as(k)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        x = x + self.o_proj.apply(self.attention.apply(q, k, v, causal=True))
        h = self.post_attention_layernorm.apply(x)
        return x + self.down_proj.apply(F.silu(self.gate_proj.apply(h)) * self.up_proj.apply(h))


class Qwen3VLVisionLayer(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        self.heads = config["num_heads"]
        prefix = f"model.visual.blocks.{index}"
        for name, key in (("qkv", "attn.qkv"), ("proj", "attn.proj"), ("fc1", "mlp.linear_fc1"), ("fc2", "mlp.linear_fc2")):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.{key}.weight", f"{prefix}.{key}.bias"))
        for name in ("norm1", "norm2"):
            self.add_module(name, LN_WEIGHT_REGISTER["torch"](f"{prefix}.{name}.weight", f"{prefix}.{name}.bias", eps=1e-6))
        self.add_module("attention", ATTN_WEIGHT_REGISTER["torch_sdpa"]())

    def forward(self, x, cos, sin, lengths):
        q, k, v = self.qkv.apply(self.norm1.apply(x)).reshape(len(x), 3, self.heads, -1).unbind(1)
        q = (q.float() * cos[:, None] + rotate_half(q.float()) * sin[:, None]).to(x.dtype)
        k = (k.float() * cos[:, None] + rotate_half(k.float()) * sin[:, None]).to(x.dtype)
        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            outputs.append(self.attention.apply(q[start:end], k[start:end], v[start:end]))
            start = end
        x = x + self.proj.apply(torch.cat(outputs))
        return x + self.fc2.apply(F.gelu(self.fc1.apply(self.norm2.apply(x)), approximate="tanh"))


class Qwen3VLPatchMerger(WeightModule):
    def __init__(self, prefix, config, postshuffle=False):
        super().__init__()
        self.dim = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.postshuffle = postshuffle
        self.add_module("norm", LN_WEIGHT_REGISTER["torch"](f"{prefix}.norm.weight", f"{prefix}.norm.bias", eps=1e-6))
        for name in ("linear_fc1", "linear_fc2"):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.{name}.weight", f"{prefix}.{name}.bias"))

    def forward(self, x):
        if self.postshuffle:
            x = x.reshape(-1, self.dim)
        x = self.norm.apply(x).reshape(-1, self.dim)
        return self.linear_fc2.apply(F.gelu(self.linear_fc1.apply(x)))


class Qwen3VLVision(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        prefix = "model.visual"
        self.add_module(
            "patch",
            CONV3D_WEIGHT_REGISTER["Default"](
                f"{prefix}.patch_embed.proj.weight", f"{prefix}.patch_embed.proj.bias", stride=(config["temporal_patch_size"], config["patch_size"], config["patch_size"])
            ),
        )
        self.add_module("position", EMBEDDING_WEIGHT_REGISTER["Default"](f"{prefix}.pos_embed.weight"))
        self.add_module("blocks", WeightModuleList(Qwen3VLVisionLayer(i, config) for i in range(config["depth"])))
        self.add_module("merger", Qwen3VLPatchMerger(f"{prefix}.merger", config))
        self.add_module("deepstack", WeightModuleList(Qwen3VLPatchMerger(f"{prefix}.deepstack_merger_list.{i}", config, True) for i in range(len(config["deepstack_visual_indexes"]))))

    def forward(self, pixels, grid):
        cfg = self.config
        device = pixels.device
        merge = cfg["spatial_merge_size"]
        side = math.isqrt(cfg["num_position_embeddings"])
        corners, corner_weights, position_ids, lengths = [], [], [], []
        for temporal, height, width in grid.tolist():
            h = torch.linspace(0, side - 1, height, device=device)
            w = torch.linspace(0, side - 1, width, device=device)
            hf, wf = h.int(), w.int()
            hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
            dh, dw = h - hf, w - wf
            indices = torch.stack([(a[:, None] * side + b[None]).flatten() for a, b in ((hf, wf), (hf, wc), (hc, wf), (hc, wc))])
            weights = torch.stack([(a[:, None] * b[None]).flatten() for a, b in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))])
            hp, wp = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
            shape = (height // merge, merge, width // merge, merge)
            hp, wp = (p.reshape(shape).transpose(1, 2).flatten().repeat(temporal) for p in (hp, wp))
            order = hp * width + wp
            corners.append(indices[:, order])
            corner_weights.append(weights[:, order])
            position_ids.append(torch.stack((hp, wp), -1))
            lengths.extend([height * width] * temporal)
        pixels = pixels.reshape(-1, cfg["in_channels"], cfg["temporal_patch_size"], cfg["patch_size"], cfg["patch_size"])
        x = self.patch.apply(pixels.to(GET_DTYPE())).reshape(-1, cfg["hidden_size"])
        pos = (self.position.apply(torch.cat(corners, 1)) * torch.cat(corner_weights, 1)[..., None]).sum(0)
        x = x + pos.to(x.dtype)
        dim = cfg["hidden_size"] // cfg["num_heads"] // 2
        inv = (1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))).to(device)
        freq = (torch.cat(position_ids)[..., None] * inv).flatten(1)
        freq = torch.cat((freq, freq), -1)
        cos, sin = freq.cos(), freq.sin()
        deepstack = []
        for index, block in enumerate(self.blocks):
            x = block.forward(x, cos, sin, lengths)
            if index in cfg["deepstack_visual_indexes"]:
                deepstack.append(self.deepstack[len(deepstack)].forward(x))
        return self.merger.forward(x), deepstack


class QwenImage21TextEncoder(WeightModule):
    def __init__(self, config):
        super().__init__()
        root = Path(config["model_path"])
        path = root / "text_encoder"
        self.cpu_offload = config.get("text_encoder_cpu_offload", False)
        self.model_config = json.loads((path / "config.json").read_text())
        self.text_config = self.model_config["text_config"]
        self.processor = Qwen3VLProcessor.from_pretrained(root / "processor", local_files_only=True)
        self.system = "Comprehend and analyze the provided prompt."
        system_tokens = self.processor.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": self.system}]}], tokenize=True, return_dict=False)
        self.drop_index = len(system_tokens[0])
        self.image_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        self.add_module("embedding", EMBEDDING_WEIGHT_REGISTER["Default"]("model.language_model.embed_tokens.weight"))
        self.add_module("layers", WeightModuleList(Qwen3VLTextLayer(i, self.text_config) for i in range(self.text_config["num_hidden_layers"])))
        self.add_module("vision", Qwen3VLVision(self.model_config["vision_config"]))
        required = set()

        def collect(module):
            if isinstance(module, WeightModule):
                for child in module._modules.values():
                    collect(child)
            else:
                for attr in ("weight_name", "bias_name"):
                    if getattr(module, attr, None):
                        required.add(getattr(module, attr))

        collect(self)
        # safetensors interprets bare "cuda" as cuda:0, unlike Tensor.to(),
        # so resolve the selected GPU index explicitly.
        device = "cpu" if self.cpu_offload else str(torch.empty(0, device=AI_DEVICE).device)
        weights = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device=device) as handle:
                for name in required.intersection(handle.keys()):
                    weights[name] = handle.get_tensor(name).to(GET_DTYPE())
        if missing := required - weights.keys():
            raise ValueError(f"Missing Qwen3-VL weights: {sorted(missing)}")
        self.load(weights)
        if self.cpu_offload:
            self.to_cpu()

    def to_cpu(self, non_blocking=False):
        if not self.cpu_offload:
            return super().to_cpu(non_blocking=non_blocking)

        # CPU loading retains immutable pinned weights. Drop device copies
        # without copying them back, including after a partial failed onload.
        def release(module):
            if isinstance(module, WeightModule):
                for child in module._modules.values():
                    release(child)
            else:
                for name in ("weight", "bias"):
                    tensor = getattr(module, f"pin_{name}", None)
                    if tensor is not None:
                        setattr(module, name, tensor)

        release(self)

    def _position_ids(self, ids, image_mask, grid):
        merge = self.model_config["vision_config"]["spatial_merge_size"]
        grids = iter(grid.tolist() if grid is not None else [])
        parts = []
        offset = 0
        for is_image, group in itertools.groupby(image_mask.tolist()):
            length = sum(1 for _ in group)
            if is_image:
                temporal, height, width = next(grids)
                height, width = height // merge, width // merge
                position = torch.stack(
                    torch.meshgrid(torch.arange(temporal, device=ids.device), torch.arange(height, device=ids.device), torch.arange(width, device=ids.device), indexing="ij")
                ).reshape(3, -1)
                if position.shape[1] != length:
                    raise ValueError("Qwen3-VL image tokens and pixel grid disagree")
                parts.append(position + offset)
                offset += max(height, width)
            else:
                parts.append(torch.arange(length, device=ids.device)[None].expand(3, -1) + offset)
                offset += length
        return torch.cat(parts, -1)

    @torch.inference_mode()
    def infer(self, prompt, images=None):
        prefix = ""
        vision_images = []
        for index, image in enumerate(images or []):
            prefix += (" " if index else "") + f"<image{index + 1}><|vision_start|><|image_pad|><|vision_end|>"
            white = Image.new("RGB", image.size, "white")
            white.paste(image, mask=image.getchannel("A"))
            vision_images.append(white)
        text = f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        kwargs = {"text": [text], "padding": True, "padding_side": "left", "return_tensors": "pt"}
        if vision_images:
            kwargs["images"] = vision_images
            # Runner already resized both encoder inputs to one aligned grid.
            # A second smart-resize can change the VLM slot count at small sizes.
            kwargs["do_resize"] = False
        inputs = self.processor(**kwargs).to(AI_DEVICE)
        ids = inputs.input_ids[0]
        image_mask = ids == self.image_id
        x = self.embedding.apply(ids)
        deepstack = []
        if vision_images:
            features, deepstack = self.vision.forward(inputs.pixel_values, inputs.image_grid_thw)
            x[image_mask] = features
        positions = self._position_ids(ids, image_mask, inputs.get("image_grid_thw"))
        cfg = self.text_config
        inv = (1.0 / (cfg["rope_theta"] ** (torch.arange(0, cfg["head_dim"], 2).float() / cfg["head_dim"]))).to(x.device)
        freq = positions.float()[..., None] * inv
        mixed = freq[0].clone()
        sections = cfg["rope_scaling"]["mrope_section"]
        for axis in (1, 2):
            mixed[..., axis : sections[axis] * 3 : 3] = freq[axis, ..., axis : sections[axis] * 3 : 3]
        mixed = torch.cat((mixed, mixed), -1)
        cos, sin = mixed.cos().to(x.dtype), mixed.sin().to(x.dtype)
        for index, layer in enumerate(self.layers):
            x = layer.forward(x, cos, sin)
            if index < len(deepstack):
                x[image_mask] += deepstack[index]
        return {"prompt_embeds": x[self.drop_index :], "image_mask": image_mask[self.drop_index :]}
