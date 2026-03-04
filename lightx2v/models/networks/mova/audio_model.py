# lightx2v/models/networks/mova/audio_model.py

import torch
import torch.nn as nn
from einops import rearrange
from lightx2v.models.networks.wan.model import WanModel as LightX2VWanModel
from lightx2v.models.networks.wan.weights.audio.transformer_weights import WanAudioTransformerWeights
from lightx2v.models.networks.mova.transformer_infer import MOVATransformerInfer


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64)
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class MOVAAudioDiT(LightX2VWanModel):
    transformer_weight_class = WanAudioTransformerWeights
    transformer_infer_class = MOVATransformerInfer  # 使用自定义推理引擎
    def to_cuda(self):
        super().to_cuda()
        self.patch_embedding = self.patch_embedding.cuda()
        return self
    def to_cpu(self):
        super().to_cpu()
        self.patch_embedding = self.patch_embedding.cpu()
        return self
    def __init__(self, model_path, config, device, mova_state_dict=None, **kwargs):
        self.mova_state_dict = mova_state_dict
        config['has_image_input'] = False
        config['lazy_load'] = False
        self.out_dim = config.get('out_dim', 128)
        # 先手动创建权重容器
        self.pre_weight = self.pre_weight_class(config)
        self.transformer_weights = self.transformer_weight_class(config)
        if hasattr(self, "post_weight_class") and self.post_weight_class is not None:
            self.post_weight = self.post_weight_class(config)

        super().__init__(model_path, config, device, **kwargs)

        # 替换 patch_embedding 为 Conv1d
        in_dim = config.get("in_dim", 128)
        dim = config.get("dim", 1536)
        # 处理 patch_size，确保为整数
        raw_patch_size = config.get("patch_size", 1)
        if isinstance(raw_patch_size, (list, tuple)):
            # 假设音频只需要时间维度的压缩倍数（通常为 1）
            self.patch_size = raw_patch_size[0]
        else:
            self.patch_size = raw_patch_size
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)

        # 重新计算一维频率
        head_dim = dim // config.get("num_heads", 24)
        self.freqs = self._precompute_freqs_cis_1d(head_dim)
    @property
    def head(self):
        if not hasattr(self, 'transformer_weights'):
            return None
        return self.transformer_weights.head

    @property
    def blocks(self):
        if not hasattr(self, 'transformer_weights'):
            return []
        return self.transformer_weights.blocks

    @property
    def infer_engine(self):
        if not hasattr(self, 'transformer_infer'):
            return None
        return self.transformer_infer
    @property
    def head(self):
        return self.transformer_weights.head
    def _precompute_freqs_cis_1d(self, dim, end=16384, theta=10000.0):
        freqs = precompute_freqs_cis_1d(dim, end, theta)
        return freqs, freqs, freqs

    def patchify(self, x):
        # 确保输入 dtype 与卷积权重匹配
        x = x.to(self.patch_embedding.weight.dtype)
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b d t -> b t d')
        return x, grid_size

    def unpatchify(self, x, grid_size):
        t = grid_size[0]
        out_dim = self.out_dim
        p = self.patch_size
        # x shape: [B, L, out_dim * p]
        x = x.view(1, t, p, out_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(1, out_dim, t * p)
        return x

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if self.mova_state_dict is None:
            raise RuntimeError("mova_state_dict must be provided")

        original_keys = list(self.mova_state_dict.keys())
        print("[MOVA Audio] Original keys count:", len(original_keys))

        mapped = {}
        # 第一步：基础映射，保留原始键和扁平化键
        for key, tensor in self.mova_state_dict.items():
            if key.startswith('audio_model.'):
                key = key[len('audio_model.'):]

            # 保留原始键
            mapped[key] = tensor

            # 对 blocks 进行扁平化映射
            if key.startswith('blocks.'):
                parts = key.split('.')
                layer = parts[1]
                module_type = parts[2]

                if module_type == 'cross_attn':
                    sub = parts[3]
                    suffix = parts[4] if len(parts) > 4 else ''
                    if sub in ('q', 'k', 'v'):
                        sub_clean = sub.replace('_proj', '')
                        flat_key = f"ca.{layer}.to_{sub_clean}.{suffix}"
                    elif sub == 'o':
                        flat_key = f"ca.{layer}.to_out.{suffix}"
                    elif sub.startswith('norm_'):
                        flat_key = f"ca.{layer}.{sub}.{suffix}"
                    else:
                        flat_key = key
                    mapped[flat_key] = tensor

                elif module_type == 'self_attn':
                    sub = parts[3]
                    suffix = parts[4] if len(parts) > 4 else ''
                    if sub in ('q', 'k', 'v', 'o'):
                        sub_clean = sub.replace('_proj', '')
                        if sub == 'o':
                            flat_key = f"sa.{layer}.to_out.{suffix}"
                        else:
                            flat_key = f"sa.{layer}.to_{sub_clean}.{suffix}"
                    elif sub.startswith('norm_'):
                        flat_key = f"sa.{layer}.{sub}.{suffix}"
                    else:
                        flat_key = key
                    mapped[flat_key] = tensor

        # 第二步：合并 KV
        layer_set = set()
        for key in mapped.keys():
            if key.startswith('ca.') and '.' in key:
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_set.add(int(parts[1]))
        max_layer = max(layer_set) if layer_set else 0

        for layer in range(max_layer + 1):
            k_weight_key = f"ca.{layer}.to_k.weight"
            v_weight_key = f"ca.{layer}.to_v.weight"
            kv_weight_key = f"ca.{layer}.to_kv.weight"
            if k_weight_key in mapped and v_weight_key in mapped:
                k_weight = mapped[k_weight_key]
                v_weight = mapped[v_weight_key]
                if k_weight.shape == v_weight.shape:
                    kv_weight = torch.cat([k_weight, v_weight], dim=0)
                    mapped[kv_weight_key] = kv_weight
                    print(f"[MOVA Audio] Created {kv_weight_key} from {k_weight_key} and {v_weight_key}")
                else:
                    print(f"[MOVA Audio] Warning: Shape mismatch for {k_weight_key} and {v_weight_key}, cannot merge")

            k_bias_key = f"ca.{layer}.to_k.bias"
            v_bias_key = f"ca.{layer}.to_v.bias"
            kv_bias_key = f"ca.{layer}.to_kv.bias"
            if k_bias_key in mapped and v_bias_key in mapped:
                k_bias = mapped[k_bias_key]
                v_bias = mapped[v_bias_key]
                if k_bias.shape == v_bias.shape:
                    kv_bias = torch.cat([k_bias, v_bias], dim=0)
                    mapped[kv_bias_key] = kv_bias
                    print(f"[MOVA Audio] Created {kv_bias_key} from {k_bias_key} and {v_bias_key}")

        # 第三步：创建 norm_kv（从 norm_k 复制）
        for layer in range(max_layer + 1):
            norm_k_weight_key = f"ca.{layer}.norm_k.weight"
            norm_kv_weight_key = f"ca.{layer}.norm_kv.weight"
            if norm_k_weight_key in mapped and norm_kv_weight_key not in mapped:
                mapped[norm_kv_weight_key] = mapped[norm_k_weight_key].clone()
                print(f"[MOVA Audio] Created {norm_kv_weight_key} from {norm_k_weight_key}")

            norm_k_bias_key = f"ca.{layer}.norm_k.bias"
            norm_kv_bias_key = f"ca.{layer}.norm_kv.bias"
            if norm_k_bias_key in mapped and norm_kv_bias_key not in mapped:
                mapped[norm_kv_bias_key] = mapped[norm_k_bias_key].clone()
                print(f"[MOVA Audio] Created {norm_kv_bias_key} from {norm_k_bias_key}")

        # 第四步：创建 shift_scale_gate（从 blocks.{layer}.modulation 复制）
        for layer in range(max_layer + 1):
            mod_key = f"blocks.{layer}.modulation"
            shift_scale_gate_key = f"ca.{layer}.shift_scale_gate"
            if mod_key in mapped and shift_scale_gate_key not in mapped:
                mapped[shift_scale_gate_key] = mapped[mod_key].clone()
                print(f"[MOVA Audio] Created {shift_scale_gate_key} from {mod_key}")

        print("[MOVA Audio] Mapped keys count:", len(mapped))
        return mapped