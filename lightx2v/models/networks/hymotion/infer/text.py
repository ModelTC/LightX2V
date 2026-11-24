"""Qwen3 forward-only text conditioning (no language head or KV cache)."""

import torch
import torch.nn.functional as F

from lightx2v.models.networks.three_models.ops import Attention
from lightx2v.utils.registry_factory import ROPE_REGISTER


class Qwen3TextInfer:
    def __init__(self, config):
        self.config = config
        self.attention = Attention()
        self.rope = ROPE_REGISTER["torch_real_rope"](layout="split_half", compute_dtype=torch.bfloat16)

    def infer(self, w, ids, padding_mask):
        cfg = self.config
        x = F.embedding(ids, w.tensor("model.embed_tokens.weight"))
        b, n, _c = x.shape
        heads, kv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        positions = torch.arange(n, device=x.device, dtype=torch.float32)
        inv = 1 / (cfg["rope_theta"] ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        phase = positions[:, None] * inv[None]
        phase = torch.cat((phase, phase), dim=-1)
        cos, sin = phase.cos().to(x.dtype)[None, None], phase.sin().to(x.dtype)[None, None]
        causal = torch.ones(n, n, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & padding_mask[:, None, None, :].bool()
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}"
            y = w.rms(p + ".input_layernorm", cfg["rms_norm_eps"])(x)
            q = w.linear(p + ".self_attn.q_proj")(y).reshape(b, n, heads, d)
            k = w.linear(p + ".self_attn.k_proj")(y).reshape(b, n, kv, d)
            v = w.linear(p + ".self_attn.v_proj")(y).reshape(b, n, kv, d).transpose(1, 2)
            q = w.rms(p + ".self_attn.q_norm", cfg["rms_norm_eps"])(q).transpose(1, 2)
            k = w.rms(p + ".self_attn.k_norm", cfg["rms_norm_eps"])(k).transpose(1, 2)
            q, k = self.rope.apply(q, k, (cos, sin))
            # Match the released encoder's repeat_kv + contiguous SDPA path.
            # Native GQA with a custom mask selects a different CUDA backend.
            k = k.repeat_interleave(heads // kv, dim=1).contiguous()
            v = v.repeat_interleave(heads // kv, dim=1).contiguous()
            q = q.contiguous()
            y = self.attention(q, k, v, mask=allowed).transpose(1, 2).reshape(b, n, heads * d)
            x = x + w.linear(p + ".self_attn.o_proj")(y)
            y = w.rms(p + ".post_attention_layernorm", cfg["rms_norm_eps"])(x)
            y = F.silu(w.linear(p + ".mlp.gate_proj")(y)) * w.linear(p + ".mlp.up_proj")(y)
            x = x + w.linear(p + ".mlp.down_proj")(y)
        return w.rms("model.norm", cfg["rms_norm_eps"])(x)


def encode_text(w, texts):
    batch = w.tokenizer([w.format(t) for t in texts], truncation=True, max_length=128 + w.crop_start, padding="max_length", return_attention_mask=True, return_tensors="pt")
    ids, mask = batch.input_ids.cuda(), batch.attention_mask.cuda()
    context = w.qwen_infer.infer(w.qwen, ids, mask)[:, w.crop_start : w.crop_start + 128].contiguous()
    lengths = (mask.sum(-1) - w.crop_start).clamp(min=0, max=128)
    clip_batch = w.clip_tokenizer(texts, truncation=True, max_length=77, padding=True, return_tensors="pt")
    _, pooled = w.clip_infer.infer(w.clip, clip_batch.input_ids.cuda())
    return pooled.unsqueeze(1), context, lengths
