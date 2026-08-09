import torch
from transformers.activations import ACT2FN

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class BagelPreInfer:
    def __init__(self, config, llm_config):
        self.config = config
        self.head_dim = llm_config.get("head_dim", llm_config["hidden_size"] // llm_config["num_attention_heads"])
        if self.head_dim % 2:
            raise ValueError(f"BAGEL RoPE head_dim must be even, got {self.head_dim}.")

        rope_scaling = llm_config.get("rope_scaling")
        rope_scaling_type = None if rope_scaling is None else rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_scaling_type not in (None, "default"):
            raise NotImplementedError(f"BAGEL currently supports only default RoPE, got rope_scaling type {rope_scaling_type!r}.")

        rope_theta = llm_config.get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / self.head_dim))
        self.attention_scaling = 1.0
        self.rope = None
        self.connector_activation = ACT2FN[config.get("connector_act", "gelu_pytorch_tanh")]

    def set_rope(self, rope):
        self.rope = rope

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def embed_tokens(self, weights, packed_text_ids):
        packed_text_ids = packed_text_ids.to(AI_DEVICE)
        embeds = weights.embed_tokens.apply(packed_text_ids)
        return embeds

    @torch.no_grad()
    def _compute_rope_cos_sin(self, x, position_ids):
        # Keep BAGEL's original Qwen2 default-RoPE operation order and dtype
        # conversion so this migration does not change its numerical path.
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def prepare_rope(self, packed_sequence, packed_position_ids, device=AI_DEVICE):
        if self.rope is None:
            raise RuntimeError("BAGEL RoPE is not initialized.")

        cos, sin = self._compute_rope_cos_sin(packed_sequence, packed_position_ids.unsqueeze(0))
        raw_freqs = (cos.squeeze(0).to(device), sin.squeeze(0).to(device))
        packed_rope_freqs = self.rope.prepare_freqs(raw_freqs, rotary_dim=self.head_dim)
        packed_rope_positions = self.rope.prepare_positions(packed_rope_freqs)
        return packed_rope_freqs, packed_rope_positions

    def infer(self, weights, packed_sequence, packed_position_ids):
        return self.prepare_rope(packed_sequence, packed_position_ids)

    def vae2llm(self, weights, x):
        x = x.to(AI_DEVICE).to(torch.bfloat16)
        x = weights.vae2llm.apply(x)
        return x

    def connector(self, weights, x):
        x = x.to(AI_DEVICE).to(torch.bfloat16)
        x = weights.fc1.apply(x)
        x = self.connector_activation(x)
        x = weights.fc2.apply(x)
        return x
