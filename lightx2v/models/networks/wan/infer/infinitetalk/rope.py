import torch
from einops import rearrange, repeat


class RotaryPositionalEmbedding1D:
    def __init__(self, head_dim, base=10000):
        self.head_dim = head_dim
        self.base = base
        self._inv_freq_cache = {}

    def _get_inv_freq(self, device):
        cache_key = (device.type, device.index)
        inv_freq = self._inv_freq_cache.get(cache_key)
        if inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
            self._inv_freq_cache[cache_key] = inv_freq
        return inv_freq

    def prepare(self, rope, pos_indices):
        angles = torch.einsum("..., f -> ... f", pos_indices.float(), self._get_inv_freq(pos_indices.device))
        angles = repeat(angles, "... n -> ... (n r)", r=2)
        cos = rearrange(angles.cos().float(), "n d -> 1 1 n d")
        sin = rearrange(angles.sin().float(), "n d -> 1 1 n d")
        freqs = rope.prepare_freqs((cos, sin), rotary_dim=self.head_dim)
        return freqs, rope.prepare_positions(freqs)

    @staticmethod
    def apply_prepared(rope, x, freqs, positions):
        if positions is None:
            return rope.apply_single(x, freqs)

        x_lhd = rearrange(x, "1 h l d -> l h d")
        x_lhd = rope.apply_single(x_lhd, freqs, positions=positions)
        return rearrange(x_lhd, "l h d -> 1 h l d")

    def __call__(self, rope, x, pos_indices):
        freqs, positions = self.prepare(rope, pos_indices)
        return self.apply_prepared(rope, x, freqs, positions)
