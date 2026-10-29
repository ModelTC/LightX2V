import torch


def apply_rope(x, rope):
    cosine, sine = rope
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cosine[:, None, 0::2] - odd * sine[:, None, 1::2]
    out[..., 1::2] = even * sine[:, None, 1::2] + odd * cosine[:, None, 0::2]
    return out


class RotaryEmbedding:
    def __init__(self, *, wan, device, dtype, head_dim=128, max_length=1024, theta=10000.0):
        self.wan = wan
        self.device, self.dtype = device, dtype
        self.max_length = max_length
        self.dims = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)) if wan else (16, 24, 24)
        self.cache = {}
        self.tables = []
        for dim in self.dims:
            if wan:
                # Upstream builds these tables on CPU in float64, then casts to float32.
                inverse = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                angles = torch.outer(torch.arange(max_length, dtype=torch.float64), inverse)
                self.tables.append((angles.cos().repeat_interleave(2, -1).float().to(device), angles.sin().repeat_interleave(2, -1).float().to(device)))
            else:
                self.tables.append(1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)))

    def for_positions(self, height, width, positions):
        key = (height, width, positions)
        if key in self.cache:
            return self.cache[key]
        if self.wan and (max(height, width) > self.max_length or min(positions) < 0 or max(positions) >= self.max_length):
            raise ValueError("RelayVSR spatial or temporal position exceeds the checkpoint's RoPE range")
        frames = len(positions)
        coordinates = (
            torch.tensor(positions, device=self.device).view(frames, 1, 1).expand(frames, height, width).reshape(-1),
            torch.arange(height, device=self.device).view(1, height, 1).expand(frames, height, width).reshape(-1),
            torch.arange(width, device=self.device).view(1, 1, width).expand(frames, height, width).reshape(-1),
        )
        cosines, sines = [], []
        for coordinate, table in zip(coordinates, self.tables):
            if self.wan:
                cosine, sine = table[0][coordinate], table[1][coordinate]
            else:
                angles = coordinate.float().unsqueeze(-1) * table
                cosine, sine = angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)
            cosines.append(cosine)
            sines.append(sine)
        result = (torch.cat(cosines, -1), torch.cat(sines, -1))
        if not self.wan:
            result = tuple(value.to(self.dtype) for value in result)
        if len(self.cache) >= 32:
            self.cache.clear()
        self.cache[key] = result
        return result

    def clear(self):
        self.cache.clear()
