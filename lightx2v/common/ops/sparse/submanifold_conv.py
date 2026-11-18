import math

import torch
from torch import nn

from .neighbor_map import cpu_hash_neighbors


def _triple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * 3


class SparseConv3d(nn.Module):
    """Stride-one submanifold convolution, with [out, kx, ky, kz, in] weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None, neighbor_map=cpu_hash_neighbors):
        super().__init__()
        if _triple(stride) != (1, 1, 1) or padding not in (None, 0, (0, 0, 0)):
            raise ValueError("SparseConv3d supports stride=1 and padding=0 only")
        self.kernel_size = _triple(kernel_size)
        self.dilation = _triple(dilation)
        self.out_channels = out_channels
        self.neighbor_map = neighbor_map
        self.backend = "torch"
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.backend == "flex_gemm":
            import flex_gemm
            from flex_gemm.ops.spconv import sparse_submanifold_conv3d

            flex_gemm.ops.spconv.set_algorithm("masked_implicit_gemm_splitk")
            flex_gemm.ops.spconv.set_hashmap_ratio(2.0)
            kx, ky, kz = self.kernel_size
            cache_key = f"SubMConv3d_neighbor_cache_{kz}x{ky}x{kx}_dilation{self.dilation}"
            cache = x.get_spatial_cache(cache_key)
            output, new_cache = sparse_submanifold_conv3d(
                x.feats, x.coords, torch.Size([*x.shape, *x.spatial_shape]), self.weight, self.bias, cache, self.dilation
            )
            if cache is None:
                x.register_spatial_cache(cache_key, new_cache)
            return x.replace(output)
        features = x.feats.to(self.weight.dtype)
        output = features.new_zeros((features.shape[0], self.out_channels))
        cache_key = f"torch_SubMConv3d_neighbor_cache_{self.kernel_size}_dilation{self.dilation}"
        neighbors = x.get_spatial_cache(cache_key)
        if neighbors is None:
            neighbors = self.neighbor_map(x.coords.to(features.device), self.kernel_size, self.dilation)
            x.register_spatial_cache(cache_key, neighbors)
        if features.device.type == "npu":
            stacked_key = f"{cache_key}_stacked"
            stacked_neighbors = x.get_spatial_cache(stacked_key)
            if stacked_neighbors is None:
                offsets = tuple(offset for offset, _ in neighbors)
                indices = torch.stack([indices for _, indices in neighbors]).to(features.device)
                stacked_neighbors = (offsets, indices)
                x.register_spatial_cache(stacked_key, stacked_neighbors)

            offsets, indices = stacked_neighbors
            weights = self.weight.permute(1, 2, 3, 4, 0).reshape(-1, features.shape[1], self.out_channels)
            chunk_size = 3
            for start in range(0, len(offsets), chunk_size):
                end = min(start + chunk_size, len(offsets))
                chunk_indices = indices[start:end]
                valid = chunk_indices >= 0
                gathered = features.index_select(0, chunk_indices.clamp_min(0).reshape(-1))
                gathered = gathered.reshape(end - start, features.shape[0], features.shape[1])
                gathered.mul_(valid.to(dtype=gathered.dtype).unsqueeze(-1))
                contribution = torch.bmm(gathered, weights[start:end]).sum(dim=0)
                output.add_(contribution)
                del gathered, contribution
            if self.bias is not None:
                output.add_(self.bias)
            return x.replace(output)

        for (kx, ky, kz), indices in neighbors:
            valid = indices >= 0
            indices = indices.clamp_min(0).to(features.device)
            gathered = features.index_select(0, indices)
            contribution = gathered @ self.weight[:, kx, ky, kz, :].transpose(0, 1)
            contribution.mul_(valid.to(device=features.device, dtype=contribution.dtype).unsqueeze(1))
            output.add_(contribution)
        if self.bias is not None:
            output.add_(self.bias)
        return x.replace(output)
