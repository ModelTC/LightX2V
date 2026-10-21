import itertools

import torch


def kernel_offsets(kernel_size, dilation):
    if any(size <= 0 or size % 2 == 0 for size in kernel_size):
        raise ValueError("Submanifold convolution requires positive odd kernel sizes")
    ranges = [range(-(size // 2), size // 2 + 1) for size in kernel_size]
    for offset in itertools.product(*ranges):
        yield tuple(value + size // 2 for value, size in zip(offset, kernel_size)), tuple(value * step for value, step in zip(offset, dilation))


def cpu_hash_neighbors(coords, kernel_size, dilation):
    """Reference lookup; coordinates are unique [batch, x, y, z] rows."""
    rows = coords.detach().to(device="cpu", dtype=torch.int64).tolist()
    lookup = {tuple(row): index for index, row in enumerate(rows)}
    maps = []
    for offset, (dx, dy, dz) in kernel_offsets(kernel_size, dilation):
        indices = [lookup.get((row[0], row[1] + dx, row[2] + dy, row[3] + dz), -1) for row in rows]
        maps.append((offset, torch.tensor(indices, dtype=torch.int64)))
    return maps


def sorted_neighbors(coords, kernel_size, dilation):
    """Exact integer lookup on the coordinate device, including batch isolation."""
    coords = coords.detach().to(dtype=torch.int64)
    count = coords.shape[0]
    if count == 0:
        return [(offset, torch.empty(0, dtype=torch.int64)) for offset, _ in kernel_offsets(kernel_size, dilation)]
    extents = (coords[:, 1:].max(dim=0).values + 2).tolist()
    strides = coords.new_tensor([extents[0] * extents[1] * extents[2], extents[1] * extents[2], extents[2], 1])
    keys, order = torch.sort((coords * strides).sum(dim=1))
    maps = []
    for offset, delta in kernel_offsets(kernel_size, dilation):
        query = coords + coords.new_tensor((0, *delta))
        query_keys = (query * strides).sum(dim=1)
        position = torch.searchsorted(keys, query_keys).clamp_max(count - 1)
        candidate = order[position]
        # Out-of-grid queries can collide after linearization. Compare full rows.
        hit = (keys[position] == query_keys) & (coords[candidate] == query).all(dim=1)
        indices = torch.where(hit, candidate, torch.full_like(candidate, -1))
        maps.append((offset, indices.cpu()))
    return maps


NEIGHBOR_MAPS = {"cpu_hash": cpu_hash_neighbors, "sorted": sorted_neighbors}
