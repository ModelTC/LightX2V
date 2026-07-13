import json

import torch


def _to_tensor_tree(value):
    if isinstance(value, dict):
        return {key: _to_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        try:
            return torch.tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            return [_to_tensor_tree(item) for item in value]
    return value


def load_dataset_stats(path):
    with open(path, encoding="utf-8") as handle:
        return _to_tensor_tree(json.load(handle))


class FieldNormalizer:
    def __init__(self, stats, mode="min/max"):
        if mode == "z-score":
            scale = 1.0 / (stats["std"] + 1e-8)
            offset = -stats["mean"] * scale
        else:
            low_key, high_key = ("min", "max") if mode == "min/max" else ("q01", "q99")
            lower = stats[low_key]
            upper = stats[high_key]
            width = upper - lower
            constant = width < 1e-4
            width = width.clone()
            width[constant] = 2.0
            scale = 2.0 / width
            offset = -1.0 - scale * lower
            offset[constant] = -lower[constant]
        self.scale = scale
        self.offset = offset

    def forward(self, value):
        return (value * self.scale + self.offset).clamp_(-5.0, 5.0)

    def backward(self, value):
        return (value - self.offset) / self.scale


class LinearNormalizer:
    def __init__(self, shape_meta, stats, mode="min/max"):
        self.normalizers = {"action": {}, "state": {}}
        for group in self.normalizers:
            for item in shape_meta[group]:
                key = item["key"]
                global_stats = {
                    name.removeprefix("global_"): value
                    for name, value in stats[group][key].items()
                    if name.startswith("global_")
                }
                self.normalizers[group][key] = FieldNormalizer(global_stats, mode=mode)

    def forward(self, batch):
        for group, fields in self.normalizers.items():
            if group not in batch:
                continue
            for key, normalizer in fields.items():
                batch[group][key] = normalizer.forward(batch[group][key])
        return batch

    def backward(self, batch):
        for group, fields in self.normalizers.items():
            for key, normalizer in fields.items():
                batch[group][key] = normalizer.backward(batch[group][key])
        return batch
