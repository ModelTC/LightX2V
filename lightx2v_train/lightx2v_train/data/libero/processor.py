import torch
import torchvision.transforms.functional as vision

from .normalizer import LinearNormalizer


class ConcatLeftAlign:
    def __init__(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    @staticmethod
    def _merge(values, metadata):
        return torch.cat([values[item["key"]] for item in metadata], dim=-1)

    @staticmethod
    def _split(value, metadata):
        result = {}
        offset = 0
        for item in metadata:
            width = int(item["shape"])
            result[item["key"]] = value[..., offset : offset + width]
            offset += width
        return result

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = self._merge(batch["action"], self.action_meta)
            batch["action_dim_is_pad"] = torch.zeros(batch["action"].shape[-1], dtype=torch.bool)
        batch["state"] = self._merge(batch["state"], self.state_meta)
        batch["state_dim_is_pad"] = torch.zeros(batch["state"].shape[-1], dtype=torch.bool)
        return batch

    def backward(self, batch):
        batch["action"] = self._split(batch["action"], self.action_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)
        return batch


class FastWAMProcessor:
    def __init__(
        self,
        shape_meta,
        num_obs_steps,
        image_size=(224, 224),
        delta_action_dim_mask=(True, True, True, True, True, True, False),
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = int(num_obs_steps)
        self.image_size = tuple(image_size)
        self.action_state_merger = ConcatLeftAlign(shape_meta)
        self.delta_action_dim_mask = torch.as_tensor(delta_action_dim_mask, dtype=torch.bool)
        self._normalizer = None

    @property
    def normalizer(self):
        if self._normalizer is None:
            raise RuntimeError("FastWAM processor normalization statistics have not been loaded.")
        return self._normalizer

    def set_normalizer_from_stats(self, stats):
        self._normalizer = LinearNormalizer(self.shape_meta, stats)

    def _validate_action_state(self, data):
        for group in ("action", "state"):
            for item in self.shape_meta[group]:
                actual = data[group][item["key"]].shape[-1]
                expected = int(item["raw_shape"])
                if actual != expected:
                    raise ValueError(f"{group}.{item['key']} width mismatch: expected {expected}, got {actual}")

    def _process_images(self, images):
        processed = []
        for item in self.shape_meta["images"]:
            image = images[item["key"]]
            if image.ndim != 4:
                raise ValueError(f"Image sequence must be [T,C,H,W], got {tuple(image.shape)}")
            image = vision.resize(
                image.to(torch.float32),
                self.image_size,
                interpolation=vision.InterpolationMode.BILINEAR,
                antialias=True,
            )
            expected = (self.num_obs_steps, *tuple(item["shape"]))
            if tuple(image.shape) != expected:
                raise ValueError(f"Processed image shape mismatch: expected {expected}, got {tuple(image.shape)}")
            processed.append(image)
        return torch.stack(processed)

    def preprocess(self, data):
        self._validate_action_state(data)
        action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
        action = data["action"]["default"]
        if bool(action_is_pad.any()):
            mask = action_is_pad[:, None] & self.delta_action_dim_mask[None, :]
            action[mask] = 0.0

        transformed = self.normalizer.forward({"action": data["action"], "state": data["state"]})
        transformed = self.action_state_merger.forward(transformed)
        return {
            "idx": data["idx"],
            "instruction": str(data["task"]),
            "pixel_values": self._process_images(data["images"]),
            "image_is_pad": data["image_is_pad"],
            "action": transformed["action"],
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": transformed["action_dim_is_pad"],
            "proprio": transformed["state"],
            "proprio_is_pad": data["state_is_pad"],
            "proprio_dim_is_pad": transformed["state_dim_is_pad"],
        }
