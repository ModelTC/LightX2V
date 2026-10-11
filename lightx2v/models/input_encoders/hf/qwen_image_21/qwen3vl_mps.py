"""MPS conditioning with streamed language layers and on-demand vision weights."""

import torch
import torch.nn.functional as F

from lightx2v.common.offload.mps_weights import MpsStreamingBlockWeights
from lightx2v.common.offload.safetensors_checkpoint import SafetensorsCheckpoint
from lightx2v.utils.envs import GET_DTYPE

from .qwen3vl import Qwen3VLTextLayer, Qwen3VLVision, QwenImage21TextEncoder


class _DiskEmbedding:
    def __init__(self, checkpoint, weight_name):
        self.checkpoint = checkpoint
        self.weight_name = weight_name

    def apply(self, input_ids):
        weights = self.checkpoint.load_tensors([self.weight_name])
        # Map the vocabulary on CPU, transferring only the requested rows.
        return F.embedding(input_ids.cpu(), weights[self.weight_name]).to(device="mps", dtype=GET_DTYPE())


class _DiskVision:
    """Keep vision weights resident only while extracting reference features."""

    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.config = config
        self.weight_names = tuple(name for name in checkpoint.weight_map if name.startswith("model.visual."))

    def forward(self, pixels, grid):
        vision = Qwen3VLVision(self.config)
        try:
            vision.load({name: tensor.to(device="mps", dtype=GET_DTYPE()) for name, tensor in self.checkpoint.load_tensors(self.weight_names).items()})
            return vision.forward(pixels, grid)
        finally:
            del vision
            torch.mps.synchronize()
            torch.mps.empty_cache()


class QwenImage21MpsTextEncoder(QwenImage21TextEncoder):
    def _load_weights(self, path, config):
        checkpoint = SafetensorsCheckpoint(path)
        self.add_module("embedding", _DiskEmbedding(checkpoint, self.embedding.weight_name))
        self.add_module(
            "layers",
            MpsStreamingBlockWeights(checkpoint, lambda: Qwen3VLTextLayer(0, self.text_config, config), self.text_config["num_hidden_layers"]),
        )
        self.add_module("vision", _DiskVision(checkpoint, self.model_config["vision_config"]))

    def to_cuda(self, non_blocking=False):
        # Allocate on first iteration; no resident CPU model needs copying.
        return self

    def to_cpu(self, non_blocking=False):
        self.layers.release()
        return self
