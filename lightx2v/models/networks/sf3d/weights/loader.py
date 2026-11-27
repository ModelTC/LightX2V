from pathlib import Path

import torch
from omegaconf import OmegaConf

from lightx2v.models.networks.native_components.checkpoint import TensorWeights
from lightx2v.models.networks.native_components.ops import bind_modules
from lightx2v.utils.utils import load_safetensors_from_path

from .system import SF3D


def load_sf3d(config):
    root = Path(config["model_path"])
    cfg = OmegaConf.load(root / "config.yaml")
    cfg.pop("global_estimator_cls", None)
    cfg.pop("global_estimator", None)
    mapping = {
        "sf3d.models.camera.LinearCameraEmbedder": "camera.LinearCameraEmbedder",
        "sf3d.models.tokenizers.image.DINOV2SingleImageTokenizer": "image.DINOV2SingleImageTokenizer",
        "sf3d.models.tokenizers.triplane.TriplaneLearnablePositionalEmbedding": "triplane.TriplaneLearnablePositionalEmbedding",
        "sf3d.models.transformers.backbone.TwoStreamInterleaveTransformer": "backbone.TwoStreamInterleaveTransformer",
        "sf3d.models.network.PixelShuffleUpsampleNetwork": "network.PixelShuffleUpsampleNetwork",
        "sf3d.models.network.MaterialMLP": "network.MaterialMLP",
        "sf3d.models.image_estimator.clip_based_estimator.ClipBasedHeadEstimator": "image_estimator.ClipBasedHeadEstimator",
    }
    for key in list(cfg):
        if key.endswith("_cls"):
            if cfg[key] not in mapping:
                raise ValueError(f"Unsupported SF3D component: {cfg[key]}")
            cfg[key] = "lightx2v.models.networks.sf3d.weights." + mapping[cfg[key]]
    cfg.tets_path = config.get("tets_path", str(root / "assets/160_tets.npz"))
    if not Path(cfg.tets_path).is_file():
        raise FileNotFoundError("Prepare the fixed SF3D grid first: python scripts/sf3d/prepare_assets.py --model-path " + str(root))
    model = SF3D(cfg).eval()
    tensors = load_safetensors_from_path(str(root / "model.safetensors"))
    device = "mlu" if config.get("device", "cuda").startswith("mlu") else "cuda"
    dino = {k.removeprefix("image_tokenizer.model."): tensors.pop(k).to(device=device, dtype=torch.bfloat16) for k in list(tensors) if k.startswith("image_tokenizer.model.")}
    clip = {k.removeprefix("image_estimator.model.visual."): tensors.pop(k).to(device=device, dtype=torch.bfloat16) for k in list(tensors) if k.startswith("image_estimator.model.visual.")}
    # CLIP's text tower is stored in the training checkpoint but never used by
    # single-image material estimation. List this intentional exclusion.
    unused_text = [k for k in tensors if k.startswith("image_estimator.model.")]
    for k in unused_text:
        tensors.pop(k)
    # Illumination estimation is not part of the single-image mesh API.
    for key in list(tensors):
        if key.startswith("global_estimator."):
            tensors.pop(key)
    model.load_state_dict(tensors, strict=True)
    model = model.float().to(device)
    # Keep the field decoder and all geometry buffers in FP32. Cast neural
    # features too, so positional tokens cannot promote BF16 residual streams.
    for name in ("camera_embedder", "image_tokenizer", "tokenizer", "backbone", "post_processor", "image_estimator"):
        module = getattr(model, name, None)
        if module is not None:
            module.to(dtype=torch.bfloat16)
            for child in module.modules():
                if isinstance(child, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
                    child.float()
    bind_modules(model)
    model.requires_grad_(False)
    model.image_tokenizer.weights = TensorWeights(dino)
    model.image_estimator.visual_weights = TensorWeights(clip)
    return model
