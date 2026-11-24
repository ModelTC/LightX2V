from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_file

from lightx2v.models.networks.three_models.checkpoint import TensorWeights
from lightx2v.models.networks.three_models.ops import bind_modules

from .system import SF3D


def load_sf3d(config):
    root = Path(config["model_path"])
    cfg = OmegaConf.load(root / "config.yaml")
    mapping = {
        "sf3d.models.camera.LinearCameraEmbedder": "camera.LinearCameraEmbedder",
        "sf3d.models.tokenizers.image.DINOV2SingleImageTokenizer": "image.DINOV2SingleImageTokenizer",
        "sf3d.models.tokenizers.triplane.TriplaneLearnablePositionalEmbedding": "triplane.TriplaneLearnablePositionalEmbedding",
        "sf3d.models.transformers.backbone.TwoStreamInterleaveTransformer": "backbone.TwoStreamInterleaveTransformer",
        "sf3d.models.network.PixelShuffleUpsampleNetwork": "network.PixelShuffleUpsampleNetwork",
        "sf3d.models.network.MaterialMLP": "network.MaterialMLP",
        "sf3d.models.image_estimator.clip_based_estimator.ClipBasedHeadEstimator": "image_estimator.ClipBasedHeadEstimator",
        "sf3d.models.global_estimator.multi_head_estimator.MultiHeadEstimator": "global_estimator.MultiHeadEstimator",
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
    tensors = load_file(str(root / "model.safetensors"))
    dino = {k.removeprefix("image_tokenizer.model."): tensors.pop(k).cuda() for k in list(tensors) if k.startswith("image_tokenizer.model.")}
    clip = {k.removeprefix("image_estimator.model.visual."): tensors.pop(k).cuda() for k in list(tensors) if k.startswith("image_estimator.model.visual.")}
    # CLIP's text tower is stored in the training checkpoint but never used by
    # single-image material estimation. List this intentional exclusion.
    unused_text = [k for k in tensors if k.startswith("image_estimator.model.")]
    for k in unused_text:
        tensors.pop(k)
    model.load_state_dict(tensors, strict=True)
    model = model.float().cuda()
    counts = bind_modules(model)
    model.requires_grad_(False)
    model.image_tokenizer.weights = TensorWeights(dino)
    model.image_estimator.visual_weights = TensorWeights(clip)
    return model, counts, unused_text
