"""Thin wrapper around external ``hy3dpaint`` for Hunyuan3D mesh texture generation."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Callable, Iterator

from loguru import logger
from torchvision_fix import apply_fix

_POSTPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
_HY3DPAINT_LINK = os.path.join(_POSTPROCESS_DIR, "hy3dpaint")
_HF_PAINT_REPO_ID = "tencent/Hunyuan3D-2.1"


def default_hy3dpaint_root() -> str:
    return _HY3DPAINT_LINK


def resolve_hy3dpaint_root(hy_repo: str | None = None) -> str:
    if hy_repo is not None:
        hy3dpaint_root = os.path.join(hy_repo, "hy3dpaint")
    else:
        hy3dpaint_root = os.path.realpath(default_hy3dpaint_root())
    if not os.path.isdir(hy3dpaint_root):
        raise FileNotFoundError(
            f"Missing hy3dpaint. Clone Hunyuan3D-2.1 and symlink its hy3dpaint/ to {default_hy3dpaint_root()} (see scripts/hunyuan3d/run_hunyuan3d.sh), or pass --hy_repo /path/to/Hunyuan3D-2.1."
        )
    if hy3dpaint_root not in sys.path:
        sys.path.insert(0, hy3dpaint_root)
    return hy3dpaint_root


def _load_texture_gen_pipeline_module(hy3dpaint_root: str) -> ModuleType:
    module_name = "textureGenPipeline"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(hy3dpaint_root, "textureGenPipeline.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Missing textureGenPipeline.py: {module_path}")

    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()
    source = source.replace("        breakpoint()\n", "")

    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(source, module_path, "exec"), module.__dict__)
    return module


def resolve_paint_output_paths(save_path: str) -> tuple[str, str]:
    if save_path.endswith(".glb"):
        obj_path = f"{save_path[:-4]}.obj"
        return obj_path, save_path
    if save_path.endswith(".obj"):
        return save_path, f"{save_path[:-4]}.glb"
    obj_path = f"{save_path}.obj"
    return obj_path, f"{save_path}.glb"


def _resolve_local_paint_hf_root(model_path: str) -> str:
    """Return local HF repo root that contains hunyuan3d-paintpbr-v2-1/."""
    paint_subdir = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
    if os.path.isfile(os.path.join(model_path, "model_index.json")):
        return os.path.dirname(model_path)
    if os.path.isfile(os.path.join(paint_subdir, "model_index.json")):
        return model_path
    raise FileNotFoundError(f"Paint weights not found. Expected model_index.json under {model_path} or {paint_subdir}.")


@contextmanager
def _use_local_paint_weights(model_path: str) -> Iterator[None]:
    """Redirect upstream snapshot_download to local HF weights without patching hy3dpaint."""
    local_root = _resolve_local_paint_hf_root(model_path)
    import huggingface_hub
    from diffusers import DiffusionPipeline

    original_download: Callable = huggingface_hub.snapshot_download
    original_from_pretrained = DiffusionPipeline.from_pretrained

    def patched_download(repo_id, *args, **kwargs):
        if repo_id == _HF_PAINT_REPO_ID:
            logger.info(f"Using local paint weights from {local_root} (skip HF download)")
            return local_root
        return original_download(repo_id, *args, **kwargs)

    @classmethod
    def patched_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original_from_pretrained(*args, **kwargs)

    huggingface_hub.snapshot_download = patched_download
    DiffusionPipeline.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        huggingface_hub.snapshot_download = original_download
        DiffusionPipeline.from_pretrained = original_from_pretrained


def build_paint_config(
    model_path: str,
    hy_repo: str | None = None,
    max_num_view: int = 6,
    resolution: int = 512,
    device: str = "cuda",
    multiview_cfg_path: str | None = None,
    realesrgan_ckpt_path: str | None = None,
    custom_pipeline: str | None = None,
    dino_ckpt_path: str = "facebook/dinov2-giant",
):
    hy3dpaint_root = resolve_hy3dpaint_root(hy_repo)
    tgp = _load_texture_gen_pipeline_module(hy3dpaint_root)

    paint_conf = tgp.Hunyuan3DPaintConfig(max_num_view, resolution)
    paint_conf.device = device
    _resolve_local_paint_hf_root(model_path)
    paint_conf.multiview_pretrained_path = _HF_PAINT_REPO_ID
    paint_conf.multiview_cfg_path = multiview_cfg_path or os.path.join(hy3dpaint_root, "cfgs", "hunyuan-paint-pbr.yaml")
    paint_conf.realesrgan_ckpt_path = realesrgan_ckpt_path or os.path.join(hy3dpaint_root, "ckpt", "RealESRGAN_x4plus.pth")
    paint_conf.custom_pipeline = custom_pipeline or os.path.join(hy3dpaint_root, "hunyuanpaintpbr")
    paint_conf.dino_ckpt_path = dino_ckpt_path
    return paint_conf, tgp


class PaintPipeline:
    def __init__(
        self,
        model_path: str,
        hy_repo: str | None = None,
        max_num_view: int = 6,
        resolution: int = 512,
        device: str = "cuda",
        multiview_cfg_path: str | None = None,
        realesrgan_ckpt_path: str | None = None,
        custom_pipeline: str | None = None,
        dino_ckpt_path: str = "facebook/dinov2-giant",
    ):
        try:
            apply_fix()
        except Exception as exc:
            logger.warning(f"Failed to apply torchvision fix: {exc}")

        paint_conf, tgp = build_paint_config(
            model_path=model_path,
            hy_repo=hy_repo,
            max_num_view=max_num_view,
            resolution=resolution,
            device=device,
            multiview_cfg_path=multiview_cfg_path,
            realesrgan_ckpt_path=realesrgan_ckpt_path,
            custom_pipeline=custom_pipeline,
            dino_ckpt_path=dino_ckpt_path,
        )
        if not os.path.isfile(paint_conf.realesrgan_ckpt_path):
            raise FileNotFoundError(
                "RealESRGAN checkpoint not found: "
                f"{paint_conf.realesrgan_ckpt_path}. "
                f"Download RealESRGAN_x4plus.pth into {os.path.join(default_hy3dpaint_root(), 'ckpt')} "
                "(see scripts/hunyuan3d/run_hunyuan3d.sh)."
            )

        self.config: dict[str, Any] = {
            "hy3dpaint_root": resolve_hy3dpaint_root(hy_repo),
            "model_path": model_path,
            "max_num_view": max_num_view,
            "resolution": resolution,
            "device": device,
        }
        with _use_local_paint_weights(model_path):
            self._pipeline = tgp.Hunyuan3DPaintPipeline(paint_conf)

    def __call__(
        self,
        mesh_path: str,
        image_path: str,
        save_path: str,
        use_remesh: bool = True,
        save_glb: bool = True,
    ) -> str:
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"mesh_path does not exist: {mesh_path}")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path does not exist: {image_path}")

        obj_path, glb_path = resolve_paint_output_paths(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(glb_path)), exist_ok=True)

        logger.info(f"Running Hunyuan3D paint: mesh={mesh_path}, image={image_path}")
        self._pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=obj_path,
            use_remesh=use_remesh,
            save_glb=save_glb,
        )

        if save_glb and os.path.isfile(glb_path):
            if glb_path != save_path:
                shutil.copy2(glb_path, save_path)
            result_path = save_path
        elif os.path.isfile(obj_path):
            result_path = obj_path
        else:
            raise FileNotFoundError(f"Paint pipeline did not produce expected output under {obj_path}")

        logger.info(f"Saved textured mesh to {result_path}")
        return result_path
