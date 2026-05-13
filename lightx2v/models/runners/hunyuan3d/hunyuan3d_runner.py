"""Runner for Hunyuan3D-2.1 image-to-3D generation.

Integrates two-stage Hunyuan3D inference into LightX2V's runner architecture:

  init_modules()
    ├── load_shape_model()   # Hunyuan3DDiTFlowMatchingPipeline (DiT + VAE + conditioner)
    └── load_paint_model()   # Hunyuan3DPaintPipeline (multiview PBR diffusion + RealESRGAN)

  run_pipeline(input_info)
    ├── run_input_encoder()  # read image → rembg background removal → RGBA PIL
    ├── run_shape()          # 50-step flow matching → trimesh mesh → demo.glb
    └── run_paint()          # multiview PBR diffusion → bake → textured .glb

Config JSON mapping:
  model_path          → Hunyuan3D-2.1 weight root
                        (e.g. /data/nvme1/wushuo/hf_models/Hunyuan3D-2.1)
  hunyuan3d_repo      → Hunyuan3D-2.1 source-code repository root; added to
                        sys.path and used as working directory
                        (e.g. /data/nvme1/wushuo/lyra_proj/Hunyuan3D-2.1)
  paint_model_path    → local directory for the paint diffusion model
                        (e.g. <model_path>/hunyuan3d-paintpbr-v2-1)
  realesrgan_ckpt_path → path to RealESRGAN_x4plus.pth (relative to hunyuan3d_repo
                         or absolute)
  paint_cfg_path      → multiview YAML config path (relative to hunyuan3d_repo or
                         absolute)
  custom_pipeline     → path to hunyuanpaintpbr pipeline dir (relative to
                         hunyuan3d_repo or absolute)
  num_inference_steps → shape DiT inference steps (default 50)
  guidance_scale      → shape CFG scale (default 5.0)
  max_num_view        → max multiview count for paint (default 6)
  resolution          → paint texture resolution in px (default 512)
  run_paint           → whether to run the paint stage (default true)

InputInfo: uses I23DInputInfo (task=i23d)
  image_path       → input image
  save_result_path → output directory
  seed             → random seed
"""

from __future__ import annotations

import os
import sys

import torch
from loguru import logger
from PIL import Image

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def _ensure_hunyuan3d_on_path(hunyuan3d_repo: str) -> None:
    """Insert Hunyuan3D-2.1 repository sub-packages into sys.path."""
    if not os.path.isdir(hunyuan3d_repo):
        raise FileNotFoundError(
            f"[Hunyuan3D] hunyuan3d_repo not found: '{hunyuan3d_repo}'. "
            "Set 'hunyuan3d_repo' in the config JSON."
        )
    for subdir in ("hy3dshape", "hy3dpaint"):
        p = os.path.join(hunyuan3d_repo, subdir)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    if hunyuan3d_repo not in sys.path:
        sys.path.insert(0, hunyuan3d_repo)
    logger.info(f"[Hunyuan3D] sys.path updated with repo: {hunyuan3d_repo}")


def _resolve_path(path: str, base: str) -> str:
    """Resolve path against base if it is relative."""
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


@RUNNER_REGISTER("hunyuan3d")
class Hunyuan3DRunner(BaseRunner):
    """LightX2V runner for Hunyuan3D-2.1 image-to-3D inference."""

    def __init__(self, config):
        super().__init__(config)
        _ensure_hunyuan3d_on_path(config["hunyuan3d_repo"])

        self.shape_pipeline = None
        self.paint_pipeline = None
        self._prev_cwd = None
        self.inputs = None

    # ------------------------------------------------------------------
    # Init / Model Loading
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Init modules")
    def init_modules(self):
        hunyuan3d_repo = self.config["hunyuan3d_repo"]
        _ensure_hunyuan3d_on_path(hunyuan3d_repo)

        self._prev_cwd = os.getcwd()
        os.chdir(hunyuan3d_repo)
        logger.info(f"[Hunyuan3D] Working directory set to: {hunyuan3d_repo}")

        with ProfilingContext4DebugL1("Load shape model"):
            self.shape_pipeline = self.load_shape_model()

        if self.config.get("run_paint", True):
            with ProfilingContext4DebugL1("Load paint model"):
                self.paint_pipeline = self.load_paint_model()

        self.config.lock()

    def load_shape_model(self):
        """Load Hunyuan3DShapeModel (LightX2V-style wrapper for DiT + VAE + conditioner)."""
        from lightx2v.models.networks.hunyuan3d import Hunyuan3DShapeModel

        model_path = self.config["model_path"]
        logger.info(f"[Hunyuan3D] Loading shape model from: {model_path}")
        model = Hunyuan3DShapeModel(
            model_path=model_path,
            config=self.config,
            device="cuda",
            dtype=torch.float16,
        )
        logger.info("[Hunyuan3D] Shape model loaded.")
        return model

    def load_paint_model(self):
        """Load Hunyuan3DPaintPipeline (multiview diffusion + RealESRGAN)."""
        from lightx2v.models.video_encoders.hunyuan3d import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        cfg = self.config
        repo = cfg["hunyuan3d_repo"]

        max_num_view = cfg.get("max_num_view", 6)
        resolution = cfg.get("resolution", 512)
        paint_conf = Hunyuan3DPaintConfig(max_num_view, resolution)

        paint_model_path = cfg.get(
            "paint_model_path",
            os.path.join(cfg["model_path"], "hunyuan3d-paintpbr-v2-1"),
        )
        paint_conf.multiview_pretrained_path = paint_model_path

        paint_cfg_path = _resolve_path(
            cfg.get("paint_cfg_path", "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"), repo
        )
        paint_conf.multiview_cfg_path = paint_cfg_path

        custom_pipeline = _resolve_path(
            cfg.get("custom_pipeline", "hy3dpaint/hunyuanpaintpbr"), repo
        )
        paint_conf.custom_pipeline = custom_pipeline

        realesrgan_ckpt = _resolve_path(
            cfg.get("realesrgan_ckpt_path", "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"), repo
        )
        paint_conf.realesrgan_ckpt_path = realesrgan_ckpt

        logger.info("[Hunyuan3D] Loading paint model...")
        pipeline = Hunyuan3DPaintPipeline(paint_conf)
        logger.info("[Hunyuan3D] Paint model loaded.")
        return pipeline

    # ------------------------------------------------------------------
    # Input Encoding
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run input encoder")
    def run_input_encoder(self):
        """Read input image and apply background removal if it is RGB-only.

        Returns a PIL Image in RGBA mode.
        """
        img_path = self.input_info.image_path
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"[Hunyuan3D] Input image not found: '{img_path}'")

        with ProfilingContext4DebugL1("Read image"):
            image = Image.open(img_path).convert("RGBA")

        if image.mode == "RGB":
            with ProfilingContext4DebugL1("Background removal (rembg)"):
                from lightx2v.models.input_encoders.hf.hunyuan3d import BackgroundRemover
                rembg = BackgroundRemover()
                image = rembg(image)
                logger.info("[Hunyuan3D] Background removed via rembg.")

        logger.info(f"[Hunyuan3D] Input image loaded: {img_path}  mode={image.mode}")
        return {"image": image}

    # ------------------------------------------------------------------
    # Shape generation
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run shape generation")
    def run_shape(self, image: Image.Image, output_dir: str) -> str:
        """Run DiT flow-matching to generate a 3D mesh.

        Returns the path to the exported .glb file.
        """
        num_steps = self.config.get("num_inference_steps", 50)
        guidance_scale = self.config.get("guidance_scale", 5.0)
        seed = getattr(self.input_info, "seed", None) or self.config.get("seed", 42)

        generator = torch.Generator(device=self.shape_pipeline.weights.device).manual_seed(seed)

        logger.info(
            f"[Hunyuan3D] Shape generation: steps={num_steps} "
            f"guidance={guidance_scale} seed={seed}"
        )
        meshes = self.shape_pipeline.infer(
            image=image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        mesh = meshes[0]

        glb_path = os.path.join(output_dir, "shape.glb")
        mesh.export(glb_path)
        logger.info(f"[Hunyuan3D] Shape saved: {glb_path}")
        return glb_path

    # ------------------------------------------------------------------
    # Paint generation
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run paint generation")
    def run_paint(self, glb_path: str, image: Image.Image, output_dir: str) -> str:
        """Run multiview PBR diffusion to texture the mesh.

        Returns the path to the textured .glb file.
        """
        output_mesh_path = os.path.join(output_dir, "textured_mesh.obj")
        logger.info(f"[Hunyuan3D] Paint generation: mesh={glb_path}")
        result_path = self.paint_pipeline(
            mesh_path=glb_path,
            image_path=image,
            output_mesh_path=output_mesh_path,
            save_glb=True,
        )
        textured_glb = result_path.replace(".obj", ".glb")
        if os.path.exists(textured_glb):
            logger.info(f"[Hunyuan3D] Textured mesh saved: {textured_glb}")
            return textured_glb
        logger.info(f"[Hunyuan3D] Paint output: {result_path}")
        return result_path

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["Hunyuan3DRunner"],
    )
    def run_pipeline(self, input_info):
        """Run full Hunyuan3D inference for one image."""
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()

        self.input_info = input_info

        img_path = input_info.image_path
        save_path = getattr(input_info, "save_result_path", None) or self.config.get(
            "save_result_path", "outputs/hunyuan3d"
        )
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join(save_path, base_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"[Hunyuan3D] Starting inference  image={img_path}")
        logger.info(f"[Hunyuan3D] Output → {output_dir}")

        with ProfilingContext4DebugL1("Run input encoder"):
            self.inputs = self.run_input_encoder()

        image = self.inputs["image"]

        with ProfilingContext4DebugL1("Run shape"):
            glb_path = self.run_shape(image, output_dir)

        if self.config.get("run_paint", True) and self.paint_pipeline is not None:
            with ProfilingContext4DebugL1("Run paint"):
                textured_path = self.run_paint(glb_path, image, output_dir)
        else:
            textured_path = glb_path

        self.end_run()

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"[Hunyuan3D] Done. Output: {textured_path}")
        return {"output_path": textured_path, "output_dir": output_dir}

    def end_run(self):
        """Restore working directory and free cached inputs."""
        self.inputs = None
        self.input_info = None

        if self._prev_cwd and os.path.isdir(self._prev_cwd):
            os.chdir(self._prev_cwd)
            logger.info(f"[Hunyuan3D] Restored working directory: {self._prev_cwd}")

        if self.config.get("offload", False):
            for attr in ("shape_pipeline", "paint_pipeline"):
                m = getattr(self, attr, None)
                if m is not None:
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
