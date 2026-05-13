"""Hunyuan3D paint pipeline — copied from Hunyuan3D-2.1/hy3dpaint/textureGenPipeline.py.

Internal imports (DifferentiableRenderer, utils.*) are resolved via hy3dpaint on sys.path,
which is added by Hunyuan3DRunner.init_modules() before this module is imported.
"""

import copy
import os
import warnings

import numpy as np
import torch
import trimesh
from loguru import logger
from PIL import Image
from typing import List

from DifferentiableRenderer.MeshRender import MeshRender
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from utils.image_super_utils import imageSuperNet
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.simplify_mesh_utils import remesh_mesh
from utils.uvwrap_utils import mesh_uv_wrap

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view=6, resolution=512):
        self.device = "cuda"

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config: Hunyuan3DPaintConfig | None = None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def load_models(self):
        torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        logger.info("[Hunyuan3DPaint] Models loaded.")

    @torch.no_grad()
    def __call__(
        self,
        mesh_path: str | None = None,
        image_path: str | Image.Image | None = None,
        output_mesh_path: str | None = None,
        use_remesh: bool = True,
        save_glb: bool = True,
    ) -> str:
        """Generate PBR texture for a 3D mesh using multiview diffusion."""
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        else:
            image_prompt = image_path

        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]

        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, "textured_mesh.obj")

        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = (
            self.view_processor.bake_view_selection(
                self.config.candidate_camera_elevs,
                self.config.candidate_camera_azims,
                self.config.candidate_view_weights,
                self.config.max_selected_view_num,
            )
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(
            selected_camera_elevs, selected_camera_azims
        )

        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
        )

        enhance_images = {
            "albedo": copy.deepcopy(multiviews_pbr["albedo"]),
            "mr": copy.deepcopy(multiviews_pbr["mr"]),
        }
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])

        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                (self.config.render_size, self.config.render_size)
            )

        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"],
            selected_camera_elevs,
            selected_camera_azims,
            selected_view_weights,
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"],
            selected_camera_elevs,
            selected_camera_azims,
            selected_view_weights,
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        self.render.save_mesh(output_mesh_path, downsample=True)

        if save_glb:
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))

        logger.info(f"[Hunyuan3DPaint] Saved: {output_mesh_path}")
        return output_mesh_path
