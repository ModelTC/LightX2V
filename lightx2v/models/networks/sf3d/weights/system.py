# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union  # noqa: UP035 - exported aliases used by the split infer modules

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from lightx2v.models.networks.sf3d.utils.base import BaseModule, ImageProcessor, convert_data, dilate_fill, find_class, float32_to_uint8_np, normalize, scale_tensor
from lightx2v.models.networks.sf3d.utils.image import create_intrinsic_from_fov_deg, default_cond_c2w, get_device
from lightx2v.models.networks.sf3d.utils.isosurface import MarchingTetrahedraHelper
from lightx2v.models.networks.sf3d.utils.mesh import Mesh

try:
    from texture_baker import TextureBaker
except ImportError:
    import logging

    logging.getLogger(__name__).warning("Could not import texture_baker. Please install it via `pip install texture-baker/`")
    raise ImportError("texture_baker not found")


class SF3D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        tets_path: str
        cond_image_size: int
        isosurface_resolution: int
        isosurface_threshold: float = 10.0
        radius: float = 1.0
        background_color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
        default_fovy_deg: float = 40.0
        default_distance: float = 1.6
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)
        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)
        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)
        image_estimator_cls: str = ""
        image_estimator: dict = field(default_factory=dict)
        global_estimator_cls: str = ""
        global_estimator: dict = field(default_factory=dict)

    cfg: Config

    @property
    def device(self):
        return next(self.parameters()).device

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(self.cfg.image_tokenizer)
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.camera_embedder = find_class(self.cfg.camera_embedder_cls)(self.cfg.camera_embedder)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(self.cfg.post_processor)
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.image_estimator = find_class(self.cfg.image_estimator_cls)(self.cfg.image_estimator)
        self.global_estimator = find_class(self.cfg.global_estimator_cls)(self.cfg.global_estimator)
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer("bbox", torch.as_tensor([[-self.cfg.radius, -self.cfg.radius, -self.cfg.radius], [self.cfg.radius, self.cfg.radius, self.cfg.radius]], dtype=torch.float32))
        self.isosurface_helper = MarchingTetrahedraHelper(self.cfg.isosurface_resolution, self.cfg.tets_path)
        self.baker = TextureBaker()
        self.image_processor = ImageProcessor()

    def triplane_to_meshes(self, triplanes: Float[Tensor, "B 3 Cp Hp Wp"]) -> list[Mesh]:
        from lightx2v.models.networks.sf3d.infer.system import SF3D_triplane_to_meshes

        return SF3D_triplane_to_meshes(self, triplanes)

    def query_triplane(self, positions: Float[Tensor, "*B N 3"], triplanes: Float[Tensor, "*B 3 Cp Hp Wp"]) -> Float[Tensor, "*B N F"]:
        from lightx2v.models.networks.sf3d.infer.system import SF3D_query_triplane

        return SF3D_query_triplane(self, positions, triplanes)

    def get_scene_codes(self, batch) -> Float[Tensor, "B 3 C H W"]:
        from lightx2v.models.networks.sf3d.infer.system import SF3D_get_scene_codes

        return SF3D_get_scene_codes(self, batch)

    def run_image(
        self, image: Image.Image | list[Image.Image], bake_resolution: int, remesh: Literal["none", "triangle", "quad"] = "none", vertex_count: int = -1, estimate_illumination: bool = False
    ) -> tuple[trimesh.Trimesh | list[trimesh.Trimesh], dict[str, Any]]:
        from lightx2v.models.networks.sf3d.infer.system import SF3D_run_image

        return SF3D_run_image(self, image, bake_resolution, remesh, vertex_count, estimate_illumination)

    def prepare_image(self, image):
        from lightx2v.models.networks.sf3d.infer.system import SF3D_prepare_image

        return SF3D_prepare_image(self, image)

    def generate_mesh(
        self, batch, bake_resolution: int, remesh: Literal["none", "triangle", "quad"] = "none", vertex_count: int = -1, estimate_illumination: bool = False
    ) -> tuple[list[trimesh.Trimesh], dict[str, Any]]:
        from lightx2v.models.networks.sf3d.infer.system import SF3D_generate_mesh

        return SF3D_generate_mesh(self, batch, bake_resolution, remesh, vertex_count, estimate_illumination)


__all__ = [
    "Any",
    "F",
    "Float",
    "Image",
    "List",
    "Literal",
    "Mesh",
    "Optional",
    "Tensor",
    "Tuple",
    "Union",
    "convert_data",
    "create_intrinsic_from_fov_deg",
    "default_cond_c2w",
    "dilate_fill",
    "float32_to_uint8_np",
    "get_device",
    "normalize",
    "np",
    "nullcontext",
    "rearrange",
    "scale_tensor",
    "torch",
    "trimesh",
]
