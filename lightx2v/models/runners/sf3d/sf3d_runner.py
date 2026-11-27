import os
from pathlib import Path

import numpy as np
import rembg
import torch
from PIL import Image

from lightx2v.models.networks.sf3d.model import SF3DModel
from lightx2v.models.networks.sf3d.utils.image import remove_background, resize_foreground
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.native_inference import validate_a100_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("sf3d")
class SF3DRunner(BaseRunner):
    def get_output_extension(self, input_info):
        return ".glb"

    def init_modules(self):
        self.device = validate_a100_config(self.config)
        if self.device.type == "cuda" and self.config.get("deterministic_algorithms", True):
            # UV normals/tangents use scatter reductions. A fixed seed alone
            # does not make their CUDA accumulation order deterministic.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        self.model = SF3DModel(self.config)
        self.background = rembg.new_session(providers=["CPUExecutionProvider"])

    @torch.inference_mode()
    def run_pipeline(self, input_info):
        with Image.open(input_info.image_path) as source:
            image = source.convert("RGBA")
        image = resize_foreground(remove_background(image, self.background), self.config.get("foreground_ratio", 0.85))
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            mesh, _ = self.model.generate(image, self.config.get("bake_resolution", 1024))
        if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise FloatingPointError("SF3D produced an empty or non-finite mesh")
        output = input_info.save_result_path
        if output:
            if Path(output).suffix.lower() != ".glb":
                raise ValueError("SF3D output must end in .glb to preserve PBR textures")
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            mesh.export(output)
            self.record_output("mesh", output)
        return {"mesh": mesh}
