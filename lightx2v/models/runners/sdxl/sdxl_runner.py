from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lightx2v.models.networks.sdxl.model import SDXLModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.three_models import begin_request, finish_request, validate_a100_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("sdxl")
class SDXLRunner(BaseRunner):
    def init_modules(self):
        validate_a100_config(self.config)
        self.model = SDXLModel(self.config)

    @torch.inference_mode()
    def run_pipeline(self, input_info):
        started = begin_request()
        size = input_info.size or [1024, 1024]
        image = self.model.generate(input_info.prompt, input_info.negative_prompt, input_info.seed, size, self.config.get("infer_steps", 30), self.config.get("cfg_scale", 5.0))
        if not torch.isfinite(image).all():
            raise FloatingPointError("SDXL generated non-finite values")
        pixels = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
        output = input_info.save_result_path
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(output)
        stats = finish_request(started, output, "sdxl", input_info, size=size, infer_steps=self.config.get("infer_steps", 30))
        return {"image": image if input_info.return_result_tensor else Image.fromarray(pixels), "statistics": stats}
