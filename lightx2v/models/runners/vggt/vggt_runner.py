from pathlib import Path

import numpy as np
import torch

from lightx2v.common.ops.inference_module import validate_bf16_graph_config
from lightx2v.models.networks.vggt.utils.image_processor import load_images
from lightx2v.models.networks.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from lightx2v.models.networks.vggt.weights.loader import load_vggt
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("vggt")
class VGGTRunner(BaseRunner):
    supported_request_fields_by_task = {"recon": COMMON_REQUEST_FIELDS | {"input_path"}}

    def init_modules(self):
        validate_bf16_graph_config(self.config)
        self.device = torch.device(AI_DEVICE)
        self.model = load_vggt(self.config["model_path"], self.device, self.config.get("attn_type", "torch_sdpa"), self.config.get("linear_backend", "Default"))

    @torch.inference_mode()
    @ProfilingContext4DebugL1("Run Pipeline")
    def run_pipeline(self, input_info):
        source = input_info.input_path
        if isinstance(source, (list, tuple)):
            paths = list(source)
        else:
            path = Path(source)
            paths = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if path.is_dir() else [path]
        images, original_coords = load_images(paths, target_size=518, workers=self.config.get("preprocess_workers", 1))
        images = images.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16), ProfilingContext4DebugL2("Run VGGT"):
            predictions = self.model(images)
        # Camera decoding and exported geometry use FP32, after network inference.
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"].float(), image_size_hw=(518, 518))
        predictions.update(extrinsics=extrinsics, intrinsics=intrinsics)
        predictions = {name: value.float().cpu() for name, value in predictions.items()}
        predictions["original_coords"] = original_coords
        if input_info.save_result_path:
            output = Path(input_info.save_result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **{name: value.numpy() for name, value in predictions.items()})
        return predictions
