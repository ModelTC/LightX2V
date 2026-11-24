from pathlib import Path

import numpy as np
import torch

from lightx2v.models.networks.hymotion.model import HYMotionModel
from lightx2v.models.networks.hymotion.utils.body_model import construct_smpl_data_dict
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.three_models import begin_request, finish_request, validate_a100_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("hymotion")
class HYMotionRunner(BaseRunner):
    def get_output_extension(self, input_info):
        return ".npz"

    def init_modules(self):
        validate_a100_config(self.config)
        self.model = HYMotionModel(self.config)

    @torch.inference_mode()
    def run_pipeline(self, input_info):
        started = begin_request()
        result = self.model.generate(input_info.prompt, input_info.seed, input_info.num_frames, self.config.get("infer_steps", 50), self.config.get("cfg_scale", 5.0))
        smpl = construct_smpl_data_dict(result["rot6d"][0].clone(), result["transl"][0].clone())
        arrays = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in smpl.items()}
        arrays.update({k: v.detach().cpu().numpy() for k, v in result.items() if isinstance(v, torch.Tensor)})
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise FloatingPointError(f"Non-finite HY-Motion output: {key}")
        output = input_info.save_result_path
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "wb") as stream:
                np.savez_compressed(stream, **arrays)
            self.record_output("motion", output)
        stats = finish_request(started, output, "hymotion", input_info, num_frames=int(result["rot6d"].shape[1]), requested_num_frames=input_info.num_frames, fps=30)
        return {"motion": result, "statistics": stats}
