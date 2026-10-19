import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from lightx2v.common.ops.inference_module import validate_bf16_graph_config
from lightx2v.common.ops.sparse import SparseTensor
from lightx2v.models.input_encoders.hf.trellis2.encoder import TrellisImageEncoder
from lightx2v.models.networks.trellis2.weights.loader import load_component
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.models.schedulers.trellis2.flow_euler import FlowEulerSampler
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("trellis2")
class Trellis2Runner(BaseRunner):
    """TRELLIS.2 512 inference, including texture voxels and geometry GLB export."""

    supported_request_fields_by_task = {"i23d": COMMON_REQUEST_FIELDS | {"image_path"}}

    def init_modules(self):
        validate_bf16_graph_config(self.config)
        if self.config.get("pipeline_type", "512") != "512":
            raise ValueError("TRELLIS.2 supports pipeline_type='512' in this runner")
        self.device = torch.device(AI_DEVICE)
        root = Path(self.config["model_path"])
        self.pipeline_config = json.loads((root / "pipeline.json").read_text())["args"]
        self.models = {}
        for name in ("sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512", "shape_slat_decoder", "tex_slat_flow_model_512", "tex_slat_decoder"):
            relative = self.pipeline_config["models"][name]
            prefix = "microsoft/TRELLIS-image-large/"
            path = Path(self.config.get("base_model_path", root / "base")) / relative[len(prefix) :] if relative.startswith(prefix) else root / relative
            self.models[name] = load_component(path, self.device, self.config.get("neighbor_map", "sorted"), self.config.get("attn_type", "torch_sdpa"), self.config.get("linear_backend", "Default"))
        self.encoder = TrellisImageEncoder(
            self.config.get("dino_model_path", root / "dinov3"),
            self.config.get("rembg_model_path", root / "rembg"),
            self.device,
            self.config.get("linear_backend", "Default"),
            self.config.get("deform_conv_backend", "torch_deform_bf16"),
        )
        self.samplers = {stage: FlowEulerSampler(**self.pipeline_config[f"{stage}_sampler"]["args"]) for stage in ("sparse_structure", "shape_slat", "tex_slat")}

    def _sample(self, stage, model, noise, cond, **kwargs):
        params = self.pipeline_config[f"{stage}_sampler"]["params"]
        return self.samplers[stage].sample(model, noise, **cond, **params, **kwargs)

    def _normalization(self, stage):
        values = self.pipeline_config[f"{stage}_normalization"]
        return (torch.tensor(values[key], device=self.device)[None] for key in ("std", "mean"))

    @torch.inference_mode()
    @ProfilingContext4DebugL1("Run Pipeline")
    def run_pipeline(self, input_info):
        with Image.open(input_info.image_path) as source:
            image = source.copy()
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            cond = self.encoder(image)
            torch.manual_seed(input_info.seed)
            structure_flow = self.models["sparse_structure_flow_model"]
            noise = torch.randn(1, structure_flow.in_channels, *([structure_flow.resolution] * 3)).to(self.device)
            structure = self._sample("sparse_structure", structure_flow, noise, cond)
            decoded = self.models["sparse_structure_decoder"](structure) > 0
            if decoded.shape[2] != 32:
                ratio = decoded.shape[2] // 32
                decoded = F.max_pool3d(decoded.float(), ratio, ratio) > 0.5
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
            if coords.shape[0] == 0:
                raise ValueError("Sparse structure decoder produced no active coordinates")
            shape_flow = self.models["shape_slat_flow_model_512"]
            noise = SparseTensor(torch.randn(coords.shape[0], shape_flow.in_channels).to(self.device), coords)
            shape = self._sample("shape_slat", shape_flow, noise, cond)
            shape_std, shape_mean = self._normalization("shape_slat")
            shape = shape * shape_std + shape_mean
            shape_condition = (shape - shape_mean) / shape_std
            texture_flow = self.models["tex_slat_flow_model_512"]
            noise = shape.replace(torch.randn(coords.shape[0], texture_flow.in_channels - shape.feats.shape[1]).to(self.device))
            texture = self._sample("tex_slat", texture_flow, noise, cond, concat_cond=shape_condition)
            texture_std, texture_mean = self._normalization("tex_slat")
            texture = texture * texture_std + texture_mean
            decoder = self.models["shape_slat_decoder"]
            decoder.resolution = 512
            meshes, subdivisions = decoder(shape, return_subs=True)
            voxels = self.models["tex_slat_decoder"](texture, guide_subs=subdivisions) * 0.5 + 0.5
        vertices, faces = meshes[0]
        if input_info.save_result_path:
            import trimesh

            output = Path(input_info.save_result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False).export(output)
        return {"vertices": vertices, "faces": faces, "texture_coords": voxels.coords.cpu(), "texture_attrs": voxels.feats.float().cpu()}
