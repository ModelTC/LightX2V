"""Shape VAE decoder and mesh export for Hunyuan3D."""

from __future__ import annotations

import torch
import trimesh

from lightx2v.models.networks.hunyuan3d.utils import synchronize_timer
from lightx2v.models.networks.hunyuan3d.utils.checkpoint import (
    load_checkpoint_dict,
    load_pipeline_config,
    resolve_ckpt_paths,
    resolve_model_dir,
)
from lightx2v.models.video_encoders.hf.hunyuan3d.autoencoders import ShapeVAE, SurfaceExtractors
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


@synchronize_timer("Export to trimesh")
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                outputs.append(trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f))
        return outputs

    mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
    return trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)


class Hunyuan3DShapeVAEDecoder:
    """Decode shape latents to trimesh via Hunyuan3D ShapeVAE."""

    def __init__(self, vae: ShapeVAE, device=AI_DEVICE, dtype=torch.float16):
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, config, ckpt: dict[str, dict[str, torch.Tensor]] | None = None):
        model_path = config["model_path"]
        subfolder = config.get("subfolder", "hunyuan3d-dit-v2-1")
        use_safetensors = bool(config.get("use_safetensors", False))
        variant = config.get("variant", "fp16")
        dtype = GET_DTYPE()
        device = config.get("device", AI_DEVICE)

        model_dir = resolve_model_dir(model_path, subfolder)
        config_path, ckpt_path = resolve_ckpt_paths(model_dir, use_safetensors=use_safetensors, variant=variant)
        pipeline_cfg = load_pipeline_config(config_path)
        if ckpt is None:
            ckpt = load_checkpoint_dict(ckpt_path, use_safetensors=use_safetensors)

        vae = ShapeVAE(**pipeline_cfg["vae"]["params"])
        vae.load_state_dict(ckpt["vae"], strict=False)
        vae.to(device=device, dtype=dtype)
        vae.eval()

        return cls(vae, device=device, dtype=dtype)

    def set_surface_extractor(self, mc_algo=None):
        if mc_algo is None:
            mc_algo = "mc"
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    def decode_mesh(
        self,
        latents,
        box_v=1.01,
        mc_level=0.0,
        num_chunks=8000,
        octree_resolution=384,
        mc_algo=None,
        enable_pbar=True,
    ):
        self.set_surface_extractor(mc_algo)
        latents = (1.0 / self.vae.scale_factor) * latents
        latents = self.vae(latents)
        outputs = self.vae.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )
        mesh = export_to_trimesh(outputs)
        if isinstance(mesh, list):
            return mesh[0]
        return mesh
