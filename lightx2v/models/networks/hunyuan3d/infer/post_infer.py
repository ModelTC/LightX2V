"""Post-infer for Hunyuan3D shape generation.

Converts denoised latents into a 3D trimesh mesh:
  1. Un-scale latents by vae.scale_factor
  2. VAE decode (latents → occupancy field)
  3. Marching cubes / octree surface extraction → raw mesh vertices/faces
  4. Export to trimesh.Trimesh
"""

from __future__ import annotations

import torch
import trimesh
from loguru import logger


class Hunyuan3DPostInfer:
    def infer(
        self,
        weights,
        latents: torch.Tensor,
        box_v: float = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        octree_resolution: int = 384,
        enable_pbar: bool = True,
    ) -> list[trimesh.Trimesh]:
        """Decode latents into trimesh mesh(es).

        Args:
            weights:            Hunyuan3DShapeWeights
            latents:            denoised latent tensor from transformer_infer
            box_v:              bounding box half-size for marching cubes
            mc_level:           isosurface level
            num_chunks:         chunked evaluation granularity
            octree_resolution:  marching cubes resolution
            enable_pbar:        show tqdm progress bar

        Returns:
            List of trimesh.Trimesh objects (one per batch element)
        """
        logger.debug("[Hunyuan3DPostInfer] VAE decode + mesh extraction")
        latents = (1.0 / weights.vae.scale_factor) * latents
        latents = weights.vae(latents)

        raw_outputs = weights.vae.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            enable_pbar=enable_pbar,
        )

        meshes = self._export_to_trimesh(raw_outputs)
        logger.debug(f"[Hunyuan3DPostInfer] Extracted {len(meshes) if isinstance(meshes, list) else 1} mesh(es)")
        return meshes if isinstance(meshes, list) else [meshes]

    @staticmethod
    def _export_to_trimesh(mesh_output):
        """Convert raw mesh output to trimesh.Trimesh (flip winding order)."""
        if isinstance(mesh_output, list):
            results = []
            for m in mesh_output:
                if m is None:
                    results.append(None)
                else:
                    m.mesh_f = m.mesh_f[:, ::-1]
                    results.append(trimesh.Trimesh(m.mesh_v, m.mesh_f))
            return results
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        return trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
