from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer
from torch import Tensor

from lightx2v.models.networks.sf3d.utils.base import dot

try:
    from uv_unwrapper import Unwrapper
except ImportError:
    Unwrapper = None


class Mesh:
    def __init__(self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"]) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Float[Tensor, "Nv 3"] | None = None
        self._v_tng: Float[Tensor, "Nv 3"] | None = None
        self._v_tex: Float[Tensor, "Nt 3"] | None = None
        self.unwrapper = Unwrapper() if Unwrapper is not None else None

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self.unwrap_uv()
        return self._v_tex

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm))
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_pos_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        duv1 = tex[1] - tex[0]
        duv2 = tex[2] - tex[0]
        dpos1 = pos[1] - pos[0]
        dpos2 = pos[2] - pos[0]

        tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]

        denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        denom_safe = denom.clip(1e-6)
        tang = tng_nom / denom_safe

        # Update all 3 vertices
        for i in range(3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        # Also normalize it. Here we do not normalize the individual triangles first so larger area
        # triangles influence the tangent space more
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    @torch.no_grad()
    def unwrap_uv(
        self,
        island_padding: float = 0.02,
    ) -> Mesh:
        if self.unwrapper is None:
            raise RuntimeError("SF3D texture baking requires the optional uv_unwrapper extension")
        uv, indices = self.unwrapper(self.v_pos, self.v_nrm, self.t_pos_idx, island_padding)

        # Do store per vertex UVs.
        # This means we need to duplicate some vertices at the seams
        individual_vertices = self.v_pos[self.t_pos_idx].reshape(-1, 3)
        individual_faces = torch.arange(
            individual_vertices.shape[0],
            device=individual_vertices.device,
            dtype=self.t_pos_idx.dtype,
        ).reshape(-1, 3)
        uv_flat = uv[indices].reshape((-1, 2))
        # uv_flat[:, 1] = 1 - uv_flat[:, 1]

        self.v_pos = individual_vertices
        self.t_pos_idx = individual_faces
        self._v_tex = uv_flat
        self._v_nrm = self._compute_vertex_normal()
        self._v_tng = self._compute_vertex_tangent()
