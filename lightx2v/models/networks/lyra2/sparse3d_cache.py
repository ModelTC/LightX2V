# Sparse3DCache extracted from lyra_2._src.models.lyra2_model
# Only the Sparse3DCache class is included; all other model code lives in lyra_repo.

from __future__ import annotations

from typing import Optional

import torch

from lightx2v.models.networks.lyra2.forward_warp_utils_pytorch import unproject_points
from loguru import logger


class Sparse3DCache:
    def __init__(
        self,
        downsample: int = 4,
        store_device: str = "cuda",
        store_values: bool = False,
    ) -> None:
        self.downsample = int(downsample)
        self._store_device = str(store_device)
        self._store_values = bool(store_values)
        self._world_points: list[torch.Tensor] = []
        self._latent_indices: list[int] = []
        self._frame_ids: list[int] = []
        self._depths: list[torch.Tensor] = []
        self._w2cs: list[torch.Tensor] = []
        self._Ks: list[torch.Tensor] = []
        self._rgbs: dict[int, torch.Tensor] = {}

    @staticmethod
    def _scale_intrinsics(intrinsic: torch.Tensor, scale: float) -> torch.Tensor:
        assert intrinsic.dim() == 3 and intrinsic.shape[-2:] == (3, 3)
        K = intrinsic.clone()
        K[:, 0, 0] = K[:, 0, 0] * scale
        K[:, 1, 1] = K[:, 1, 1] * scale
        K[:, 0, 2] = K[:, 0, 2] * scale
        K[:, 1, 2] = K[:, 1, 2] * scale
        return K

    def add(
        self,
        depth_B_1_H_W: torch.Tensor,
        w2c_B_4_4: torch.Tensor,
        K_B_3_3: torch.Tensor,
        latent_index: int,
        frame_id: Optional[int] = None,
    ) -> None:
        ds = self.downsample
        depth_ds = depth_B_1_H_W[:, :, ::ds, ::ds]
        scale = 1.0 / float(ds)
        K_scaled = self._scale_intrinsics(K_B_3_3, scale)
        mask_valid = depth_ds > 0
        world_pts: torch.Tensor = unproject_points(
            depth=depth_ds,
            w2c=w2c_B_4_4,
            intrinsic=K_scaled,
            is_depth=True,
            is_ftheta=False,
            mask=mask_valid,
            return_sparse=False,
        )
        if self._store_device == "cpu":
            world_pts = world_pts.detach().to("cpu", non_blocking=True)
        self._world_points.append(world_pts)
        self._latent_indices.append(int(latent_index))
        self._frame_ids.append(int(latent_index) if frame_id is None else int(frame_id))
        if self._store_values:
            d = depth_B_1_H_W.detach()
            w = w2c_B_4_4.detach()
            k = K_B_3_3.detach()
            if self._store_device == "cpu":
                d = d.to("cpu", non_blocking=True)
                w = w.to("cpu", non_blocking=True)
                k = k.to("cpu", non_blocking=True)
            self._depths.append(d)
            self._w2cs.append(w)
            self._Ks.append(k)

    def store_rgb(self, frame_id: int, rgb: torch.Tensor) -> None:
        t = rgb.detach()
        if self._store_device == "cpu":
            t = t.to("cpu", non_blocking=True)
        self._rgbs[int(frame_id)] = t

    def get_rgb_by_frame_id(self, frame_id: int) -> torch.Tensor:
        fid = int(frame_id)
        if fid not in self._rgbs:
            raise KeyError(f"frame_id={fid} not found in Sparse3DCache RGB storage")
        return self._rgbs[fid]

    def get_rgbd_by_frame_id(self, frame_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._store_values:
            raise RuntimeError("Sparse3DCache.get_rgbd_by_frame_id requires store_values=True")
        for i in range(len(self._frame_ids) - 1, -1, -1):
            if int(self._frame_ids[i]) == int(frame_id):
                return self._depths[i], self._w2cs[i], self._Ks[i]
        raise KeyError(f"frame_id={int(frame_id)} not found in Sparse3DCache")

    def update_by_frame_id(
        self,
        frame_id: int,
        depth_B_1_H_W: torch.Tensor,
        w2c_B_4_4: torch.Tensor,
        K_B_3_3: torch.Tensor,
    ) -> bool:
        fid = int(frame_id)
        idx = None
        for i in range(len(self._frame_ids)):
            if int(self._frame_ids[i]) == fid:
                idx = i
                break
        if idx is None:
            return False

        compute_device = depth_B_1_H_W.device
        _depth = depth_B_1_H_W.to(compute_device)
        _w2c = w2c_B_4_4.to(compute_device)
        _K = K_B_3_3.to(compute_device)

        ds = self.downsample
        depth_ds = _depth[:, :, ::ds, ::ds]
        scale = 1.0 / float(ds)
        K_scaled = self._scale_intrinsics(_K, scale)
        mask_valid = depth_ds > 0
        world_pts: torch.Tensor = unproject_points(
            depth=depth_ds,
            w2c=_w2c,
            intrinsic=K_scaled,
            is_depth=True,
            is_ftheta=False,
            mask=mask_valid,
            return_sparse=False,
        )
        if self._store_device == "cpu":
            world_pts = world_pts.detach().to("cpu", non_blocking=True)
        self._world_points[idx] = world_pts
        if self._store_values:
            d = depth_B_1_H_W.detach()
            w = w2c_B_4_4.detach()
            k = K_B_3_3.detach()
            if self._store_device == "cpu":
                d = d.to("cpu", non_blocking=True)
                w = w.to("cpu", non_blocking=True)
                k = k.to("cpu", non_blocking=True)
            self._depths[idx] = d
            self._w2cs[idx] = w
            self._Ks[idx] = k
        return True

    @torch.no_grad()
    def retrieve(
        self,
        target_w2c_B_4_4: torch.Tensor,
        target_K_B_3_3: torch.Tensor,
        target_hw: tuple[int, int],
        num_latents: int,
        skip_last_n: int = 0,
        random: bool = False,
        max_coverage: bool = False,
        depth_threshold: float = 0.1,
    ) -> list[tuple[int, int]]:
        Ht, Wt = target_hw
        num_total = len(self._world_points)
        if num_total == 0 or num_latents <= 0:
            return []
        device = target_w2c_B_4_4.device
        ds = self.downsample
        scale = 1.0 / float(ds)
        Ht_ds = int((Ht + ds - 1) // ds)
        Wt_ds = int((Wt + ds - 1) // ds)

        if target_w2c_B_4_4.dim() == 4:
            num_views = int(target_w2c_B_4_4.shape[1])
            w2c_views = [target_w2c_B_4_4[:, v] for v in range(num_views)]
            K_views = [target_K_B_3_3[:, v] for v in range(num_views)]
        else:
            num_views = 1
            w2c_views = [target_w2c_B_4_4]
            K_views = [target_K_B_3_3]

        s = int(skip_last_n) if skip_last_n is not None else 0
        avail = max(0, num_total - max(0, s))
        if avail <= 0:
            return []

        pts_list = self._world_points[:avail]
        pts_stacked = torch.stack([p.to(device=device) for p in pts_list], dim=0)
        C, Bp, Hp, Wp, _ = pts_stacked.shape

        ones_hw = torch.ones(C, Bp, Hp, Wp, 1, device=device, dtype=pts_stacked.dtype)
        pts_homo = torch.cat([pts_stacked, ones_hw], dim=-1).unsqueeze(-1)

        K_ds_views = [self._scale_intrinsics(K_v, scale) for K_v in K_views]
        w2c_stack = torch.stack(w2c_views, dim=0)
        K_ds_stack = torch.stack(K_ds_views, dim=0)

        cam_homo = torch.matmul(
            w2c_stack[:, None, :, None, None],
            pts_homo[None],
        )
        cam_pts = cam_homo[..., :3, :]

        proj = torch.matmul(
            K_ds_stack[:, None, :, None, None],
            cam_pts,
        )

        z_all = proj[..., 2, 0]
        u_all = proj[..., 0, 0] / (z_all + 1e-7)
        v_all = proj[..., 1, 0] / (z_all + 1e-7)
        x_all = u_all.round().long()
        y_all = v_all.round().long()
        valid = (z_all > 0) & (x_all >= 0) & (x_all < Wt_ds) & (y_all >= 0) & (y_all < Ht_ds)

        if not valid.any():
            logger.info(
                f"Sparse3DCache.retrieve: no valid projections for any of {avail} candidates "
                f"(frame_ids={self._frame_ids[:avail]})"
            )
            return []

        view_ids, cand_ids, b_idx, _, _ = valid.nonzero(as_tuple=True)
        y_idx = y_all[valid]
        x_idx = x_all[valid]
        z_vals = z_all[valid].to(torch.float32)

        Btot = Bp
        pixels_per_view = Btot * Ht_ds * Wt_ds
        lin_keys = view_ids * pixels_per_view + b_idx * (Ht_ds * Wt_ds) + y_idx * Wt_ds + x_idx
        n_keys = num_views * pixels_per_view

        inf_val = torch.tensor(float("inf"), device=device, dtype=z_vals.dtype)
        min_depth = torch.full((n_keys,), inf_val, device=device, dtype=z_vals.dtype)
        min_depth.scatter_reduce_(0, lin_keys, z_vals, reduce="amin", include_self=True)

        min_d_for_pts = min_depth[lin_keys]
        num_cands = avail
        if max_coverage:
            keep = z_vals <= (min_d_for_pts + float(depth_threshold))
            if not keep.any():
                return []

            lin_keys_keep = lin_keys[keep]
            cand_keep = cand_ids[keep].to(torch.long)

            flat_idx = cand_keep * n_keys + lin_keys_keep
            mask_flat = torch.zeros((num_cands * n_keys,), device=device, dtype=torch.bool)
            mask_flat.scatter_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.bool))
            mask = mask_flat.view(num_cands, n_keys)

            k = min(int(num_latents), num_cands)
            if k <= 0:
                return []

            avail_frame_ids = self._frame_ids[:avail]
            max_frame_id = max(avail_frame_ids)
            excluded: set[int] = set()
            if max_frame_id > 0:
                last_cand_idx = int(max(range(avail), key=lambda i: avail_frame_ids[i]))
                covered = mask[last_cand_idx].clone()
                excluded.add(last_cand_idx)
                logger.info(
                    f"Sparse3DCache.retrieve(max_coverage): pre-covering pixels from temporally closest "
                    f"frame_id={avail_frame_ids[last_cand_idx]} (cand_idx={last_cand_idx}, "
                    f"pixels={int(covered.sum().item())})"
                )
            else:
                covered = torch.zeros((n_keys,), device=device, dtype=torch.bool)

            selected: list[int] = []
            for _ in range(k):
                additional = (mask & (~covered)).sum(dim=1)
                exclude_indices = list(selected) + list(excluded)
                if len(exclude_indices) > 0:
                    additional[torch.tensor(exclude_indices, device=device)] = -1
                best = int(torch.argmax(additional).item())
                if additional[best].item() <= 0:
                    break
                selected.append(best)
                covered |= mask[best]

            if len(selected) == 0:
                return []
            top_ids = selected
        else:
            is_min = z_vals <= (min_d_for_pts + 1e-6)
            big_int = torch.iinfo(torch.long).max
            cid_masked = torch.where(
                is_min, cand_ids.to(torch.long), torch.full_like(cand_ids, big_int, dtype=torch.long)
            )

            owner_lin_tmp = torch.full((n_keys,), big_int, device=device, dtype=torch.long)
            owner_lin_tmp.scatter_reduce_(0, lin_keys, cid_masked, reduce="amin", include_self=True)
            owner_lin = torch.where(owner_lin_tmp == big_int, torch.full_like(owner_lin_tmp, -1), owner_lin_tmp)

            valid_owner = owner_lin[owner_lin >= 0]
            counts = torch.bincount(valid_owner, minlength=num_cands)

            scores_t = counts.float()
            scores = scores_t.tolist()

            score_map = {
                int(self._latent_indices[i]): {"score": float(scores[i]), "frame_id": int(self._frame_ids[i])}
                for i in range(num_cands)
            }
            logger.info(f"Sparse3DCache.retrieve scores (latent_index -> score): {score_map}")

            if random and num_latents > 0:
                max_score = scores_t.max() if scores_t.numel() > 0 else scores_t.new_tensor(1.0)
                weights = torch.clamp(scores_t, min=0.0) + max_score * 0.02

                k = min(int(num_latents), scores_t.shape[0])
                if k <= 0:
                    return []
                sampled_ids = torch.multinomial(weights, num_samples=k, replacement=False)
                top_ids = [int(i) for i in sampled_ids.tolist()]
            else:
                top_ids = sorted(range(num_cands), key=lambda i: scores[i], reverse=True)[:num_latents]

        top_ids_reversed = top_ids[::-1]
        return [(self._latent_indices[i], self._frame_ids[i]) for i in top_ids_reversed]
