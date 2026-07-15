from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from ..heads.camera_head import CameraHead
from ..heads.dense_head import DPTHead
from ..layers.block import Block, DistBlock
from ..utils.camera_utils import (
    extrinsics_to_vector,
    vector_to_camera_matrices,
)
from ..utils.priors import normalize_depth, normalize_poses
from .rasterization import GaussianSplatRenderer
from .visual_transformer import VisualGeometryTransformer


class WorldMirror(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        model_size="large",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        gs_dim=256,
        num_register_tokens=4,
        enable_cond=True,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_depth_mask=True,
        enable_norm=True,
        enable_gs=True,
        enable_bf16=False,
        patch_embed="dinov2_vitl14_reg",
        fixed_patch_embed=False,
        sampling_strategy="uniform",
        dpt_gradient_checkpoint=False,
        condition_strategy=["token", "pow3r", "token"],
        rope_base=100.0,
        normalized_rope=True,
        rope_normalize_coords="separate",
        rope_shift_coords=None,
        rope_jitter_coords=None,
        rope_rescale_coords=None,
        sp_size=1,
        # Legacy parameters (ignored, kept for checkpoint compatibility)
        set_sky_region_to_maxdepth=False,
        disable_gs_depth=False,
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.model_size = model_size
        if model_size == "large":
            embed_dim = 1024
            depth = 24
            num_heads = 16
            mlp_ratio = 4.0
            gs_dim = 256
            num_register_tokens = 4
        elif model_size == "base":
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4.0
            gs_dim = 256
            num_register_tokens = 4
        elif model_size == "small":
            embed_dim = 384
            depth = 12
            num_heads = 6
            mlp_ratio = 4.0
            gs_dim = 128
            num_register_tokens = 4
        elif model_size is None:
            pass
        print(
            f"[WorldMirror] model_size: {model_size}, embed_dim: {embed_dim}, "
            f"depth: {depth}, num_heads: {num_heads}, mlp_ratio: {mlp_ratio}, "
            f"gs_dim: {gs_dim}, num_register_tokens: {num_register_tokens}"
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.gs_dim = gs_dim
        self.num_register_tokens = num_register_tokens

        self.normalized_rope = normalized_rope
        self.rope_normalize_coords = rope_normalize_coords
        self.rope_shift_coords = rope_shift_coords
        self.rope_jitter_coords = rope_jitter_coords
        self.rope_rescale_coords = rope_rescale_coords

        self.enable_cam = enable_cam
        self.enable_pts = enable_pts
        self.enable_depth = enable_depth
        self.enable_depth_mask = enable_depth_mask
        self.enable_cond = enable_cond
        self.enable_norm = enable_norm
        self.enable_gs = enable_gs
        self.enable_bf16 = enable_bf16
        self.patch_embed = patch_embed
        self.sampling = sampling_strategy
        self.dpt_checkpoint = dpt_gradient_checkpoint
        self.cond_methods = condition_strategy
        self.config = self._store_config()
        self.sp_size = sp_size

        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            block_fn=Block if self.sp_size == 1 else DistBlock,
            normalized_rope=normalized_rope,
            rope_normalize_coords=rope_normalize_coords,
            rope_shift_coords=rope_shift_coords,
            rope_jitter_coords=rope_jitter_coords,
            rope_rescale_coords=rope_rescale_coords,
            enable_cond=enable_cond,
            sampling_strategy=sampling_strategy,
            patch_embed=patch_embed,
            fixed_patch_embed=fixed_patch_embed,
            condition_strategy=condition_strategy,
            intermediate_idxs=self.intermediate_layer_idx[model_size],
        )

        self._init_heads(embed_dim, patch_size, gs_dim)

        if enable_bf16:
            self.to = self._bf16_to

    def _store_config(self):
        """Save the model configuration."""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "gs_dim": self.gs_dim,
            "num_register_tokens": self.num_register_tokens,
            "normalized_rope": self.normalized_rope,
            "rope_normalize_coords": self.rope_normalize_coords,
            "rope_shift_coords": self.rope_shift_coords,
            "rope_jitter_coords": self.rope_jitter_coords,
            "rope_rescale_coords": self.rope_rescale_coords,
            "enable_cam": self.enable_cam,
            "enable_pts": self.enable_pts,
            "enable_depth": self.enable_depth,
            "enable_depth_mask": self.enable_depth_mask,
            "enable_norm": self.enable_norm,
            "enable_gs": self.enable_gs,
            "patch_embed": self.patch_embed,
            "sampling_strategy": self.sampling,
            "dpt_gradient_checkpoint": self.dpt_checkpoint,
            "condition_strategy": self.cond_methods,
            "model_size": self.model_size,
        }

    def _init_heads(self, dim, patch_size, gs_dim):
        """Initialize all prediction heads."""

        if self.enable_cam:
            self.cam_head = CameraHead(
                dim_in=2 * dim,
                block_fn=Block if self.sp_size == 1 else DistBlock,
            )

        if self.enable_pts:
            self.pts_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
                gradient_checkpoint=self.dpt_checkpoint,
            )

        if self.enable_depth:
            self.depth_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2 if not self.enable_depth_mask else 3,
                patch_size=patch_size,
                activation="exp+expp1" if not self.enable_depth_mask else "exp+expp1+linear",
                enable_depth_mask=self.enable_depth_mask,
                gradient_checkpoint=self.dpt_checkpoint,
            )

        if self.enable_norm:
            self.norm_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="norm+expp1",
                gradient_checkpoint=self.dpt_checkpoint,
            )

        if self.enable_gs:
            self.gs_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2 if not self.enable_depth_mask else 3,
                patch_size=patch_size,
                features=gs_dim,
                is_gsdpt=True,
                activation="exp+expp1" if not self.enable_depth_mask else "exp+expp1+linear",
                enable_depth_mask=self.enable_depth_mask,
                gradient_checkpoint=self.dpt_checkpoint,
            )
            self.gs_renderer = GaussianSplatRenderer(
                feature_dim=gs_dim,
                sh_degree=0,
                enable_prune=True,
                voxel_size=0.002,
            )

    def _bf16_to(self, *args, **kwargs):
        """Custom to() for bf16 mode: selectively move heads to target device/dtype."""
        self.visual_geometry_transformer = self.visual_geometry_transformer.to(*args, **kwargs)
        if self.enable_cam:
            self.cam_head = self.cam_head.to(*args, **kwargs)
        if self.enable_pts:
            self.pts_head = self.pts_head.to(*args, **kwargs)
        if self.enable_depth:
            self.depth_head = self.depth_head.to(*args, **kwargs)
        if self.enable_norm:
            self.norm_head = self.norm_head.to(*args, **kwargs)
        if self.enable_gs:
            self.gs_head = self.gs_head.to(*args, **kwargs)
            self.gs_renderer = self.gs_renderer.to(*args, **kwargs)
        return self

    def forward(
        self,
        views: Dict[str, torch.Tensor],
        cond_flags: List[int] = [0, 0, 0],
        is_inference=True,
        sp_size=1,
        sp_group=None,
    ):
        """Execute forward pass through the WorldMirror model.

        Args:
            views: Input data dictionary containing 'img' and optional priors.
            cond_flags: Conditioning flags [pose, depth, intrinsics].
            is_inference: Whether running in inference mode.
            sp_size: Sequence parallel size (>1 for multi-GPU).
            sp_group: Process group for SP communication.

        Returns:
            dict: Prediction results dictionary.
        """
        if self.enable_bf16:
            views["img"] = views["img"].to(torch.bfloat16)

        imgs = views["img"]
        use_cond = sum(cond_flags) > 0

        if use_cond:
            priors = self.extract_priors(views)
            token_list, patch_start_idx = self.visual_geometry_transformer(
                imgs,
                priors,
                cond_flags=cond_flags,
                enable_bf16=self.enable_bf16,
                sp_size=sp_size,
                sp_group=sp_group,
            )
        else:
            token_list, patch_start_idx = self.visual_geometry_transformer(
                imgs,
                enable_bf16=self.enable_bf16,
                sp_size=sp_size,
                sp_group=sp_group,
            )

        with torch.amp.autocast("cuda", enabled=(not self.enable_bf16), dtype=torch.float32):
            if sp_size > 1:
                preds = self._gen_all_preds_frame_sp(
                    token_list,
                    imgs,
                    patch_start_idx,
                    views,
                    cond_flags,
                    is_inference,
                    sp_size,
                    sp_group,
                )
            else:
                preds = self._gen_all_preds(
                    token_list,
                    imgs,
                    patch_start_idx,
                    views,
                    cond_flags,
                    is_inference,
                )

        return preds

    def _gen_all_preds_frame_sp(
        self,
        token_list,
        imgs,
        patch_start_idx,
        views,
        cond_flags,
        is_inference,
        sp_size,
        sp_group,
    ):
        """Generate predictions with frame-parallel DPT heads for SP inference.

        Splits S frames across sp_size ranks. Each rank processes S/sp_size frames
        through ALL head types, then Allgather to reconstruct full results.
        CameraHead runs on all frames on every rank (cross-view attention needed).
        """
        preds = {}
        rank = dist.get_rank()
        rank_in_sp = dist.get_group_rank(sp_group, rank)

        B, S, C_img, H, W = imgs.shape

        # Determine frame assignment for this rank
        if S >= sp_size:
            base_chunk = S // sp_size
            remainder = S % sp_size
            if rank_in_sp < remainder:
                my_count = base_chunk + 1
                my_start = rank_in_sp * (base_chunk + 1)
            else:
                my_count = base_chunk
                my_start = remainder * (base_chunk + 1) + (rank_in_sp - remainder) * base_chunk
        else:
            if rank_in_sp < S:
                my_count = 1
                my_start = rank_in_sp
            else:
                my_count = 0
                my_start = S

        my_end = my_start + my_count
        has_frames = my_count > 0

        if has_frames:
            token_list_chunk = [t[:, my_start:my_end].contiguous() for t in token_list]
            imgs_chunk = imgs[:, my_start:my_end].contiguous()

        # Camera head: runs on ALL frames on every rank (cross-view attention)
        if self.enable_cam:
            cam_seq = self.cam_head(token_list)
            cam_params = cam_seq[-1]
            preds["camera_params"] = cam_params
            c2w_mat, int_mat = self.transform_camera_vector(cam_params, H, W)
            preds["camera_poses"] = c2w_mat
            preds["camera_intrs"] = int_mat

        # DPT heads: frame-parallel
        if self.enable_depth:
            if has_frames:
                if self.enable_depth_mask:
                    depth_chunk, depth_conf_chunk, depth_mask_logits_chunk = self.depth_head(
                        token_list_chunk,
                        images=imgs_chunk,
                        patch_start_idx=patch_start_idx,
                    )
                else:
                    depth_chunk, depth_conf_chunk = self.depth_head(
                        token_list_chunk,
                        images=imgs_chunk,
                        patch_start_idx=patch_start_idx,
                    )
            else:
                depth_chunk = torch.zeros(B, 0, H, W, 1, dtype=imgs.dtype, device=imgs.device)
                depth_conf_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)
                if self.enable_depth_mask:
                    depth_mask_logits_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)

            preds["depth"] = self._frame_allgather_variable(depth_chunk, my_count, S, sp_size, sp_group, dim=1)
            preds["depth_conf"] = self._frame_allgather_variable(depth_conf_chunk, my_count, S, sp_size, sp_group, dim=1)
            if self.enable_depth_mask:
                depth_mask_logits_full = self._frame_allgather_variable(
                    depth_mask_logits_chunk,
                    my_count,
                    S,
                    sp_size,
                    sp_group,
                    dim=1,
                )
                preds["depth_mask_logits"] = depth_mask_logits_full
                preds["depth_mask"] = depth_mask_logits_full.sigmoid()

        if self.enable_pts:
            if has_frames:
                pts_chunk, pts_conf_chunk = self.pts_head(
                    token_list_chunk,
                    images=imgs_chunk,
                    patch_start_idx=patch_start_idx,
                )
            else:
                pts_chunk = torch.zeros(B, 0, H, W, 3, dtype=imgs.dtype, device=imgs.device)
                pts_conf_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)

            preds["pts3d"] = self._frame_allgather_variable(pts_chunk, my_count, S, sp_size, sp_group, dim=1)
            preds["pts3d_conf"] = self._frame_allgather_variable(pts_conf_chunk, my_count, S, sp_size, sp_group, dim=1)

        if self.enable_norm:
            if has_frames:
                normals_chunk, norm_conf_chunk = self.norm_head(
                    token_list_chunk,
                    images=imgs_chunk,
                    patch_start_idx=patch_start_idx,
                )
            else:
                normals_chunk = torch.zeros(B, 0, H, W, 3, dtype=imgs.dtype, device=imgs.device)
                norm_conf_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)

            preds["normals"] = self._frame_allgather_variable(normals_chunk, my_count, S, sp_size, sp_group, dim=1)
            preds["normals_conf"] = self._frame_allgather_variable(norm_conf_chunk, my_count, S, sp_size, sp_group, dim=1)

        # GS head: frame-parallel, then render on full gathered data
        if self.enable_gs:
            context_preds, context_nums = self.prepare_contexts(views, cond_flags, is_inference)
            gs_token_list = context_preds.get("token_list", token_list)
            gs_imgs = context_preds.get("imgs", imgs)
            gs_S = gs_imgs.shape[1]

            if gs_S == S and has_frames:
                gs_token_chunk = [t[:, my_start:my_end].contiguous() for t in gs_token_list]
                gs_imgs_chunk = gs_imgs[:, my_start:my_end].contiguous()

                if self.enable_depth_mask:
                    gs_feat_chunk, gs_depth_chunk, gs_depth_conf_chunk, gs_dmask_chunk = self.gs_head(
                        gs_token_chunk,
                        images=gs_imgs_chunk,
                        patch_start_idx=patch_start_idx,
                    )
                else:
                    gs_feat_chunk, gs_depth_chunk, gs_depth_conf_chunk = self.gs_head(
                        gs_token_chunk,
                        images=gs_imgs_chunk,
                        patch_start_idx=patch_start_idx,
                    )

                gs_feat = self._frame_allgather_variable(gs_feat_chunk, my_count, gs_S, sp_size, sp_group, dim=1)
                gs_depth = self._frame_allgather_variable(gs_depth_chunk, my_count, gs_S, sp_size, sp_group, dim=1)
                gs_depth_conf = self._frame_allgather_variable(gs_depth_conf_chunk, my_count, gs_S, sp_size, sp_group, dim=1)
                if self.enable_depth_mask:
                    gs_depth_mask_logits = self._frame_allgather_variable(
                        gs_dmask_chunk,
                        my_count,
                        gs_S,
                        sp_size,
                        sp_group,
                        dim=1,
                    )
                    preds["gs_depth_mask_logits"] = gs_depth_mask_logits
                    preds["gs_depth_mask"] = gs_depth_mask_logits.sigmoid()

            elif gs_S == S and not has_frames:
                gs_feat_c = self.gs_dim // 2
                gs_feat_chunk = torch.zeros(B, 0, gs_feat_c, H, W, dtype=imgs.dtype, device=imgs.device)
                gs_depth_chunk = torch.zeros(B, 0, H, W, 1, dtype=imgs.dtype, device=imgs.device)
                gs_depth_conf_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)

                gs_feat = self._frame_allgather_variable(gs_feat_chunk, 0, gs_S, sp_size, sp_group, dim=1)
                gs_depth = self._frame_allgather_variable(gs_depth_chunk, 0, gs_S, sp_size, sp_group, dim=1)
                gs_depth_conf = self._frame_allgather_variable(gs_depth_conf_chunk, 0, gs_S, sp_size, sp_group, dim=1)
                if self.enable_depth_mask:
                    gs_dmask_chunk = torch.zeros(B, 0, H, W, dtype=imgs.dtype, device=imgs.device)
                    gs_depth_mask_logits = self._frame_allgather_variable(
                        gs_dmask_chunk,
                        0,
                        gs_S,
                        sp_size,
                        sp_group,
                        dim=1,
                    )
                    preds["gs_depth_mask_logits"] = gs_depth_mask_logits
                    preds["gs_depth_mask"] = gs_depth_mask_logits.sigmoid()
            else:
                if self.enable_depth_mask:
                    gs_feat, gs_depth, gs_depth_conf, gs_depth_mask_logits = self.gs_head(
                        gs_token_list,
                        images=gs_imgs,
                        patch_start_idx=patch_start_idx,
                    )
                    preds["gs_depth_mask_logits"] = gs_depth_mask_logits
                    preds["gs_depth_mask"] = gs_depth_mask_logits.sigmoid()
                else:
                    gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                        gs_token_list,
                        images=gs_imgs,
                        patch_start_idx=patch_start_idx,
                    )

            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf

            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference,
            )

        return preds

    def _frame_allgather_variable(self, chunk, my_count, total_S, sp_size, sp_group, dim=1):
        """Allgather tensors with potentially variable chunk sizes across ranks.

        Pads each chunk to max_chunk_size, allgathers, then extracts valid frames
        from each rank's chunk to reconstruct the correct frame order.
        """
        if sp_size <= 1:
            return chunk

        if total_S >= sp_size:
            base_chunk = total_S // sp_size
            remainder = total_S % sp_size
            counts = [(base_chunk + 1) if r < remainder else base_chunk for r in range(sp_size)]
        else:
            counts = [1 if r < total_S else 0 for r in range(sp_size)]

        max_chunk = max(counts)

        current_size = chunk.shape[dim]
        if current_size < max_chunk:
            pad_size = max_chunk - current_size
            pad_shape = list(chunk.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, padding], dim=dim)

        chunk = chunk.contiguous()
        gathered_list = [torch.zeros_like(chunk) for _ in range(sp_size)]
        dist.all_gather(gathered_list, chunk, group=sp_group)

        valid_chunks = []
        for r in range(sp_size):
            cnt = counts[r]
            if cnt > 0:
                slices = [slice(None)] * gathered_list[r].dim()
                slices[dim] = slice(0, cnt)
                valid_chunks.append(gathered_list[r][tuple(slices)])

        return torch.cat(valid_chunks, dim=dim).contiguous()

    def _gen_all_preds(self, token_list, imgs, patch_start_idx, views, cond_flags, is_inference):
        """Generate all enabled predictions (single-GPU path)."""
        preds = {}

        if self.enable_cam:
            cam_seq = self.cam_head(token_list)
            cam_params = cam_seq[-1]
            preds["camera_params"] = cam_params
            c2w_mat, int_mat = self.transform_camera_vector(cam_params, imgs.shape[-2], imgs.shape[-1])
            preds["camera_poses"] = c2w_mat
            preds["camera_intrs"] = int_mat

        if self.enable_depth:
            if self.enable_depth_mask:
                depth, depth_conf, depth_mask_logits = self.depth_head(
                    token_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                )
                preds["depth_mask_logits"] = depth_mask_logits
                preds["depth_mask"] = depth_mask_logits.sigmoid()
            else:
                depth, depth_conf = self.depth_head(
                    token_list,
                    images=imgs,
                    patch_start_idx=patch_start_idx,
                )
            preds["depth"] = depth
            preds["depth_conf"] = depth_conf

        if self.enable_pts:
            pts, pts_conf = self.pts_head(
                token_list,
                images=imgs,
                patch_start_idx=patch_start_idx,
            )
            preds["pts3d"] = pts
            preds["pts3d_conf"] = pts_conf

        if self.enable_norm:
            normals, norm_conf = self.norm_head(
                token_list,
                images=imgs,
                patch_start_idx=patch_start_idx,
            )
            preds["normals"] = normals
            preds["normals_conf"] = norm_conf

        if self.enable_gs:
            context_preds, context_nums = self.prepare_contexts(views, cond_flags, is_inference)
            if self.enable_depth_mask:
                gs_feat, gs_depth, gs_depth_conf, gs_depth_mask_logits = self.gs_head(
                    context_preds.get("token_list", token_list),
                    images=context_preds.get("imgs", imgs),
                    patch_start_idx=patch_start_idx,
                )
                preds["gs_depth_mask_logits"] = gs_depth_mask_logits
                preds["gs_depth_mask"] = gs_depth_mask_logits.sigmoid()
            else:
                gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                    context_preds.get("token_list", token_list),
                    images=context_preds.get("imgs", imgs),
                    patch_start_idx=patch_start_idx,
                )

            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf

            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference,
            )

        return preds

    def extract_priors(self, views):
        """Extract and normalize geometric priors from input views.

        Returns (depths, rays, poses) tuple — each may be None if unavailable.
        """
        h, w = views["img"].shape[-2:]
        depths = rays = poses = None

        if "camera_poses" in views:
            extrinsics = views["camera_poses"][:, :, :3]
            extrinsics = normalize_poses(extrinsics)
            cam_params = extrinsics_to_vector(extrinsics)
            poses = cam_params[:, :, :7]
            if self.enable_bf16:
                poses = poses.to(torch.bfloat16)

        if "depthmap" in views:
            depth_h, depth_w = views["depthmap"].shape[-2:]
            depths = views["depthmap"]
            if depth_h != h or depth_w != w:
                depths = F.interpolate(depths, size=(h, w), mode="bilinear", align_corners=False)
            depths = normalize_depth(depths)
            if self.enable_bf16:
                depths = depths.to(torch.bfloat16)

        if "camera_intrs" in views:
            intrinsics = views["camera_intrs"][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)
            if self.enable_bf16:
                rays = rays.to(torch.bfloat16)

        return (depths, rays, poses)

    def transform_camera_vector(self, camera_params, h, w):
        """Convert camera parameter vector to c2w and intrinsic matrices."""
        ext_mat, int_mat = vector_to_camera_matrices(camera_params, image_hw=(h, w))
        homo_row = torch.tensor([0, 0, 0, 1], device=ext_mat.device).view(1, 1, 1, 4)
        homo_row = homo_row.repeat(ext_mat.shape[0], ext_mat.shape[1], 1, 1)
        w2c_mat = torch.cat([ext_mat, homo_row], dim=2)
        try:
            c2w_mat = torch.linalg.inv(w2c_mat)
        except Exception as e:
            print(f"[WorldMirror] linalg.inv fallback to CPU: {e}")
            c2w_mat = torch.linalg.inv(w2c_mat.cpu()).to(camera_params.device)
        return c2w_mat, int_mat

    def prepare_contexts(self, views, cond_flags, is_inference):
        """Prepare context views for GS rendering (training only, passthrough in inference)."""
        context_preds = {}
        if is_inference:
            return context_preds, views["img"].shape[1]

        assert self.enable_cam and self.enable_gs
        if "is_target" not in views:
            context_nums = views["img"].shape[1]
        else:
            context_nums = (~views["is_target"][0]).sum().item()
        context_imgs = views["img"][:, :context_nums]

        use_cond = sum(cond_flags) > 0

        if self.enable_bf16:
            context_imgs = context_imgs.to(torch.bfloat16)

        with torch.amp.autocast("cuda", enabled=(not self.enable_bf16), dtype=torch.bfloat16):
            if use_cond:
                priors = self.extract_priors(views)
                context_priors = (prior[:, :context_nums] if prior is not None else None for prior in priors)
                context_token_list, _ = self.visual_geometry_transformer(
                    context_imgs,
                    context_priors,
                    cond_flags=cond_flags,
                    enable_bf16=self.enable_bf16,
                )
            else:
                context_token_list, _ = self.visual_geometry_transformer(
                    context_imgs,
                    enable_bf16=self.enable_bf16,
                )

        context_cam_seq = self.cam_head(context_token_list)
        context_cam_params = context_cam_seq[-1]
        context_c2w_mat, context_int_mat = self.transform_camera_vector(context_cam_params, context_imgs.shape[-2], context_imgs.shape[-1])
        context_preds["camera_poses"] = context_c2w_mat
        context_preds["camera_intrs"] = context_int_mat
        context_preds["token_list"] = context_token_list
        context_preds["imgs"] = context_imgs

        return context_preds, context_nums
