from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

try:
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
except ImportError:
    rasterization = None
    DefaultStrategy = None
from torch import Tensor

from ..utils import act_gs, sh_utils
from ..utils.frustum import calculate_unprojected_mask
from ..utils.geometry import depth_to_world_coords_points


class Rasterizer:
    def __init__(self, rasterization_mode="classic", packed=True, abs_grad=True, with_eval3d=False, camera_model="pinhole", sparse_grad=False, distributed=False, grad_strategy=DefaultStrategy):
        self.rasterization_mode = rasterization_mode
        self.packed = packed
        self.abs_grad = abs_grad
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.grad_strategy = grad_strategy
        self.distributed = distributed
        self.with_eval3d = with_eval3d

    def rasterize_splats(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=(self.abs_grad if isinstance(self.grad_strategy, DefaultStrategy) else False),
            sparse_grad=self.sparse_grad,
            rasterize_mode=self.rasterization_mode,
            distributed=self.distributed,
            camera_model=self.camera_model,
            with_eval3d=self.with_eval3d,
            render_mode="RGB+ED",
            **kwargs,
        )
        return render_colors[..., :3], render_colors[..., 3:], render_alphas

    def rasterize_batches(self, means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kwargs):
        rendered_colors, rendered_depths, rendered_alphas = [], [], []
        batch_size = len(means)
        for i in range(batch_size):
            means_i = means[i]  # [N, 4]
            quats_i = quats[i]  # [N, 4]
            scales_i = scales[i]  # [N, 3]
            opacities_i = opacities[i]  # [N,]
            colors_i = colors[i]  # [N, 3]
            viewmats_i = viewmats[i]  # [V, 4, 4]
            Ks_i = Ks[i]  # [V, 3, 3]
            render_colors_i, render_depths_i, render_alphas_i = self.rasterize_splats(means_i, quats_i, scales_i, opacities_i, colors_i, viewmats_i, Ks_i, width, height, **kwargs)

            rendered_colors.append(render_colors_i)  # V H W 3
            rendered_depths.append(render_depths_i)  # V H W 1
            rendered_alphas.append(render_alphas_i)  # V H W 1

        rendered_colors = torch.stack(rendered_colors, dim=0)  # B V H W 3
        rendered_depths = torch.stack(rendered_depths, dim=0)  # B V H W 1
        rendered_alphas = torch.stack(rendered_alphas, dim=0)  # B V H W 1

        return rendered_colors, rendered_depths, rendered_alphas


class GaussianSplatRenderer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,  # Output channels of gs_feat_head
        sh_degree: int = 0,
        enable_prune: bool = True,
        voxel_size: float = 0.002,  # Default voxel size for prune_gs
        enable_conf_filter: bool = False,  # Enable confidence filtering
        conf_threshold_percent: float = 30.0,  # Confidence threshold percentage
        max_gaussians: int = 5000000,  # Maximum number of Gaussians
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.nums_sh = (sh_degree + 1) ** 2
        self.voxel_size = voxel_size
        self.enable_prune = enable_prune
        self.enable_conf_filter = enable_conf_filter
        self.conf_threshold_percent = conf_threshold_percent
        self.max_gaussians = max_gaussians

        # Predict Gaussian parameters from GS features (quaternions/scales/opacities/SH/weights)
        splits_and_inits = [
            (4, 1.0, 0.0),  # quats
            (3, 0.00003, -7.0),  # scales
            (1, 1.0, -2.0),  # opacities
            (3 * self.nums_sh, 1.0, 0.0),  # residual_sh
            (1, 1.0, -2.0),  # weights
        ]
        gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1

        self.gs_head = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
        )
        # Initialize weights and biases of the final layer by segments
        final_conv_layer = self.gs_head[-1]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            nn.init.xavier_uniform_(final_conv_layer.weight[start_channels : start_channels + out_channel], s)
            nn.init.constant_(final_conv_layer.bias[start_channels : start_channels + out_channel], b)
            start_channels += out_channel

        # Rasterizer
        self.rasterizer = Rasterizer()

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def render(
        self,
        gs_feats: torch.Tensor,  # [B, S, 3, H, W]
        images: torch.Tensor,  # [B, S+V, 3, H, W]
        predictions: Dict[str, torch.Tensor],  # From WorldMirror: pose/depth/pts3d etc
        views: Dict[str, torch.Tensor],
        context_predictions: Dict[str, torch.Tensor],
        is_inference: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        B, _, _, H, W = images.shape
        S = context_predictions.get("imgs", images).shape[1]  # context view nums
        V = images.shape[1] - S  # target view nums

        # 1) Predict GS features from tokens, then convert to Gaussian parameters
        gs_feats_reshape = rearrange(gs_feats, "b s c h w -> (b s) c h w")
        # Align input dtype with gs_head weights (handles fp32 input from
        # precision-critical DPT output_conv2 when model runs in bf16 mode)
        head_dtype = next(self.gs_head.parameters()).dtype
        if gs_feats_reshape.dtype != head_dtype:
            gs_feats_reshape = gs_feats_reshape.to(head_dtype)
        gs_params = self.gs_head(gs_feats_reshape)
        gt_colors = images.permute(0, 1, 3, 4, 2)

        # 2) Select rendering cameras
        if self.training:
            # Using all gt cameras
            render_viewmats, render_Ks = self.prepare_cameras(views, S + V)
            gt_valid_masks_src = views["valid_mask"][:, :S]  # [B, S, H, W]
            gt_valid_masks_tgt = views["valid_mask"][:, S:]  # [B, V, H, W]
            unproject_masks = calculate_unprojected_mask(views, S)  # [B, V, H, W]
            valid_masks = torch.cat([gt_valid_masks_src, (gt_valid_masks_tgt & unproject_masks)], dim=1)
        else:
            # Re-predict the camera for novel views and perform translation scale alignment
            pred_all_extrinsic, pred_all_intrinsic = self.prepare_cameras(predictions, S + V)
            scale_factor = torch.ones(B, device=images.device)
            if "camera_poses" in context_predictions:
                pred_context_extrinsic, _ = self.prepare_cameras(context_predictions, S)
                scale_factor = pred_context_extrinsic[:, :, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True) / (pred_all_extrinsic[:, :S, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True) + 1e-6)

            pred_all_extrinsic[..., :3, 3] = pred_all_extrinsic[..., :3, 3] * scale_factor.unsqueeze(-1)
            render_viewmats, render_Ks = pred_all_extrinsic, pred_all_intrinsic
            valid_masks = views.get("valid_mask", torch.ones(B, S + V, H, W, dtype=bool, device=images.device))

        # 3) Generate splats from gs_params + predictions, and perform voxel merging
        if self.training:
            splats = self.prepare_splats(
                views,
                predictions,
                images,
                gs_params,
                S,
                position_from="gsdepth+gtcamera",
            )
        elif not is_inference:
            splats = self.prepare_splats(
                views,
                predictions,
                images,
                gs_params,
                S,
                context_predictions,
                position_from="gsdepth+predcamera",
            )
        else:
            splats = self.prepare_splats(
                views,
                predictions,
                images,
                gs_params,
                S,
                position_from="gsdepth+predcamera",
            )

        if is_inference:
            predictions["splats"] = splats
            return predictions

        # Apply confidence filtering before pruning
        if self.enable_conf_filter and "gs_depth_conf" in predictions:
            splats = self.apply_confidence_filter(splats, predictions["gs_depth_conf"])

        if self.enable_prune:
            splats = self.prune_gs(splats, voxel_size=self.voxel_size)

        predictions["splats"] = splats

        # 4) Rasterization rendering (training: chunked rendering + novel view valid mask correction; evaluation: view-by-view)

        # Prevent OOM by using chunked rendering
        rendered_colors_list, rendered_depths_list, rendered_alphas_list = [], [], []
        chunk_size = 2
        for i in range(0, gt_colors.shape[1], chunk_size):
            end_idx = min(i + chunk_size, gt_colors.shape[1])
            viewmats_i = render_viewmats[:, i:end_idx]
            Ks_i = render_Ks[:, i:end_idx]

            rendered_colors, rendered_depths, rendered_alphas = self.rasterizer.rasterize_batches(
                splats["means"],
                splats["quats"],
                splats["scales"],
                splats["opacities"],
                splats["sh"] if "sh" in splats else splats["colors"],
                viewmats_i.detach(),
                Ks_i.detach(),
                width=images.shape[-1],
                height=images.shape[-2],
                sh_degree=min(self.sh_degree, 0) if "sh" in splats else None,
            )
            rendered_colors_list.append(rendered_colors)
            rendered_depths_list.append(rendered_depths)
            rendered_alphas_list.append(rendered_alphas)

        rendered_colors = torch.cat(rendered_colors_list, dim=1)
        rendered_depths = torch.cat(rendered_depths_list, dim=1)
        rendered_alphas = torch.cat(rendered_alphas_list, dim=1)

        if self.training and V > 0:
            nvs_rendered_mask = rendered_alphas[:, S:, ..., 0].detach() > 0.1
            valid_masks[:, S:] = nvs_rendered_mask & valid_masks[:, S:]

        # 5) return predictions
        predictions["rendered_colors"] = rendered_colors
        predictions["rendered_depths"] = rendered_depths
        predictions["rendered_alphas"] = rendered_alphas
        predictions["gt_colors"] = gt_colors.float()
        predictions["gt_depths"] = views.get("depthmap")
        predictions["valid_masks"] = valid_masks.bool()
        predictions["rendered_extrinsics"] = render_viewmats
        predictions["rendered_intrinsics"] = render_Ks

        return predictions

    def apply_confidence_filter(self, splats, gs_depth_conf):
        """
        Apply confidence filtering to Gaussian splats before pruning.
        Discard bottom p% confidence points, keep top (100-p)%.

        Args:
            splats: Dictionary containing Gaussian parameters
            gs_depth_conf: Confidence tensor [B, S, H, W]

        Returns:
            Filtered splats dictionary
        """
        if not self.enable_conf_filter or gs_depth_conf is None:
            return splats

        device = splats["means"].device
        B, N = splats["means"].shape[:2]

        # Flatten confidence: [B, S, H, W] -> [B, N]
        conf = gs_depth_conf.flatten(1).to(device)
        # Mask invalid/very small values
        conf = conf.masked_fill(conf <= 1e-5, float("-inf"))

        # Keep top (100-p)% points, discard bottom p%
        if self.conf_threshold_percent > 0:
            keep_from_percent = int(np.ceil(N * (100.0 - self.conf_threshold_percent) / 100.0))
        else:
            keep_from_percent = N
        K = max(1, min(self.max_gaussians, keep_from_percent))

        # Select top-K indices for each batch (deterministic, no randomness)
        topk_idx = torch.topk(conf, K, dim=1, largest=True, sorted=False).indices  # [B, K]

        filtered = {}
        mask_keys = ["means", "quats", "scales", "opacities", "sh", "weights"]

        for key in splats.keys():
            if key in mask_keys and key in splats:
                x = splats[key]
                if x.ndim == 2:  # [B, N]
                    filtered[key] = torch.gather(x, 1, topk_idx)
                else:
                    # Expand indices to match tensor dimensions
                    expand_idx = topk_idx.clone()
                    for i in range(x.ndim - 2):
                        expand_idx = expand_idx.unsqueeze(-1)
                    expand_idx = expand_idx.expand(-1, -1, *x.shape[2:])
                    filtered[key] = torch.gather(x, 1, expand_idx)
            else:
                filtered[key] = splats[key]

        return filtered

    def prune_gs(self, splats, voxel_size=0.002, filter_mask=None):
        """
        Prune Gaussian splats by optional mask filtering + voxel merging.

        Args:
            splats: Dictionary containing Gaussian parameters.
                    Each value is [B, S*H*W, ...] (batch of per-pixel gaussians).
            voxel_size: Size of voxels for spatial grouping.
            filter_mask: Optional bool tensor [B, S*H*W] or numpy [S, H, W].
                         True = keep, False = discard.  Applied before voxel merge.

        Returns:
            Dictionary with pruned/merged splats (list-of-tensors per batch).
        """
        B = splats["means"].shape[0]
        merged_splats_list = []
        device = splats["means"].device

        for i in range(B):
            # Extract splats for current batch
            splats_i = {k: splats[k][i] for k in ["means", "quats", "scales", "opacities", "sh", "weights"]}

            # --- Apply filter_mask (discard unwanted gaussians before merge) ---
            if filter_mask is not None:
                if isinstance(filter_mask, np.ndarray):
                    fm = torch.from_numpy(filter_mask.reshape(-1)).to(device)
                elif filter_mask.dim() == 3:
                    # [S, H, W] -> flatten
                    fm = filter_mask.reshape(-1).to(device)
                else:
                    fm = filter_mask[i].to(device)
                fm = fm.bool()
                splats_i = {k: v[fm] for k, v in splats_i.items()}

            N_in = splats_i["means"].shape[0]
            if N_in == 0:
                # All filtered out — push empty tensors
                merged_splats_list.append(
                    {
                        "means": torch.zeros((0, 3), device=device),
                        "quats": torch.zeros((0, 4), device=device),
                        "scales": torch.zeros((0, 3), device=device),
                        "opacities": torch.zeros(0, device=device),
                        "sh": torch.zeros((0, self.nums_sh, 3), device=device),
                    }
                )
                continue

            # Compute voxel indices
            coords = splats_i["means"]
            voxel_indices = (coords / voxel_size).floor().long()
            min_indices = voxel_indices.min(dim=0)[0]
            voxel_indices = voxel_indices - min_indices
            max_dims = voxel_indices.max(dim=0)[0] + 1

            # Flatten 3D voxel indices to 1D
            flat_indices = voxel_indices[:, 0] * max_dims[1] * max_dims[2] + voxel_indices[:, 1] * max_dims[2] + voxel_indices[:, 2]

            # Find unique voxels and inverse mapping
            unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            K = len(unique_voxels)

            # Initialize merged splats
            merged = {
                "means": torch.zeros((K, 3), device=device),
                "quats": torch.zeros((K, 4), device=device),
                "scales": torch.zeros((K, 3), device=device),
                "opacities": torch.zeros(K, device=device),
                "sh": torch.zeros((K, self.nums_sh, 3), device=device),
            }

            # Get weights and compute weight sums per voxel
            weights = splats_i["weights"]
            weight_sums = torch.zeros(K, device=device)
            weight_sums.scatter_add_(0, inverse_indices, weights)
            weight_sums = torch.clamp(weight_sums, min=1e-8)

            # Merge means (weighted average)
            for d in range(3):
                merged["means"][:, d].scatter_add_(0, inverse_indices, splats_i["means"][:, d] * weights)
            merged["means"] = merged["means"] / weight_sums.unsqueeze(1)

            # Merge spherical harmonics (weighted average)
            for d in range(3):
                merged["sh"][:, 0, d].scatter_add_(0, inverse_indices, splats_i["sh"][:, 0, d] * weights)
            merged["sh"] = merged["sh"] / weight_sums.unsqueeze(-1).unsqueeze(-1)

            # Merge opacities (weighted sum of squares)
            merged["opacities"].scatter_add_(0, inverse_indices, weights * weights)
            merged["opacities"] = merged["opacities"] / weight_sums

            # Merge scales (weighted average)
            for d in range(3):
                merged["scales"][:, d].scatter_add_(0, inverse_indices, splats_i["scales"][:, d] * weights)
            merged["scales"] = merged["scales"] / weight_sums.unsqueeze(1)

            # Merge quaternions (weighted average + normalization)
            for d in range(4):
                merged["quats"][:, d].scatter_add_(0, inverse_indices, splats_i["quats"][:, d] * weights)
            quat_norms = torch.norm(merged["quats"], dim=1, keepdim=True)
            merged["quats"] = merged["quats"] / torch.clamp(quat_norms, min=1e-8)

            merged_splats_list.append(merged)

        # Reorganize output
        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats"]:
            output[key] = [merged[key] for merged in merged_splats_list]

        return output

    def prepare_splats(self, views, predictions, images, gs_params, context_nums, context_predictions={}, position_from="gsdepth+gtcamera"):
        """
        Prepare Gaussian splats from model predictions and input data.

        Args:
            views: Dictionary containing view data (camera poses, intrinsics, etc.)
            predictions: Model predictions including depth, pose_enc, etc.
            images: Input images [B, S_all, 3, H, W]
            gs_params: Gaussian splatting parameters from model
            context_predictions: Optional context predictions for camera poses
            position_from: Method to compute 3D positions ("pts3d", "gsdepth+gtcamera", "gsdepth+predcamera",
            "depth_head+gtcamera", "depth_head+predcamera")
            debug: Whether to use debug mode with ground truth data

        Returns:
            splats: Dictionary containing prepared Gaussian splat parameters
        """
        B, _, _, H, W = images.shape
        S = context_nums
        splats = {}

        # Only take parameters from source view branch
        gs_params = rearrange(gs_params, "(b s) c h w -> b s h w c", b=B)
        splats["gs_feats"] = gs_params.reshape(B, S * H * W, -1)

        # Split Gaussian parameters
        quats, scales, opacities, residual_sh, weights = torch.split(gs_params, [4, 3, 1, self.nums_sh * 3, 1], dim=-1)

        # Apply activation functions to Gaussian parameters
        splats["quats"] = act_gs.reg_dense_rotation(quats.reshape(B, S * H * W, 4))
        splats["scales"] = act_gs.reg_dense_scales(scales.reshape(B, S * H * W, 3)).clamp_max(0.3)
        splats["opacities"] = act_gs.reg_dense_opacities(opacities.reshape(B, S * H * W))
        residual_sh = act_gs.reg_dense_sh(residual_sh.reshape(B, S * H * W, self.nums_sh * 3))

        # Handle spherical harmonics (SH) coefficients
        new_sh = torch.zeros_like(residual_sh)
        new_sh[..., 0, :] = sh_utils.RGB2SH(images[:, :S].permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3))
        splats["sh"] = new_sh + residual_sh
        splats["residual_sh"] = residual_sh

        splats["weights"] = act_gs.reg_dense_weights(weights.reshape(B, S * H * W))

        # Compute 3D positions based on specified method
        if position_from == "pts3d":
            pts3d = predictions["pts3d"][:, :S].reshape(B, S * H * W, 3)
            splats["means"] = pts3d
        elif position_from == "gsdepth+gtcamera":
            depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
            pose4x4 = views["camera_poses"][:, :S].reshape(B * S, 4, 4)
            intrinsic = views["camera_intrs"][:, :S].reshape(B * S, 3, 3)
            pts3d, _, _ = depth_to_world_coords_points(depth, pose4x4, intrinsic)
            pts3d = pts3d.reshape(B, S * H * W, 3)
            splats["means"] = pts3d

        elif position_from == "gsdepth+predcamera":
            depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
            pose4x4 = context_predictions.get("camera_poses", predictions["camera_poses"])[:, :S].reshape(B * S, 4, 4)
            intrinsic = context_predictions.get("camera_intrs", predictions["camera_intrs"])[:, :S].reshape(B * S, 3, 3)
            pts3d, _, _ = depth_to_world_coords_points(depth, pose4x4.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S * H * W, 3)
            splats["means"] = pts3d
        else:
            raise ValueError(f"Invalid position_from={position_from}")

        return splats

    def prepare_cameras(self, views, nums):
        viewmats = views["camera_poses"][:, :nums]
        Ks = views["camera_intrs"][:, :nums]
        return viewmats, Ks
