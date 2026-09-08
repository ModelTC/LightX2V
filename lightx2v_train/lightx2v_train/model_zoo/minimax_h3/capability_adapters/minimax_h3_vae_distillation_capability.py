"""MiniMax-H3 operations for decoder-only VAE distillation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_capabilities import (
    BoundCapability,
    LossResult,
    VAEDistillationCapability,
    VAEDistillationStepContext,
)
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    CLIP_LENGTH,
    FRAME_OVERLAP,
    FRAME_PRE_PADDING,
    LATENTS_PER_CHUNK,
    SPATIAL_COMPRESSION_RATIO,
    TEMPORAL_COMPRESSION_RATIO,
    TOKEN_OVERLAP,
    TOKENS_CHUNK_SIZE,
    spatial_tile_layout,
    temporal_decode_windows,
)


_LOSS_NAMES = (
    "reconstruction",
    "teacher_output",
    "temporal_velocity",
    "temporal_acceleration",
    "perceptual",
    "feature",
    "spatial_gradient",
    "frequency",
    "laplacian",
    "seam",
    "adversarial",
    "auxiliary",
)
_ROUTES = ("single", "horizontal_pair", "vertical_pair", "quad", "temporal_pair", "temporal_quad")


@dataclass(frozen=True, slots=True)
class _DistillationStage:
    name: str
    start_iter: int
    routes: dict[str, float]
    weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class _SampleGroup:
    route: str
    temporal_start: int
    temporal_count: int
    row_start: int
    row_count: int
    column_start: int
    column_count: int


class MiniMaxH3VAEDistillationCapability(BoundCapability, VAEDistillationCapability):
    def __init__(self, model, config):
        super().__init__(model)
        self.use_spatial_tiling = config.get("use_spatial_tiling", True)
        self.teacher_feature_indices = tuple(int(index) for index in config.get("teacher_feature_indices", (17, 35)))
        self.feature_loss_type = config.get("feature_loss_type", "mse")
        if self.feature_loss_type not in {"mse", "masked_mse"}:
            raise ValueError(f"Unknown VAE feature loss: {self.feature_loss_type}")
        self.teacher_tile_batch_size = int(config.get("teacher_tile_batch_size", 1))
        self.student_tile_batch_size = config.get("student_tile_batch_size")
        self.student_window_checkpointing = bool(config.get("student_window_checkpointing", False))
        self.charbonnier_epsilon = float(config.get("charbonnier_epsilon", 1e-3))
        self.perceptual_num_frames = int(config.get("perceptual_num_frames", 2))
        self.perceptual_frame_batch_size = int(
            config.get("perceptual_frame_batch_size", self.perceptual_num_frames)
        )
        self.perceptual_gradient_checkpointing = bool(
            config.get("perceptual_gradient_checkpointing", False)
        )
        self.frequency_num_frames = int(config.get("frequency_num_frames", 2))
        self.laplacian_num_frames = int(config.get("laplacian_num_frames", 2))
        self.laplacian_frame_batch_size = int(
            config.get("laplacian_frame_batch_size", self.laplacian_num_frames)
        )
        self.laplacian_levels = int(config.get("laplacian_levels", 3))
        self.perceptual_network = config.get("perceptual_network", "vgg")
        self.route_seed = int(config.get("route_seed", 42))
        self.stages = self._build_stages(config)
        self.uses_feature_loss = any(stage.weights["feature"] > 0 for stage in self.stages)
        auxiliary_starts = [stage.start_iter for stage in self.stages if stage.weights["auxiliary"] > 0]
        self.uses_auxiliary_loss = bool(auxiliary_starts)
        self.auxiliary_start_iter = min(auxiliary_starts) if auxiliary_starts else 0
        auxiliary = config.get("auxiliary_decoder", {})
        self.auxiliary_feature_indices = tuple(auxiliary.get("student_feature_indices", (8, 9, 10)))
        self.auxiliary_ramp_iters = int(auxiliary.get("ramp_iters", 500))
        self.auxiliary_gradient_checkpointing = auxiliary.get("gradient_checkpointing", True)
        self.auxiliary_weights = {
            "reconstruction": float(auxiliary.get("reconstruction_weight", 1.0)),
            "spatial_gradient": float(auxiliary.get("spatial_gradient_weight", 0.1)),
            "spatiotemporal": float(auxiliary.get("spatiotemporal_weight", 0.1)),
        }
        if self.uses_auxiliary_loss:
            if not self.auxiliary_feature_indices or len(set(self.auxiliary_feature_indices)) != len(self.auxiliary_feature_indices):
                raise ValueError("Auxiliary student feature indices must be nonempty and unique.")
            if min(self.auxiliary_feature_indices) < 0 or max(self.auxiliary_feature_indices) >= len(self.teacher_feature_indices):
                raise ValueError("Auxiliary student feature index has no corresponding teacher anchor.")
            if self.auxiliary_ramp_iters < 0 or min(self.auxiliary_weights.values()) < 0 or sum(self.auxiliary_weights.values()) <= 0:
                raise ValueError("Auxiliary weights must be nonnegative with positive total, and ramp_iters must be nonnegative.")
        if self.uses_feature_loss and (
            not self.teacher_feature_indices
            or min(self.teacher_feature_indices) < 0
            or tuple(sorted(set(self.teacher_feature_indices))) != self.teacher_feature_indices
        ):
            raise ValueError("Teacher feature block indices must be nonempty, unique, nonnegative and ordered.")
        self._perceptual_model = None
        self._last_stage_index = None
        if self.perceptual_num_frames < 1 or self.perceptual_frame_batch_size < 1:
            raise ValueError("Perceptual frame counts must be positive.")
        if self.student_tile_batch_size is not None and self.student_tile_batch_size < 1:
            raise ValueError("Student tile batch size must be positive.")
        if self.laplacian_num_frames < 1 or self.laplacian_frame_batch_size < 1 or self.laplacian_levels < 1:
            raise ValueError("Laplacian frame counts and levels must be positive.")

    @staticmethod
    def _build_stages(config):
        base_weights = {
            "reconstruction": float(config.get("reconstruction_weight", 1.0)),
            "teacher_output": float(config.get("teacher_output_weight", 1.0)),
            "temporal_velocity": float(config.get("temporal_velocity_weight", config.get("temporal_weight", 0.1))),
            "temporal_acceleration": float(config.get("temporal_acceleration_weight", 0.0)),
            "perceptual": float(config.get("perceptual_weight", 0.0)),
            "feature": float(config.get("feature_weight", 0.0)),
            "spatial_gradient": float(config.get("spatial_gradient_weight", 0.0)),
            "frequency": float(config.get("frequency_weight", 0.0)),
            "laplacian": float(config.get("laplacian_weight", 0.0)),
            "seam": float(config.get("seam_weight", 0.0)),
            "adversarial": float(config.get("adversarial_weight", 0.0)),
            "auxiliary": float(config.get("auxiliary_weight", 0.0)),
        }
        stage_configs = config.get("stages")
        if not stage_configs:
            stage_configs = [{"name": "distillation", "start_iter": 0, "routes": {"single": 1.0}}]

        stages = []
        for stage_config in stage_configs:
            unknown_weights = set(stage_config.get("weights", {})) - set(_LOSS_NAMES)
            unknown_routes = set(stage_config.get("routes", {})) - set(_ROUTES)
            if unknown_weights:
                raise ValueError(f"Unknown VAE distillation loss weights: {sorted(unknown_weights)}")
            if unknown_routes:
                raise ValueError(f"Unknown VAE distillation routes: {sorted(unknown_routes)}")
            weights = dict(base_weights)
            weights.update({name: float(value) for name, value in stage_config.get("weights", {}).items()})
            routes = {name: float(value) for name, value in stage_config.get("routes", {"single": 1.0}).items()}
            if sum(routes.values()) <= 0:
                raise ValueError(f"VAE distillation stage {stage_config['name']!r} has no positive route weight.")
            if sum(weights.values()) <= 0:
                raise ValueError(f"VAE distillation stage {stage_config['name']!r} has no positive loss weight.")
            stages.append(
                _DistillationStage(
                    name=stage_config["name"],
                    start_iter=int(stage_config["start_iter"]),
                    routes=routes,
                    weights=weights,
                )
            )
        starts = [stage.start_iter for stage in stages]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("VAE distillation stages must start at 0 with strictly increasing start_iter values.")
        return tuple(stages)

    def _stage_for(self, iteration):
        stage_index = max(index for index, stage in enumerate(self.stages) if stage.start_iter <= iteration)
        if stage_index != self._last_stage_index:
            stage = self.stages[stage_index]
            logger.info(
                "[train] VAE distillation stage={} index={} start_iter={} routes={} weights={}",
                stage.name,
                stage_index,
                stage.start_iter,
                stage.routes,
                stage.weights,
            )
            self._last_stage_index = stage_index
        return stage_index, self.stages[stage_index]

    def _perceptual_loss(self, prediction, target):
        if self._perceptual_model is None:
            try:
                import lpips
            except ImportError as error:
                raise ImportError("Perceptual VAE distillation requires the lpips package.") from error
            self._perceptual_model = (
                lpips.LPIPS(net=self.perceptual_network)
                .requires_grad_(False)
                .eval()
                .to(self.model.device)
            )

        indices = self._sample_frame_indices(prediction.shape[2], self.perceptual_num_frames, prediction.device)
        prediction = prediction.index_select(2, indices)
        target = target.index_select(2, indices)
        prediction = prediction.permute(0, 2, 1, 3, 4).flatten(0, 1) * 2.0 - 1.0
        target = target.permute(0, 2, 1, 3, 4).flatten(0, 1) * 2.0 - 1.0
        total = prediction.new_zeros((), dtype=torch.float32)
        count = 0
        for start in range(0, prediction.shape[0], self.perceptual_frame_batch_size):
            end = start + self.perceptual_frame_batch_size
            inputs = (prediction[start:end], target[start:end])
            if self.perceptual_gradient_checkpointing and torch.is_grad_enabled():
                values = checkpoint(self._perceptual_model, *inputs, use_reentrant=False).float()
            else:
                values = self._perceptual_model(*inputs).float()
            total = total + values.sum()
            count += values.numel()
        return total / count

    @staticmethod
    def _sample_frame_indices(num_frames, count, device):
        count = min(num_frames, count)
        return torch.randperm(num_frames, device=device)[:count].sort().values

    @staticmethod
    def _sample_index(length):
        return int(torch.randint(length, ()).item())

    @staticmethod
    def _charbonnier(prediction, target, epsilon):
        return (torch.sqrt((prediction.float() - target.float()).square() + epsilon**2) - epsilon).mean()

    def _sample_route(self, stage, context, layout, num_temporal_windows):
        available = {
            "single": True,
            "horizontal_pair": self.use_spatial_tiling and len(layout.width_slices) >= 2,
            "vertical_pair": self.use_spatial_tiling and len(layout.height_slices) >= 2,
            "quad": self.use_spatial_tiling and len(layout.height_slices) >= 2 and len(layout.width_slices) >= 2,
            "temporal_pair": num_temporal_windows >= 2,
            "temporal_quad": num_temporal_windows >= 4,
        }
        names = [name for name in _ROUTES if available[name] and stage.routes.get(name, 0.0) > 0]
        if not names:
            raise ValueError(f"VAE distillation stage {stage.name!r} has no route compatible with this sample geometry.")
        weights = torch.tensor([stage.routes[name] for name in names], dtype=torch.float64)
        generator = torch.Generator().manual_seed(
            self.route_seed + context.iteration * 1_000_003 + context.micro_step
        )
        return names[int(torch.multinomial(weights, 1, generator=generator).item())]

    def _sample_group(self, route, layout, num_temporal_windows):
        temporal_count = {"temporal_pair": 2, "temporal_quad": 4}.get(route, 1)
        row_count = 2 if route in {"vertical_pair", "quad"} else 1
        column_count = 2 if route in {"horizontal_pair", "quad"} else 1
        return _SampleGroup(
            route=route,
            temporal_start=self._sample_index(num_temporal_windows - temporal_count + 1),
            temporal_count=temporal_count,
            row_start=self._sample_index(len(layout.height_slices) - row_count + 1),
            row_count=row_count,
            column_start=self._sample_index(len(layout.width_slices) - column_count + 1),
            column_count=column_count,
        )

    def _sample_training_group(self, latents, stage, context):
        student = self.model.denoiser_module()
        if self.use_spatial_tiling:
            layout = spatial_tile_layout(
                latents.shape[-2],
                latents.shape[-1],
                tile_height=student.tile_sample_min_height,
                tile_width=student.tile_sample_min_width,
                overlap_height=student.tile_sample_min_overlap_height,
                overlap_width=student.tile_sample_min_overlap_width,
            )
        else:
            layout = spatial_tile_layout(
                latents.shape[-2],
                latents.shape[-1],
                tile_height=latents.shape[-2] * SPATIAL_COMPRESSION_RATIO,
                tile_width=latents.shape[-1] * SPATIAL_COMPRESSION_RATIO,
                overlap_height=0,
                overlap_width=0,
            )
        temporal_windows = temporal_decode_windows(latents.shape[2])
        route = self._sample_route(stage, context, layout, len(temporal_windows))
        return layout, temporal_windows, self._sample_group(route, layout, len(temporal_windows))

    @staticmethod
    def _supervised_student_frames(raw_video):
        chunk_frames = TOKENS_CHUNK_SIZE * TEMPORAL_COMPRESSION_RATIO
        main = raw_video[:, :, FRAME_PRE_PADDING:chunk_frames]
        overlap_start = chunk_frames + FRAME_PRE_PADDING
        overlap_end = (TOKENS_CHUNK_SIZE + TOKEN_OVERLAP) * TEMPORAL_COMPRESSION_RATIO
        overlap = raw_video[:, :, overlap_start:overlap_end]
        return torch.cat((main, overlap), dim=2)

    def _latent_tiles(self, latents, layout, temporal_windows, group):
        tiles = []
        for temporal_index in range(group.temporal_start, group.temporal_start + group.temporal_count):
            frame_start, frame_end = temporal_windows[temporal_index]
            for row_index in range(group.row_start, group.row_start + group.row_count):
                height_slice = layout.height_slices[row_index]
                for column_index in range(group.column_start, group.column_start + group.column_count):
                    width_slice = layout.width_slices[column_index]
                    tiles.append(
                        latents[:, :, frame_start:frame_end, height_slice, width_slice].contiguous()
                    )
        return torch.cat(tiles, dim=0)

    def _stitch_student_tiles(self, supervised_tiles, layout, group):
        student = self.model.denoiser_module()
        height_overlaps = layout.height_overlaps[
            group.row_start : group.row_start + group.row_count - 1
        ]
        width_overlaps = layout.width_overlaps[
            group.column_start : group.column_start + group.column_count - 1
        ]
        spatial_windows = []
        tile_index = 0
        for _ in range(group.temporal_count):
            rows = []
            for _ in range(group.row_count):
                row = []
                for _ in range(group.column_count):
                    row.append(supervised_tiles[tile_index : tile_index + 1])
                    tile_index += 1
                rows.append(row)
            spatial_windows.append(student.stitch_tiles(rows, height_overlaps, width_overlaps))

        if len(spatial_windows) == 1:
            return spatial_windows[0]
        pieces = [spatial_windows[0][:, :, :CLIP_LENGTH]]
        previous_tail = spatial_windows[0][:, :, CLIP_LENGTH:]
        for window in spatial_windows[1:]:
            main = window[:, :, :CLIP_LENGTH]
            pieces.append(student.blend(previous_tail, main, FRAME_OVERLAP, -3))
            previous_tail = window[:, :, CLIP_LENGTH:]
        pieces.append(previous_tail)
        return torch.cat(pieces, dim=2)

    def _frame_aligned_latent_condition(self, latents, temporal_windows, group):
        windows = []
        for start, end in temporal_windows[group.temporal_start : group.temporal_start + group.temporal_count]:
            # H3 predicts four raw frames per token, then drops padding in each five-token chunk.
            raw_condition = latents[:, :, start:end].repeat_interleave(TEMPORAL_COMPRESSION_RATIO, dim=2)
            windows.append(self._supervised_student_frames(raw_condition))

        pieces = [windows[0][:, :, :CLIP_LENGTH]]
        previous_tail = windows[0][:, :, CLIP_LENGTH:]
        student = self.model.denoiser_module()
        for window in windows[1:]:
            pieces.append(student.blend(previous_tail, window[:, :, :CLIP_LENGTH], FRAME_OVERLAP, -3))
            previous_tail = window[:, :, CLIP_LENGTH:]
        pieces.append(previous_tail)
        return torch.cat(pieces, dim=2)

    def _valid_stitched_crop(self, stitched, video, latents, source_num_frames, layout, temporal_windows, group, *, return_slice=False):
        first_height = layout.height_slices[group.row_start]
        last_height = layout.height_slices[group.row_start + group.row_count - 1]
        first_width = layout.width_slices[group.column_start]
        last_width = layout.width_slices[group.column_start + group.column_count - 1]

        pixel_y = first_height.start * SPATIAL_COMPRESSION_RATIO
        pixel_end_y = last_height.stop * SPATIAL_COMPRESSION_RATIO
        pixel_x = first_width.start * SPATIAL_COMPRESSION_RATIO
        pixel_end_x = last_width.stop * SPATIAL_COMPRESSION_RATIO
        frame_start = group.temporal_start * CLIP_LENGTH
        frame_end = (group.temporal_start + group.temporal_count) * CLIP_LENGTH + LATENTS_PER_CHUNK
        frame_end = min(frame_end, source_num_frames)
        target = video[:, :, frame_start:frame_end, pixel_y:pixel_end_y, pixel_x:pixel_end_x]
        stitched = stitched[:, :, : target.shape[2]]

        front = FRAME_OVERLAP if group.temporal_start > 0 else 0
        back = FRAME_OVERLAP if group.temporal_start + group.temporal_count < len(temporal_windows) else 0
        top = layout.height_overlaps[group.row_start - 1] if group.row_start > 0 else 0
        bottom_index = group.row_start + group.row_count - 1
        bottom = layout.height_overlaps[bottom_index] if bottom_index < len(layout.height_overlaps) else 0
        left = layout.width_overlaps[group.column_start - 1] if group.column_start > 0 else 0
        right_index = group.column_start + group.column_count - 1
        right = layout.width_overlaps[right_index] if right_index < len(layout.width_overlaps) else 0

        time_end = stitched.shape[2] - back if back else stitched.shape[2]
        height_end = stitched.shape[-2] - bottom if bottom else stitched.shape[-2]
        width_end = stitched.shape[-1] - right if right else stitched.shape[-1]
        valid_slice = (
            slice(None),
            slice(None),
            slice(front, time_end),
            slice(top, height_end),
            slice(left, width_end),
        )
        prediction = stitched[valid_slice]
        target = target[valid_slice]
        if 0 in prediction.shape[2:]:
            raise RuntimeError(
                f"VAE distillation route {group.route!r} produced an empty supervised crop; "
                "adjust the tile size or overlap."
            )

        latent_condition = latents[
            :,
            :,
            :,
            first_height.start:last_height.stop,
            first_width.start:last_width.stop,
        ]
        latent_top = top // SPATIAL_COMPRESSION_RATIO
        latent_bottom = bottom // SPATIAL_COMPRESSION_RATIO
        latent_left = left // SPATIAL_COMPRESSION_RATIO
        latent_right = right // SPATIAL_COMPRESSION_RATIO
        latent_height_end = latent_condition.shape[-2] - latent_bottom if latent_bottom else latent_condition.shape[-2]
        latent_width_end = latent_condition.shape[-1] - latent_right if latent_right else latent_condition.shape[-1]
        latent_condition = latent_condition[
            ...,
            latent_top:latent_height_end,
            latent_left:latent_width_end,
        ].contiguous()
        latent_condition = self._frame_aligned_latent_condition(latent_condition, temporal_windows, group)
        latent_condition = latent_condition[:, :, front:time_end].contiguous()

        seam_mask = self._seam_mask(
            prediction,
            layout,
            group,
            valid_frame_start=frame_start + front,
            valid_pixel_y=pixel_y + top,
            valid_pixel_x=pixel_x + left,
        )
        result = (prediction, target, latent_condition, seam_mask)
        return (*result, valid_slice) if return_slice else result

    @staticmethod
    def _seam_mask(prediction, layout, group, *, valid_frame_start, valid_pixel_y, valid_pixel_x):
        mask = torch.zeros(
            (prediction.shape[0], 1, prediction.shape[2], prediction.shape[-2], prediction.shape[-1]),
            device=prediction.device,
            dtype=torch.bool,
        )
        for row_index in range(group.row_start + 1, group.row_start + group.row_count):
            seam_start = layout.height_slices[row_index].start * SPATIAL_COMPRESSION_RATIO - valid_pixel_y
            extent = layout.height_overlaps[row_index - 1]
            start = max(0, seam_start)
            end = min(mask.shape[-2], seam_start + extent)
            if start < end:
                mask[..., start:end, :] = True
        for column_index in range(group.column_start + 1, group.column_start + group.column_count):
            seam_start = layout.width_slices[column_index].start * SPATIAL_COMPRESSION_RATIO - valid_pixel_x
            extent = layout.width_overlaps[column_index - 1]
            start = max(0, seam_start)
            end = min(mask.shape[-1], seam_start + extent)
            if start < end:
                mask[..., start:end] = True
        for temporal_index in range(group.temporal_start + 1, group.temporal_start + group.temporal_count):
            seam_start = temporal_index * CLIP_LENGTH - valid_frame_start
            start = max(0, seam_start)
            end = min(mask.shape[2], seam_start + FRAME_OVERLAP)
            if start < end:
                mask[:, :, start:end] = True
        return mask

    def _teacher_losses(self, latent_tiles, student_raw, student_features, weights, *, return_raw=False):
        zero = student_raw.float().new_zeros(())
        if not (weights["teacher_output"] or weights["feature"] or return_raw):
            return zero, zero

        teacher_output = zero
        feature = zero
        raw_outputs = []
        total_tiles = latent_tiles.shape[0]
        for start in range(0, total_tiles, self.teacher_tile_batch_size):
            end = min(total_tiles, start + self.teacher_tile_batch_size)
            tile_count = end - start
            teacher_result = self.model.teacher_decode_window(
                latent_tiles[start:end],
                self.teacher_feature_indices if weights["feature"] else (),
            )
            if weights["feature"]:
                teacher_raw, teacher_features = teacher_result
                chunk_feature = torch.stack(
                    [
                        self._feature_loss(student[start:end], teacher)
                        for student, teacher in zip(student_features, teacher_features, strict=True)
                    ]
                ).mean()
                feature = feature + chunk_feature * tile_count
            else:
                teacher_raw = teacher_result
            if return_raw:
                raw_outputs.append(teacher_raw.detach())
            if weights["teacher_output"]:
                teacher_output = teacher_output + F.l1_loss(
                    self._supervised_student_frames(student_raw[start:end]).float(),
                    self._supervised_student_frames(teacher_raw).float(),
                ) * tile_count
        result = (teacher_output / total_tiles, feature / total_tiles)
        return (*result, torch.cat(raw_outputs)) if return_raw else result

    def _feature_loss(self, student, teacher):
        student, teacher = student.float(), teacher.detach().float()
        if student.shape != teacher.shape:
            raise ValueError(f"VAE feature shapes differ: student={student.shape}, teacher={teacher.shape}")
        if self.feature_loss_type == "masked_mse":
            # TinyFusion masks each side independently, then averages over all elements.
            dims = tuple(range(1, student.ndim))
            student_var, student_mean = torch.var_mean(student, dim=dims, keepdim=True)
            teacher_var, teacher_mean = torch.var_mean(teacher, dim=dims, keepdim=True)
            student = student * ((student - student_mean).abs() < 2 * student_var.sqrt())
            teacher = teacher * ((teacher - teacher_mean).abs() < 2 * teacher_var.sqrt())
        return F.mse_loss(student, teacher)

    def _spatial_gradient_loss(self, prediction, target):
        horizontal = self._charbonnier(
            prediction[..., 1:] - prediction[..., :-1],
            target[..., 1:] - target[..., :-1],
            self.charbonnier_epsilon,
        )
        vertical = self._charbonnier(
            prediction[..., 1:, :] - prediction[..., :-1, :],
            target[..., 1:, :] - target[..., :-1, :],
            self.charbonnier_epsilon,
        )
        return 0.5 * (horizontal + vertical)

    def _spatiotemporal_loss(self, prediction, target):
        # Compare changing edges, not adjacent frames against a static target.
        prediction = prediction.float()
        target = target.detach().float()
        if prediction.shape[2] < 2:
            return prediction.sum() * 0.0
        prediction_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
        target_delta = target[:, :, 1:] - target[:, :, :-1]
        losses = [
            self._charbonnier(prediction_delta.diff(dim=axis), target_delta.diff(dim=axis), self.charbonnier_epsilon)
            for axis in (-2, -1) if prediction.shape[axis] > 1
        ]
        return torch.stack(losses).mean() if losses else prediction.sum() * 0.0

    def _auxiliary_loss(self, prediction, target):
        prediction, target = prediction.float(), target.detach().float()
        losses = {
            "reconstruction": self._charbonnier(prediction, target, self.charbonnier_epsilon),
            "spatial_gradient": self._spatial_gradient_loss(prediction, target),
            "spatiotemporal": self._spatiotemporal_loss(prediction, target),
        }
        loss = sum(self.auxiliary_weights[name] * value for name, value in losses.items())
        return loss, {f"auxiliary_{name}": value for name, value in losses.items()}

    def _auxiliary_anchor(self, context):
        generator = torch.Generator().manual_seed(
            self.route_seed + 97_000_019 + context.iteration * 1_000_003 + context.micro_step
        )
        choice = int(torch.randint(len(self.auxiliary_feature_indices), (), generator=generator).item())
        return self.auxiliary_feature_indices[choice]

    def _auxiliary_ramp(self, iteration):
        if not self.uses_auxiliary_loss or iteration < self.auxiliary_start_iter:
            return 0.0
        if self.auxiliary_ramp_iters == 0:
            return 1.0
        return min(1.0, (iteration - self.auxiliary_start_iter + 1) / self.auxiliary_ramp_iters)

    def _decode_auxiliary_tiles(self, full_tokens, latent_shape, student_index):
        outputs = []
        for start in range(0, full_tokens.shape[0], self.teacher_tile_batch_size):
            outputs.append(self.model.teacher_decode_suffix(
                full_tokens[start : start + self.teacher_tile_batch_size],
                latent_shape,
                self.teacher_feature_indices[student_index],
                gradient_checkpointing=self.auxiliary_gradient_checkpointing,
            ))
        return torch.cat(outputs)

    def _frequency_loss(self, prediction, target):
        indices = self._sample_frame_indices(prediction.shape[2], self.frequency_num_frames, prediction.device)
        prediction = prediction.index_select(2, indices).float()
        target = target.index_select(2, indices).float()
        prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
        target = target - target.mean(dim=(-2, -1), keepdim=True)
        prediction_spectrum = torch.log1p(torch.fft.rfft2(prediction, norm="ortho").abs())
        target_spectrum = torch.log1p(torch.fft.rfft2(target, norm="ortho").abs())
        frequency_y = torch.fft.fftfreq(prediction.shape[-2], device=prediction.device).abs()
        frequency_x = torch.fft.rfftfreq(prediction.shape[-1], device=prediction.device).abs()
        radial = torch.sqrt(frequency_y[:, None].square() + frequency_x[None, :].square())
        radial = radial / radial.mean().clamp_min(1e-6)
        radial[0, 0] = 0.0
        return ((prediction_spectrum - target_spectrum).abs() * radial).mean()

    @staticmethod
    def _gaussian_blur(frames):
        coefficients = frames.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))
        kernel = coefficients[:, None] * coefficients[None, :]
        kernel = (kernel / kernel.sum()).view(1, 1, 5, 5).expand(frames.shape[1], 1, 5, 5)
        return F.conv2d(F.pad(frames, (2, 2, 2, 2), mode="reflect"), kernel, groups=frames.shape[1])

    def _laplacian_loss(self, prediction, target):
        indices = self._sample_frame_indices(prediction.shape[2], self.laplacian_num_frames, prediction.device)
        prediction = prediction.index_select(2, indices).permute(0, 2, 1, 3, 4).flatten(0, 1)
        target = target.index_select(2, indices).permute(0, 2, 1, 3, 4).flatten(0, 1)
        total = prediction.float().new_zeros(())
        normalizer = 0.0
        for start in range(0, prediction.shape[0], self.laplacian_frame_batch_size):
            stop = start + self.laplacian_frame_batch_size
            prediction_level = prediction[start:stop].float()
            target_level = target[start:stop].float()
            frame_count = prediction_level.shape[0]
            for level in range(self.laplacian_levels):
                prediction_blur = self._gaussian_blur(prediction_level)
                target_blur = self._gaussian_blur(target_level)
                level_weight = 0.5**level
                total = total + self._charbonnier(
                    prediction_level - prediction_blur,
                    target_level - target_blur,
                    self.charbonnier_epsilon,
                ) * frame_count * level_weight
                normalizer += frame_count * level_weight
                if level + 1 < self.laplacian_levels:
                    prediction_level = F.avg_pool2d(prediction_blur, 2)
                    target_level = F.avg_pool2d(target_blur, 2)
        return total / normalizer

    def _seam_loss(self, prediction, target, seam_mask):
        if not seam_mask.any():
            return prediction.float().new_zeros(())
        error = torch.sqrt((prediction.float() - target.float()).square() + self.charbonnier_epsilon**2)
        error = error - self.charbonnier_epsilon
        return (error * seam_mask).sum() / (seam_mask.sum() * prediction.shape[1])

    @staticmethod
    def _source_num_frames(batch):
        return int(batch["meta"]["source_num_frames"].item())

    def _decode_student_tiles(self, latent_tiles, running_dtype, *, auxiliary_feature_index=None):
        def decode(window):
            if auxiliary_feature_index is not None:
                return self.model.student_decode_window_with_aux(
                    window, running_dtype, auxiliary_feature_index=auxiliary_feature_index,
                    return_features=self.uses_feature_loss,
                )
            return self.model.student_decode_window(
                window, running_dtype, return_features=self.uses_feature_loss,
            )

        batch_size = self.student_tile_batch_size or latent_tiles.shape[0]
        results = []
        for start in range(0, latent_tiles.shape[0], batch_size):
            window = latent_tiles[start : start + batch_size]
            if self.student_window_checkpointing and torch.is_grad_enabled():
                result = checkpoint(decode, window, use_reentrant=False)
            else:
                result = decode(window)
            results.append(result)
        if auxiliary_feature_index is not None:
            raw, features, full_tokens = zip(*results, strict=True)
            return torch.cat(raw), tuple(torch.cat(values) for values in zip(*features, strict=True)), torch.cat(full_tokens)
        if self.uses_feature_loss:
            raw, features = zip(*results, strict=True)
            return torch.cat(raw), tuple(torch.cat(values) for values in zip(*features, strict=True))
        return torch.cat(results), ()

    def _pixel_losses(self, prediction_raw, target, latent_condition, seam_mask, weights, stage, context):
        """Shared RGB objectives for encoder and decoder recovery."""
        target_raw = self.model.preprocess_video(target)
        zero = prediction_raw.float().new_zeros(())
        reconstruction = self._charbonnier(prediction_raw, target_raw, self.charbonnier_epsilon)
        temporal_velocity = zero
        if weights["temporal_velocity"]:
            temporal_velocity = self._charbonnier(
                prediction_raw[:, :, 1:] - prediction_raw[:, :, :-1],
                target_raw[:, :, 1:] - target_raw[:, :, :-1],
                self.charbonnier_epsilon,
            )
        temporal_acceleration = zero
        if weights["temporal_acceleration"]:
            temporal_acceleration = self._charbonnier(
                prediction_raw[:, :, 2:] - 2 * prediction_raw[:, :, 1:-1] + prediction_raw[:, :, :-2],
                target_raw[:, :, 2:] - 2 * target_raw[:, :, 1:-1] + target_raw[:, :, :-2],
                self.charbonnier_epsilon,
            )
        spatial_gradient = zero
        if weights["spatial_gradient"]:
            spatial_gradient = self._spatial_gradient_loss(prediction_raw, target_raw)
        frequency = zero
        if weights["frequency"]:
            frequency = self._frequency_loss(prediction_raw, target_raw)
        laplacian = zero
        if weights["laplacian"]:
            laplacian = self._laplacian_loss(prediction_raw, target_raw)
        seam = zero
        if weights["seam"]:
            seam = self._seam_loss(prediction_raw, target_raw, seam_mask)

        prediction_rgb = None
        perceptual = zero
        if weights["perceptual"]:
            prediction_rgb = self.model.postprocess_raw_video(prediction_raw)
            perceptual = self._perceptual_loss(prediction_rgb, target)

        adversarial = zero
        adversarial_metrics = {
            "discriminator": zero,
            "discriminator_real": zero,
            "discriminator_fake": zero,
            "adversarial_feature_distance": zero,
            "adversarial_correction_rms": zero,
            "adversarial_ramp": 0.0,
        }
        if weights["adversarial"]:
            if context.adversarial_objective is None:
                raise RuntimeError("The active VAE distillation stage requires training.vae_distillation.gan.enabled=true.")
            if prediction_rgb is None:
                prediction_rgb = self.model.postprocess_raw_video(prediction_raw)
            adversarial, adversarial_metrics = context.adversarial_objective(
                prediction_rgb,
                target,
                latent_condition.detach(),
                context.iteration - stage.start_iter,
            )

        losses = {
            "reconstruction": reconstruction,
            "temporal_velocity": temporal_velocity,
            "temporal_acceleration": temporal_acceleration,
            "perceptual": perceptual,
            "spatial_gradient": spatial_gradient,
            "frequency": frequency,
            "laplacian": laplacian,
            "seam": seam,
            "adversarial": adversarial,
        }
        return losses, adversarial_metrics

    def compute_loss(self, batch, context: VAEDistillationStepContext) -> LossResult:
        stage_index, stage = self._stage_for(context.iteration)
        weights = dict(stage.weights)
        auxiliary_ramp = self._auxiliary_ramp(context.iteration) if weights["auxiliary"] else 0.0
        weights["auxiliary"] *= auxiliary_ramp
        auxiliary_index = self._auxiliary_anchor(context) if weights["auxiliary"] else None
        video = batch["inputs"]["video"]
        normalized_latents = batch["inputs"].get("latents")
        if normalized_latents is None:
            video = video.to(device=self.model.device, dtype=torch.float32)
            normalized_latents = self.model.encode_video(video)
        normalized_latents = normalized_latents.to(device=self.model.device, dtype=torch.float32)
        video = video.to(device=self.model.device, dtype=torch.float32)

        layout, temporal_windows, group = self._sample_training_group(normalized_latents, stage, context)
        latent_tiles = self._latent_tiles(normalized_latents, layout, temporal_windows, group)
        student_result = self._decode_student_tiles(
            latent_tiles, context.running_dtype, auxiliary_feature_index=auxiliary_index,
        )
        if auxiliary_index is None:
            student_raw, student_features = student_result
        else:
            student_raw, student_features, auxiliary_tokens = student_result
        supervised_tiles = self._supervised_student_frames(student_raw)
        stitched = self._stitch_student_tiles(supervised_tiles, layout, group)
        prediction_raw, target, latent_condition, seam_mask, valid_slice = self._valid_stitched_crop(
            stitched,
            video,
            normalized_latents,
            self._source_num_frames(batch),
            layout,
            temporal_windows,
            group,
            return_slice=True,
        )
        zero = prediction_raw.float().new_zeros(())
        teacher_result = self._teacher_losses(
            latent_tiles,
            student_raw,
            student_features,
            weights,
            return_raw=auxiliary_index is not None,
        )
        auxiliary = zero
        auxiliary_metrics = {f"auxiliary_{name}": zero for name in self.auxiliary_weights}
        if auxiliary_index is None:
            teacher_output, feature = teacher_result
        else:
            teacher_output, feature, teacher_raw = teacher_result
            auxiliary_raw = self._decode_auxiliary_tiles(auxiliary_tokens, latent_tiles.shape[2:], auxiliary_index)
            auxiliary_stitched = self._stitch_student_tiles(self._supervised_student_frames(auxiliary_raw), layout, group)
            teacher_stitched = self._stitch_student_tiles(self._supervised_student_frames(teacher_raw), layout, group)
            # Use the main branch's exact crop, including clip padding and incomplete overlap boundaries.
            auxiliary, auxiliary_metrics = self._auxiliary_loss(auxiliary_stitched[valid_slice], teacher_stitched[valid_slice])
        if self.uses_feature_loss and not weights["feature"]:
            feature = sum(value.float().sum() for value in student_features) * 0.0
        losses, adversarial_metrics = self._pixel_losses(
            prediction_raw, target, latent_condition, seam_mask, weights, stage, context,
        )
        losses.update(teacher_output=teacher_output, feature=feature, auxiliary=auxiliary)
        loss = sum(weights[name] * losses[name] for name in _LOSS_NAMES)
        metrics = dict(losses)
        metrics.update({f"weighted_{name}": weights[name] * losses[name] for name in _LOSS_NAMES})
        metrics.update(adversarial_metrics)
        if self.uses_auxiliary_loss:
            metrics.update(auxiliary_metrics)
            metrics.update(
                auxiliary_ramp=auxiliary_ramp,
                auxiliary_student_layer=float(auxiliary_index + 1) if auxiliary_index is not None else 0.0,
                auxiliary_teacher_layer=float(self.teacher_feature_indices[auxiliary_index] + 1) if auxiliary_index is not None else 0.0,
            )
        metrics.update(
            {
                "stage_index": float(stage_index),
                "supervised_frames": float(prediction_raw.shape[2]),
                "supervised_height": float(prediction_raw.shape[-2]),
                "supervised_width": float(prediction_raw.shape[-1]),
                **{f"route_{name}": float(group.route == name) for name in _ROUTES},
            }
        )
        return LossResult(loss=loss, metrics=metrics)
