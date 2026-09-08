"""Pruned encoder recovery through the frozen, original H3 decoder."""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_capabilities import LossResult
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    CLIP_LENGTH,
    LATENTS_PER_CHUNK,
    SPATIAL_COMPRESSION_RATIO,
    video_latent_num_frames,
)

from .minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
    _LOSS_NAMES,
    _ROUTES,
)


def posterior_alignment(student_moments, teacher_moments, latent_std):
    """Expected normalized-latent MSE under the same posterior sampling noise."""
    student_mean, student_logvar = student_moments.float().chunk(2, dim=1)
    teacher_mean, teacher_logvar = teacher_moments.detach().float().chunk(2, dim=1)
    student_std = (0.5 * student_logvar.clamp(-30.0, 20.0)).exp()
    teacher_std = (0.5 * teacher_logvar.clamp(-30.0, 20.0)).exp()
    mean_loss = ((student_mean - teacher_mean) / latent_std).square().mean()
    std_loss = ((student_std - teacher_std) / latent_std).square().mean()
    return mean_loss + std_loss, mean_loss, std_loss


def group_latent_region(layout, windows, group):
    return (
        slice(windows[group.temporal_start][0], windows[group.temporal_start + group.temporal_count - 1][1]),
        slice(layout.height_slices[group.row_start].start, layout.height_slices[group.row_start + group.row_count - 1].stop),
        slice(layout.width_slices[group.column_start].start, layout.width_slices[group.column_start + group.column_count - 1].stop),
    )


def intersecting_tiles(slices, region):
    return {index for index, tile in enumerate(slices) if tile.start < region.stop and tile.stop > region.start}


class MiniMaxH3EncoderDistillationCapability(MiniMaxH3VAEDistillationCapability):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.posterior_weight = float(config.get("posterior_weight", 1.0))
        if self.posterior_weight < 0:
            raise ValueError("posterior_weight must be nonnegative.")
        if self.teacher_feature_indices != tuple(range(6)):
            raise ValueError("Encoder feature anchors must be the six original stage boundaries [0,1,2,3,4,5].")
        if not self.use_spatial_tiling:
            raise ValueError("Encoder distillation uses the released H3 encoder's spatial tiling geometry.")

    def _encode_tile(self, pixels, running_dtype, need_features, auxiliary_index):
        # Checkpoint the whole tile calculation, including the feature MSE. Returning
        # full early-stage features would retain high-resolution Conv3d activations.
        student_result = self.model.student_encode_clip(
            pixels, running_dtype, return_features=need_features, auxiliary_index=auxiliary_index,
        )
        if auxiliary_index is not None:
            student, student_features, anchor = student_result
            auxiliary = self.model.teacher_encoder_suffix(anchor, auxiliary_index)
        elif need_features:
            student, student_features = student_result
            auxiliary = student.new_empty(0)
        else:
            student = student_result
            student_features = ()
            auxiliary = student.new_empty(0)
        teacher_result = self.model.teacher_encode_clip(pixels, return_features=need_features)
        feature = student.float().new_zeros(())
        if need_features:
            teacher, teacher_features = teacher_result
            feature = torch.stack([
                self._feature_loss(left, right)
                for left, right in zip(student_features, teacher_features, strict=True)
            ]).mean()
        else:
            teacher = teacher_result
        return student.float(), teacher.detach().float(), auxiliary.float(), feature

    def _encode_group(self, video, layout, windows, group, running_dtype, need_features, auxiliary_index):
        """Encode only dependencies of a decoder crop, with exact native blending.

        All intersecting encoder tiles (including neighbouring halo tiles) are
        computed. Zero placeholders outside the dependency region let us reuse
        H3's original stitch order without changing its corner-blending policy.
        """
        student = self.model.denoiser_module()
        region = group_latent_region(layout, windows, group)
        time_slice, height_slice, width_slice = region
        row_indices = intersecting_tiles(layout.height_slices, height_slice)
        column_indices = intersecting_tiles(layout.width_slices, width_slice)
        first_clip = time_slice.start // LATENTS_PER_CHUNK
        last_clip = math.ceil(time_slice.stop / LATENTS_PER_CHUNK)
        full_time = video_latent_num_frames(video.shape[2])
        height_overlaps = tuple(value // SPATIAL_COMPRESSION_RATIO for value in layout.height_overlaps)
        width_overlaps = tuple(value // SPATIAL_COMPRESSION_RATIO for value in layout.width_overlaps)
        streams = [[], [], []] if auxiliary_index is not None else [[], []]
        feature_losses = []
        latent_channels = len(self.model.teacher_vae.config.latents_mean)

        def encode(pixels):
            return self._encode_tile(pixels, running_dtype, need_features, auxiliary_index)

        for clip_index in range(first_clip, last_clip):
            grids = [[] for _ in streams]
            for row_index, height in enumerate(layout.height_slices):
                rows = [[] for _ in streams]
                for column_index, width in enumerate(layout.width_slices):
                    if row_index in row_indices and column_index in column_indices:
                        pixels = video[
                            :, :, clip_index * CLIP_LENGTH : (clip_index + 1) * CLIP_LENGTH,
                            height.start * SPATIAL_COMPRESSION_RATIO : height.stop * SPATIAL_COMPRESSION_RATIO,
                            width.start * SPATIAL_COMPRESSION_RATIO : width.stop * SPATIAL_COMPRESSION_RATIO,
                        ].to(device=self.model.device, dtype=torch.float32)
                        if pixels.shape[2] < CLIP_LENGTH:
                            pixels = torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, CLIP_LENGTH - pixels.shape[2], -1, -1)), dim=2)
                        pixels = self.model.preprocess_video(pixels)
                        outputs = (
                            checkpoint(encode, pixels, use_reentrant=False)
                            if self.student_window_checkpointing and torch.is_grad_enabled()
                            else encode(pixels)
                        )
                        feature_losses.append(outputs[3])
                        for row, output in zip(rows, outputs[:len(rows)], strict=True):
                            row.append(output)
                    else:
                        shape = (1, 2 * latent_channels, LATENTS_PER_CHUNK, height.stop - height.start, width.stop - width.start)
                        empty = torch.zeros((), device=self.model.device).expand(shape)
                        for row in rows:
                            row.append(empty)
                for grid, row in zip(grids, rows, strict=True):
                    grid.append(row)
            for stream, grid in zip(streams, grids, strict=True):
                stream.append(student.stitch_tiles(grid, height_overlaps, width_overlaps))

        moments = []
        for stream in streams:
            value = torch.cat(stream, dim=2)
            offset = first_clip * LATENTS_PER_CHUNK
            value = value[:, :, :full_time - offset]
            # Only the final three tokens of the complete video are removed.
            value = F.pad(value, (0, 0, 0, 0, offset, full_time - offset - value.shape[2]))
            moments.append(value)
        return moments, torch.stack(feature_losses).mean(), region

    def _normalized_mode(self, moments):
        mean = moments.float().chunk(2, dim=1)[0]
        offset = mean.new_tensor(self.model.teacher_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        scale = mean.new_tensor(self.model.teacher_vae.config.latents_std).view(1, -1, 1, 1, 1)
        return (mean - offset) / scale

    def _decode_latent_tiles(self, latents, running_dtype, *, teacher=False):
        outputs = []
        # The decoder is frozen on both paths; only the student path records
        # autograd and checkpoints transformer blocks to transmit dL/dz_S.
        for window in latents.split(self.teacher_tile_batch_size, dim=0):
            if teacher:
                output = self.model.teacher_decode_window(window)
            else:
                output = self.model.decode_student_latent_window(window, running_dtype)
            outputs.append(output)
        return torch.cat(outputs)

    def compute_loss(self, batch, context):
        stage_index, stage = self._stage_for(context.iteration)
        weights = dict(stage.weights)
        auxiliary_ramp = self._auxiliary_ramp(context.iteration) if weights["auxiliary"] else 0.0
        weights["auxiliary"] *= auxiliary_ramp
        auxiliary_index = self._auxiliary_anchor(context) if weights["auxiliary"] else None
        video = batch["inputs"]["video"]
        if video.shape[0] != 1:
            raise ValueError("H3 encoder distillation uses one video per micro-batch; use gradient accumulation for larger batches.")
        latent_shape = (
            1, len(self.model.teacher_vae.config.latents_mean), video_latent_num_frames(video.shape[2]),
            video.shape[-2] // SPATIAL_COMPRESSION_RATIO, video.shape[-1] // SPATIAL_COMPRESSION_RATIO,
        )
        geometry = torch.empty(latent_shape, device="meta")
        layout, windows, group = self._sample_training_group(geometry, stage, context)
        moments, feature, region = self._encode_group(
            video, layout, windows, group, context.running_dtype, bool(weights["feature"]), auxiliary_index,
        )
        student_moments, teacher_moments = moments[:2]
        std = teacher_moments.new_tensor(self.model.teacher_vae.config.latents_std).view(1, -1, 1, 1, 1)
        region_index = (slice(None), slice(None), *region)
        posterior, posterior_mean, posterior_std = posterior_alignment(
            student_moments[region_index], teacher_moments[region_index], std,
        )
        student_latents = self._normalized_mode(student_moments)
        teacher_latents = self._normalized_mode(teacher_moments).detach()
        student_tiles = self._latent_tiles(student_latents, layout, windows, group)
        raw = self._decode_latent_tiles(student_tiles, context.running_dtype)
        stitched = self._stitch_student_tiles(self._supervised_student_frames(raw), layout, group)
        prediction, target, condition, seam_mask, valid_slice = self._valid_stitched_crop(
            stitched, video.to(device=self.model.device, dtype=torch.float32), teacher_latents,
            self._source_num_frames(batch), layout, windows, group, return_slice=True,
        )
        losses, adversarial_metrics = self._pixel_losses(
            prediction, target, condition.detach(), seam_mask, weights, stage, context,
        )
        zero = prediction.new_zeros(())
        auxiliary, teacher_output = zero, zero
        auxiliary_metrics = {f"auxiliary_{name}": zero for name in self.auxiliary_weights}
        if auxiliary_index is not None or weights["teacher_output"]:
            teacher_tiles = self._latent_tiles(teacher_latents, layout, windows, group)
            teacher_raw = self._decode_latent_tiles(teacher_tiles, context.running_dtype, teacher=True)
            teacher_stitched = self._stitch_student_tiles(self._supervised_student_frames(teacher_raw), layout, group)
            if weights["teacher_output"]:
                teacher_output = F.l1_loss(prediction.float(), teacher_stitched[valid_slice].detach().float())
            if auxiliary_index is not None:
                auxiliary_latents = self._normalized_mode(moments[2])
                auxiliary_tiles = self._latent_tiles(auxiliary_latents, layout, windows, group)
                auxiliary_raw = self._decode_latent_tiles(auxiliary_tiles, context.running_dtype)
                auxiliary_stitched = self._stitch_student_tiles(self._supervised_student_frames(auxiliary_raw), layout, group)
                auxiliary, auxiliary_metrics = self._auxiliary_loss(
                    auxiliary_stitched[valid_slice], teacher_stitched[valid_slice],
                )
        losses.update(feature=feature, teacher_output=teacher_output, auxiliary=auxiliary)
        loss = sum(weights[name] * losses[name] for name in _LOSS_NAMES) + self.posterior_weight * posterior
        metrics = dict(losses)
        metrics.update({f"weighted_{name}": weights[name] * losses[name] for name in _LOSS_NAMES})
        metrics.update(adversarial_metrics)
        metrics.update(auxiliary_metrics)
        metrics.update(
            posterior=posterior, posterior_mean=posterior_mean, posterior_std=posterior_std,
            weighted_posterior=self.posterior_weight * posterior,
            stage_index=float(stage_index), auxiliary_ramp=auxiliary_ramp,
            auxiliary_encoder_stage=float(auxiliary_index + 1) if auxiliary_index is not None else 0.0,
            supervised_frames=float(prediction.shape[2]), supervised_height=float(prediction.shape[-2]),
            supervised_width=float(prediction.shape[-1]),
            **{f"route_{name}": float(group.route == name) for name in _ROUTES},
        )
        return LossResult(loss=loss, metrics=metrics)
