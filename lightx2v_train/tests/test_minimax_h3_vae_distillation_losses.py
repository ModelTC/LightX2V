import unittest

import torch

from lightx2v_train.model_capabilities import VAEDistillationStepContext
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
    _SampleGroup,
)
from lightx2v_train.model_zoo.native.minimax_h3.vae_protocol import MiniMaxH3VideoVAE
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    CLIP_LENGTH,
    FRAME_OVERLAP,
    LATENTS_PER_CHUNK,
    SPATIAL_COMPRESSION_RATIO,
    spatial_tile_layout,
)


class _Student:
    tile_sample_min_height = 32
    tile_sample_min_width = 32
    tile_sample_min_overlap_height = 16
    tile_sample_min_overlap_width = 16

    blend = staticmethod(MiniMaxH3VideoVAE.blend)

    def stitch_tiles(self, rows, height_overlaps, width_overlaps):
        return MiniMaxH3VideoVAE.stitch_tiles(self, rows, height_overlaps, width_overlaps)


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.student = _Student()

    def denoiser_module(self):
        return self.student


def _capability(config=None):
    model = _Model()
    capability = MiniMaxH3VAEDistillationCapability(model, config or {})
    capability._test_model_owner = model
    return capability


def _spatial_external_trims(layout, group):
    top = layout.height_overlaps[group.row_start - 1] if group.row_start else 0
    bottom_index = group.row_start + group.row_count - 1
    bottom = layout.height_overlaps[bottom_index] if bottom_index < len(layout.height_overlaps) else 0
    left = layout.width_overlaps[group.column_start - 1] if group.column_start else 0
    right_index = group.column_start + group.column_count - 1
    right = layout.width_overlaps[right_index] if right_index < len(layout.width_overlaps) else 0
    return top, bottom, left, right


def _trim_spatial(tensor, top, bottom, left, right):
    height_end = tensor.shape[-2] - bottom if bottom else tensor.shape[-2]
    width_end = tensor.shape[-1] - right if right else tensor.shape[-1]
    return tensor[..., top:height_end, left:width_end]


def _stitch_temporal(windows):
    pieces = [windows[0][:, :, :CLIP_LENGTH]]
    previous_tail = windows[0][:, :, CLIP_LENGTH:]
    for window in windows[1:]:
        pieces.append(MiniMaxH3VideoVAE.blend(previous_tail, window[:, :, :CLIP_LENGTH], FRAME_OVERLAP, -3))
        previous_tail = window[:, :, CLIP_LENGTH:]
    pieces.append(previous_tail)
    return torch.cat(pieces, dim=2)


class StageAndRouteTest(unittest.TestCase):
    def test_stage_boundaries(self):
        capability = _capability(
            {
                "stages": [
                    {"name": "bootstrap", "start_iter": 0, "routes": {"single": 1.0}},
                    {"name": "stitch", "start_iter": 10, "routes": {"horizontal_pair": 1.0}},
                    {"name": "detail", "start_iter": 20, "routes": {"quad": 1.0}},
                ]
            }
        )

        expected = {
            0: (0, "bootstrap"),
            9: (0, "bootstrap"),
            10: (1, "stitch"),
            19: (1, "stitch"),
            20: (2, "detail"),
            10_000: (2, "detail"),
        }
        for iteration, result in expected.items():
            with self.subTest(iteration=iteration):
                index, stage = capability._stage_for(iteration)
                self.assertEqual((index, stage.name), result)

    def test_route_sequence_is_deterministic_for_iteration_and_micro_step(self):
        config = {
            "route_seed": 1234,
            "stages": [
                {
                    "name": "mixed",
                    "start_iter": 0,
                    "routes": {
                        "single": 1.0,
                        "horizontal_pair": 1.0,
                        "vertical_pair": 1.0,
                        "quad": 1.0,
                        "temporal_pair": 1.0,
                    },
                }
            ],
        }
        layout = spatial_tile_layout(
            4,
            5,
            tile_height=32,
            tile_width=32,
            overlap_height=16,
            overlap_width=16,
        )
        capabilities = (_capability(config), _capability(config))

        sequences = []
        for capability in capabilities:
            stage = capability.stages[0]
            sequences.append(
                [
                    capability._sample_route(
                        stage,
                        VAEDistillationStepContext(
                            running_dtype=torch.float32,
                            iteration=iteration,
                            micro_step=micro_step,
                        ),
                        layout,
                        num_temporal_windows=4,
                    )
                    for iteration in range(4)
                    for micro_step in range(8)
                ]
            )

        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(set(sequences[0]), {"single", "horizontal_pair", "vertical_pair", "quad", "temporal_pair"})

    def test_temporal_quad_selects_four_consecutive_windows(self):
        capability = _capability(
            {
                "stages": [
                    {
                        "name": "long_temporal",
                        "start_iter": 0,
                        "routes": {"temporal_quad": 1.0},
                    }
                ]
            }
        )
        layout = spatial_tile_layout(
            4,
            5,
            tile_height=32,
            tile_width=32,
            overlap_height=16,
            overlap_width=16,
        )
        stage = capability.stages[0]
        context = VAEDistillationStepContext(
            running_dtype=torch.float32,
            iteration=0,
            micro_step=0,
        )

        route = capability._sample_route(stage, context, layout, num_temporal_windows=7)
        group = capability._sample_group(route, layout, num_temporal_windows=7)

        self.assertEqual(route, "temporal_quad")
        self.assertEqual(group.temporal_count, 4)
        self.assertLessEqual(group.temporal_start, 3)


class StitchedGeometryTest(unittest.TestCase):
    def setUp(self):
        self.capability = _capability()
        self.student = self.capability.model.denoiser_module()

    def test_spatial_local_routes_match_full_stitch_after_external_trim(self):
        layout = spatial_tile_layout(
            4,
            5,
            tile_height=32,
            tile_width=32,
            overlap_height=16,
            overlap_width=16,
        )
        generator = torch.Generator().manual_seed(7)
        rows = [
            [torch.randn(1, 2, 3, 32, 32, generator=generator) for _ in layout.width_slices]
            for _ in layout.height_slices
        ]
        full = self.student.stitch_tiles(rows, layout.height_overlaps, layout.width_overlaps)
        route_shapes = {
            "single": (1, 1),
            "horizontal_pair": (1, 2),
            "vertical_pair": (2, 1),
            "quad": (2, 2),
        }

        for route, (row_count, column_count) in route_shapes.items():
            for row_start in range(len(rows) - row_count + 1):
                for column_start in range(len(rows[0]) - column_count + 1):
                    with self.subTest(route=route, row=row_start, column=column_start):
                        group = _SampleGroup(
                            route=route,
                            temporal_start=0,
                            temporal_count=1,
                            row_start=row_start,
                            row_count=row_count,
                            column_start=column_start,
                            column_count=column_count,
                        )
                        local_tiles = torch.cat(
                            [
                                rows[row][column]
                                for row in range(row_start, row_start + row_count)
                                for column in range(column_start, column_start + column_count)
                            ],
                            dim=0,
                        )
                        local = self.capability._stitch_student_tiles(local_tiles, layout, group)
                        top, bottom, left, right = _spatial_external_trims(layout, group)
                        local = _trim_spatial(local, top, bottom, left, right)

                        first_height = layout.height_slices[row_start]
                        last_height = layout.height_slices[row_start + row_count - 1]
                        first_width = layout.width_slices[column_start]
                        last_width = layout.width_slices[column_start + column_count - 1]
                        y_start = first_height.start * SPATIAL_COMPRESSION_RATIO + top
                        y_end = last_height.stop * SPATIAL_COMPRESSION_RATIO - bottom
                        x_start = first_width.start * SPATIAL_COMPRESSION_RATIO + left
                        x_end = last_width.stop * SPATIAL_COMPRESSION_RATIO - right
                        torch.testing.assert_close(local, full[..., y_start:y_end, x_start:x_end])

    def test_temporal_single_and_pair_match_full_stitch_after_external_trim(self):
        layout = spatial_tile_layout(
            2,
            2,
            tile_height=32,
            tile_width=32,
            overlap_height=0,
            overlap_width=0,
        )
        generator = torch.Generator().manual_seed(11)
        windows = [
            torch.randn(1, 2, CLIP_LENGTH + LATENTS_PER_CHUNK, 8, 8, generator=generator)
            for _ in range(5)
        ]
        full = _stitch_temporal(windows)

        for temporal_count, route in ((1, "single"), (2, "temporal_pair")):
            for temporal_start in range(len(windows) - temporal_count + 1):
                with self.subTest(route=route, temporal_start=temporal_start):
                    group = _SampleGroup(
                        route=route,
                        temporal_start=temporal_start,
                        temporal_count=temporal_count,
                        row_start=0,
                        row_count=1,
                        column_start=0,
                        column_count=1,
                    )
                    local = self.capability._stitch_student_tiles(
                        torch.cat(windows[temporal_start : temporal_start + temporal_count], dim=0),
                        layout,
                        group,
                    )
                    front = FRAME_OVERLAP if temporal_start else 0
                    back = FRAME_OVERLAP if temporal_start + temporal_count < len(windows) else 0
                    local_end = local.shape[2] - back if back else local.shape[2]
                    local = local[:, :, front:local_end]
                    global_start = temporal_start * CLIP_LENGTH + front
                    global_end = (
                        (temporal_start + temporal_count) * CLIP_LENGTH
                        + LATENTS_PER_CHUNK
                        - back
                    )
                    torch.testing.assert_close(local, full[:, :, global_start:global_end])


class DistillationLossTest(unittest.TestCase):
    def setUp(self):
        self.capability = _capability({"frequency_num_frames": 2})
        self.target = torch.zeros(1, 3, 4, 8, 10)

    def _assert_zero_and_perturbed_backward(self, loss_fn, perturbation):
        equal_prediction = self.target.clone().requires_grad_(True)
        equal_loss = loss_fn(equal_prediction, self.target)
        self.assertEqual(equal_loss.item(), 0.0)

        prediction = perturbation(self.target.clone()).requires_grad_(True)
        loss = loss_fn(prediction, self.target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_charbonnier_loss(self):
        self._assert_zero_and_perturbed_backward(
            lambda prediction, target: self.capability._charbonnier(
                prediction,
                target,
                self.capability.charbonnier_epsilon,
            ),
            lambda value: value + 0.25,
        )

    def test_spatial_gradient_loss(self):
        checkerboard = (torch.arange(8)[:, None] + torch.arange(10)[None, :]).remainder(2).float()
        self._assert_zero_and_perturbed_backward(
            self.capability._spatial_gradient_loss,
            lambda value: value + checkerboard.view(1, 1, 1, 8, 10),
        )

    def test_frequency_loss(self):
        checkerboard = (torch.arange(8)[:, None] + torch.arange(10)[None, :]).remainder(2).float()
        self._assert_zero_and_perturbed_backward(
            self.capability._frequency_loss,
            lambda value: value + checkerboard.view(1, 1, 1, 8, 10),
        )

    def test_laplacian_loss(self):
        checkerboard = (torch.arange(8)[:, None] + torch.arange(10)[None, :]).remainder(2).float()
        capability = _capability(
            {
                "laplacian_num_frames": 4,
                "laplacian_frame_batch_size": 2,
                "laplacian_levels": 2,
            }
        )
        self._assert_zero_and_perturbed_backward(
            capability._laplacian_loss,
            lambda value: value + checkerboard.view(1, 1, 1, 8, 10),
        )

    def test_perceptual_loss_processes_frames_in_bounded_batches(self):
        class RecordingPerceptual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, prediction, target):
                self.batch_sizes.append(prediction.shape[0])
                return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)

        capability = _capability(
            {
                "perceptual_num_frames": 5,
                "perceptual_frame_batch_size": 2,
            }
        )
        perceptual = RecordingPerceptual()
        capability._perceptual_model = perceptual
        prediction = torch.ones(1, 3, 5, 8, 10, requires_grad=True)
        target = torch.zeros_like(prediction)

        loss = capability._perceptual_loss(prediction, target)
        loss.backward()

        self.assertEqual(perceptual.batch_sizes, [2, 2, 1])
        self.assertEqual(loss.item(), 4.0)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_seam_loss(self):
        seam_mask = torch.zeros(1, 1, 4, 8, 10, dtype=torch.bool)
        seam_mask[..., 3:5, :] = True

        def seam_loss(prediction, target):
            return self.capability._seam_loss(prediction, target, seam_mask)

        def perturb_seam(value):
            value[..., 3:5, :] = 0.5
            return value

        self._assert_zero_and_perturbed_backward(seam_loss, perturb_seam)


if __name__ == "__main__":
    unittest.main()
