from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

import lightx2v.models.runners.wan.wan_animate2_runner as runner_module
import lightx2v.common.ops.attn.flex_attn as flex_attn_module
from lightx2v.common.ops.attn.flex_attn import _FlexMaskCache
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.infer.animate2.transformer_infer import WanAnimate2TransformerInfer
from lightx2v.models.runners.wan.wan_animate2_runner import _Animate2VideoRecorder, WanAnimate2Runner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


class WanAnimate2TailWindowTest(TestCase):
    def test_only_canonical_animate2_internal_name_skips_legacy_pose_weights(self):
        model_cls = "wan2.2_animate2_distilled"
        weights = WanPreWeights(
            {
                "in_dim": 36,
                "dim": 5120,
                "task": "animate",
                "model_cls": model_cls,
                "use_image_encoder": True,
                "layer_norm_type": "torch",
            }
        )
        self.assertNotIn("pose_patch_embedding", weights._modules)
        self.assertIn(model_cls, RUNNER_REGISTER)

    def test_odd_output_is_padded_for_yuv420p_browser_compatibility(self):
        recorder = _Animate2VideoRecorder.__new__(_Animate2VideoRecorder)
        recorder.width = 711
        recorder.height = 1264
        recorder.fps = 24
        recorder.video_port = 12345
        recorder.livestream_url = "/tmp/output.mp4"
        recorder.ffmpeg_log_level = "error"
        recorder.video_queue = None
        recorder.video_conn = None
        recorder.video_socket = None
        recorder.video_thread = None
        recorder.stoppable_t = None
        recorder.returncode = None

        with patch("subprocess.Popen", return_value=Mock(pid=1)) as popen:
            recorder.start_ffmpeg_process_local()

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-pix_fmt") + 1], "rgb24")
        self.assertEqual(command[command.index("-vf") + 1], "pad=ceil(iw/2)*2:ceil(ih/2)*2")
        output_pix_fmt_index = len(command) - 1 - command[::-1].index("-pix_fmt")
        self.assertEqual(command[output_pix_fmt_index + 1], "yuv420p")

    def test_tail_window_uses_its_actual_length_for_attention_geometry(self):
        with patch.object(runner_module, "AI_DEVICE", "cpu"), patch.object(runner_module, "GET_DTYPE", return_value=torch.float32):
            runner = WanAnimate2Runner.__new__(WanAnimate2Runner)
            runner.config = {
                "target_video_length": 81,
                "target_width": 1280,
                "target_height": 720,
            }
            runner.segment_plan = [(0, 81), (80, 109)]
            runner.reference_image = np.zeros((16, 16, 3), dtype=np.uint8)
            runner.driving_frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(109)]
            runner.previous_frame = torch.zeros(3, 1, 16, 16)
            runner.generation_reference_latents = torch.zeros(16, 1, 2, 2)
            runner.generation_clip = torch.zeros(1)
            runner.input_info = SimpleNamespace(latent_shape=None)
            runner.inputs = {}

            runner._vae_encode = lambda pixels: torch.zeros(16, 8, 2, 2)
            runner._i2v_mask = lambda latent_t, latent_h, latent_w, mask_len: torch.zeros(4, latent_t, latent_h, latent_w)
            runner.run_image_encoder = lambda image: torch.zeros(1)
            runner._build_reference_cache = lambda reference_latents: object()

            runner._build_segment_inputs(segment_idx=1)

            self.assertEqual(runner.inputs["animate2"]["clip_len"], 29)
            self.assertEqual(runner.inputs["animate2"]["origin_len"], 29)
            self.assertEqual(runner.inputs["animate2"]["reference_latents"].shape[1], 8)
            self.assertEqual(runner.inputs["animate2"]["generation_y"].shape[1], 9)
            self.assertEqual(runner.input_info.latent_shape, [16, 9, 2, 2])

    def test_tail_window_flex_layout_matches_generation_and_reference_grids(self):
        captured = {}

        def fake_create_block_mask(mask_mod, *, Q_LEN, KV_LEN, **kwargs):
            captured.update(mask_mod=mask_mod, q_len=Q_LEN, kv_len=KV_LEN, kwargs=kwargs)
            return object()

        with patch.object(flex_attn_module, "create_block_mask", side_effect=fake_create_block_mask):
            _, q_total, ref_total, frame_capacity = _FlexMaskCache().get(29, [1280, 720], "cpu")

        # 29 pixels frames encode to 8 reference latent frames. Generation
        # prepends one reference-image latent, so its grid has 9 frames.
        self.assertEqual(frame_capacity, 1280 * 720 // 256)
        self.assertEqual((q_total, ref_total), (32512, 28800))
        self.assertEqual((captured["q_len"], captured["kv_len"]), (32512, 61312))

        generation = torch.ones(9 * frame_capacity, 1, 1)
        reference = torch.ones(8 * frame_capacity, 1, 1)
        generation_packed = WanAnimate2TransformerInfer._pack_stream(
            generation, (9, 45, 80), frame_capacity, q_total
        )
        reference_packed = WanAnimate2TransformerInfer._pack_stream(
            reference, (8, 45, 80), frame_capacity, ref_total
        )

        self.assertEqual(tuple(generation_packed.shape), (q_total, 1, 1))
        self.assertEqual(tuple(reference_packed.shape), (ref_total, 1, 1))
        self.assertTrue(torch.all(generation_packed[: 9 * frame_capacity] == 1))
        self.assertTrue(torch.all(reference_packed == 1))
        self.assertEqual(int(generation_packed[9 * frame_capacity :].count_nonzero()), 0)

        # The previous full-window value would select a different, oversized
        # layout and must not be reused for this tail segment.
        with patch.object(flex_attn_module, "create_block_mask", side_effect=fake_create_block_mask):
            _, old_q_total, old_ref_total, _ = _FlexMaskCache().get(81, [1280, 720], "cpu")
        self.assertNotEqual((old_q_total, old_ref_total), (q_total, ref_total))
