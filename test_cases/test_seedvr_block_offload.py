import ast
import json
import os
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]


class SeedVRBlockOffloadTest(unittest.TestCase):
    def make_buffer(self):
        transformer_weights = import_module("lightx2v.models.networks.seedvr.weights.transformer_weights")
        SeedVRTransformerBlockWeights = transformer_weights.SeedVRTransformerBlockWeights
        buffer = SeedVRTransformerBlockWeights.__new__(SeedVRTransformerBlockWeights)
        buffer.config = {
            "num_layers": 32,
            "mm_layers": 10,
            "window": [(4, 3, 3)] * 32,
            "window_method": ["regular", "shifted"] * 16,
            "last_layer_vid_only": True,
        }
        buffer.create_cuda_buffer = True
        buffer.alias_shared_to_vid = True
        return buffer

    def test_shared_block_keys_and_metadata_are_mapped_to_vid_buffer(self):
        WeightModule = import_module("lightx2v.common.modules.weight_module").WeightModule
        buffer = self.make_buffer()
        tensor = object()
        source = {"blocks.10.attn.proj_qkv.all.weight": tensor}

        with patch.object(WeightModule, "load_state_dict", autospec=True, return_value="loaded") as parent_load:
            result = buffer.load_state_dict(source, 10)

        mapped = parent_load.call_args.args[1]
        self.assertEqual(result, "loaded")
        self.assertIs(mapped["blocks.10.attn.proj_qkv.vid.weight"], tensor)
        self.assertNotIn("blocks.10.attn.proj_qkv.all.weight", mapped)
        self.assertTrue(buffer.shared_weights)
        self.assertFalse(buffer.vid_only)
        self.assertEqual(buffer.window_method, "regular")

    def test_last_block_metadata_is_updated(self):
        WeightModule = import_module("lightx2v.common.modules.weight_module").WeightModule
        buffer = self.make_buffer()

        with patch.object(WeightModule, "load_state_dict", autospec=True, return_value=None):
            buffer.load_state_dict({}, 31)

        self.assertTrue(buffer.shared_weights)
        self.assertTrue(buffer.vid_only)
        self.assertEqual(buffer.window_method, "shifted")

    def test_pipeline_uses_runner_clip_offload_key(self):
        tree = ast.parse((REPO_ROOT / "lightx2v/pipeline.py").read_text())
        pipeline_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LightX2VPipeline")
        method = next(node for node in pipeline_class.body if isinstance(node, ast.FunctionDef) and node.name == "enable_offload")
        assigned_attributes = {target.attr for node in ast.walk(method) if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Attribute)}

        self.assertIn("clip_cpu_offload", assigned_attributes)
        self.assertNotIn("clip_encoder_offload", assigned_attributes)

    def test_seedvr_configs_enable_block_offload(self):
        paths = [
            REPO_ROOT / "configs/seedvr/seedvr2_3b.json",
            REPO_ROOT / "configs/seedvr/seedvr2_7b.json",
            REPO_ROOT / "configs/seedvr/4090/seedvr2_3b.json",
            REPO_ROOT / "configs/seedvr/4090/seedvr2_7b.json",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(json.loads(path.read_text())["offload_granularity"], "block")

    def test_seedvr_runner_profiles_major_stages(self):
        source = (REPO_ROOT / "lightx2v/models/runners/seedvr/seedvr_runner.py").read_text()
        stages = [
            '"RUN pipeline"',
            '"Run VAE Encoder"',
            '"Run DiT"',
            '"Run VAE Decoder"',
            'f"Segment {idx + 1}/{len(segments)}',
        ]
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertIn(stage, source)

        noisy_stages = ["SeedVR Read Input", "SeedVR VAE Decode Core", "SeedVR Color Fix", "SeedVR Save Segment"]
        for stage in noisy_stages:
            with self.subTest(noisy_stage=stage):
                self.assertNotIn(stage, source)

    def test_segment_fallback_honors_requested_time_range(self):
        source = (REPO_ROOT / "lightx2v/models/runners/seedvr/seedvr_runner.py").read_text()

        self.assertIn("frame_time < float(start_pts or 0)", source)
        self.assertIn("frame_time >= float(end_pts)", source)
        self.assertIn('self._sr_video_backend = "torchcodec"', source)

    def test_block_offload_waits_for_segment_inputs_and_resets_block_zero(self):
        offload_source = (REPO_ROOT / "lightx2v/models/networks/seedvr/infer/offload/transformer_infer.py").read_text()
        model_source = (REPO_ROOT / "lightx2v/models/networks/seedvr/model.py").read_text()

        self.assertIn("compute_stream.wait_stream(current_stream)", offload_source)
        self.assertIn("offload_manager.need_init_first_buffer = True", model_source)

    def test_seedvr_key_stages_enable_peak_memory_profiling(self):
        runner_source = (REPO_ROOT / "lightx2v/models/runners/seedvr/seedvr_runner.py").read_text()
        profiler_source = (REPO_ROOT / "lightx2v/utils/profiler.py").read_text()

        self.assertGreaterEqual(runner_source.count("profile_memory=True"), 4)
        self.assertIn("reset_peak_memory_stats", profiler_source)
        self.assertIn("peak_allocated", profiler_source)
        self.assertIn("peak_reserved", profiler_source)


if __name__ == "__main__":
    unittest.main()
