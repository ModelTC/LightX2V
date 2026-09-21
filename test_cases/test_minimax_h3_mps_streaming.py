"""Run with PLATFORM=mps DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 python -m unittest test_cases.test_minimax_h3_mps_streaming."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

if os.environ.get("PLATFORM") != "mps" or not torch.backends.mps.is_available():
    raise unittest.SkipTest("Requires the MPS platform and an Apple GPU")

from lightx2v.common.offload.mps_manager import MpsSharedWeightAsyncStreamManager
from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3TransformerWeights
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner


class MiniMaxH3MPSStreamingTest(unittest.TestCase):
    def test_text_only_streaming_does_not_load_visual_components(self):
        encoder = MiniMaxH3Qwen3VLTextEncoder.__new__(MiniMaxH3Qwen3VLTextEncoder)
        encoder.disk_streaming = True
        encoder.load_tokenizer = Mock()
        encoder.load_text_encoder = Mock()
        encoder.load_processor = Mock(side_effect=AssertionError("Text streaming must not load the processor"))
        encoder.load_vision_encoder = Mock(side_effect=AssertionError("Text streaming must not load vision weights"))
        encoder.load()
        encoder.load_tokenizer.assert_called_once()
        encoder.load_text_encoder.assert_called_once()

    def test_regular_encoder_still_preloads_visual_components(self):
        encoder = MiniMaxH3Qwen3VLTextEncoder.__new__(MiniMaxH3Qwen3VLTextEncoder)
        encoder.disk_streaming = False
        encoder.load_tokenizer = Mock()
        encoder.load_text_encoder = Mock()
        encoder.load_processor = Mock()
        encoder.load_vision_encoder = Mock()
        encoder.load()
        encoder.load_processor.assert_called_once()
        encoder.load_vision_encoder.assert_called_once()

    def test_disk_streaming_rejects_shared_cpu_loading(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with shared_cpu_weights"):
            MiniMaxH3Model("unused", {"dit_disk_streaming": True, "shared_cpu_weights": True}, "mps")
        with self.assertRaisesRegex(ValueError, "cannot be combined with text_encoder_shared_cpu_weights"):
            MiniMaxH3Qwen3VLTextEncoder({"text_encoder_disk_streaming": True, "text_encoder_shared_cpu_weights": True})

    def test_supported_tasks_preserve_both_loading_modes(self):
        runner = MiniMaxH3Runner.__new__(MiniMaxH3Runner)
        runner.config = {"model_variant": "fl2av", "text_encoder_disk_streaming": True}
        self.assertEqual(runner.get_supported_tasks(), ("t2av",))
        runner.config["text_encoder_disk_streaming"] = False
        self.assertIn("ref2av", runner.get_supported_tasks())
        runner.config["model_variant"] = "ref2av"
        self.assertEqual(runner.get_supported_tasks(), ("ref2av",))

    def test_runner_releases_each_buffer_once(self):
        for explicit_release in (False, True):
            for resident in (False, True):
                with self.subTest(explicit_release=explicit_release, resident=resident):
                    runner = MiniMaxH3Runner.__new__(MiniMaxH3Runner)
                    runner.config = {
                        "model_variant": "fl2av",
                        "cpu_offload": True,
                        "dit_disk_streaming": True,
                        "text_encoder_disk_streaming": True,
                        "dit_release_block_offload_buffers": explicit_release,
                    }
                    runner.model = Mock(block_offload=True, prepost_resident=resident)
                    runner.maybe_empty_cache = Mock()
                    runner._offload_transformer()
                    self.assertEqual(runner.model.pre_weight.to_cpu.call_count, int(not resident))
                    self.assertEqual(runner.model.post_weight.to_cpu.call_count, int(not resident))
                    self.assertEqual(runner.model.release_block_offload_buffers.call_count, int(explicit_release))
                    self.assertEqual(runner.model.release_disk_streaming_buffer.call_count, int(not explicit_release))

    @staticmethod
    def _write_checkpoint(root):
        shapes = {
            "norm1.weight": (8,),
            "norm2.weight": (8,),
            "attn.to_q.weight": (8, 8),
            "attn.to_k.weight": (8, 8),
            "attn.to_v.weight": (8, 8),
            "attn.to_out.0.weight": (8, 8),
            "attn.norm_q.weight": (4,),
            "attn.norm_k.weight": (4,),
            "ff.net.0.proj.weight": (32, 8),
            "ff.net.2.weight": (8, 16),
        }
        generator = torch.Generator().manual_seed(1497)
        tensors = {f"transformer_blocks.{index}.{name}": torch.randn(shape, generator=generator).to(torch.bfloat16) for index in range(2) for name, shape in shapes.items()}
        save_file(tensors, str(root / "diffusion_pytorch_model.safetensors"))
        return tensors

    def _assert_block_weights(self, block, index, tensors):
        for path, module in (
            ("attn.to_q", block.attn.to_q),
            ("attn.to_k", block.attn.to_k),
            ("attn.to_v", block.attn.to_v),
            ("attn.to_out.0", block.attn.to_out),
            ("ff.net.0.proj", block.ff.in_proj),
            ("ff.net.2", block.ff.out_proj),
        ):
            expected = tensors[f"transformer_blocks.{index}.{path}.weight"]
            torch.testing.assert_close(module.weight.cpu(), expected.t(), rtol=0, atol=0)
        if block.attn.has_fused_qkv:
            expected = torch.cat([tensors[f"transformer_blocks.{index}.attn.to_{part}.weight"] for part in ("q", "k", "v")])
            torch.testing.assert_close(block.attn.to_qkv.weight.cpu(), expected.t(), rtol=0, atol=0)

    def test_streamed_weights_survive_block_changes_and_buffer_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = self._write_checkpoint(Path(directory))
            for shared in (False, True):
                for fused in (False, True):
                    if shared and not hasattr(torch.mps, "_host_alias_storage"):
                        continue
                    with self.subTest(shared=shared, fused=fused):
                        config = {
                            "num_layers": 2,
                            "cpu_offload": True,
                            "offload_granularity": "block",
                            "dit_disk_streaming": True,
                            "dit_mps_shared_buffer": shared,
                            "dit_original_ckpt": directory,
                            "use_adaln_cache": True,
                            "attn_type": "torch_sdpa",
                            "use_fused_qkv": fused,
                        }
                        model = MiniMaxH3Model.__new__(MiniMaxH3Model)
                        model.config = config
                        model.block_offload = True
                        model.transformer_weights = MiniMaxH3TransformerWeights(config)
                        manager = MpsSharedWeightAsyncStreamManager() if shared else None
                        model.transformer_infer = SimpleNamespace(offload_manager=manager, compiled_blocks={})
                        try:
                            # The release flag must never enter the CUDA-only loader
                            # or assume that disk streaming has resident CPU blocks.
                            with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("MPS called CUDA")):
                                for _ in range(2):
                                    model.release_block_offload_buffers()
                                    self.assertIsNone(model.transformer_weights.streaming_block)
                                    model.ensure_block_offload_buffers()
                                    weights = model.transformer_weights
                                    if shared:
                                        manager.init_cuda_buffer(weights.offload_block_cuda_buffers)
                                        manager.init_first_buffer(weights)
                                        for index in (0, 1, 0):
                                            block = manager.cuda_buffers[0]
                                            weights.prepare_streaming_block(block, index)
                                            manager.prefetch_weights(1 - index, weights)
                                            self._assert_block_weights(block, index, tensors)
                                            manager.swap_blocks()
                                    else:
                                        for index in (0, 1, 0):
                                            self._assert_block_weights(weights.load_streaming_block(index), index, tensors)
                        finally:
                            model.release_disk_streaming_buffer()


if __name__ == "__main__":
    unittest.main()
