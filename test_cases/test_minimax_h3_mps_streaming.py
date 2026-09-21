"""Run with PLATFORM=mps DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 python -m unittest test_cases.test_minimax_h3_mps_streaming."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

if os.environ.get("PLATFORM") != "mps" or not torch.backends.mps.is_available():
    raise unittest.SkipTest("Requires the MPS platform and an Apple GPU")

from lightx2v.common.offload.mps_manager import MpsSharedWeightAsyncStreamManager
from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight
from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MiniMaxH3Qwen3VLTextEncoder, _Qwen3VLTextBackboneWeights
from lightx2v.models.networks.minimax_h3.infer.fused_qkv import prepare_qkv_norm_rope
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3AttentionWeights, MiniMaxH3TransformerWeights
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner


class MiniMaxH3MPSStreamingTest(unittest.TestCase):
    def test_ops_register_without_importing_triton(self):
        # Use a fresh process so earlier tests cannot hide eager kernel imports.
        script = textwrap.dedent("""
            import importlib.abc
            import sys
            import torch

            class WithoutTriton(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "triton" or fullname.startswith("triton."):
                        raise ModuleNotFoundError("Triton is unavailable", name="triton")

            sys.meta_path.insert(0, WithoutTriton())
            import lightx2v.common.ops
            from lightx2v.utils.registry_factory import (
                ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER,
                RMS_WEIGHT_REGISTER, ROPE_REGISTER,
            )
            from lightx2v.common.ops.attn.ulysses_prepost import create_ulysses_prepost_backend

            assert {"torch_sdpa", "flash_attn2", "svg_attn", "svg2_attn", "ulysses"} <= ATTN_WEIGHT_REGISTER.keys()
            assert "flashinfer_rope" in ROPE_REGISTER
            ATTN_WEIGHT_REGISTER["torch_sdpa"]()
            create_ulysses_prepost_backend("torch")
            x = torch.randn(2, 8)
            norm = LN_WEIGHT_REGISTER["torch"]()
            torch.testing.assert_close(norm.apply(x), torch.nn.functional.layer_norm(x, (8,), eps=norm.eps))
            assert "triton" not in sys.modules
            assert "lightx2v.common.ops.attn.kernels.svg" not in sys.modules
            assert "lightx2v.common.ops.attn.svg2_attn_utils" not in sys.modules

            # Selecting an implementation that needs Triton must fail during
            # construction, instead of storing None and failing during inference.
            factories = (
                lambda: LN_WEIGHT_REGISTER["Triton"](),
                lambda: RMS_WEIGHT_REGISTER["one-pass"]("norm.weight"),
                lambda: MM_WEIGHT_REGISTER["fp8-triton"]("proj.weight", None),
                lambda: MM_WEIGHT_REGISTER["int8-triton"]("proj.weight", None),
            )
            for factory in factories:
                try:
                    factory()
                except ModuleNotFoundError as error:
                    assert error.name == "triton", error
                else:
                    raise AssertionError("A Triton implementation accepted a missing dependency")
        """)
        for platform in ("mps", "cuda"):
            with self.subTest(platform=platform):
                env = {**os.environ, "PLATFORM": platform, "SKIP_PLATFORM_CHECK": "1"}
                result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rope_precision_preserves_non_mps_fusion(self):
        q = torch.zeros((2, 8), dtype=torch.bfloat16)
        norm = SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16), sensitive_layer_dtype=torch.bfloat16, infer_dtype=torch.bfloat16)
        freqs = (torch.ones((2, 4)), torch.zeros((2, 4)))
        for device in ("mps", "cuda", "xpu"):
            with self.subTest(device=device), patch("lightx2v.models.networks.minimax_h3.weights.transformer_weights.AI_DEVICE", device):
                rope = MiniMaxH3AttentionWeights("transformer_blocks.0.attn", {"attn_type": "torch_sdpa"}).rope
                self.assertEqual(rope.compute_dtype, torch.bfloat16 if device == "mps" else torch.float32)
                prepared = prepare_qkv_norm_rope(q, q, q, norm, norm, rope, freqs)
                self.assertEqual(prepared is not None, device != "mps")

    def test_query_chunking_supports_other_shapes_and_grouped_query_attention(self):
        generator = torch.Generator().manual_seed(1497)
        for batch, heads, kv_heads in ((1, 2, 2), (2, 4, 4), (2, 4, 2)):
            with self.subTest(batch=batch, heads=heads, kv_heads=kv_heads):
                q = torch.randn((batch, 9, heads, 8), generator=generator).to("mps")
                k = torch.randn((batch, 11, kv_heads, 8), generator=generator).to("mps")
                v = torch.randn((batch, 11, kv_heads, 8), generator=generator).to("mps")
                expected = TorchSDPAWeight().apply(q, k, v)
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                    actual = TorchSDPAWeight().apply(q, k, v, mps_sdpa_query_chunk_size=3)
                    self.assertEqual([call.args[0].shape[2] for call in sdpa.call_args_list], [3, 3, 3])
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_query_chunking_preserves_masked_causal_and_cpu_attention(self):
        generator = torch.Generator().manual_seed(1497)
        source = torch.randn((1, 9, 2, 8), generator=generator)
        cases = (("cpu", {}), ("mps", {"causal": True}), ("mps", {"attn_mask": torch.ones((9, 9), dtype=torch.bool, device="mps")}))
        for device, kwargs in cases:
            with self.subTest(device=device, kwargs=tuple(kwargs)):
                q = source.to(device)
                expected = TorchSDPAWeight().apply(q, q, q, **kwargs)
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                    actual = TorchSDPAWeight().apply(q, q, q, mps_sdpa_query_chunk_size=3, **kwargs)
                    self.assertEqual(sdpa.call_count, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_and_streaming_text_weights_share_checkpoint_validation(self):
        text_config = {
            "hidden_size": 8,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
        }
        config = {"text_encoder_host_pinned": False}
        encoder = MiniMaxH3Qwen3VLTextEncoder
        with tempfile.TemporaryDirectory() as directory, patch("lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl.MINIMAX_H3_TEXT_ENCODER_LAYER", 2):
            generator = torch.Generator().manual_seed(1497)
            tensors = {name: torch.randn(shape, generator=generator).to(torch.bfloat16) for name, shape in encoder._expected_weight_shapes(text_config).items()}
            checkpoint_path = str(Path(directory) / "model.safetensors")
            save_file(tensors, checkpoint_path)
            native = _Qwen3VLTextBackboneWeights(config, text_config, num_layers=2)
            self.assertEqual(encoder._load_native_weights(native, directory, text_config), "BF16")
            torch.testing.assert_close(native.embed_tokens.weight, tensors[native.embed_tokens.weight_name], rtol=0, atol=0)
            for layer in native.layers:
                for module in layer.weight_modules():
                    for name, attr, transpose in module.base_attrs:
                        expected = tensors[name].t() if transpose else tensors[name]
                        torch.testing.assert_close(getattr(module, attr), expected, rtol=0, atol=0)

            streaming = _Qwen3VLTextBackboneWeights(config, text_config, num_layers=2, disk_streaming=True)
            weight_map, dtype = encoder._preflight_native_checkpoint(streaming, directory, text_config)
            self.assertEqual(dtype, "BF16")
            try:
                streaming.init_disk_streaming(directory, weight_map)
                for index in (0, 1, 0):
                    layer = streaming.load_streaming_layer(index)
                    for actual, expected in zip(layer.weight_modules(), native.layers[index].weight_modules()):
                        torch.testing.assert_close(actual.weight.cpu(), expected.weight, rtol=0, atol=0)
            finally:
                streaming.release_disk_streaming_buffer()

            # A malformed tensor in the last layer must fail before loading any
            # host weights, in both resident and disk-streaming modes.
            del native, streaming
            tensors["model.language_model.layers.1.mlp.down_proj.weight"] = torch.zeros((8, 15), dtype=torch.bfloat16)
            save_file(tensors, checkpoint_path)
            for loader in (encoder._load_native_weights, encoder._preflight_native_checkpoint):
                with self.subTest(loader=loader.__name__):
                    backbone = _Qwen3VLTextBackboneWeights(config, text_config, num_layers=2)
                    with patch.object(encoder, "_adopt_pageable_weights", side_effect=AssertionError("Loaded weights before validation")):
                        with self.assertRaisesRegex(ValueError, "Unexpected checkpoint shape"):
                            loader(backbone, directory, text_config)
                    self.assertIsNone(getattr(backbone.embed_tokens, "weight", None))
                    self.assertIsNone(getattr(backbone.embed_tokens, "pin_weight", None))

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
        self.assertEqual(block.attn.has_fused_qkv, block.attn.use_fused_qkv)
        if block.attn.use_fused_qkv:
            expected = torch.cat([tensors[f"transformer_blocks.{index}.attn.to_{part}.weight"] for part in ("q", "k", "v")])
            torch.testing.assert_close(block.attn.to_qkv.weight.cpu(), expected.t(), rtol=0, atol=0)
            hidden_states = torch.arange(24, dtype=torch.bfloat16, device="mps").reshape(3, 8) / 16
            separate = torch.cat([module.apply(hidden_states) for module in (block.attn.to_q, block.attn.to_k, block.attn.to_v)], dim=-1)
            torch.testing.assert_close(block.attn.to_qkv.apply(hidden_states), separate, rtol=1e-2, atol=1e-2)

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
