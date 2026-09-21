"""Run with PLATFORM=mps DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 python -m unittest test_cases.test_minimax_h3_mps_streaming."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

if os.environ.get("PLATFORM") != "mps" or not torch.backends.mps.is_available():
    raise unittest.SkipTest("Requires the MPS platform and an Apple GPU")

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight
from lightx2v.common.ops.attn.torch_sdpa_mps import TorchSDPAMPSWeight
from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MiniMaxH3Qwen3VLTextEncoder, _Qwen3VLTextBackboneWeights
from lightx2v.models.networks.minimax_h3.infer.fused_qkv import prepare_qkv_norm_rope
from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3MpsOffloadTransformerInfer, MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.weights.streaming_weights import MiniMaxH3StreamingTransformerWeights
from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3AttentionWeights
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoCausalConv3d


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
                SPARSE_MASK_GENERATOR_REGISTER, SPARSE_OPERATOR_REGISTER,
            )
            from lightx2v.common.ops.attn.ulysses_prepost import create_ulysses_prepost_backend

            assert {"torch_sdpa", "torch_sdpa_mps", "flash_attn2", "svg_attn", "svg2_attn", "ulysses"} <= ATTN_WEIGHT_REGISTER.keys()
            assert "flashinfer_rope" in ROPE_REGISTER
            ATTN_WEIGHT_REGISTER["torch_sdpa"]()
            ATTN_WEIGHT_REGISTER["torch_sdpa_mps"]()
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
                lambda: ATTN_WEIGHT_REGISTER["dynamic_sparse_attn"](),
                lambda: ATTN_WEIGHT_REGISTER["spas_flash_attn4"](),
                lambda: ATTN_WEIGHT_REGISTER["spas_sage_attn2"](),
                lambda: ATTN_WEIGHT_REGISTER["spas_sage_attn3"](),
                lambda: ATTN_WEIGHT_REGISTER["svg2_attn"](),
                lambda: SPARSE_MASK_GENERATOR_REGISTER["sla_mask_generator"](),
                lambda: SPARSE_MASK_GENERATOR_REGISTER["sparge_mask_generator"](),
                lambda: SPARSE_OPERATOR_REGISTER["sla_triton_operator"](),
                lambda: SPARSE_OPERATOR_REGISTER["spas_sage2_operator"](),
                lambda: SPARSE_OPERATOR_REGISTER["spas_sage3_operator"](),
                lambda: SPARSE_OPERATOR_REGISTER["spas_fa4_operator"](),
                lambda: create_ulysses_prepost_backend("triton"),
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
        for batch_shape, heads, kv_heads, dtype in (((), 2, 2, torch.float32), ((1,), 2, 2, torch.float32), ((2,), 4, 4, torch.float32), ((2,), 4, 2, torch.float32), ((), 4, 2, torch.bfloat16)):
            with self.subTest(batch_shape=batch_shape, heads=heads, kv_heads=kv_heads, dtype=dtype):
                q = torch.randn((*batch_shape, 9, heads, 8), generator=generator).to(device="mps", dtype=dtype)
                k = torch.randn((*batch_shape, 11, kv_heads, 8), generator=generator).to(device="mps", dtype=dtype)
                v = torch.randn((*batch_shape, 11, kv_heads, 8), generator=generator).to(device="mps", dtype=dtype)
                expected = TorchSDPAWeight().apply(q, k, v)
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                    actual = TorchSDPAMPSWeight(query_chunk_size=4).apply(q, k, v)
                    self.assertEqual([call.args[0].shape[2] for call in sdpa.call_args_list], [4, 4, 1])
                    self.assertEqual([call.args[1].shape[2] for call in sdpa.call_args_list], [11, 11, 11])
                torch.testing.assert_close(actual, expected)

    def test_query_chunking_preserves_masked_causal_and_disabled_attention(self):
        generator = torch.Generator().manual_seed(1497)
        q = torch.randn((1, 9, 2, 8), generator=generator).to("mps")
        cases = (
            (3, {"causal": True}),
            (3, {"attn_mask": torch.ones((9, 9), dtype=torch.bool, device="mps").tril()}),
            (3, {"attn_mask": torch.randn((9, 9), generator=generator).to("mps")}),
            (3, {"drop_rate": 0.1}),
            (0, {}),
            (-1, {}),
            (9, {}),
        )
        for chunk_size, kwargs in cases:
            with self.subTest(chunk_size=chunk_size, kwargs=tuple(kwargs)):
                # MPS SDPA rejects dropout, so verify its forwarding on CPU.
                inputs = q.cpu() if "drop_rate" in kwargs else q
                torch.manual_seed(1497)
                expected = TorchSDPAWeight().apply(inputs, inputs, inputs, **kwargs)
                torch.manual_seed(1497)
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                    actual = TorchSDPAMPSWeight(query_chunk_size=chunk_size).apply(inputs, inputs, inputs, **kwargs)
                    self.assertEqual(sdpa.call_count, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_plain_sdpa_ignores_mps_chunk_size(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device):
                q = torch.randn((9, 2, 8), device=device)
                expected = TorchSDPAWeight().apply(q, q, q)
                with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                    actual = TorchSDPAWeight().apply(q, q, q, mps_sdpa_query_chunk_size=3)
                    self.assertEqual(sdpa.call_count, 1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mps_configs_select_query_chunking(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs/platforms/mps"
        for filename in ("minimax_h3_t2av.json", "minimax_h3_t2av_4step_512_22.json"):
            with self.subTest(config=filename):
                config = json.loads((config_dir / filename).read_text())
                config["use_adaln_cache"] = False
                weights = MiniMaxH3AttentionWeights("transformer_blocks.0.attn", config)
                self.assertIsInstance(weights.calculate, TorchSDPAMPSWeight)
                infer = MiniMaxH3TransformerInfer(config)
                infer.scheduler = SimpleNamespace()
                infer.block_idx = 0
                chunk_size = config["mps_sdpa_query_chunk_size"]
                q = torch.randn((chunk_size + 1, 2, 8), device="mps", dtype=torch.bfloat16)
                expected = TorchSDPAWeight().apply(q, q, q)
                pre_infer_out = SimpleNamespace(rotary_emb=None, sequence_parallel_state=None)
                with (
                    patch.object(infer, "_prepare_qkv", return_value=(q, q, q)),
                    patch.object(weights.to_out, "apply", side_effect=lambda x: x),
                    patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa,
                ):
                    actual = infer._attention(weights, None, pre_infer_out)
                    self.assertEqual([call.args[0].shape[2] for call in sdpa.call_args_list], [chunk_size, 1])
                torch.testing.assert_close(actual, expected)

    def test_vae_temporal_padding_survives_meta_loading_and_cpu_offload(self):
        source = torch.randn((2, 2, 4, 5, 7), generator=torch.Generator().manual_seed(1497))
        reference = torch.nn.Conv3d(2, 3, kernel_size=3)
        for temporal_padding in (0, 2):
            expected_padding = F.pad(source, (0, 0, 0, 0, temporal_padding, 0))
            with torch.inference_mode():
                expected = reference(expected_padding)
            for platform, device in (("mps", "mps"), ("cuda", "cpu"), ("xpu", "cpu")):
                with self.subTest(platform=platform, temporal_padding=temporal_padding):
                    # Production builds the VAE on meta before loading CPU weights.
                    with patch("lightx2v.models.video_encoders.hf.minimax_h3.video_vae.AI_DEVICE", platform), torch.device("meta"):
                        conv = MiniMaxH3VideoCausalConv3d(2, 3, kernel_size=3, temporal_padding=temporal_padding)
                    conv.to_empty(device="cpu")
                    conv.load_state_dict(reference.state_dict())
                    for _ in range(2):
                        conv.to(device)
                        inputs = source.to(device)
                        with torch.inference_mode():
                            torch.testing.assert_close(conv._pad_temporal(inputs).cpu(), expected_padding, rtol=0, atol=0)
                            torch.testing.assert_close(conv(inputs).cpu(), expected)
                        conv.to("cpu")
                    if platform == "cuda":
                        # Check tracing without requiring NVIDIA hardware on this host.
                        with torch.inference_mode():
                            compiled = torch.compile(conv, backend="eager", fullgraph=True)
                            torch.testing.assert_close(compiled(source), expected)

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

    @unittest.skipUnless(hasattr(torch.mps, "_host_alias_storage"), "Requires shared MPS storage")
    def test_text_prefetch_overlaps_reads_and_recovers(self):
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
        with tempfile.TemporaryDirectory() as directory, patch("lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl.MINIMAX_H3_TEXT_ENCODER_LAYER", 3):
            generator = torch.Generator().manual_seed(1497)
            tensors = {name: torch.randn(shape, generator=generator).to(torch.bfloat16) for name, shape in encoder._expected_weight_shapes(text_config).items()}
            # Use indexed subdirectories and cross a shard boundary during prefetch.
            (Path(directory) / "shards").mkdir()
            indexed_weights = {}
            for index in (0, 1):
                shard_name = f"shards/model-{index}.safetensors"
                shard = {name: tensor for name, tensor in tensors.items() if (".layers.1." in name) == bool(index)}
                save_file(shard, str(Path(directory) / shard_name))
                indexed_weights.update(dict.fromkeys(shard, shard_name))
            (Path(directory) / "model.safetensors.index.json").write_text(json.dumps({"weight_map": indexed_weights}))
            synchronous = _Qwen3VLTextBackboneWeights(config, text_config, num_layers=3, disk_streaming=True)
            streaming = _Qwen3VLTextBackboneWeights({**config, "text_encoder_prefetch": True}, text_config, num_layers=3, disk_streaming=True)
            weight_map, _ = encoder._preflight_native_checkpoint(synchronous, directory, text_config)
            input_ids = torch.tensor([1, 3, 5, 7], device="mps")
            try:
                synchronous.init_disk_streaming(directory, weight_map)
                expected = synchronous.forward(input_ids)
                synchronous.release_disk_streaming_buffer()
                torch.testing.assert_close(synchronous.forward(input_ids), expected, rtol=0, atol=0)
                streaming.init_disk_streaming(directory, weight_map)
                buffers = list(streaming.offload_cuda_buffers)
                addresses = [layer.mlp.down_proj.weight.data_ptr() for layer in buffers]
                self.assertEqual(len(set(addresses)), 2)
                started, proceed = threading.Event(), threading.Event()
                reads = []
                load_block = streaming.load_block_into
                forward = buffers[0].forward

                def read(layer, index):
                    reads.append((index, threading.get_ident()))
                    if index == 1:
                        started.set()
                        if not proceed.wait(5):
                            raise TimeoutError("Prefetch blocked the compute thread")
                    load_block(layer, index)

                def compute(hidden, positions):
                    self.assertTrue(started.wait(5), "The next layer was not prefetched before compute")
                    proceed.set()
                    return forward(hidden, positions)

                with patch.object(streaming, "load_block_into", side_effect=read), patch.object(buffers[0], "forward", side_effect=compute):
                    actual = streaming.forward(input_ids)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual([index for index, _ in reads], [0, 1, 2])
                self.assertEqual(reads[0][1], threading.get_ident())
                self.assertTrue(all(worker != threading.get_ident() for _, worker in reads[1:]))
                torch.testing.assert_close(streaming.forward(input_ids), expected, rtol=0, atol=0)
                self.assertEqual([layer.mlp.down_proj.weight.data_ptr() for layer in buffers], addresses)

                def fail_read(layer, index):
                    if index == 1:
                        raise OSError("test prefetch failure")
                    load_block(layer, index)

                for failure in ("read", "compute"):
                    with self.subTest(failure=failure):
                        injection = (
                            patch.object(streaming, "load_block_into", side_effect=fail_read)
                            if failure == "read"
                            else patch.object(buffers[0], "forward", side_effect=RuntimeError("test compute failure"))
                        )
                        with injection, self.assertRaisesRegex((OSError, RuntimeError), "test .* failure"):
                            streaming.forward(input_ids)
                        self.assertIsNone(streaming.offload_manager.executor)
                        self.assertEqual(streaming.offload_manager.prefetch_futures, [])
                        torch.testing.assert_close(streaming.forward(input_ids), expected, rtol=0, atol=0)

                streaming.release_disk_streaming_buffer()
                self.assertIsNone(streaming.offload_manager)
                self.assertIsNone(streaming.offload_cuda_buffers)
                self.assertTrue(all(not layer.shared_host_tensors for layer in buffers))
                self.assertTrue(all(module.weight is None and module.weight_cuda_buffer is None for layer in buffers for module in layer.weight_modules()))
                torch.testing.assert_close(streaming.forward(input_ids), expected, rtol=0, atol=0)
            finally:
                synchronous.release_disk_streaming_buffer()
                streaming.release_disk_streaming_buffer()

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

    def test_model_rejects_unsupported_streaming_modes_before_loading(self):
        config = {"dit_disk_streaming": True, "cpu_offload": True, "offload_granularity": "block", "use_adaln_cache": True, "adaln_cache_dir": "unused"}
        cases = (
            ({"lazy_load": True}, NotImplementedError, "cannot be combined with lazy_load"),
            ({"dit_quantized": True, "dit_quant_scheme": "int8-q8f", "dit_quantized_ckpt": "unused"}, NotImplementedError, "does not support quantized"),
            ({"tensor_parallel": True}, NotImplementedError, "does not support tensor parallel"),
            ({"cpu_offload": False}, ValueError, "requires cpu_offload=true"),
            ({"offload_granularity": "model"}, ValueError, "requires offload_granularity='block'"),
            ({"dit_mps_shared_buffer": True, "dit_disk_streaming": False}, ValueError, "requires MPS and dit_disk_streaming=true"),
        )
        with patch("lightx2v.models.networks.minimax_h3.weights.streaming_weights.MiniMaxH3ShardCheckpoint") as checkpoint:
            for overrides, error, message in cases:
                with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                    MiniMaxH3Model("unused", {**config, **overrides}, "mps")
            checkpoint.assert_not_called()

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

    def test_cache_cleanup_thresholds_across_backends(self):
        gib = 1024**3
        runner = MiniMaxH3Runner.__new__(MiniMaxH3Runner)
        runner.config = {}
        # Headroom and reclaimable GiB, force, collect garbage, expected clear/GC.
        cases = (
            (5, 3, False, False, False, False),
            (5, 3, False, True, False, True),
            (3, 3, False, False, True, True),
            (3, 1, False, False, False, True),
            (5, 0, True, False, True, True),
            (4, 3, False, False, False, False),
            (3, 2, False, False, True, True),
        )
        for device in ("mps", "cuda", "xpu"):
            for headroom, reclaimable, force, collect, cleared, collected in cases:
                with self.subTest(device=device, headroom=headroom, reclaimable=reclaimable, force=force, collect=collect):
                    backend = SimpleNamespace(empty_cache=Mock())
                    if device == "mps":
                        backend.recommended_max_memory = Mock(return_value=(8 + reclaimable + headroom) * gib)
                        backend.driver_allocated_memory = Mock(return_value=(8 + reclaimable) * gib)
                        backend.current_allocated_memory = Mock(return_value=8 * gib)
                    else:
                        backend.mem_get_info = Mock(return_value=(headroom * gib, 32 * gib))
                        backend.memory_reserved = Mock(return_value=(8 + reclaimable) * gib)
                        backend.memory_allocated = Mock(return_value=8 * gib)
                    with (
                        patch.multiple("lightx2v.models.runners.default_runner", AI_DEVICE=device, torch_device_module=backend),
                        patch("lightx2v.models.runners.default_runner.gc.collect") as gc_collect,
                    ):
                        self.assertEqual(runner.maybe_empty_cache(force=force, collect_garbage=collect), cleared)
                    self.assertEqual(backend.empty_cache.call_count, int(cleared))
                    self.assertEqual(gc_collect.call_count, int(collected))

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
        self.assertEqual(block.attn.has_fused_qkv, block.attn.use_fused_qkv and not block.attn.to_q.has_lora_branch)
        if block.attn.use_fused_qkv:
            expected = torch.cat([tensors[f"transformer_blocks.{index}.attn.to_{part}.weight"] for part in ("q", "k", "v")])
            torch.testing.assert_close(block.attn.to_qkv.weight.cpu(), expected.t(), rtol=0, atol=0)
            if block.attn.has_fused_qkv:
                hidden_states = torch.arange(24, dtype=torch.bfloat16, device="mps").reshape(3, 8) / 16
                separate = torch.cat([module.apply(hidden_states) for module in (block.attn.to_q, block.attn.to_k, block.attn.to_v)], dim=-1)
                torch.testing.assert_close(block.attn.to_qkv.apply(hidden_states), separate, rtol=1e-2, atol=1e-2)

    def test_offload_infer_selection(self):
        cases = (
            ({"cpu_offload": False}, MiniMaxH3TransformerInfer),
            ({"cpu_offload": True, "offload_granularity": "model"}, MiniMaxH3OffloadTransformerInfer),
            ({"cpu_offload": True, "offload_granularity": "block"}, MiniMaxH3OffloadTransformerInfer),
            ({"cpu_offload": True, "dit_disk_streaming": True}, MiniMaxH3TransformerInfer),
            ({"cpu_offload": True, "dit_disk_streaming": True, "dit_mps_shared_buffer": True}, MiniMaxH3MpsOffloadTransformerInfer),
        )
        for config, expected in cases:
            with self.subTest(config=config):
                model = MiniMaxH3Model.__new__(MiniMaxH3Model)
                model.config = config
                model.cpu_offload = config["cpu_offload"]
                model._init_infer_class()
                self.assertIs(model.transformer_infer_class, expected)

    def test_streamed_weights_survive_block_changes_and_buffer_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = self._write_checkpoint(Path(directory))
            adapter_dir = Path(directory) / "adapter"
            adapter_dir.mkdir()
            lora = {}
            generator = torch.Generator().manual_seed(1497)
            for index in range(2):
                prefix = f"transformer_blocks.{index}.attn.to_q"
                lora[f"{prefix}.lora_down.weight"] = torch.randn((2, 8), generator=generator).to(torch.bfloat16) / 16
                lora[f"{prefix}.lora_up.weight"] = torch.randn((8, 2), generator=generator).to(torch.bfloat16) / 16
                lora[f"{prefix}.alpha"] = torch.tensor(2.0, dtype=torch.bfloat16)
            lora_path = str(adapter_dir / "lora.safetensors")
            save_file(lora, lora_path)
            hidden_states = torch.arange(24, dtype=torch.bfloat16, device="mps").reshape(3, 8) / 16
            pre_infer_out = SimpleNamespace(hidden_states=hidden_states)
            for shared in (False, True):
                for fused, use_lora in ((False, False), (True, False), (False, True), (True, True)):
                    if shared and not hasattr(torch.mps, "_host_alias_storage"):
                        continue
                    with self.subTest(shared=shared, fused=fused, lora=use_lora):
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
                        model.cpu_offload = True
                        model.block_offload = True
                        model.device = "mps"
                        model.lora_path = lora_path if use_lora else None
                        model.lora_strength = 1.0
                        model.lora_alpha = None
                        # This fixture contains only the main transformer blocks.
                        model.pre_weight_class = lambda config: WeightModule()
                        model.post_weight_class = lambda config: WeightModule()
                        model._init_weights()
                        self.assertIsInstance(model.transformer_weights, MiniMaxH3StreamingTransformerWeights)
                        model._init_infer_class()
                        # Exercise the real offload loop without a full AdaLN cache/model.
                        infer = model.transformer_infer = model.transformer_infer_class({**config, "use_adaln_cache": False})
                        weights = model.transformer_weights
                        expected = hidden_states
                        for index in range(2):
                            prefix = f"transformer_blocks.{index}.attn.to_q"
                            value = expected @ tensors[f"{prefix}.weight"].to("mps").t()
                            if use_lora:
                                value = value + (expected @ lora[f"{prefix}.lora_down.weight"].to("mps").t()) @ lora[f"{prefix}.lora_up.weight"].to("mps").t()
                            expected = value

                        def run_block(index, block, hidden, pre):
                            self._assert_block_weights(block, index, tensors)
                            return block.attn.to_q.apply(hidden)

                        try:
                            # The release flag must never enter the CUDA-only loader
                            # or assume that disk streaming has resident CPU blocks.
                            with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("MPS called CUDA")), patch.object(infer, "run_block", side_effect=run_block):
                                for _ in range(2):
                                    old_blocks = list(weights.offload_block_cuda_buffers) if shared else [weights.streaming_block]
                                    tensor_refs = []
                                    for block in old_blocks:
                                        for module in (block.attn.to_q, block.attn.to_qkv, block.ff.in_proj, block.norm1):
                                            tensor_refs.extend(
                                                weakref.ref(tensor)
                                                for attr in ("weight", "weight_cuda_buffer", "lora_down", "lora_up")
                                                for tensor in (getattr(module, attr, None),)
                                                if isinstance(tensor, torch.Tensor)
                                            )
                                        if shared:
                                            tensor_refs.extend(weakref.ref(tensor) for tensor in block.shared_host_tensors.values())
                                    infer.get_compiled_block(0, weights.streaming_block)
                                    model.release_block_offload_buffers()
                                    self.assertIsNone(model.transformer_weights.streaming_block)
                                    self.assertEqual(infer.compiled_blocks, {})
                                    self.assertTrue(all(ref() is None for ref in tensor_refs))
                                    if shared:
                                        self.assertTrue(all(not block.shared_host_tensors for block in old_blocks))
                                    model.ensure_block_offload_buffers()
                                    for _ in range(2):
                                        torch.testing.assert_close(infer.infer(weights, pre_infer_out), expected, rtol=0, atol=0)
                                if shared:
                                    for failure in ("read", "compute"):
                                        target, method = (weights, "load_block_into") if failure == "read" else (infer, "run_block")
                                        with patch.object(target, method, side_effect=RuntimeError(f"test {failure} failure")), self.assertRaisesRegex(RuntimeError, f"test {failure} failure"):
                                            infer.infer(weights, pre_infer_out)
                                        self.assertIsNone(infer.offload_manager.executor)
                                        self.assertEqual(infer.offload_manager.cuda_buffers, [])
                                        torch.testing.assert_close(infer.infer(weights, pre_infer_out), expected, rtol=0, atol=0)
                        finally:
                            model.release_disk_streaming_buffer()


if __name__ == "__main__":
    unittest.main()
