"""CPU coverage for Turbo-style block AdaLN adapters on the real H3 model."""

import math
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av import MiniMaxH3T2AVModel
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class, init_empty_minimax_h3_transformer
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import _lora_init_state, stream_load_minimax_h3_transformer

_TARGETS = [*MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS, "adaln_proj.linear"]
_PATTERN = {"adaln_proj.linear": 16}


def _transformer(*, production_adaln=False, device="cpu"):
    hidden = 5376 if production_adaln else 16
    with torch.device(device):
        return _transformer_class()(
            num_attention_heads=42 if production_adaln else 2,
            attention_head_dim=128 if production_adaln else 8,
            hidden_size=hidden,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 1, 1),
            text_dim=12,
            freq_dim=4,
            time_embed_hidden_dim=16,
            time_embed_dim=2688 if production_adaln else 8,
            rope_freq_dim=1,
        )


def _owner(transformer):
    owner = object.__new__(MiniMaxH3T2AVModel)
    owner.config = {"training": {"dmd": {"official_pdmd": True}}}
    owner.transformer = transformer
    return owner


def _configure(owner):
    owner.add_lora(128, 128, _TARGETS, rank_pattern=_PATTERN, alpha_pattern=_PATTERN)
    owner.set_lora_param_dtype("fp32")
    owner.set_lora_trainable()


def _adapter_modules(transformer):
    return {name: module for name, module in transformer.named_modules() if hasattr(module, "lora_A")}


def _check_adapter_configuration(transformer):
    adapters = _adapter_modules(transformer)
    expected = {
        f"{block}.{suffix}"
        for block in ("token_refiner.refiner_blocks.0", "transformer_blocks.0", "transformer_blocks.1")
        for suffix in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    } | {f"transformer_blocks.{index}.adaln_proj.linear" for index in range(2)}
    assert set(adapters) == expected
    assert "norm_out.linear" not in adapters
    for name, module in adapters.items():
        rank = 16 if name.endswith("adaln_proj.linear") else 128
        assert module.r["default"] == rank
        assert module.lora_alpha["default"] == rank
        assert module.scaling["default"] == 1
        assert module.lora_A["default"].weight.shape[0] == rank
        assert module.lora_B["default"].weight.shape[1] == rank
    for name, parameter in transformer.named_parameters():
        assert parameter.requires_grad == ("lora" in name)
        if "lora" in name:
            assert parameter.dtype == torch.float32


def _streamed_worker(rank, world_size, init_file, transformer_dir):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}", timeout=timedelta(seconds=60))
    try:
        transformer = init_empty_minimax_h3_transformer(transformer_dir, torch_dtype=torch.bfloat16)
        owner = _owner(transformer)
        _configure(owner)
        _check_adapter_configuration(transformer)
        expected = _lora_init_state(transformer, seed=123, rank_zero=True, init_lora_weights=True)
        base_dtypes = {name: parameter.dtype for name, parameter in transformer.named_parameters() if "lora" not in name}
        mesh = init_device_mesh("cpu", (world_size,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in [*transformer.token_refiner.refiner_blocks, *transformer.transformer_blocks]:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        fully_shard(transformer, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
        stream_load_minimax_h3_transformer(transformer, transformer_dir, device=torch.device("cpu"), lora_seed=123, init_lora_weights=True)
        _check_adapter_configuration(transformer)
        for name, parameter in transformer.named_parameters():
            assert isinstance(parameter, DTensor) and not parameter.is_meta
            if "lora" in name:
                torch.testing.assert_close(parameter.full_tensor(), expected[name], rtol=0, atol=0)
            else:
                assert parameter.dtype == base_dtypes[name]

        layout = build_packed_sequence(torch.ones(4, dtype=torch.long), 2, 2, 2, 3, patch_size=(1, 1, 1))
        times, time_indices = build_row_timesteps(layout, 0.5, 0.25)
        torch.manual_seed(456 + rank)
        video, audio = transformer(
            hidden_states=torch.randn(1, 8, 2),
            audio_hidden_states=torch.randn(1, 6, 2),
            encoder_hidden_states=torch.randn(1, 4, 12, dtype=torch.bfloat16),
            timestep=times,
            timestep_indices=time_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
        loss = video.float().square().mean() + audio.float().square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        for name, parameter in transformer.named_parameters():
            if "lora" in name:
                assert parameter.grad is not None and parameter.grad.dtype == torch.float32
                assert bool(torch.isfinite(parameter.grad.to_local()).all())
            else:
                assert parameter.grad is None
    finally:
        dist.destroy_process_group()


class MiniMaxH3MixedRankLoRATest(unittest.TestCase):
    def setUp(self):
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)

    def test_real_h3_targets_use_per_module_rank_alpha_and_frozen_base(self):
        transformer = _transformer().to(dtype=torch.bfloat16)
        originals = {name: (parameter, parameter.detach().clone()) for name, parameter in transformer.named_parameters()}
        owner = _owner(transformer)
        _configure(owner)
        _check_adapter_configuration(transformer)
        for name, parameter in transformer.named_parameters():
            if "lora" in name:
                if ".lora_B." in name:
                    self.assertEqual(torch.count_nonzero(parameter).item(), 0)
                else:
                    self.assertGreater(torch.count_nonzero(parameter).item(), 0)
                    # Injection initializes in the base BF16 dtype before the
                    # explicit FP32-master cast; allow one BF16 rounding unit.
                    self.assertLessEqual(parameter.abs().max().item(), 1.01 / math.sqrt(parameter.shape[1]))
            else:
                original, value = originals[name.replace(".base_layer.", ".")]
                self.assertIs(parameter, original)
                torch.testing.assert_close(parameter, value, rtol=0, atol=0)

    def test_production_fused_adaln_shapes_stay_meta_and_exclude_final_norm(self):
        transformer = _transformer(production_adaln=True, device="meta")
        _configure(_owner(transformer))
        _check_adapter_configuration(transformer)
        self.assertTrue(all(parameter.is_meta for parameter in transformer.parameters()))
        for block in transformer.transformer_blocks:
            module = block.adaln_proj.linear
            self.assertEqual(tuple(module.lora_A["default"].weight.shape), (16, 2688))
            self.assertEqual(tuple(module.lora_B["default"].weight.shape), (96768, 16))

    def test_raw_adapter_export_and_same_configuration_resume_keep_mixed_scaling(self):
        owner = _owner(_transformer())
        _configure(owner)
        with torch.no_grad():
            for module in _adapter_modules(owner.transformer).values():
                module.lora_B["default"].weight.uniform_(-0.1, 0.1)
        expected = {name: parameter.detach().clone() for name, parameter in owner.transformer.named_parameters() if "lora" in name}
        with tempfile.TemporaryDirectory() as directory:
            owner.save_lora_weights(directory)
            saved = load_file(str(Path(directory) / "pytorch_lora_weights.safetensors"))
            self.assertEqual(len(saved), len(expected))
            restored = _owner(_transformer())
            _configure(restored)
            restored.load_lora_weights_for_resume(directory)
        _check_adapter_configuration(restored.transformer)
        for name, parameter in restored.transformer.named_parameters():
            if "lora" in name:
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        for name, module in _adapter_modules(owner.transformer).items():
            torch.testing.assert_close(module.get_delta_weight("default"), _adapter_modules(restored.transformer)[name].get_delta_weight("default"), rtol=0, atol=0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires CPU Gloo")
    def test_fsdp_streamed_mixed_adapters_keep_initialization_precision_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            transformer_dir = str(Path(directory) / "transformer")
            model = _transformer().to(dtype=torch.bfloat16)
            for name, module in model.named_modules():
                if name and any(pattern in name for pattern in model._keep_in_fp32_modules):
                    module.to(dtype=torch.float32)
            model.save_pretrained(transformer_dir, safe_serialization=True, max_shard_size="20KB")
            del model
            mp.spawn(_streamed_worker, args=(2, str(Path(directory) / "gloo_init"), transformer_dir), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
