"""CPU regressions for official-PDMD LoRA initialization and legacy parity."""

import copy
import math
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, inject_adapter_in_model

from lightx2v_train.model_zoo.minimax_h3 import minimax_h3_t2av as wrapper_module
from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import _lora_init_state


class _MetaAdapter(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(23, 7, bias=False, dtype=dtype, device="meta")})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(7, 11, bias=False, dtype=dtype, device="meta")})


def _mixed_meta_adapters():
    model = torch.nn.Module()
    model.fp32 = _MetaAdapter(torch.float32)
    model.bf16 = _MetaAdapter(torch.bfloat16)
    model.fp64 = _MetaAdapter(torch.float64)
    return model


def _reference_state(model, seed, *, kaiming=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = {}
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            value = torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="cpu")
            if kaiming:
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
            else:
                # This is the pre-change streamed initializer, including its
                # original parameter order, dtype, CPU RNG, and normal draw.
                value.normal_(mean=0.0, std=1.0 / parameter.shape[0], generator=generator)
            expected[name] = value
        elif ".lora_B." in name:
            expected[name] = torch.zeros(tuple(parameter.shape), dtype=parameter.dtype, device="cpu")
    return expected


class MiniMaxH3LoRAInitializationTest(unittest.TestCase):
    def assert_states_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            with self.subTest(parameter=name):
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    @staticmethod
    def _wrapper(config, transformer=None):
        model = object.__new__(wrapper_module.MiniMaxH3T2AVModel)
        model.config = config
        if transformer is None:
            transformer = torch.nn.Module()
            transformer.projection = torch.nn.Linear(23, 11, bias=False, dtype=torch.bfloat16)
        model.transformer = transformer
        return model

    def test_stream_legacy_default_is_bitwise_unchanged_and_preserves_rng(self):
        model = _mixed_meta_adapters()
        before = torch.get_rng_state().clone()
        actual = _lora_init_state(model, seed=123, rank_zero=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assert_states_equal(actual, _reference_state(model, 123))
        explicit = _lora_init_state(model, seed=123, rank_zero=True, init_lora_weights="gaussian")
        self.assert_states_equal(actual, explicit)

    def test_stream_official_matches_kaiming_sqrt5_and_preserves_all_dtypes(self):
        model = _mixed_meta_adapters()
        before = torch.get_rng_state().clone()
        actual = _lora_init_state(model, seed=123, rank_zero=True, init_lora_weights=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assert_states_equal(actual, _reference_state(model, 123, kaiming=True))
        for name, value in actual.items():
            if ".lora_B." in name:
                self.assertEqual(torch.count_nonzero(value).item(), 0)
            elif value.dtype == torch.float32:
                self.assertLessEqual(value.abs().max().item(), 1.0 / math.sqrt(value.shape[1]))
        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))

    def test_stream_seeds_remain_independent_and_repeatable(self):
        model = _mixed_meta_adapters()
        for mode in ("gaussian", True):
            with self.subTest(mode=mode):
                first = _lora_init_state(model, seed=123, rank_zero=True, init_lora_weights=mode)
                repeat = _lora_init_state(model, seed=123, rank_zero=True, init_lora_weights=mode)
                other = _lora_init_state(model, seed=124, rank_zero=True, init_lora_weights=mode)
                self.assert_states_equal(first, repeat)
                self.assertTrue(any(not torch.equal(first[name], other[name]) for name in first if ".lora_A." in name))
                self.assertTrue(all(torch.equal(first[name], other[name]) for name in first if ".lora_B." in name))

    def test_nonzero_rank_does_not_materialize_parameters_or_consume_rng(self):
        model = _mixed_meta_adapters()
        before = torch.get_rng_state().clone()
        with patch.object(model, "named_parameters") as parameters:
            for mode in ("gaussian", True):
                self.assertEqual(_lora_init_state(model, seed=123, rank_zero=False, init_lora_weights=mode), {})
        parameters.assert_not_called()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_full_fake_or_teacher_without_lora_is_unchanged(self):
        model = torch.nn.Linear(7, 11, dtype=torch.float32)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        before = torch.get_rng_state().clone()
        for mode in ("gaussian", True):
            self.assertEqual(_lora_init_state(model, seed=123, rank_zero=True, init_lora_weights=mode), {})
            self.assert_states_equal(model.state_dict(), initial)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_only_official_flag_selects_kaiming(self):
        for config in ({}, {"training": {}}, {"training": {"dmd": {"official_pdmd": False}}}):
            with self.subTest(config=config):
                self.assertEqual(self._wrapper(config)._lora_init_weights(), "gaussian")
        self.assertIs(self._wrapper({"training": {"dmd": {"official_pdmd": True}}})._lora_init_weights(), True)

    def test_peft_legacy_initialization_and_rng_match_previous_call_exactly(self):
        self._check_peft_initialization(official=False, expected_mode="gaussian")

    def test_peft_official_uses_kaiming_without_changing_rank_alpha_or_precision(self):
        self._check_peft_initialization(official=True, expected_mode=True)

    def _check_peft_initialization(self, *, official, expected_mode):
        wrapper = self._wrapper({"training": {"dmd": {"official_pdmd": official}}})
        reference = copy.deepcopy(wrapper.transformer)
        base_weight = wrapper.transformer.projection.weight.detach().clone()
        saved_rng = torch.get_rng_state().clone()
        try:
            torch.manual_seed(128)
            with patch.object(wrapper_module, "sync_sequence_parallel_parameters"):
                wrapper.add_lora(128, 8, ["projection"])
            actual_rng = torch.get_rng_state().clone()
            torch.manual_seed(128)
            reference = inject_adapter_in_model(
                LoraConfig(r=128, lora_alpha=8, init_lora_weights=expected_mode, target_modules=["projection"]),
                reference,
                adapter_name="default",
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), actual_rng))
            self.assert_states_equal(wrapper.transformer.state_dict(), reference.state_dict())
        finally:
            torch.set_rng_state(saved_rng)
        adapter = wrapper.transformer.projection
        self.assertEqual(adapter.r["default"], 128)
        self.assertEqual(adapter.lora_alpha["default"], 8)
        self.assertEqual(adapter.scaling["default"], 8 / 128)
        self.assertEqual(adapter.lora_A["default"].weight.dtype, torch.bfloat16)
        self.assertEqual(adapter.lora_B["default"].weight.dtype, torch.bfloat16)
        self.assertEqual(torch.count_nonzero(adapter.lora_B["default"].weight).item(), 0)
        torch.testing.assert_close(adapter.base_layer.weight, base_weight, rtol=0, atol=0)

    def test_wrapper_passes_same_official_mode_to_streamed_loader(self):
        for official, expected_mode in ((False, "gaussian"), (True, True)):
            with self.subTest(official=official):
                wrapper = self._wrapper({"training": {"dmd": {"official_pdmd": official}}})
                wrapper.device = torch.device("cpu")
                wrapper._stream_load_pending = True
                wrapper._stream_load_transformer_dir = "/unused/transformer"
                wrapper._stream_load_lora_seed = 17
                with patch.object(wrapper_module, "stream_load_minimax_h3_transformer") as load:
                    wrapper.after_fsdp2_shard({})
                self.assertEqual(load.call_args.kwargs["init_lora_weights"], expected_mode)
                self.assertEqual(load.call_args.kwargs["lora_seed"], 17)
                self.assertFalse(wrapper._stream_load_pending)

    def test_default_target_coverage_includes_attention_and_ffn_in_both_block_types(self):
        cls = _transformer_class()
        with torch.device("meta"):
            transformer = cls(
                num_attention_heads=2,
                attention_head_dim=8,
                hidden_size=16,
                num_layers=2,
                num_refiner_layers=1,
                ffn_dim=32,
                in_channels=2,
                audio_in_channels=2,
                patch_size=(1, 1, 1),
                text_dim=12,
                freq_dim=4,
                time_embed_hidden_dim=16,
                time_embed_dim=8,
                rope_freq_dim=1,
            )
        targets = MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS
        matched = {name for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear) and any(name.endswith("." + target) for target in targets)}
        expected = {
            f"{block}.{suffix}"
            for block in ("token_refiner.refiner_blocks.0", "transformer_blocks.0", "transformer_blocks.1")
            for suffix in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
        }
        self.assertEqual(matched, expected)


if __name__ == "__main__":
    unittest.main()
