"""CPU contracts for PEFT-aware selective gathering of sharded LoRA exports."""

import unittest
import warnings
from unittest.mock import patch

import torch
from diffusers.loaders import PeftAdapterMixin
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from lightx2v_train.model_zoo import base as base_module
from lightx2v_train.model_zoo.base import BaseModel


class _ExportDenoiser(torch.nn.Module, PeftAdapterMixin):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 5, dtype=torch.bfloat16)
        self.projection = torch.nn.Linear(5, 3, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        self.a_aux = torch.nn.Parameter(torch.tensor([1.25, 2.5], dtype=torch.float32))
        self.z_aux = torch.nn.Parameter(torch.tensor([3.5], dtype=torch.bfloat16))

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return None


class _ExportModel(BaseModel):
    def __init__(self, **adapter_options):
        with patch("torch.cuda.is_available", return_value=False):
            super().__init__({"model": {"running_dtype": "bf16"}})
        self.transformer = _ExportDenoiser()
        self.transformer.add_adapter(self.adapter_config(**adapter_options))

    @staticmethod
    def adapter_config(**options):
        return LoraConfig(r=2, lora_alpha=4, **{"target_modules": ["projection"], **options})

    def denoiser_module(self):
        return self.transformer


class _GatheredTensor:
    def __init__(self, name, value, events):
        self.name, self.value, self.events = name, value, events

    def cpu(self):
        self.events.append(("cpu", self.name))
        return self.value.cpu()


class _TrackedShard:
    """A DTensor stand-in: PEFT must only select it, never inspect its values."""

    def __init__(self, name, value, events):
        self.name, self.value, self.events = name, value, events

    def detach(self):
        self.events.append(("detach", self.name))
        return self

    def full_tensor(self):
        self.events.append(("gather", self.name))
        return _GatheredTensor(self.name, self.value, self.events)


class LoRASelectiveExportTest(unittest.TestCase):
    VARIANTS = (
        ("plain", {}),
        ("bias_all", {"bias": "all"}),
        ("bias_lora_only", {"bias": "lora_only"}),
        ("dora", {"use_dora": True}),
        ("modules_to_save", {"modules_to_save": ["head"]}),
        ("embedding", {"target_modules": ["projection", "embed_tokens"]}),
    )

    def setUp(self):
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.filterwarnings("ignore", message="Setting `save_embedding_layers` to `True`")

    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(actual[name].device.type, "cpu", name)
            self.assertTrue(actual[name].is_contiguous(), name)
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def _check_sharded_export(self, model, *, adapter_name=None, main_rank=True, auxiliary_names=("z_aux", "a_aux", "z_aux")):
        # Exports must not silently drop adapters temporarily frozen for scoring
        # or inference, nor any frozen embedding/bias selected by PEFT.
        model.transformer.requires_grad_(False)
        parameters = {name: (parameter, parameter.dtype, parameter.requires_grad) for name, parameter in model.transformer.named_parameters()}
        original = model.transformer.state_dict()
        kwargs = {} if adapter_name is None else {"adapter_name": adapter_name}
        expected_peft = get_peft_model_state_dict(model.transformer, state_dict=original, **kwargs)
        expected_aux = {name: original[name] for name in set(auxiliary_names)}
        events = []
        # Reverse source insertion order to exercise deterministic gather order.
        sharded = {name: _TrackedShard(name, value, events) for name, value in reversed(list(original.items()))}
        selected_shards = get_peft_model_state_dict(model.transformer, state_dict=sharded, **kwargs)
        expected_gathers = [selected_shards[name].name for name in sorted(selected_shards)] + sorted(set(auxiliary_names))

        def select_adapter(*args, **options):
            events.append(("select", adapter_name))
            return get_peft_model_state_dict(*args, **options)

        with (
            patch.object(base_module, "DTensor", _TrackedShard),
            patch.object(base_module, "is_fsdp2_module", return_value=True),
            patch.object(base_module, "is_main_process", return_value=main_rank),
            patch.object(base_module, "get_state_dict", return_value=(sharded, {})) as get_state,
            patch.object(base_module, "get_peft_model_state_dict", side_effect=select_adapter),
        ):
            actual_peft, actual_aux = model._get_lora_and_auxiliary_state_dict_for_save(adapter_name, auxiliary_names)

        get_state.assert_called_once()
        self.assertEqual(get_state.call_args.args, (model.transformer, ()))
        options = get_state.call_args.kwargs["options"]
        self.assertFalse(options.full_state_dict)
        self.assertFalse(options.cpu_offload)
        self.assertFalse(options.ignore_frozen_params)
        self.assertFalse(options.strict)
        self.assertEqual(events[0], ("select", adapter_name))
        self.assertEqual([name for event, name in events if event == "gather"], expected_gathers)
        self.assertEqual([name for event, name in events if event == "detach"], expected_gathers)
        self.assertEqual([name for event, name in events if event == "cpu"], expected_gathers if main_rank else [])
        self.assertNotIn("projection.base_layer.weight", expected_gathers)
        if main_rank:
            self.assert_state_equal(actual_peft, expected_peft)
            self.assert_state_equal(actual_aux, expected_aux)
        else:
            self.assertEqual((actual_peft, actual_aux), ({}, {}))
        for name, parameter in model.transformer.named_parameters():
            before, dtype, requires_grad = parameters[name]
            self.assertIs(parameter, before)
            self.assertEqual(parameter.dtype, dtype)
            self.assertEqual(parameter.requires_grad, requires_grad)
        return expected_peft, expected_gathers

    def test_sharded_variants_gather_only_peft_required_tensors_and_explicit_auxiliary(self):
        for variant, options in self.VARIANTS:
            for main_rank in (True, False):
                with self.subTest(variant=variant, main_rank=main_rank):
                    expected, gathered = self._check_sharded_export(_ExportModel(**options), main_rank=main_rank)
                    if variant == "bias_all":
                        self.assertIn("head.bias", expected)
                        self.assertIn("projection.base_layer.bias", expected)
                    elif variant == "dora":
                        self.assertIn("projection.lora_magnitude_vector", expected)
                        self.assertIn("projection.lora_magnitude_vector.default.weight", gathered)
                    elif variant == "modules_to_save":
                        self.assertIn("head.weight", expected)
                        self.assertIn("head.bias", expected)
                        self.assertIn("head.modules_to_save.default.weight", gathered)
                    elif variant == "embedding":
                        self.assertIn("embed_tokens.lora_embedding_A", expected)
                        self.assertIn("embed_tokens.lora_embedding_B", expected)
                        self.assertIn("embed_tokens.base_layer.weight", gathered)

    def test_named_adapter_export_never_gathers_the_other_adapter(self):
        for adapter_name in (None, "second"):
            for main_rank in (True, False):
                with self.subTest(adapter_name=adapter_name, main_rank=main_rank):
                    model = _ExportModel(modules_to_save=["head"])
                    model.transformer.add_adapter(model.adapter_config(modules_to_save=["head"]), adapter_name="second")
                    with torch.no_grad():
                        for name, parameter in model.transformer.named_parameters():
                            if ".second." in name:
                                parameter.fill_(0.5)
                            elif ".default." in name:
                                parameter.fill_(0.25)
                    expected, gathered = self._check_sharded_export(model, adapter_name=adapter_name, main_rank=main_rank)
                    excluded = "default" if adapter_name == "second" else "second"
                    self.assertFalse(any(f".{excluded}." in name for name in gathered))
                    expected_value = 0.5 if adapter_name == "second" else 0.25
                    self.assertTrue(all(torch.equal(value, torch.full_like(value, expected_value)) for value in expected.values()))

    def test_missing_auxiliary_fails_on_every_rank_before_any_gather(self):
        for main_rank in (True, False):
            with self.subTest(main_rank=main_rank):
                model = _ExportModel()
                events = []
                sharded = {name: _TrackedShard(name, value, events) for name, value in model.transformer.state_dict().items()}
                with (
                    patch.object(base_module, "DTensor", _TrackedShard),
                    patch.object(base_module, "is_fsdp2_module", return_value=True),
                    patch.object(base_module, "is_main_process", return_value=main_rank),
                    patch.object(base_module, "get_state_dict", return_value=(sharded, {})),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Auxiliary parameters.*missing"):
                        model._get_lora_and_auxiliary_state_dict_for_save(auxiliary_parameter_names=("a_aux", "missing_aux"))
                self.assertEqual(events, [])

    def test_unsharded_export_retains_existing_peft_selection_without_collectives(self):
        for variant, options in self.VARIANTS:
            with self.subTest(variant=variant):
                model = _ExportModel(**options)
                model.transformer.requires_grad_(False)
                original = model.transformer.state_dict()
                expected = get_peft_model_state_dict(model.transformer, state_dict=original)
                with (
                    patch.object(base_module, "is_fsdp2_module", return_value=False),
                    patch.object(base_module, "get_state_dict") as get_state,
                    patch.object(model, "_gather_selected_state_dict_for_save") as gather,
                ):
                    actual, auxiliary = model._get_lora_and_auxiliary_state_dict_for_save(auxiliary_parameter_names=("z_aux", "a_aux"))
                get_state.assert_not_called()
                gather.assert_not_called()
                self.assert_state_equal(actual, expected)
                self.assert_state_equal(auxiliary, {name: original[name] for name in ("a_aux", "z_aux")})

    def test_selected_gather_accepts_replicated_plain_tensors_too(self):
        for main_rank in (True, False):
            with self.subTest(main_rank=main_rank):
                values = {
                    "replicated": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t(),
                    "sharded": torch.ones(2, dtype=torch.float32),
                }
                events = []
                selected = {"sharded": _TrackedShard("sharded", values["sharded"], events), "replicated": values["replicated"]}
                with patch.object(base_module, "DTensor", _TrackedShard), patch.object(base_module, "is_main_process", return_value=main_rank):
                    actual = BaseModel._gather_selected_state_dict_for_save(selected)
                self.assertEqual([name for event, name in events if event == "gather"], ["sharded"])
                if main_rank:
                    self.assert_state_equal(actual, values)
                else:
                    self.assertEqual(actual, {})


if __name__ == "__main__":
    unittest.main()
