"""CPU/meta regressions for opt-in, model-independent LoRA parameter precision."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import load_file

from lightx2v_train.model_capabilities import CheckpointCapability, TrainableModelCapability
from lightx2v_train.model_zoo import base as base_module
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.runtime.fsdp import _build_mp_policy
from lightx2v_train.trainers.base import BaseTrainer
from lightx2v_train.trainers.dmd.config import DmdConfig


class _TinyDenoiser(torch.nn.Module, PeftAdapterMixin):
    def __init__(self, dtype, device):
        super().__init__()
        self.projection = torch.nn.Linear(5, 3, dtype=dtype, device=device)
        self.untargeted = torch.nn.Linear(3, 3, dtype=dtype, device=device)
        self.register_buffer("scale", torch.ones((), dtype=dtype, device=device))

    def forward(self, inputs):
        return self.untargeted(self.projection(inputs)) * self.scale


class _TinyModel(BaseModel):
    pipeline_cls = object

    def __init__(self, dtype=torch.bfloat16, device="cpu"):
        config = {
            "model": {"running_dtype": "bf16"},
            "distributed": {
                "fsdp2": {
                    "enabled": True,
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "output_dtype": None,
                        "cast_forward_inputs": False,
                    },
                }
            },
        }
        # Keep this suite CPU-only even when the host has CUDA devices.
        with patch("torch.cuda.is_available", return_value=False):
            super().__init__(config)
        self.transformer = _TinyDenoiser(dtype, device)

    def denoiser_module(self):
        return self.transformer


class LoRAParameterPrecisionTest(unittest.TestCase):
    def setUp(self):
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)

    @staticmethod
    def _capability(model):
        return model.ensure_capabilities().require(TrainableModelCapability)

    @staticmethod
    def _lora_parameters(model):
        return {name: parameter for name, parameter in model.transformer.named_parameters() if "lora" in name}

    def _configure(self, model, **options):
        capability = self._capability(model)
        capability.configure("lora", {"rank": 2, "alpha": 4, "target_modules": ["projection"], **options})
        return capability

    def _assert_precision(self, model, base_dtype, lora_dtype, device="cpu"):
        lora = self._lora_parameters(model)
        self.assertEqual(len(lora), 2)
        self.assertTrue(any(".lora_A." in name for name in lora))
        self.assertTrue(any(".lora_B." in name for name in lora))
        for name, parameter in model.transformer.named_parameters():
            with self.subTest(parameter=name):
                is_lora = "lora" in name
                self.assertEqual(parameter.dtype, lora_dtype if is_lora else base_dtype)
                self.assertEqual(parameter.device, torch.device(device))
                self.assertEqual(parameter.requires_grad, is_lora)
        self.assertEqual(model.transformer.scale.dtype, base_dtype)
        self.assertTrue(model.transformer.training)

    def test_omitted_and_null_option_preserve_original_precision(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for options in ({}, {"param_dtype": None}):
                with self.subTest(dtype=dtype, options=options):
                    model = _TinyModel(dtype)
                    with patch.object(model, "set_lora_param_dtype", wraps=model.set_lora_param_dtype) as cast:
                        capability = self._configure(model, **options)
                        capability.restore("lora")
                    cast.assert_not_called()
                    self._assert_precision(model, dtype, dtype)

    def test_public_capability_accepts_dtype_aliases_on_cpu_and_meta(self):
        aliases = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        for name, target_dtype in aliases.items():
            for device in ("cpu", "meta"):
                with self.subTest(param_dtype=name, device=device):
                    model = _TinyModel(torch.bfloat16, device)
                    base_weight = model.transformer.projection.weight
                    base_bias = model.transformer.projection.bias
                    base_values = [parameter.detach().clone() for parameter in model.transformer.parameters()]
                    capability = self._configure(model, param_dtype=name)
                    self._assert_precision(model, torch.bfloat16, target_dtype, device)
                    self.assertIs(model.transformer.projection.base_layer.weight, base_weight)
                    self.assertIs(model.transformer.projection.base_layer.bias, base_bias)
                    self.assertEqual({id(parameter) for parameter in capability.parameters()}, {id(parameter) for parameter in self._lora_parameters(model).values()})
                    if device == "cpu":
                        frozen = [parameter for name, parameter in model.transformer.named_parameters() if "lora" not in name]
                        for parameter, expected in zip(frozen, base_values, strict=True):
                            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_explicit_fp32_preserves_parameter_identity_values_and_existing_gradients(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for device in ("cpu", "meta"):
                with self.subTest(dtype=dtype, device=device):
                    model = _TinyModel(dtype, device)
                    model.add_lora(2, 4, ["projection"])
                    parameters = dict(model.transformer.named_parameters())
                    values = {name: parameter.detach().clone() for name, parameter in parameters.items()}
                    for parameter in parameters.values():
                        parameter.grad = torch.full_like(parameter, 0.25)
                    model.set_lora_param_dtype("fp32")
                    self._assert_precision(model, dtype, torch.float32, device)
                    for name, parameter in model.transformer.named_parameters():
                        self.assertIs(parameter, parameters[name])
                        self.assertEqual(parameter.grad.dtype, parameter.dtype)
                        self.assertEqual(parameter.grad.device, parameter.device)
                        if device == "cpu":
                            expected = values[name].float() if "lora" in name else values[name]
                            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25), rtol=0, atol=0)

    def test_restore_reenables_lora_without_recasting_or_reinjection(self):
        for options, dtype in (({}, torch.bfloat16), ({"param_dtype": "fp32"}, torch.float32)):
            with self.subTest(options=options):
                model = _TinyModel()
                capability = self._configure(model, **options)
                parameters = dict(model.transformer.named_parameters())
                pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
                model.transformer.requires_grad_(False)
                model.transformer.eval()
                with (
                    patch.object(model, "add_lora", wraps=model.add_lora) as add_lora,
                    patch.object(model, "set_lora_param_dtype", wraps=model.set_lora_param_dtype) as cast,
                    patch.object(base_module, "is_fsdp2_module", return_value=True),
                ):
                    capability.restore("lora")
                    capability.restore("lora")
                add_lora.assert_not_called()
                cast.assert_not_called()
                self._assert_precision(model, torch.bfloat16, dtype)
                for name, parameter in model.transformer.named_parameters():
                    self.assertIs(parameter, parameters[name])
                    self.assertEqual(parameter.data_ptr(), pointers[name])

    def test_plain_adapter_injection_and_training_selection_do_not_cast(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                model = _TinyModel(dtype)
                model.add_lora(2, 4, ["projection"])
                model.set_lora_trainable()
                self._assert_precision(model, dtype, dtype)

    def test_full_fake_training_ignores_lora_precision_option(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                model = _TinyModel(dtype)
                parameters = dict(model.transformer.named_parameters())
                capability = self._capability(model)
                with patch.object(model, "set_lora_param_dtype", wraps=model.set_lora_param_dtype) as cast:
                    capability.configure("full", {"param_dtype": "fp32"})
                    model.transformer.requires_grad_(False)
                    capability.restore("full")
                cast.assert_not_called()
                self.assertFalse(self._lora_parameters(model))
                for name, parameter in model.transformer.named_parameters():
                    self.assertIs(parameter, parameters[name])
                    self.assertEqual(parameter.dtype, dtype)
                    self.assertTrue(parameter.requires_grad)

    def test_fp32_option_does_not_mutate_config_or_bf16_fsdp_compute_policy(self):
        model = _TinyModel()
        original = copy.deepcopy(model.config)
        self._configure(model, param_dtype="fp32")
        self.assertEqual(model.config, original)
        self.assertEqual(model.running_dtype, torch.bfloat16)
        policy = _build_mp_policy(model.config["distributed"]["fsdp2"]["mixed_precision"])
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertIsNone(policy.output_dtype)
        self.assertFalse(policy.cast_forward_inputs)

    def test_base_trainer_forwards_optional_lora_dtype(self):
        for options, expected_dtype in (({}, torch.bfloat16), ({"param_dtype": None}, torch.bfloat16), ({"param_dtype": "fp32"}, torch.float32)):
            with self.subTest(options=options):
                model = _TinyModel()
                config = copy.deepcopy(model.config)
                config["training"] = {
                    "train_type": "lora",
                    "lora": {"rank": 2, "alpha": 4, "target_modules": ["projection"], **options},
                    "lr_warmup_iters": 0,
                    "max_train_iters": 1,
                    "output_dir": "/unused",
                    "gradient_accumulation_iters": 1,
                    "save_every_iters": 1,
                    "save_total_limit": 1,
                }
                config["inference"] = {"method": "none"}
                original = copy.deepcopy(config)
                with (
                    patch("lightx2v_train.trainers.base.RectifiedFlowMatchingScheduler"),
                    patch("lightx2v_train.trainers.base.build_monitor"),
                ):
                    trainer = BaseTrainer(config)
                self.assertEqual(trainer.lora_param_dtype, options.get("param_dtype"))
                trainer._setup_trainable_model(model)
                self._assert_precision(model, torch.bfloat16, expected_dtype)
                self.assertEqual(config, original)

    def test_dmd_config_preserves_role_local_lora_precision(self):
        for fake_train_type in ("lora", "full"):
            with self.subTest(fake_train_type=fake_train_type):
                config = {
                    "training": {
                        "student": {"train_type": "lora", "lora": {"rank": 2, "alpha": 4, "param_dtype": "fp32"}},
                        "fake": {"train_type": fake_train_type, "lora": {"rank": 2, "alpha": 4, "param_dtype": "bf16"}},
                        "dmd": {},
                    }
                }
                original = copy.deepcopy(config)
                parsed = DmdConfig.from_mapping(config)
                self.assertEqual(parsed.student_lora["param_dtype"], "fp32")
                if fake_train_type == "lora":
                    self.assertEqual(parsed.fake_lora["param_dtype"], "bf16")
                else:
                    self.assertIsNone(parsed.fake_lora)
                parsed.student_lora["param_dtype"] = "fp16"
                self.assertEqual(config, original)

    def test_invalid_dtype_is_rejected(self):
        for dtype in ("fp64", "float64", "int8", "invalid"):
            with self.subTest(param_dtype=dtype):
                model = _TinyModel()
                with self.assertRaises(ValueError):
                    self._configure(model, param_dtype=dtype)

    def test_dtype_change_after_fsdp_wrap_is_rejected_without_mutating_parameters(self):
        model = _TinyModel()
        self._configure(model)
        parameters = dict(model.transformer.named_parameters())
        values = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        with patch.object(base_module, "is_fsdp2_module", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "FSDP"):
                model.set_lora_param_dtype("fp32")
        for name, parameter in model.transformer.named_parameters():
            self.assertIs(parameter, parameters[name])
            self.assertEqual(parameter.dtype, torch.bfloat16)
            torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    def test_real_peft_backward_and_adamw_follow_optional_parameter_dtype(self):
        for options, lora_dtype in (({}, torch.bfloat16), ({"param_dtype": "fp32"}, torch.float32)):
            with self.subTest(options=options):
                model = _TinyModel()
                capability = self._configure(model, **options)
                frozen = {name: parameter.detach().clone() for name, parameter in model.transformer.named_parameters() if "lora" not in name}
                optimizer = torch.optim.AdamW(capability.parameters(), lr=0.01)
                output = model.transformer(torch.ones(2, 5, dtype=torch.bfloat16))
                self.assertEqual(output.dtype, torch.bfloat16)
                output.float().square().mean().backward()
                optimizer.step()
                for name, parameter in model.transformer.named_parameters():
                    if "lora" in name:
                        self.assertEqual(parameter.dtype, lora_dtype)
                        self.assertEqual(parameter.grad.dtype, lora_dtype)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, lora_dtype)
                        self.assertEqual(optimizer.state[parameter]["exp_avg_sq"].dtype, lora_dtype)
                    else:
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)

    def test_bf16_checkpoint_and_optimizer_resume_into_opt_in_fp32_lora(self):
        source = _TinyModel()
        self._configure(source)
        source_optimizer = torch.optim.AdamW(source.trainable_parameters(), lr=0.01)
        for parameter in self._lora_parameters(source).values():
            parameter.grad = torch.full_like(parameter, 0.25)
        source_optimizer.step()
        source_values = {name: parameter.detach().float().clone() for name, parameter in self._lora_parameters(source).items()}
        old_optimizer_state = copy.deepcopy(source_optimizer.state_dict())
        self.assertTrue(all(state["exp_avg"].dtype == torch.bfloat16 for state in old_optimizer_state["state"].values()))

        target = _TinyModel()
        capability = self._configure(target, param_dtype="fp32")
        parameters = dict(target.transformer.named_parameters())
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items() if "lora" not in name}
        optimizer = torch.optim.AdamW(capability.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            source.ensure_capabilities().require(CheckpointCapability).save_weights(checkpoint_dir, "lora")
            checkpoint_path = Path(checkpoint_dir) / "pytorch_lora_weights.safetensors"
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in load_file(checkpoint_path).values()))
            target.ensure_capabilities().require(CheckpointCapability).load_weights(checkpoint_dir, "lora")
            optimizer.load_state_dict(old_optimizer_state)
            capability.restore("lora")
            self._assert_precision(target, torch.bfloat16, torch.float32)
            for name, parameter in target.transformer.named_parameters():
                self.assertIs(parameter, parameters[name])
                if "lora" in name:
                    torch.testing.assert_close(parameter, source_values[name], rtol=0, atol=0)
                    self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)
                    self.assertEqual(optimizer.state[parameter]["exp_avg_sq"].dtype, torch.float32)
                else:
                    torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)

            target.transformer(torch.ones(2, 5, dtype=torch.bfloat16)).float().square().mean().backward()
            optimizer.step()
            target.ensure_capabilities().require(CheckpointCapability).save_weights(checkpoint_dir, "lora")
            self.assertTrue(all(value.dtype == torch.float32 for value in load_file(checkpoint_path).values()))


if __name__ == "__main__":
    unittest.main()
