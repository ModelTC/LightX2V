"""CPU contracts for teacher-local precision and heterogeneous Wan checkpoints."""

import copy
import importlib.machinery
import importlib.util
import sys
import types
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from lightx2v_train.model_capabilities import DistributionMatchingCapability, ParallelCapability, TrainableModelCapability
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.model_zoo.capability_adapters.common import CommonParallelCapability
from lightx2v_train.runtime.fsdp import _build_mp_policy
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime

# The loader tests explicitly skip text encoders. Permit running in a minimal
# CPU environment without ftfy, but fail if that out-of-scope path is invoked.
_missing_ftfy = importlib.util.find_spec("ftfy") is None
if _missing_ftfy:
    _ftfy = types.ModuleType("ftfy")
    _ftfy.__spec__ = importlib.machinery.ModuleSpec("ftfy", loader=None)
    _ftfy.fix_text = Mock(side_effect=AssertionError("Role-precision tests must not run a text encoder"))
    sys.modules["ftfy"] = _ftfy
# Native T5 evaluates current_device() in a constructor default at import time;
# the constructor itself is never invoked by these transformer-only tests.
try:
    with patch("torch.cuda.current_device", return_value=0):
        from lightx2v_train.model_zoo.wan.wan_t2v import WanModel, WanT2VModel
finally:
    # Remove only our shim, not modules legitimately imported by Wan/PEFT.
    if _missing_ftfy:
        sys.modules.pop("ftfy", None)


class DmdRolePrecisionTest(TestCase):
    def fixture(self, teacher_bf16=False):
        config = {
            "model": {
                "name": "wan_t2v",
                "pretrained_model_name_or_path": "/mock/Wan2.1-T2V-1.3B",
                "running_dtype": "fp32",
                "transformer_param_dtype": "fp32",
                "vae_dtype": "fp32",
                "t5_dtype": "fp32",
                "teacher": {
                    "name": "wan_t2v_14b",
                    "pretrained_model_name_or_path": "/mock/Wan2.1-T2V-14B",
                },
            },
            "distributed": {
                "backend": "nccl",
                "sequence_parallel": {"enabled": False, "size": 1},
                "fsdp2": {
                    "enabled": True,
                    "size": 2,
                    "reshard_after_forward": {"root_reshard": False, "block_reshard": True},
                    "mixed_precision": {
                        "param_dtype": "fp32",
                        "reduce_dtype": "fp32",
                        "output_dtype": None,
                        "cast_forward_inputs": False,
                    },
                },
            },
        }
        if teacher_bf16:
            config["model"]["teacher"].update(
                {
                    "running_dtype": "bf16",
                    "transformer_param_dtype": "bf16",
                    "distributed": {"fsdp2": {"mixed_precision": {"param_dtype": "bf16"}}},
                }
            )
        runtime = _DmdRuntime.__new__(_DmdRuntime)
        runtime.config = config
        runtime.model_config = config["model"]
        base = {key: copy.deepcopy(value) for key, value in config["model"].items() if key not in {"teacher", "fake"}}
        return runtime, base

    def test_no_override_preserves_global_precision_and_existing_model_overrides(self):
        runtime, base = self.fixture()
        before = copy.deepcopy(runtime.config)
        fake = runtime._build_dmd_role_config("fake", base)
        teacher = runtime._build_dmd_role_config("teacher", base)
        for role_config in (fake, teacher):
            self.assertEqual(role_config["distributed"], runtime.config["distributed"])
            self.assertEqual(role_config["model"]["running_dtype"], "fp32")
            self.assertEqual(role_config["model"]["transformer_param_dtype"], "fp32")
        self.assertEqual(fake["model"]["name"], "wan_t2v")
        self.assertEqual(teacher["model"]["name"], "wan_t2v_14b")
        self.assertTrue(teacher["model"]["pretrained_model_name_or_path"].endswith("14B"))
        self.assertEqual(runtime.config, before)

    def test_teacher_bf16_deep_merges_precision_without_changing_student_or_fake(self):
        runtime, base = self.fixture(teacher_bf16=True)
        before = copy.deepcopy(runtime.config)
        teacher = runtime._build_dmd_role_config("teacher", base)
        fake = runtime._build_dmd_role_config("fake", base)
        teacher_fsdp = teacher["distributed"]["fsdp2"]
        teacher_policy = _build_mp_policy(teacher_fsdp["mixed_precision"])
        fake_policy = _build_mp_policy(fake["distributed"]["fsdp2"]["mixed_precision"])
        self.assertEqual(teacher_policy.param_dtype, torch.bfloat16)
        self.assertEqual(teacher_policy.reduce_dtype, torch.float32)
        self.assertIsNone(teacher_policy.output_dtype)
        self.assertFalse(teacher_policy.cast_forward_inputs)
        self.assertEqual(fake_policy.param_dtype, torch.float32)
        self.assertEqual(teacher_fsdp["size"], 2)
        self.assertEqual(teacher_fsdp["reshard_after_forward"], before["distributed"]["fsdp2"]["reshard_after_forward"])
        self.assertNotIn("distributed", teacher["model"])
        self.assertEqual(runtime.config, before)
        teacher_fsdp["mixed_precision"]["reduce_dtype"] = "bf16"
        self.assertEqual(fake["distributed"]["fsdp2"]["mixed_precision"]["reduce_dtype"], "fp32")
        self.assertEqual(runtime.config, before)

    def test_fake_precision_override_does_not_change_teacher(self):
        runtime, base = self.fixture()
        runtime.model_config["fake"] = {"distributed": {"fsdp2": {"mixed_precision": {"output_dtype": "bf16"}}}}
        fake = runtime._build_dmd_role_config("fake", base)
        teacher = runtime._build_dmd_role_config("teacher", base)
        self.assertEqual(fake["distributed"]["fsdp2"]["mixed_precision"]["output_dtype"], "bf16")
        self.assertIsNone(teacher["distributed"]["fsdp2"]["mixed_precision"]["output_dtype"])

    def test_rejects_malformed_precision_and_role_specific_topology(self):
        invalid = [
            None,
            {"sequence_parallel": {"size": 1}},
            {"fsdp2": None},
            {"fsdp2": {"size": 1}},
            {"fsdp2": {"enabled": False}},
            {"fsdp2": {"mixed_precision": "bf16"}},
            {"fsdp2": {"mixed_precision": {"param_dytpe": "bf16"}}},
        ]
        for override in invalid:
            with self.subTest(override=override):
                runtime, base = self.fixture()
                runtime.model_config["teacher"]["distributed"] = override
                with self.assertRaisesRegex(ValueError, "model.teacher.distributed"):
                    runtime._build_dmd_role_config("teacher", base)
        runtime, base = self.fixture()
        runtime.model_config["teacher"] = "bf16"
        with self.assertRaisesRegex(ValueError, "model.teacher must be a mapping"):
            runtime._build_dmd_role_config("teacher", base)

    @staticmethod
    def tiny_transformer(*args, **kwargs):
        del args, kwargs
        transformer = torch.nn.Linear(2, 2)
        transformer.patch_size = (1, 2, 2)
        transformer.text_len = 512
        return transformer

    def test_real_wan_loader_routes_14b_teacher_and_preserves_shared_frozen_components(self):
        runtime, base = self.fixture(teacher_bf16=True)
        options = dict(load_transformer=True, load_vae=False, load_condition_encoder=False)
        # Exercise the actual registry, Wan wrapper and checkpoint loader path;
        # replace only the enormous transformer weights with a tiny CPU module.
        with patch("torch.cuda.is_available", return_value=False), patch.object(WanModel, "from_pretrained", side_effect=self.tiny_transformer) as load:
            student = build_loaded_model(runtime.config, **options)
            teacher_config = runtime._build_dmd_role_config("teacher", base)
            teacher = build_loaded_model(teacher_config, **options)
        self.assertIsInstance(student, WanT2VModel)
        self.assertIsInstance(teacher, WanT2VModel)
        self.assertEqual(load.call_args_list[0].args[0], "/mock/Wan2.1-T2V-1.3B")
        self.assertEqual(load.call_args_list[0].kwargs["torch_dtype"], torch.float32)
        self.assertEqual(load.call_args_list[1].args[0], "/mock/Wan2.1-T2V-14B")
        self.assertEqual(load.call_args_list[1].kwargs["torch_dtype"], torch.bfloat16)
        self.assertEqual(next(student.denoiser_module().parameters()).dtype, torch.float32)
        self.assertEqual(next(teacher.denoiser_module().parameters()).dtype, torch.bfloat16)
        self.assertEqual(student.running_dtype, torch.float32)
        self.assertEqual(teacher.running_dtype, torch.bfloat16)
        student.vae, student.text_encoder, student.text_pipeline = object(), object(), object()
        teacher.reuse_frozen_components_from(student)
        self.assertIs(teacher.vae, student.vae)
        self.assertIs(teacher.text_encoder, student.text_encoder)
        self.assertIs(teacher.text_pipeline, student.text_pipeline)
        self.assertEqual(student.running_dtype, torch.float32)
        self.assertEqual(student.vae_dtype, torch.float32)
        self.assertEqual(student.t5_dtype, torch.float32)

    def test_setup_passes_each_role_precision_to_parallel_apply(self):
        runtime, _ = self.fixture(teacher_bf16=True)
        options = dict(load_transformer=True, load_vae=False, load_condition_encoder=False)
        with patch("torch.cuda.is_available", return_value=False), patch.object(WanModel, "from_pretrained", side_effect=self.tiny_transformer):
            runtime.model = build_loaded_model(runtime.config, **options)
        runtime.student = runtime.model.capabilities.require(DistributionMatchingCapability)
        runtime.parallel = runtime.model.capabilities.require(ParallelCapability)
        runtime.trainable_model = runtime.model.capabilities.require(TrainableModelCapability)
        runtime.student_train_type = runtime.fake_train_type = "full"
        runtime.student_lora_config = runtime.fake_lora_config = None
        runtime.gradient_checkpointing = False
        runtime.infer_every_iters = None
        runtime.max_train_iters = 1
        runtime.fake_update_ratio = 5
        runtime.random_schedule_enabled = False
        runtime.defer_ida_setup = True
        runtime.fake_optimizer_learning_rate = 1e-5
        runtime.fake_optimizer_adam_beta1 = 0.0
        runtime.fake_optimizer_adam_beta2 = 0.999
        runtime.fake_optimizer_weight_decay = 0.01
        runtime.fake_optimizer_adam_epsilon = 1e-8
        runtime._build_optimizer = Mock(return_value=object())
        runtime._build_lr_scheduler = Mock(return_value=object())
        runtime.model.vae, runtime.model.text_encoder = object(), object()
        before = copy.deepcopy(runtime.config)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(WanModel, "from_pretrained", side_effect=self.tiny_transformer),
            patch.object(CommonParallelCapability, "apply", autospec=True) as apply,
            patch("lightx2v_train.trainers.dmd.runtime.DMDFlowMatchingScheduler"),
        ):
            runtime.setup()
        self.assertEqual(apply.call_count, 3)
        role_configs = [call.args[1] for call in apply.call_args_list]
        self.assertEqual([config["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"] for config in role_configs], ["fp32", "fp32", "bf16"])
        self.assertEqual([config["model"]["running_dtype"] for config in role_configs], ["fp32", "fp32", "bf16"])
        self.assertEqual(runtime.config, before)
        self.assertIs(runtime.teacher_model.vae, runtime.model.vae)
        self.assertIs(runtime.fake_model.text_encoder, runtime.model.text_encoder)
        self.assertTrue(all(parameter.requires_grad for parameter in runtime.student.denoiser().parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in runtime.fake.denoiser().parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in runtime.teacher.denoiser().parameters()))
        self.assertFalse(runtime.teacher.denoiser().training)


if __name__ == "__main__":
    main()
