"""Student-only adapter transfer across Turbo -> DMD/PDMD, including FSDP2."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_minimax_h3_mixed_rank_lora import _owner, _transformer
from lightx2v_train.model_zoo.native.minimax_h3.modeling import init_empty_minimax_h3_transformer
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import stream_load_minimax_h3_transformer
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime
from lightx2v_train.trainers.dmd.student_warmstart import load_student_adapter, student_adapter_path
from lightx2v_train.trainers.dmd.trainer import DmdTrainer


def _configured(transformer):
    owner = _owner(transformer)
    owner.add_lora(
        128,
        128,
        [*MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS, "adaln_proj.linear", "norm_out.linear"],
        rank_pattern={"adaln_proj.linear": 16, "norm_out.linear": 16},
        alpha_pattern={"adaln_proj.linear": 16, "norm_out.linear": 16},
    )
    owner.set_lora_param_dtype("fp32")
    owner.set_lora_trainable()
    return owner


def _fill_adapter(owner):
    with torch.no_grad():
        for index, (name, parameter) in enumerate(owner.transformer.named_parameters()):
            if ".lora_" in name:
                parameter.fill_((index + 1) / 1000)


def _fsdp_warmstart_worker(rank, world_size, init_file, directory, device_type="cpu"):
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}", timeout=timedelta(seconds=90))
    try:
        root = Path(directory)
        owner = _configured(init_empty_minimax_h3_transformer(root / "transformer", torch_dtype=torch.bfloat16))
        module = owner.transformer
        mesh = init_device_mesh(device_type, (world_size,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in [*module.token_refiner.refiner_blocks, *module.transformer_blocks]:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        fully_shard(module, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
        stream_load_minimax_h3_transformer(module, root / "transformer", device=device, lora_seed=123, init_lora_weights=True)
        identities = {name: id(parameter) for name, parameter in module.named_parameters()}
        frozen_before = {name: parameter.to_local().clone() for name, parameter in module.named_parameters() if ".lora_" not in name}
        optimizer = torch.optim.AdamW([parameter for parameter in module.parameters() if parameter.requires_grad])
        load_student_adapter(module, root / "student_ema", strict=True)
        expected = torch.load(root / "expected_lora.pt", weights_only=True)
        for name, parameter in module.named_parameters():
            assert id(parameter) == identities[name]
            assert isinstance(parameter, DTensor) and not parameter.is_meta
            if ".lora_" in name:
                torch.testing.assert_close(parameter.full_tensor().cpu(), expected[name], rtol=0, atol=0)
                assert parameter.dtype == torch.float32
            else:
                torch.testing.assert_close(parameter.to_local(), frozen_before[name], rtol=0, atol=0)
        assert not optimizer.state
        assert module.norm_out.linear.r["default"] == 16
        assert module.norm_out.linear.scaling["default"] == 1
    finally:
        dist.destroy_process_group()


class StudentWarmstartTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = _configured(_transformer())
        _fill_adapter(self.source)
        self.export = self.root / "student_ema"
        self.source.save_lora_weights(str(self.export))
        torch.save({name: parameter.detach().clone() for name, parameter in self.source.transformer.named_parameters() if ".lora_" in name}, self.root / "expected_lora.pt")

    def test_ema_directory_or_explicit_file_loads_only_student_adapters(self):
        target = _configured(_transformer())
        before = {name: parameter.detach().clone() for name, parameter in target.transformer.named_parameters() if ".lora_" not in name}
        expected = dict(self.source.transformer.named_parameters())
        for path in (self.export, self.export / "pytorch_lora_weights.safetensors"):
            load_student_adapter(target.transformer, path)
            for name, parameter in target.transformer.named_parameters():
                torch.testing.assert_close(parameter, expected[name] if ".lora_" in name else before[name], rtol=0, atol=0)
        self.assertIsNone(student_adapter_path(self.root))  # Never infer EMA.

    def test_diffusers_down_up_keys_and_prefix_are_supported(self):
        raw = load_file(str(self.export / "pytorch_lora_weights.safetensors"))
        aliases = {"transformer." + key.replace(".lora_A.weight", ".lora.down.weight").replace(".lora_B.weight", ".lora.up.weight"): value for key, value in raw.items()}
        save_file(aliases, self.export / "pytorch_lora_weights.safetensors")
        target = _configured(_transformer())
        load_student_adapter(target.transformer, self.export)
        expected = dict(self.source.transformer.named_parameters())
        for name, parameter in target.transformer.named_parameters():
            if ".lora_" in name:
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_missing_norm_out_or_wrong_rank_fails_before_any_tensor_is_changed(self):
        path = self.export / "pytorch_lora_weights.safetensors"
        raw = load_file(str(path))
        target = _configured(_transformer())
        initial = {name: parameter.detach().clone() for name, parameter in target.transformer.named_parameters()}
        missing = {key: value for key, value in raw.items() if not key.startswith("norm_out.linear.")}
        save_file(missing, path)
        with self.assertRaisesRegex(RuntimeError, "missing=.*norm_out.linear"):
            load_student_adapter(target.transformer, self.export)
        wrong = dict(raw)
        key = next(key for key in wrong if ".lora_A." in key)
        wrong[key] = wrong[key][:1]
        save_file(wrong, path)
        with self.assertRaisesRegex(RuntimeError, "shape_mismatches"):
            load_student_adapter(target.transformer, self.export)
        for name, parameter in target.transformer.named_parameters():
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)

    def test_same_shape_but_different_alpha_fails(self):
        target = _configured(_transformer())
        target.transformer.norm_out.linear.scaling["default"] = 8
        with self.assertRaisesRegex(RuntimeError, "rank/alpha configuration mismatch"):
            load_student_adapter(target.transformer, self.export)

    def test_adapter_transfer_runs_after_parallel_setup_and_not_during_resume(self):
        for resume in (None, "existing_pdmd_checkpoint"):
            owner = object.__new__(DmdTrainer)
            owner.student_checkpoint_path = str(self.export)
            owner.student_checkpoint_strict = True
            owner.student_train_type = "lora"
            owner.model = SimpleNamespace(denoiser_module=lambda: "sharded_student")
            owner._load_student_checkpoint = Mock()
            owner._setup_fake_real_resources = Mock(side_effect=RuntimeError("stop after transfer"))
            events = []
            with (
                patch.object(_DmdRuntime, "setup", side_effect=lambda **kwargs: events.append("parallel_ready")),
                patch("lightx2v_train.trainers.dmd.trainer.load_student_adapter", side_effect=lambda *args, **kwargs: events.append("adapter_loaded")) as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after transfer"):
                    owner.setup(resume_ckpt_path=resume)
            self.assertEqual(events, ["parallel_ready", "adapter_loaded"] if resume is None else ["parallel_ready"])
            self.assertEqual(load.call_count, int(resume is None))
            owner._load_student_checkpoint.assert_not_called()

    def test_legacy_full_checkpoint_still_loads_before_parallel_setup(self):
        owner = object.__new__(DmdTrainer)
        owner.student_checkpoint_path = str(self.root / "model_state.pt")
        owner.student_checkpoint_strict = True
        owner._setup_fake_real_resources = Mock(side_effect=RuntimeError("stop"))
        events = []
        owner._load_student_checkpoint = Mock(side_effect=lambda *args, **kwargs: events.append("legacy_loaded"))
        with patch.object(_DmdRuntime, "setup", side_effect=lambda **kwargs: events.append("parallel_ready")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                owner.setup()
        self.assertEqual(events, ["legacy_loaded", "parallel_ready"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires CPU Gloo")
    def test_two_rank_meta_streamed_h3_load_preserves_shards_base_and_optimizer(self):
        self._streamed_checkpoint()
        mp.spawn(_fsdp_warmstart_worker, args=(2, str(self.root / "gloo_init"), str(self.root)), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_two_gpu_nccl_meta_streamed_h3_load_preserves_shards_base_and_optimizer(self):
        self._streamed_checkpoint()
        mp.spawn(_fsdp_warmstart_worker, args=(2, str(self.root / "nccl_init"), str(self.root), "cuda"), nprocs=2, join=True)

    def _streamed_checkpoint(self):
        model = _transformer().to(dtype=torch.bfloat16)
        for name, module in model.named_modules():
            if name and any(pattern in name for pattern in model._keep_in_fp32_modules):
                module.to(dtype=torch.float32)
        model.save_pretrained(self.root / "transformer", safe_serialization=True, max_shard_size="20KB")


if __name__ == "__main__":
    unittest.main()
