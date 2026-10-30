"""Frozen preset layout, export and real two-rank streamed-FSDP contracts."""

import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from lightx2v_train.model_capabilities import DistributionMatchingCapability, TrainableModelCapability
from lightx2v_train.model_zoo.capability_adapters.common import CommonTrainableCapability
from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_minimax_h3_mixed_rank_lora import _configure, _owner, _transformer
from lightx2v_train.model_zoo.minimax_h3.training_adapter import AdapterOptions, merge_training_adapter
from lightx2v_train.model_zoo.native.minimax_h3.modeling import init_empty_minimax_h3_transformer
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime


def _preset(model, path, layout="comfy"):
    generator = torch.Generator().manual_seed(95)
    result = {}
    for group, blocks in (("blocks", model.transformer_blocks), ("token_refiner.blocks", model.token_refiner.refiner_blocks)):
        for index, block in enumerate(blocks):
            for suffix, target in (("attn.qkv_proj", block.attn.to_q), ("attn.out_proj", block.attn.to_out[0]), ("mlp.fc1", block.ff.net[0].proj), ("mlp.fc2", block.ff.net[2])):
                out_features, in_features = target.weight.shape
                rank = 3
                a = torch.randint(-4, 5, (rank, in_features), generator=generator).float() / 64
                b = torch.randint(-4, 5, (out_features * (3 if suffix.endswith("qkv_proj") else 1), rank), generator=generator).float() / 64
                if layout == "official" and suffix.endswith("qkv_proj"):
                    heads, dim = model.config.num_attention_heads, model.config.attention_head_dim
                    b = b.reshape(3, heads, dim, rank).permute(1, 0, 2, 3).reshape(-1, rank)
                result[f"{group}.{index}.{suffix}.lora_A.default.weight"] = a.to(torch.bfloat16)
                result[f"{group}.{index}.{suffix}.lora_B.default.weight"] = b.to(torch.bfloat16)
    save_file(result, path)
    return result


def _expected_state(model, factors, scale=1.0):
    result = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for group, target in (("blocks", "transformer_blocks"), ("token_refiner.blocks", "token_refiner.refiner_blocks")):
        count = len(model.transformer_blocks) if group == "blocks" else len(model.token_refiner.refiner_blocks)
        for index in range(count):
            for suffix, destination in (("attn.qkv_proj", None), ("attn.out_proj", "attn.to_out.0"), ("mlp.fc1", "ff.net.0.proj"), ("mlp.fc2", "ff.net.2")):
                source = f"{group}.{index}.{suffix}"
                a, b = factors[source + ".lora_A.default.weight"].float(), factors[source + ".lora_B.default.weight"].float()
                if suffix == "attn.qkv_proj":
                    changes = [(f"attn.to_{name}", part @ a) for name, part in zip(("q", "k", "v"), b.chunk(3))]
                elif suffix == "mlp.fc1":
                    first, second = b.chunk(2)
                    changes = [(destination, torch.cat((second, first)) @ a)]
                else:
                    changes = [(destination, b @ a)]
                for name, delta in changes:
                    key = f"{target}.{index}.{name}.weight"
                    result[key] = (result[key].float() + delta * scale).to(result[key].dtype)
    return result


def _fsdp_worker(rank, world_size, init_file, directory, device_type):
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}", timeout=timedelta(seconds=120))
    try:
        root = Path(directory)
        transformer = init_empty_minimax_h3_transformer(root / "model", torch_dtype=torch.bfloat16)
        owner = _owner(transformer)
        owner.device = device
        owner._training_adapter_options = AdapterOptions.from_mapping({"enabled": True, "path": str(root / "preset.safetensors"), "scale": 0.5})
        owner._training_adapter_role = None
        owner._training_adapter_applied = False
        owner._stream_load_pending = True
        owner._stream_load_transformer_dir = root / "model"
        owner._stream_load_lora_seed = 123
        owner.prepare_training_adapter_role("fake")
        assert not owner._training_adapter_applied
        _configure(owner)
        mesh = init_device_mesh(device_type, (world_size,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in [*transformer.token_refiner.refiner_blocks, *transformer.transformer_blocks]:
            fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        fully_shard(transformer, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
        identities = {name: id(parameter) for name, parameter in transformer.named_parameters()}
        owner.after_fsdp2_shard({})
        assert owner._training_adapter_applied
        owner.after_fsdp2_shard({})  # Cannot merge twice.
        expected = torch.load(root / "expected.pt", weights_only=True)
        for name, parameter in transformer.named_parameters():
            assert id(parameter) == identities[name] and isinstance(parameter, DTensor)
            if ".lora_" not in name:
                torch.testing.assert_close(parameter.full_tensor().cpu(), expected[name.replace(".base_layer.", ".")], rtol=0, atol=0)
                assert not parameter.requires_grad
            else:
                assert parameter.requires_grad and parameter.dtype == torch.float32
        layout = build_packed_sequence(torch.ones(4, dtype=torch.long), 2, 2, 2, 3, patch_size=(1, 1, 1)).to(device)
        times, indices = build_row_timesteps(layout, 0.5, 0.25)
        video, audio = transformer(
            hidden_states=torch.randn(1, 8, 2, device=device),
            audio_hidden_states=torch.randn(1, 6, 2, device=device),
            encoder_hidden_states=torch.randn(1, 4, 12, dtype=torch.bfloat16, device=device),
            timestep=times,
            timestep_indices=indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
        loss = video.float().square().mean() + audio.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in transformer.named_parameters():
            if ".lora_" in name:
                assert parameter.grad is not None and torch.isfinite(parameter.grad.to_local()).all()
            else:
                assert parameter.grad is None
    finally:
        dist.destroy_process_group()


class TrainingAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = _transformer().to(dtype=torch.bfloat16)
        self.path = self.root / "preset.safetensors"
        self.factors = _preset(self.model, self.path)

    def test_default_off_does_not_require_or_inspect_a_path(self):
        option = AdapterOptions.from_mapping({"enabled": False, "path": "/does/not/exist"})
        self.assertEqual(option.metadata(), {"enabled": False})
        self.assertEqual(merge_training_adapter(self.model, option), 0)
        score_preset = AdapterOptions.from_mapping({"enabled": True, "path": str(self.path)})
        self.assertEqual(score_preset.roles, ("fake", "teacher"))
        self.assertFalse(score_preset.metadata()["inference_requires_preset"])
        for bad in ("false", 0, None):
            with self.subTest(enabled=bad), self.assertRaises(ValueError):
                AdapterOptions.from_mapping({"enabled": bad})
        for role in ("student", "fake", "teacher"):
            selected = AdapterOptions.from_mapping({"enabled": True, "path": str(self.path), "roles": [role]})
            self.assertEqual(selected.metadata()["inference_requires_preset"], role == "student")

    def test_comfy_qkv_and_ffn_gate_half_mapping_preserve_parameter_objects(self):
        expected = _expected_state(self.model, self.factors, scale=0.5)
        identities = {name: id(parameter) for name, parameter in self.model.named_parameters()}
        count = merge_training_adapter(self.model, AdapterOptions.from_mapping({"enabled": True, "path": str(self.path), "scale": 0.5}))
        self.assertEqual(count, 18)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(id(parameter), identities[name])
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_official_interleaved_qkv_produces_same_update_as_comfy_variant(self):
        official = self.root / "official.safetensors"
        _preset(self.model, official, layout="official")
        expected = _expected_state(self.model, self.factors)
        merge_training_adapter(self.model, AdapterOptions.from_mapping({"enabled": True, "path": str(official), "qkv_layout": "official"}))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_invalid_coverage_shape_and_nonfinite_data_fail_before_mutation(self):
        initial = copy.deepcopy(self.model.state_dict())
        for mode in ("missing", "shape", "nonfinite"):
            factors = dict(self.factors)
            key = next(iter(factors))
            if mode == "missing":
                factors.pop(key)
            elif mode == "shape":
                factors[key] = factors[key][:1]
            else:
                factors[key] = factors[key].clone().fill_(float("nan"))
            save_file(factors, self.path)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                merge_training_adapter(self.model, AdapterOptions.from_mapping({"enabled": True, "path": str(self.path)}))
            for name, parameter in self.model.named_parameters():
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)

    def test_role_filter_idempotence_and_learned_only_export_sidecar(self):
        owner = _owner(self.model)
        # Explicit legacy/custom student presets retain their dependency marker.
        owner._training_adapter_options = AdapterOptions.from_mapping({"enabled": True, "path": str(self.path), "roles": ["student", "fake"]})
        owner._training_adapter_role, owner._training_adapter_applied = None, False
        owner._stream_load_pending = False
        baseline = copy.deepcopy(self.model.state_dict())
        owner.prepare_training_adapter_role("teacher")
        self.assertFalse(owner._training_adapter_applied)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, baseline[key], rtol=0, atol=0)
        owner.prepare_training_adapter_role("student")
        expected = copy.deepcopy(self.model.state_dict())
        owner.prepare_training_adapter_role("student")
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        _configure(owner)
        self.model.transformer_blocks[0].attn.to_q(torch.randn(1, 2, 16, dtype=torch.bfloat16)).float().sum().backward()
        for name, parameter in self.model.named_parameters():
            if ".lora_" not in name:
                self.assertIsNone(parameter.grad)
        output = self.root / "export"
        owner.save_lora_weights(output)
        exported = load_file(output / "pytorch_lora_weights.safetensors")
        self.assertTrue(all("lora" in key for key in exported))
        metadata = json.loads((output / "training_adapter.json").read_text())
        self.assertTrue(metadata["inference_requires_preset"])
        self.assertEqual(metadata["scale"], 1.0)
        self.assertEqual(metadata["roles"], ["student", "fake"])
        self.assertEqual(len(metadata["sha256"]), 64)
        self.assertEqual(metadata["model_name"], "minimax_h3_t2av")
        self.assertEqual(metadata["transformer_component"], "transformer")
        self.assertEqual(metadata["export_role"], "student")
        with self.assertRaisesRegex(RuntimeError, "Cannot change"):
            owner.prepare_training_adapter_role("teacher")

    def _score_owner(self):
        model = _transformer().to(dtype=torch.bfloat16)
        model.load_state_dict(self.model.state_dict())
        model.requires_grad_(False)
        owner = _owner(model)
        owner._training_adapter_options = AdapterOptions.from_mapping({"enabled": True, "path": str(self.path)})
        owner._training_adapter_role, owner._training_adapter_applied = None, False
        owner._stream_load_pending = False
        capability = SimpleNamespace(model=owner, student_sla=SimpleNamespace(enabled=False))
        trainable = CommonTrainableCapability(owner)
        distribution = SimpleNamespace(prepare_role=lambda role: MiniMaxH3DistributionMatchingCapability.prepare_role(capability, role))
        choices = {DistributionMatchingCapability: distribution, TrainableModelCapability: trainable}
        owner.ensure_capabilities = lambda: SimpleNamespace(require=lambda interface: choices[interface])
        return owner, distribution

    def test_score_roles_keep_student_original_teacher_frozen_and_fake_full_trainable(self):
        baseline = copy.deepcopy(self.model.state_dict())
        expected_score = _expected_state(self.model, self.factors)
        lora = {
            "rank": 128,
            "alpha": 128,
            "param_dtype": "fp32",
            "target_modules": [*MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS, "adaln_proj.linear", "norm_out.linear"],
            "rank_pattern": {"adaln_proj.linear": 16, "norm_out.linear": 16},
            "alpha_pattern": {"adaln_proj.linear": 16, "norm_out.linear": 16},
        }
        runtime = SimpleNamespace(student_train_type="lora", student_lora_config=lora, fake_train_type="full", fake_lora_config=None)

        student, _ = self._score_owner()
        _DmdRuntime._setup_trainable_model(runtime, student, role="student")
        self.assertFalse(student._training_adapter_applied)
        for name, parameter in student.transformer.named_parameters():
            self.assertEqual(parameter.requires_grad, ".lora_" in name)
            if ".lora_" not in name:
                torch.testing.assert_close(parameter, baseline[name.replace(".base_layer.", ".")], rtol=0, atol=0)

        teacher, distribution = self._score_owner()
        distribution.prepare_role("teacher")
        self.assertTrue(teacher._training_adapter_applied)
        self.assertFalse(teacher.transformer.training)
        for name, parameter in teacher.transformer.named_parameters():
            self.assertFalse(parameter.requires_grad)
            torch.testing.assert_close(parameter, expected_score[name], rtol=0, atol=0)

        fake, _ = self._score_owner()
        _DmdRuntime._setup_trainable_model(runtime, fake, role="fake")
        self.assertTrue(fake._training_adapter_applied)
        for name, parameter in fake.transformer.named_parameters():
            self.assertTrue(parameter.requires_grad)
            torch.testing.assert_close(parameter, expected_score[name], rtol=0, atol=0)
        layer = fake.transformer.transformer_blocks[0].attn.to_q
        before = layer.weight.detach().clone()
        optimizer = torch.optim.SGD(fake.transformer.parameters(), lr=0.1)
        layer(torch.ones(1, 2, 16, dtype=torch.bfloat16)).float().sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(layer.weight, before))
        self.assertIsNone(teacher.transformer.transformer_blocks[0].attn.to_q.weight.grad)

        export = self.root / "score_side_student_export"
        student.save_lora_weights(export)
        saved = load_file(export / "pytorch_lora_weights.safetensors")
        self.assertTrue(all("lora" in key for key in saved))
        metadata = json.loads((export / "training_adapter.json").read_text())
        self.assertFalse(metadata["enabled"])
        self.assertFalse(metadata["inference_requires_preset"])
        self.assertEqual(metadata["export_semantics"], "learned_lora_only")
        self.assertEqual(metadata["model_name"], "minimax_h3_t2av")
        self.assertEqual(metadata["export_role"], "student")
        self.assertEqual(metadata["score_training_adapter"]["roles"], ["fake", "teacher"])
        self.assertEqual(metadata["score_training_adapter"]["sha256"], student.training_adapter_metadata()["sha256"])

    def _streamed_fixture(self):
        for name, module in self.model.named_modules():
            if name and any(pattern in name for pattern in self.model._keep_in_fp32_modules):
                module.to(dtype=torch.float32)
        self.model.save_pretrained(self.root / "model", safe_serialization=True, max_shard_size="20KB")
        torch.save(_expected_state(self.model, self.factors, scale=0.5), self.root / "expected.pt")

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires two CPU Gloo ranks")
    def test_two_rank_meta_streamed_fsdp_merge_and_backward(self):
        self._streamed_fixture()
        mp.spawn(_fsdp_worker, args=(2, str(self.root / "cpu_init"), str(self.root), "cpu"), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_two_gpu_nccl_meta_streamed_fsdp_merge_and_backward(self):
        self._streamed_fixture()
        mp.spawn(_fsdp_worker, args=(2, str(self.root / "gpu_init"), str(self.root), "cuda"), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
