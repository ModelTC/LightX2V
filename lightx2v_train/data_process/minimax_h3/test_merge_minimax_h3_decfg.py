"""Offline score export matches the online BF16 teacher and FP32 fake."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

TRAIN_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_test_offline_h3_merge", TRAIN_ROOT / "scripts/merge_minimax_h3_decfg.py")
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class OfflineDeCFGTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        self.adapters = self.root / "adapters"
        self.base.mkdir()
        self.adapters.mkdir()
        (self.base / "vae").mkdir()
        (self.base / "vae/config.json").write_text("{}")
        self.expected = {}
        self.states = {}
        generator = torch.Generator().manual_seed(17)
        for component, adapter in (("transformer", "model_for_comfy_dit.safetensors"), ("transformer_ref", "model_ref2va_for_comfy_dit.safetensors")):
            directory = self.base / component
            directory.mkdir()
            config = {"_class_name": "MiniMaxH3Transformer3DModel", "num_layers": 1, "num_refiner_layers": 1, "attention_head_dim": 2}
            (directory / "config.json").write_text(json.dumps(config))
            state = {"proj_in.weight": torch.randn(3, 4, generator=generator), "an_integer_buffer": torch.arange(3), "a_scalar": torch.tensor(2.0)}
            expected = {key: value.clone() for key, value in state.items()}
            factors = {}
            for group, target in (("blocks", "transformer_blocks"), ("token_refiner.blocks", "token_refiner.refiner_blocks")):
                for suffix, destinations, rows in (
                    ("attn.qkv_proj", ("attn.to_q", "attn.to_k", "attn.to_v"), 12),
                    ("attn.out_proj", ("attn.to_out.0",), 4),
                    ("mlp.fc1", ("ff.net.0.proj",), 8),
                    ("mlp.fc2", ("ff.net.2",), 4),
                ):
                    key = f"{group}.0.{suffix}"
                    a = (torch.randint(-4, 5, (3, 4), generator=generator).float() / 64).bfloat16()
                    b = (torch.randint(-4, 5, (rows, 3), generator=generator).float() / 64).bfloat16()
                    factors[f"{key}.lora_A.default.weight"] = a
                    factors[f"{key}.lora_B.default.weight"] = b
                    if suffix == "attn.qkv_proj":
                        deltas = [part.float() @ a.float() for part in b.chunk(3)]
                    elif suffix == "mlp.fc1":
                        first, second = b.chunk(2)
                        deltas = [torch.cat((second, first)).float() @ a.float()]
                    else:
                        deltas = [b.float() @ a.float()]
                    for name, delta in zip(destinations, deltas):
                        weight = (torch.randn(delta.shape, generator=generator) / 10).bfloat16()
                        state[f"{target}.0.{name}.weight"] = weight
                        expected[f"{target}.0.{name}.weight"] = weight.float().add(delta, alpha=0.5)
            shard_names = ["part-1.safetensors", "part-2.safetensors"]
            keys = sorted(state)
            weight_map = {}
            for shard, subset in zip(shard_names, (keys[::2], keys[1::2])):
                save_file({key: state[key] for key in subset}, directory / shard)
                weight_map.update({key: shard for key in subset})
            (directory / MERGE.INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
            save_file(factors, self.adapters / adapter)
            self.states[component] = state
            self.expected[component] = expected

    def invoke(self, **kwargs):
        return MERGE.merge(self.base, self.adapters, self.root / "merged", scale=0.5, tile_rows=3, **kwargs)

    def test_both_partitions_streamed_fp32_fake_and_bf16_teacher_match_online_merge(self):
        manifest = self.invoke()
        output = self.root / "merged"
        self.assertEqual(set(manifest["components"]), {"transformer", "transformer_ref"})
        self.assertFalse(manifest["student_uses_decfg"])
        self.assertTrue((output / "vae").is_symlink())
        self.assertEqual((output / "vae").resolve(), self.base / "vae")
        self.assertEqual(json.loads((output / MERGE.MANIFEST_NAME).read_text()), manifest)
        for component, metadata in manifest["components"].items():
            helper_spec = importlib.util.spec_from_file_location("_test_premerged_score_validator", TRAIN_ROOT / "lightx2v_train/model_zoo/minimax_h3/premerged_decfg.py")
            validator = importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(validator)
            validated = validator.validate_premerged_score_base(output, component_name=component, expected_adapter=metadata["training_adapter"], required=True, verify_weights=True)
            self.assertEqual(validated["component"], metadata)
            state = {}
            for shard in metadata["shards"]:
                path = output / component / shard["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), shard["sha256"])
                self.assertEqual(path.stat().st_size, shard["size_bytes"])
                state.update(load_file(path))
            self.assertEqual(set(state), set(self.states[component]))
            for key, reference in self.expected[component].items():
                torch.testing.assert_close(state[key], reference, rtol=0, atol=0)
                torch.testing.assert_close(state[key].bfloat16(), reference.bfloat16(), rtol=0, atol=0)
            self.assertEqual(metadata["adapted_weights"], 12)
            self.assertEqual(metadata["training_adapter"]["roles"], ["fake", "teacher"])
            self.assertEqual(json.loads((output / component / MERGE.MANIFEST_NAME).read_text()), manifest)

    def test_single_file_base_always_publishes_a_sharded_loader_index(self):
        directory = self.base / "transformer"
        for path in directory.glob("*.safetensors"):
            path.unlink()
        (directory / MERGE.INDEX_NAME).unlink()
        save_file(self.states["transformer"], directory / "diffusion_pytorch_model.safetensors")
        manifest = self.invoke(tasks=("fl2av",))
        output = self.root / "merged/transformer"
        self.assertEqual(len(manifest["components"]["transformer"]["shards"]), 1)
        index = json.loads((output / MERGE.INDEX_NAME).read_text())
        self.assertEqual(set(index["weight_map"]), set(self.states["transformer"]))
        self.assertFalse((self.root / "merged/transformer_ref").exists())
        for key, tensor in load_file(output / "diffusion_pytorch_model.safetensors").items():
            torch.testing.assert_close(tensor, self.expected["transformer"][key], rtol=0, atol=0)

    def test_original_interleaved_qkv_adapter_reproduces_comfy_merge(self):
        for adapter in self.adapters.glob("*_for_comfy_dit.safetensors"):
            factors = load_file(adapter)
            for key, value in list(factors.items()):
                if ".attn.qkv_proj.lora_B." in key:
                    factors[key] = value.reshape(3, 2, 2, 3).permute(1, 0, 2, 3).reshape(12, 3).contiguous()
            save_file(factors, adapter.with_name(adapter.name.replace("_for_comfy_dit", "")))
        manifest = self.invoke(qkv_layout="official")
        for component, metadata in manifest["components"].items():
            for shard in metadata["shards"]:
                for key, tensor in load_file(self.root / "merged" / component / shard["path"]).items():
                    torch.testing.assert_close(tensor, self.expected[component][key], rtol=0, atol=0)

    def test_no_overwrite_or_second_merge(self):
        self.invoke()
        with self.assertRaises(FileExistsError):
            self.invoke()
        with self.assertRaisesRegex(ValueError, "already contains"):
            MERGE.merge(self.root / "merged", self.adapters, self.root / "twice", scale=0.5)
        self.assertFalse((self.root / "twice").exists())

    def test_non_bf16_adapted_source_rejected_before_output_creation(self):
        path = self.base / "transformer/part-1.safetensors"
        state = load_file(path)
        key = next(key for key in state if ".attn." in key and key.endswith(".weight"))
        state[key] = state[key].float()
        save_file(state, path)
        with self.assertRaisesRegex(ValueError, "source must be BF16"):
            self.invoke()
        self.assertFalse((self.root / "merged").exists())

    def test_missing_adapter_pair_and_nan_fail_before_publication(self):
        path = self.adapters / "model_for_comfy_dit.safetensors"
        original = load_file(path)
        for mode in ("missing", "nan"):
            factors = dict(original)
            if mode == "missing":
                factors.pop(next(iter(factors)))
            else:
                factors[next(iter(factors))] = factors[next(iter(factors))].clone().fill_(float("nan"))
            save_file(factors, path)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.invoke()
            self.assertFalse((self.root / "merged").exists())

    def test_write_failure_leaves_no_complete_output_and_preserves_original(self):
        before = {path: path.read_bytes() for path in self.base.glob("*/*.safetensors")}
        with patch.object(MERGE, "_write_shard", side_effect=RuntimeError("simulated disk failure")), self.assertRaisesRegex(RuntimeError, "disk failure"):
            self.invoke()
        self.assertFalse((self.root / "merged").exists())
        self.assertEqual(list(self.root.glob(".merged.merge-*")), [])
        self.assertEqual(before, {path: path.read_bytes() for path in self.base.glob("*/*.safetensors")})


if __name__ == "__main__":
    unittest.main()
