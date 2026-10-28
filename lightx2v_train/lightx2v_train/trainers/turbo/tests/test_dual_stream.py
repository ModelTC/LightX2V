"""Separate cached T2AV consistency targets from online three-task conditions."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from lightx2v_train.trainers.dmd.trainer import DmdTrainer
from lightx2v_train.trainers.turbo.config import TurboConfig
from lightx2v_train.trainers.turbo.tests.test_trainer import CursorSampler, Joint, make_trainer, save_toy_roles, state_snapshot


def dual_stream(trainer, stream):
    loader = trainer.dataloader_cm if stream == "cm" else trainer.dataloader_train
    sampler = loader.sampler
    while True:
        index = sampler.cursor
        sampler.cursor += 1
        rows = trainer.cm_read_ids if stream == "cm" else trainer.read_ids
        rows.append(index)
        sample = {
            "index": index,
            "stream": stream,
            "shape": "toy_av",
            "condition": -0.2 - index * 0.001 if stream == "cm" else 0.3 + index * 0.001,
            "task": "t2av" if stream == "cm" else ("t2av", "i2av", "fl2av")[index % 3],
        }
        if stream == "cm":
            generator = torch.Generator().manual_seed(18000 + index)
            sample["clean"] = Joint(torch.randn(1, 2, 2, generator=generator), torch.randn(1, 3, 2, generator=generator), sample["shape"])
        # Online rows deliberately contain no video/audio clean target at all.
        yield sample


def make_dual_trainer(output, *, auto_resume=False, accumulation=2):
    trainer = make_trainer(output, auto_resume=auto_resume, accumulation=accumulation)
    trainer.public_config["model"]["name"] = "minimax_h3_t2av"
    trainer.public_config["data"]["cm"] = {"name": "minimax_h3_cm_dataset", "data_path": "toy_cm", "tasks": ["t2av"], "batch_size": 1}
    trainer.public_config["data"]["train"].update(name="minimax_h3_cache_dataset", data_path="toy_online", tasks=["t2av", "i2av", "fl2av"], route_mode="stratified")
    trainer.config = copy.deepcopy(trainer.public_config)
    trainer.consumed_cm_microbatches = 0
    trainer.cm_read_ids, trainer.extracted_cm_ids, trainer.noise_reference_ids, trainer.objective_conditions = [], [], [], []
    trainer.dataloader_cm = SimpleNamespace(sampler=CursorSampler(), dataset=SimpleNamespace(samples=[{"type": "metadata", "row": {"stream": "cm", "task": "t2av", "index": i}} for i in range(32)]))
    trainer.dataloader_train.dataset.samples = [{"type": "metadata", "row": {"stream": "online", "task": ("t2av", "i2av", "fl2av")[i % 3], "index": i}} for i in range(32)]
    trainer._iter_train_samples = lambda: dual_stream(trainer, "online")
    trainer._iter_cm_samples = lambda: dual_stream(trainer, "cm")
    trainer._encode_conditions = lambda sample: (sample["condition"], None)
    trainer._latent_shape = lambda sample: sample["shape"]

    original_extract = trainer.student.extract_turbo_latents

    def extract(sample, shape):
        assert sample["stream"] == "cm", "Online condition rows must never be used as CM targets"
        trainer.extracted_cm_ids.append(sample["index"])
        return original_extract(sample, shape)

    def reference(owner, sample):
        assert sample["stream"] == "online" and "clean" not in sample
        owner.noise_reference_ids.append(sample["index"])
        return Joint(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2), sample["shape"])

    trainer.student.extract_turbo_latents = extract
    trainer._noise_reference = MethodType(reference, trainer)
    for objective in ("dcm", "scm", "dmd", "critic"):
        original = getattr(trainer, f"_{objective}_loss")

        def record(clean, condition, *args, _original=original, _name=objective):
            trainer.objective_conditions.append((_name, condition))
            if _name in ("dcm", "scm"):
                assert condition < 0, "CM must use its own paired T2AV condition"
            else:
                assert condition > 0, "DMD/critic must use online three-task conditions"
                assert not bool(clean.video.any()) and not bool(clean.audio.any())
            return _original(clean, condition, *args)

        setattr(trainer, f"_{objective}_loss", record)
    return trainer


def dual_snapshot(trainer):
    return dict(state_snapshot(trainer), consumed_cm_microbatches=trainer.consumed_cm_microbatches)


class DualStreamTests(unittest.TestCase):
    def setUp(self):
        self.setup_patch = patch.object(DmdTrainer, "setup", autospec=True)
        self.save_patch = patch.object(DmdTrainer, "save_checkpoint", new=save_toy_roles)
        self.setup_patch.start()
        self.save_patch.start()
        self.addCleanup(self.setup_patch.stop)
        self.addCleanup(self.save_patch.stop)

    def assert_nested_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right, strict=True):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_default_32_gpu_sample_counts(self):
        config = TurboConfig.from_mapping({"training": {"method": "turbo", "max_train_iters": 7000}, "model": {"name": "minimax_h3_t2av"}})
        for accumulation in (1, 2):
            self.assertEqual(config.consumed_cm_microbatches(7000, accumulation), 7000 * accumulation)
            self.assertEqual(config.consumed_dmd_microbatches(2500, accumulation), 0)
            self.assertEqual(config.consumed_dmd_microbatches(2501, accumulation), 5 * accumulation)
            self.assertEqual(config.consumed_dmd_microbatches(7000, accumulation), 22500 * accumulation)
        self.assertEqual(32 * (config.consumed_cm_microbatches(7000) + config.consumed_dmd_microbatches(7000)), 944000)

    def test_two_distinct_streams_for_real_losses_accumulation_and_sync(self):
        with TemporaryDirectory() as output:
            trainer = make_dual_trainer(output)
            trainer.train()
            self.assertEqual(trainer.cm_read_ids, list(range(10)))
            self.assertEqual(trainer.read_ids, list(range(20)))
            self.assertEqual(trainer.extracted_cm_ids, trainer.cm_read_ids)
            self.assertEqual(trainer.noise_reference_ids, trainer.read_ids)
            self.assertEqual(trainer.consumed_cm_microbatches, 10)
            self.assertEqual(trainer.consumed_microbatches, 20)
            self.assertEqual(trainer.fake_updates, 8)
            self.assertEqual(trainer.student_sync, [False, True] * 3 + [False, False, False, True] * 2)
            self.assertEqual(trainer.fake_sync, [False, True] * 8)
            counts = {objective: sum(name == objective for name, _ in trainer.objective_conditions) for objective in ("dcm", "scm", "dmd", "critic")}
            self.assertEqual(counts, {"dcm": 4, "scm": 6, "dmd": 4, "critic": 16})
            directory = Path(output) / "checkpoint-000000005"
            marker = json.loads((directory / "turbo.complete.json").read_text())
            receipt = json.loads((directory / "turbo.rank00000.json").read_text())
            self.assertEqual(marker["version"], 2)
            self.assertEqual(receipt["consumed_cm_microbatches"], 10)
            self.assertEqual(receipt["consumed_microbatches"], 20)
            self.assertIn("cm_dataset", receipt["recipe"])
            self.assertIn("cm_data", receipt["recipe"])

    def test_cm_stages_do_not_consume_online_conditions(self):
        with TemporaryDirectory() as output:
            trainer = make_dual_trainer(output)
            trainer.max_train_iters = 3
            trainer.train()
            self.assertEqual(trainer.cm_read_ids, list(range(6)))
            self.assertEqual(trainer.read_ids, [])
            self.assertEqual(trainer.noise_reference_ids, [])
            self.assertEqual(trainer.consumed_microbatches, 0)
            self.assertEqual(trainer.consumed_cm_microbatches, 6)

    def test_both_data_cursors_and_full_training_state_resume_exactly(self):
        with TemporaryDirectory() as root:
            full = make_dual_trainer(Path(root) / "full")
            full.train()
            expected = dual_snapshot(full)
            for boundary in (2, 3, 4):
                with self.subTest(boundary=boundary):
                    output = Path(root) / f"split{boundary}"
                    first = make_dual_trainer(output)
                    first.max_train_iters = boundary
                    first.train()
                    second = make_dual_trainer(output, auto_resume=True)
                    torch.manual_seed(15555)
                    second.train()
                    self.assert_nested_equal(expected, dual_snapshot(second))
                    self.assertEqual(first.read_ids + second.read_ids, full.read_ids)
                    self.assertEqual(first.cm_read_ids + second.cm_read_ids, full.cm_read_ids)
                    self.assertEqual(second.dataloader_train.sampler.configuration["start_microbatch"], first.consumed_microbatches)
                    self.assertEqual(second.dataloader_cm.sampler.configuration["start_microbatch"], first.consumed_cm_microbatches)

    def test_each_fingerprint_and_each_cursor_are_independently_validated(self):
        with TemporaryDirectory() as output:
            trainer = make_dual_trainer(output)
            trainer.max_train_iters = 3
            trainer.train()
            directory = Path(output) / "checkpoint-000000003"
            for stream in ("cm", "online"):
                with self.subTest(stream=stream):
                    resumed = make_dual_trainer(output, auto_resume=True)
                    loader = resumed.dataloader_cm if stream == "cm" else resumed.dataloader_train
                    loader.dataset.samples.reverse()
                    with self.assertRaisesRegex(RuntimeError, "recipe, counters or data cursor"):
                        resumed._resume_cursor(str(directory), 3)
            sidecar = directory / "turbo.rank00000.json"
            original = sidecar.read_text()
            for cursor in ("consumed_microbatches", "consumed_cm_microbatches"):
                with self.subTest(cursor=cursor):
                    bad = json.loads(original)
                    bad[cursor] += 1
                    sidecar.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(RuntimeError, "recipe, counters or data cursor"):
                        make_dual_trainer(output, auto_resume=True)._resume_cursor(str(directory), 3)
            sidecar.write_text(original)

    def test_dual_checkpoint_cannot_resume_as_legacy_single_stream(self):
        with TemporaryDirectory() as output:
            trainer = make_dual_trainer(output)
            trainer.max_train_iters = 2
            trainer.train()
            directory = Path(output) / "checkpoint-000000002"
            with self.assertRaisesRegex(RuntimeError, "completion metadata mismatch"):
                make_trainer(output, auto_resume=True)._resume_cursor(str(directory), 2)


if __name__ == "__main__":
    unittest.main()
