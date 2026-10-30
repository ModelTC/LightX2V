"""Global curriculum coverage and resume under real H3 trainer interfaces."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.utils.data import Dataset

from lightx2v_train.data.minimax_h3_curriculum_cache_dataset import (
    MiniMaxH3CurriculumCacheDataset,
    MiniMaxH3CurriculumSampler,
)
from lightx2v_train.utils.registry import build_data


class MetadataDataset(Dataset):
    dataset_repeat = 1

    def __init__(self, rows, source):
        self.samples = [{"type": "metadata", "row": {**row, "condition_path": f"/cache/{source}/{index}.pt"}, "base_dir": "/cache"} for index, row in enumerate(rows)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"inputs": {}, "conditioning": {}, "meta": {"cache_index": index}}


def row(task="t2av", index=0, *, images=1, video=False, native=False):
    return {
        "task": task,
        "source_id": f"{task}:{index}",
        "cache_fingerprint": f"fingerprint:{task}:{index}",
        "target_height": 1024 if native else 768,
        "target_width": 992 if native else 1344,
        "num_frames": 124,
        "reference_image_count": 0 if video else images,
        "reference_video_count": int(video),
        "reference_audio_count": 0,
        "packed_sequence_tokens_124": 17000 + index,
    }


def fl_dataset(new_count=37, old_counts=(17, 41, 53)):
    new = MetadataDataset([row(index=index, native=index % 2 == 0) for index in range(new_count)], "new")
    old = MetadataDataset([row(task, index) for task, count in zip(("t2av", "i2av", "fl2av"), old_counts) for index in range(count)], "old")
    return MiniMaxH3CurriculumCacheDataset(new, old)


def sampler(dataset, rank=0, world=32, kind="fl2av", **kwargs):
    return MiniMaxH3CurriculumSampler(dataset, kind=kind, rank=rank, num_replicas=world, **kwargs)


class MiniMaxH3CurriculumTests(unittest.TestCase):
    def test_all_32_ranks_finish_full_new_then_full_old_without_dropping_tails(self):
        dataset = fl_dataset()
        samplers = [sampler(dataset, rank) for rank in range(32)]
        for item in samplers:
            item.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=2, fake_update_ratio=10)
        rounds = samplers[0].microbatches_per_round
        self.assertEqual(rounds, 6)
        observed = {"new": [], "old": []}
        for step in range(rounds):
            phases = {item.phase_at(step) for item in samplers}
            self.assertEqual(len(phases), 1)
            phase = phases.pop()
            values = [item.sample_index(step) for item in samplers]
            self.assertEqual(len(set(values)), 32)
            observed[phase].extend(values)
        self.assertEqual(set(observed["new"]), set(range(37)))
        self.assertEqual(set(observed["old"]), set(range(37, len(dataset))))
        self.assertEqual([samplers[0].phase_at(step) for step in range(rounds)], ["new", "new", "old", "old", "old", "old"])
        self.assertEqual(samplers[0].phase_at(rounds), "new")
        metadata = samplers[0].checkpoint_metadata()
        self.assertEqual(metadata["repeated_rows_per_round"], {"new": 27, "old": 17})

    def test_new_old_order_repeats_each_round_and_shuffles_each_source(self):
        dataset = fl_dataset()
        item = sampler(dataset, world=4)
        item.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=10)
        length = len(item)
        first = [item.sample_index(index) for index in range(length)]
        second = [item.sample_index(length + index) for index in range(length)]
        self.assertNotEqual(first, second)
        for start in (0, length, 2 * length):
            phases = [item.phase_at(start + index) for index in range(length)]
            boundary = phases.index("old")
            self.assertTrue(all(value == "new" for value in phases[:boundary]))
            self.assertTrue(all(value == "old" for value in phases[boundary:]))
        item.set_epoch(1)
        self.assertEqual(list(item), second)

    def test_official_pdmd_resume_is_exact_at_arbitrary_fresh_microbatch_offsets(self):
        dataset = fl_dataset()
        original = sampler(dataset, rank=7)
        original.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=2, fake_update_ratio=10)
        for offset in (1, 2, 5, 6, 13, 101_001):
            with self.subTest(offset=offset):
                resumed = sampler(dataset, rank=7)
                resumed.configure_from_microbatch_offset(start_microbatch=offset, gradient_accumulation_iters=2, fake_update_ratio=10)
                self.assertEqual(list(resumed), [original.sample_index(offset + index) for index in range(len(resumed))])
                resumed.set_epoch(1)
                self.assertEqual(list(resumed), [original.sample_index(offset + len(resumed) + index) for index in range(len(resumed))])
                self.assertEqual(original.checkpoint_metadata(), resumed.checkpoint_metadata())

    def test_ordinary_dmd_resume_counts_student_and_fake_fresh_batches_in_actual_order(self):
        dataset = fl_dataset()
        for order in ("student_first", "fake_first"):
            original = sampler(dataset, world=4, dmd_update_order=order)
            original.configure(start_iteration=0, gradient_accumulation_iters=2, fake_update_ratio=10)
            resumed = sampler(dataset, world=4, dmd_update_order=order)
            resumed.configure(start_iteration=9, gradient_accumulation_iters=2, fake_update_ratio=10)
            self.assertEqual(list(resumed), [original.sample_index(9 * 22 + index) for index in range(len(resumed))])
            order_stages = [("student", 0)] + [("fake", index) for index in range(10)]
            if order == "fake_first":
                order_stages = order_stages[1:] + order_stages[:1]
            for offset, (stage, fake_index) in enumerate(order_stages):
                for micro in range(2):
                    self.assertEqual(
                        original.microbatch_loss_scale(outer_iteration=9, stage=stage, micro_step=micro, fake_update_index=fake_index),
                        original.cursor_microbatch_loss_scale(9 * 22 + offset * 2 + micro),
                    )

    def test_old_three_tasks_rotate_11_11_10_while_available_then_cover_remaining_rows(self):
        dataset = fl_dataset(new_count=37, old_counts=(41, 71, 83))
        item = sampler(dataset)
        item.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=10)
        first_old = next(index for index in range(len(item)) if item.phase_at(index) == "old")
        counts = Counter(dataset.samples[item.sample_index(first_old, rank)]["row"]["task"] for rank in range(32))
        self.assertEqual(sorted(counts.values()), [10, 11, 11])
        for task, count in counts.items():
            values = [sampler(dataset, rank) for rank in range(32) if dataset.samples[item.sample_index(first_old, rank)]["row"]["task"] == task]
            for child in values:
                child.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=10)
            self.assertAlmostEqual(sum(child.cursor_microbatch_loss_scale(first_old) for child in values), 32 / 3)

    def test_new_ref_accepts_all_native_image_video_and_av_while_old_ref_is_image_only(self):
        new = MetadataDataset([row("ref2av", index, video=index % 2 == 1, native=True) for index in range(37)], "new")
        old = MetadataDataset([row("ref2av", index, images=1 + index % 6) for index in range(65)], "old")
        dataset = MiniMaxH3CurriculumCacheDataset(new, old)
        item = sampler(dataset, kind="ref2av", world=4)
        item.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=2, fake_update_ratio=10)
        observed = {"new": set(), "old": set()}
        for step in range(len(item)):
            batch = [item.sample_index(step, rank) for rank in range(4)]
            observed[item.phase_at(step)].update(batch)
            image_counts = {dataset.samples[index]["row"]["reference_image_count"] for index in batch}
            self.assertEqual(len(image_counts), 1)
        self.assertEqual(observed["new"], set(range(37)))
        self.assertEqual(observed["old"], set(range(37, 102)))
        self.assertFalse(item.is_minimax_h3_task_cycle_sampler)
        self.assertTrue(item.is_minimax_h3_ref_cost_sampler)
        self.assertTrue(all(item.cursor_microbatch_loss_scale(step) == 1 for step in range(len(item))))

    def test_actual_dmd_stage_uses_curriculum_route_scale_for_fake_first_consumption(self):
        from lightx2v_train.trainers.dmd.trainer import DmdTrainer

        item = sampler(fl_dataset(), world=4, dmd_update_order="fake_first")
        item.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=2)
        expected_scale = item.microbatch_loss_scale(outer_iteration=3, stage="student", micro_step=0)
        self.assertNotEqual(expected_scale, 1)
        trainer = DmdTrainer.__new__(DmdTrainer)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
        trainer.lr_scheduler = Mock()
        trainer.trainable_params = [parameter]
        trainer.max_grad_norm = 1e6
        trainer._set_student_gradient_sync = Mock()
        trainer._sync_sequence_parallel_grads = Mock()
        trainer._after_student_optimizer_step = Mock()
        trainer.real_data_fake_trick = SimpleNamespace(enabled_for=lambda region: False)
        trainer.diversity_trick = SimpleNamespace(enabled=False)
        trainer.dataloader_train = SimpleNamespace(sampler=item)
        trainer._encode_conditions = Mock(return_value=({}, None))
        trainer._latent_shape = Mock(return_value=None)
        trainer.sample_initial_latents = Mock(return_value=None)
        trainer.forward_loss = Mock(side_effect=lambda *args, **kwargs: {"loss": 2 * parameter})
        result = trainer._train_one_stage(iter([{}]), "student", 1, outer_iteration=3)
        torch.testing.assert_close(parameter, torch.tensor(1 - 0.2 * expected_scale))
        self.assertEqual(result["loss"], 2)

    def test_dataset_reports_source_and_local_index_without_changing_underlying_sample(self):
        dataset = fl_dataset()
        for index, expected in ((0, ("new", 0)), (37, ("old", 0)), (40, ("old", 3))):
            item = dataset[index]
            self.assertEqual((item["meta"]["curriculum_source"], item["meta"]["curriculum_source_index"]), expected)
        self.assertNotIn("curriculum_source", dataset.sources["old"][0]["meta"])

    def test_metadata_fingerprint_and_schedule_changes_are_visible_to_checkpoint_validation(self):
        dataset = fl_dataset()
        item = sampler(dataset)
        item.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=10)
        original = item.checkpoint_metadata()
        changed = copy.deepcopy(dataset)
        changed.samples[0]["row"]["cache_fingerprint"] = "different"
        rebuilt = MiniMaxH3CurriculumCacheDataset(changed.sources["new"], changed.sources["old"])
        other = sampler(rebuilt)
        other.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=10)
        self.assertNotEqual(original["dataset_fingerprint"], other.checkpoint_metadata()["dataset_fingerprint"])
        ordinary = sampler(dataset)
        ordinary.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=10)
        self.assertNotEqual(original["data_consumption_mode"], ordinary.checkpoint_metadata()["data_consumption_mode"])

    def test_registry_injects_actual_training_order_and_loads_both_metadata_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for source, rows in (("new", [row()]), ("old", [row(task, index) for index, task in enumerate(("t2av", "i2av", "fl2av"))])):
                path = root / f"{source}.jsonl"
                records = []
                for index, record in enumerate(rows):
                    condition = root / f"{source}_{index}.pt"
                    torch.save({"conditioning": {"positive": {"prompt_embeds": torch.ones(1, 2, 5120), "text_token_tags": torch.ones(2, dtype=torch.long)}}}, condition)
                    records.append({**record, "condition_path": str(condition)})
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
                paths[source] = str(path)
            config = {
                "data": {
                    "train": {
                        "name": "minimax_h3_curriculum_cache_dataset",
                        "kind": "fl2av",
                        "num_workers": 0,
                        "pin_memory": False,
                        "sources": {name: {"data_path": path} for name, path in paths.items()},
                    }
                },
                "training": {"method": "dmd", "dmd": {"update_order": "fake_first", "official_pdmd": True}},
            }
            loader = build_data(config, "train")
            self.assertEqual(loader.sampler.dmd_update_order, "fake_first")
            self.assertTrue(loader.sampler.official_pdmd)
            loader.sampler.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=10)
            item = next(iter(loader))
            self.assertEqual(item["meta"]["curriculum_source"], ["new"])
            self.assertEqual(config["data"]["train"].get("dmd_update_order"), None)

    def test_invalid_rows_or_configuration_are_rejected(self):
        dataset = fl_dataset()
        item = sampler(dataset)
        for offset in (-1, 0.5, True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                item.configure_from_microbatch_offset(start_microbatch=offset)
        malformed = copy.deepcopy(dataset)
        malformed.samples[0]["row"]["target_height"] = 769
        with self.assertRaisesRegex(ValueError, "32-aligned"):
            sampler(malformed)
        duplicate = MetadataDataset([row()], "new")
        with self.assertRaisesRegex(ValueError, "repeats one condition"):
            MiniMaxH3CurriculumCacheDataset(duplicate, duplicate)


if __name__ == "__main__":
    unittest.main()
