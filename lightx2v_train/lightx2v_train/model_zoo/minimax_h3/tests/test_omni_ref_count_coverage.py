import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, MiniMaxH3ReferenceCostSampler


class _Dataset:
    def __init__(self, sizes):
        self.samples = []
        for count, size in sizes.items():
            for offset in range(size):
                index = len(self.samples)
                self.samples.append(
                    {
                        "type": "metadata",
                        "row": {
                            "condition_path": f"condition_{index}.pt",
                            # Deliberately no portrait samples; native data need not
                            # have all image-count/orientation combinations.
                            "target_orientation": "landscape",
                            "reference_image_count": count,
                            "reference_video_count": 0,
                            "reference_audio_count": 0,
                            "packed_sequence_tokens_124": 1000 * count + offset,
                        },
                    }
                )


class OmniReferenceCountCoverageTest(unittest.TestCase):
    def _sampler(self, sizes=None, *, rank=0, world=4, start=0, **overrides):
        options = dict(
            num_replicas=world,
            rank=rank,
            seed=17,
            batch_mode="count_coverage",
            balance_orientation=False,
            balance_image_counts=False,
            strict_full_epoch=False,
            remainder_policy="rotating_drop",
            image_counts=list(range(1, 10)),
            require_all_image_counts=False,
        )
        options.update(overrides)
        sampler = MiniMaxH3ReferenceCostSampler(_Dataset(sizes or {1: 13, 2: 3, 5: 4}), **options)
        sampler.configure(start_iteration=start, gradient_accumulation_iters=2, fake_update_ratio=5)
        return sampler

    def test_preserves_unbalanced_natural_counts_and_maximizes_batch_coverage(self):
        sampler = self._sampler()
        batches = sampler._batches_for_epoch(0)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(20)))
        self.assertEqual(len(set(flattened)), 20)
        self.assertEqual(sampler.image_counts, (1, 2, 5))
        counts = Counter(sampler.records[index]["image_count"] for index in flattened)
        self.assertEqual(counts, {1: 13, 2: 3, 5: 4})
        presence = Counter(count for batch in batches for count in {sampler.records[index]["image_count"] for index in batch})
        self.assertEqual(presence, {1: 5, 2: 3, 5: 4})
        self.assertEqual(sampler.epoch_rows, 20)
        self.assertEqual(sampler.rows_per_orientation, None)

    def test_global_32_slots_cover_all_available_counts_when_enough_rows(self):
        sizes = {count: 8 for count in range(1, 10)}
        sizes[1] += 24
        sampler = self._sampler(sizes, world=32)
        self.assertEqual(sampler.epoch_rows, 96)
        for batch in sampler._batches_for_epoch(0):
            self.assertEqual(len(set(batch)), 32)
            self.assertEqual({sampler.records[index]["image_count"] for index in batch}, set(range(1, 10)))

    def test_rotating_tail_drops_less_than_world_and_eventually_covers_all(self):
        sampler = self._sampler({1: 15, 4: 4, 9: 3})
        self.assertEqual(sampler.epoch_rows, 20)
        selections = [{index for batch in sampler._batches_for_epoch(epoch) for index in batch} for epoch in range(2)]
        self.assertEqual([len(indices) for indices in selections], [20, 20])
        self.assertNotEqual(selections[0], selections[1])
        self.assertEqual(selections[0] | selections[1], set(range(22)))
        metadata = sampler.checkpoint_metadata()
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["dropped_rows"], 2)
        self.assertEqual(metadata["batch_mode"], "count_coverage")
        self.assertEqual(metadata["coverage_unit"], "global_data_parallel_microbatch")

    def test_rank_streams_and_resume_match_absolute_batch_offsets(self):
        samplers = [self._sampler(rank=rank) for rank in range(4)]
        actual = list(zip(*(list(iter(sampler)) for sampler in samplers)))
        self.assertEqual(actual, list(samplers[0]._batches_for_epoch(0)))
        resumed = self._sampler(start=2)
        self.assertEqual(list(iter(resumed)), [samplers[0].sample_index(offset) for offset in range(24, 29)])
        resumed.set_epoch(1)
        self.assertEqual(list(iter(resumed)), [samplers[0].sample_index(offset) for offset in range(29, 34)])

    def test_more_counts_than_slots_remains_unique_and_complete(self):
        sampler = self._sampler({count: 2 for count in range(1, 10)}, world=3)
        batches = sampler._batches_for_epoch(0)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(18)))
        for batch in batches:
            self.assertEqual(len(set(batch)), 3)

    def test_seed_and_epoch_are_deterministic(self):
        first = self._sampler()
        same = self._sampler()
        other = self._sampler(seed=18)
        self.assertEqual(first._batches_for_epoch(0), same._batches_for_epoch(0))
        self.assertNotEqual(first._batches_for_epoch(0), first._batches_for_epoch(1))
        self.assertNotEqual(first._batches_for_epoch(0), other._batches_for_epoch(0))

    def test_ref_image_count_alias(self):
        dataset = _Dataset({1: 4})
        for sample in dataset.samples:
            sample["row"]["ref_image_count"] = sample["row"].pop("reference_image_count")
        sampler = MiniMaxH3ReferenceCostSampler(
            dataset,
            num_replicas=4,
            rank=0,
            batch_mode="count_coverage",
            balance_orientation=False,
        )
        self.assertEqual(sampler.image_counts, (1,))

    def test_rejects_downsampling_options_and_empty_global_batch(self):
        with self.assertRaisesRegex(ValueError, "natural dataset frequencies"):
            self._sampler(balance_orientation=True)
        with self.assertRaisesRegex(ValueError, "natural dataset frequencies"):
            self._sampler(balance_image_counts=True)
        with self.assertRaisesRegex(ValueError, "at least one global microbatch"):
            self._sampler({1: 3})
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._sampler({1: 5}, strict_full_epoch=True, remainder_policy="strict")

    def test_condition_only_bf16_cache_loads_without_target_latents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition_path = root / "condition.pt"
            torch.save(
                {
                    "positive": {
                        "prompt_embeds": torch.ones(1, 7, 4, dtype=torch.bfloat16),
                        "text_token_tags": torch.ones(1, 7, dtype=torch.long),
                        "references": [{"kind": "image", "video_latents": torch.ones(1, 4, 2, dtype=torch.bfloat16)}],
                    }
                },
                condition_path,
            )
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "sample_id": "omni-1",
                        "condition_path": "condition.pt",
                        "caption": "fallback prompt",
                        "prompt_source": "prompt_en",
                        "ref_image_count": 1,
                        "reference_image_count": 1,
                        "reference_video_count": 0,
                        "reference_audio_count": 0,
                        "target_height": 768,
                        "target_width": 1344,
                        "target_num_frames": 124,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sample = LatentDataset(str(metadata))[0]
            self.assertFalse(sample["inputs"])
            self.assertEqual(sample["conditioning"]["positive"]["prompt_embeds"].dtype, torch.bfloat16)
            self.assertEqual(sample["conditioning"]["positive"]["text_token_tags"].dtype, torch.long)
            self.assertEqual(sample["conditioning"]["positive"]["references"][0]["video_latents"].dtype, torch.bfloat16)
            self.assertEqual(sample["meta"]["num_frames"], 124)
            self.assertEqual(sample["meta"]["ref_image_count"], 1)
            self.assertEqual(sample["meta"]["prompt_source"], "prompt_en")


if __name__ == "__main__":
    unittest.main()
