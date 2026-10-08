"""CPU regression for the original 32-rank Ref image/audio sampling stream.

The frozen reference below is the balanced-count branch of the pre-migration
``data/video_dataset.py`` sampler.  It deliberately lives in this repository:
running these tests must not require a checkout of the old training code.
"""

import unittest
from collections import Counter

import torch

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler


def _legacy_rows(max_count=5, with_audio=True):
    rows = []
    for count in range(1, max_count + 1):
        for orientation_index, orientation in enumerate(("landscape", "portrait")):
            # Unequal cells and non-divisible tails exercise actual rotating
            # downsampling. Equal costs also exercise stable randomized ties.
            for offset in range(34 + 7 * (max_count - count) + 3 * orientation_index):
                rows.append(
                    {
                        "condition_path": f"condition_{len(rows):04d}.pt",
                        "target_orientation": orientation,
                        "reference_image_count": count,
                        "reference_video_count": 0,
                        "reference_audio_count": offset % 4 if with_audio else 0,
                        "packed_sequence_tokens_124": 10_000 * count + offset // 3,
                    }
                )
    return rows


class _Dataset:
    def __init__(self, rows):
        self.samples = [{"type": "metadata", "row": row} for row in rows]


def _old_epoch(rows, epoch, seed=42, image_counts=tuple(range(1, 6))):
    """Frozen legacy uniform-count / 50:50 / rotating-drop policy, 32 ranks."""
    pools = {
        (count, orientation): [index for index, row in enumerate(rows) if row["reference_image_count"] == count and row["target_orientation"] == orientation]
        for count in image_counts
        for orientation in ("landscape", "portrait")
    }
    rows_per_cell = min(map(len, pools.values())) // 16 * 16
    paired = []
    for count_index, count in enumerate(image_counts):
        selected = {}
        for orientation_index, orientation in enumerate(("landscape", "portrait")):
            pool = pools[count, orientation]
            generator = torch.Generator().manual_seed(seed + 65_537 * (count_index + 1) + 8_191 * (orientation_index + 1))
            order = torch.randperm(len(pool), generator=generator).tolist()
            start = epoch * rows_per_cell % len(pool)
            subset = [pool[order[(start + offset) % len(pool)]] for offset in range(rows_per_cell)]
            generator = torch.Generator().manual_seed(seed + epoch * 1_000_003 + 10_007 * (count_index + 1) + 101 * orientation_index)
            order = torch.randperm(len(subset), generator=generator).tolist()
            subset = [subset[index] for index in order]
            subset.sort(key=lambda index: rows[index]["packed_sequence_tokens_124"])
            selected[orientation] = subset
        for start in range(0, rows_per_cell, 16):
            paired.append(selected["landscape"][start : start + 16] + selected["portrait"][start : start + 16])
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003 + 97_409)
    batches = []
    for batch_index in torch.randperm(len(paired), generator=generator).tolist():
        rank_order = torch.randperm(32, generator=generator).tolist()
        batches.append(tuple(paired[batch_index][rank] for rank in rank_order))
    return tuple(batches)


class LegacyRefShift12SamplerTest(unittest.TestCase):
    def sampler(self, *, rank=0, start=0, seed=42, accumulation=1):
        sampler = MiniMaxH3ReferenceCostSampler(
            _Dataset(_legacy_rows()),
            num_replicas=32,
            rank=rank,
            seed=seed,
            cost_key="packed_sequence_tokens_124",
            require_compute_cost=True,
            require_image_only=False,
            image_counts=(1, 2, 3, 4, 5),
            require_all_image_counts=True,
            balance_image_counts=True,
            balance_orientation=True,
            strict_full_epoch=False,
            remainder_policy="rotating_drop",
            batch_mode="cost_local",
        )
        sampler.configure(start_iteration=start, gradient_accumulation_iters=accumulation, fake_update_ratio=5)
        return sampler

    def test_all_rank_indices_match_old_implementation_across_epochs(self):
        rows = _legacy_rows()
        for seed in (42, 7):
            sampler = self.sampler(seed=seed)
            for epoch in range(5):
                with self.subTest(seed=seed, epoch=epoch):
                    self.assertEqual(sampler._batches_for_epoch(epoch), _old_epoch(rows, epoch, seed))

    def test_32_rank_batch_is_one_count_with_16_landscape_and_16_portrait(self):
        sampler = self.sampler()
        rows = _legacy_rows()
        covered = set()
        for epoch in range(5):
            indices = []
            for batch in sampler._batches_for_epoch(epoch):
                self.assertEqual(len(set(batch)), 32)
                self.assertEqual(len({rows[index]["reference_image_count"] for index in batch}), 1)
                self.assertEqual(Counter(rows[index]["target_orientation"] for index in batch), {"landscape": 16, "portrait": 16})
                indices.extend(batch)
            self.assertEqual(len(indices), len(set(indices)))
            self.assertEqual(Counter(rows[index]["reference_image_count"] for index in indices), {count: 64 for count in range(1, 6)})
            covered.update(indices)
        self.assertEqual(covered, set(range(len(rows))))

    def test_resume_advances_shared_student_then_five_fake_stream(self):
        # Ten batches per data epoch, six per outer iteration: both resumes
        # and role transitions routinely cross the dataloader epoch boundary.
        rows = _legacy_rows()
        reference = tuple(batch for epoch in range(12) for batch in _old_epoch(rows, epoch))
        for accumulation, start in ((1, 0), (1, 3), (1, 7), (2, 3)):
            for rank in (0, 11, 31):
                sampler = self.sampler(rank=rank, start=start, accumulation=accumulation)
                self.assertEqual(sampler.samples_per_outer_iteration, 6 * accumulation)
                for loader_epoch in (0, 1):
                    sampler.set_epoch(loader_epoch)
                    offset = start * 6 * accumulation + loader_epoch * sampler.num_global_batches
                    self.assertEqual(list(sampler), [batch[rank] for batch in reference[offset : offset + len(sampler)]])

    def test_checkpoint_preserves_legacy_sampler_contract(self):
        metadata = self.sampler().checkpoint_metadata()
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["route_mode"], "ref_cost_bucket")
        self.assertEqual(metadata["image_counts"], [1, 2, 3, 4, 5])
        self.assertEqual(metadata["data_parallel_world_size"], 32)
        self.assertEqual(metadata["gradient_accumulation_iters"], 1)
        self.assertEqual(metadata["fake_update_ratio"], 5)
        self.assertEqual(metadata["rows_per_image_orientation_cell"], 32)
        self.assertEqual(metadata["epoch_rows"], 320)
        self.assertNotIn("batch_mode", metadata)


class OmniLegacyBalancedSamplerTest(unittest.TestCase):
    """New Omni data uses the old strategy, not the old 1..5 count filter."""

    def sampler(self, rows=None, **overrides):
        options = dict(
            num_replicas=32,
            rank=0,
            seed=42,
            image_counts=list(range(1, 10)),
            require_all_image_counts=False,
            require_image_only=True,
            balance_image_counts=True,
            balance_orientation=True,
            strict_full_epoch=False,
            remainder_policy="rotating_drop",
            batch_mode="cost_local",
        )
        options.update(overrides)
        sampler = MiniMaxH3ReferenceCostSampler(_Dataset(_legacy_rows(6, False) if rows is None else rows), **options)
        sampler.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=5)
        return sampler

    def test_all_observed_counts_including_six_follow_old_policy(self):
        rows = _legacy_rows(6, False)
        sampler = self.sampler(rows)
        self.assertEqual(sampler.image_counts, (1, 2, 3, 4, 5, 6))
        self.assertEqual(sampler.rows_per_image_orientation_cell, 32)
        self.assertEqual(sampler.epoch_rows, 384)
        covered = set()
        for epoch in range(5):
            batches = sampler._batches_for_epoch(epoch)
            self.assertEqual(batches, _old_epoch(rows, epoch, image_counts=(1, 2, 3, 4, 5, 6)))
            indices = [index for batch in batches for index in batch]
            self.assertEqual(len(set(indices)), 384)
            self.assertEqual(Counter(rows[index]["reference_image_count"] for index in indices), {count: 64 for count in range(1, 7)})
            for batch in batches:
                self.assertEqual(len({rows[index]["reference_image_count"] for index in batch}), 1)
                self.assertEqual(Counter(rows[index]["target_orientation"] for index in batch), {"landscape": 16, "portrait": 16})
            covered.update(indices)
        self.assertEqual(covered, set(range(len(rows))))

    def test_missing_orientation_for_observed_count_is_not_silently_dropped(self):
        rows = [row for row in _legacy_rows(6, False) if not (row["reference_image_count"] == 6 and row["target_orientation"] == "portrait")]
        with self.assertRaisesRegex(ValueError, "every image-count/orientation cell"):
            self.sampler(rows)

    def test_sparse_count_orientation_requires_at_least_16_rows(self):
        rows = _legacy_rows(6, False)
        cell = [row for row in rows if row["reference_image_count"] == 6 and row["target_orientation"] == "portrait"]
        rows = [row for row in rows if row not in cell] + cell[:15]
        with self.assertRaisesRegex(ValueError, "every image-count/orientation cell"):
            self.sampler(rows)

    def test_absent_counts_7_to_9_are_allowed_but_audio_is_rejected(self):
        rows = _legacy_rows(6, False)
        self.assertEqual(self.sampler(rows).image_counts, (1, 2, 3, 4, 5, 6))
        rows[0]["reference_audio_count"] = 1
        with self.assertRaisesRegex(ValueError, "image-only"):
            self.sampler(rows)


if __name__ == "__main__":
    unittest.main()
