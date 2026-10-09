"""Same-count random DP batches without orientation quotas or cost grouping."""

import copy
import unittest
from collections import Counter

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler
from lightx2v_train.model_zoo.minimax_h3.tests.test_minimax_h3_ref_cost_sampler import _MetadataDataset, _rows, _uneven_rows


def _count_rows(sizes=(64, 96, 128, 160, 192, 224)):
    rows = []
    for count, size in enumerate(sizes, 1):
        for index in range(size):
            rows.append(
                {
                    "condition_path": f"count{count}_{index:05d}.pt",
                    # Count one has only landscape; the others are deliberately
                    # far from 50/50 and need not have sixteen portrait rows.
                    "target_orientation": "portrait" if count > 1 and index < count else "landscape",
                    "reference_image_count": count,
                    "reference_video_count": 0,
                    "reference_audio_count": 0,
                    "packed_sequence_tokens_124": 10_000 * count + index * 100,
                }
            )
    return rows


class MiniMaxH3CountRandomSamplerTest(unittest.TestCase):
    def sampler(self, rows=None, **overrides):
        options = {
            "num_replicas": 32,
            "rank": 0,
            "seed": 42,
            "image_counts": None,
            "batch_mode": "count_random",
            "balance_orientation": False,
            "balance_image_counts": True,
            "strict_full_epoch": False,
            "remainder_policy": "rotating_drop",
        }
        options.update(overrides)
        sampler = MiniMaxH3ReferenceCostSampler(_MetadataDataset(_count_rows() if rows is None else rows), **options)
        sampler.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=1)
        return sampler

    @staticmethod
    def batches(sampler, epoch):
        return [tuple(sampler.sample_index(epoch * sampler.num_global_batches + index, rank=rank) for rank in range(sampler.num_replicas)) for index in range(sampler.num_global_batches)]

    def test_dp32_balances_counts_but_accepts_skewed_and_single_orientation_buckets(self):
        rows = _count_rows()
        sampler = self.sampler(rows)
        self.assertEqual(sampler.rows_per_image_count, dict.fromkeys(range(1, 7), 64))
        self.assertEqual(sampler.epoch_rows, 384)
        self.assertEqual(sampler.num_global_batches, 12)
        self.assertIsNone(sampler.rows_per_orientation)
        self.assertIsNone(sampler.rows_per_image_orientation_cell)
        for epoch in range(4):
            batches = self.batches(sampler, epoch)
            flat = [index for batch in batches for index in batch]
            self.assertEqual(len(flat), len(set(flat)))
            counts = Counter()
            for batch in batches:
                image_counts = {rows[index]["reference_image_count"] for index in batch}
                self.assertEqual(len(image_counts), 1)
                counts.update(image_counts)
                self.assertEqual(len(batch), 32)
                self.assertLessEqual(sum(rows[index]["target_orientation"] == "portrait" for index in batch), 6)
            self.assertEqual(counts, dict.fromkeys(range(1, 7), 2))

    def test_majority_selection_rotates_and_covers_every_physical_row(self):
        rows = _count_rows()
        sampler = self.sampler(rows)
        selections = [{index for batch in self.batches(sampler, epoch) for index in batch} for epoch in range(4)]
        self.assertNotEqual(selections[0], selections[1])
        self.assertEqual(set.union(*selections), set(range(len(rows))))
        # Count two has 96 rows and a quota of 64: over its three-epoch cycle
        # every row occurs twice, including its scarce portrait examples.
        frequencies = Counter(index for epoch in range(3) for batch in self.batches(sampler, epoch) for index in batch if rows[index]["reference_image_count"] == 2)
        self.assertEqual(len(frequencies), 96)
        self.assertEqual(set(frequencies.values()), {2})

    def test_orientation_and_cost_values_never_control_selection_or_grouping(self):
        rows = _count_rows()
        altered = copy.deepcopy(rows)
        for index, row in enumerate(altered):
            row["target_orientation"] = "portrait"
            row["packed_sequence_tokens_124"] = 1_000_000 - index * 17
        baseline, changed = self.sampler(rows), self.sampler(altered)
        self.assertNotEqual(baseline.dataset_fingerprint, changed.dataset_fingerprint)
        for epoch in range(3):
            self.assertEqual(self.batches(baseline, epoch), self.batches(changed, epoch))
        # Rows have strictly increasing costs within each count. A batch
        # spanning more than one sorted 32-row window proves no cost-local sort.
        self.assertTrue(any(max(batch) - min(batch) > 31 for batch in self.batches(baseline, 0)))

    def test_seed_epoch_rank_and_resume_are_deterministic_across_epoch_boundaries(self):
        original = self.sampler()
        identical = self.sampler()
        self.assertEqual(self.batches(original, 2), self.batches(identical, 2))
        self.assertNotEqual(self.batches(original, 0), self.batches(self.sampler(seed=7), 0))
        self.assertNotEqual(self.batches(original, 0), self.batches(original, 1))
        for rank in (0, 7, 31):
            resumed = self.sampler(rank=rank)
            resumed.configure(start_iteration=5, gradient_accumulation_iters=2, fake_update_ratio=1)
            for loader_epoch in (0, 1):
                resumed.set_epoch(loader_epoch)
                begin = 20 + loader_epoch * original.num_global_batches
                expected = [original.sample_index(index, rank=rank) for index in range(begin, begin + original.num_global_batches)]
                self.assertEqual(list(resumed), expected)

    def test_quota_uses_count_totals_not_orientation_cells_and_rounds_down(self):
        sampler = self.sampler(_count_rows((65, 98)))
        self.assertEqual(sampler.rows_per_image_count, {1: 64, 2: 64})
        self.assertEqual(sampler.dropped_per_image_count, {1: 1, 2: 34})
        self.assertEqual(sampler.num_global_batches, 4)
        metadata = sampler.checkpoint_metadata()
        self.assertEqual(metadata["route_mode"], "ref_count_random")
        self.assertEqual(metadata["schema_version"], 4)
        self.assertEqual(metadata["batch_mode"], "count_random")
        self.assertFalse(metadata["balance_orientation"])
        self.assertEqual(metadata["orientation_sampling"], "natural_within_count")
        self.assertEqual(metadata["cost_grouping"], "none")
        self.assertEqual(metadata["rows_per_image_count"], {1: 64, 2: 64})
        self.assertEqual(metadata["dropped_rows"], 35)
        self.assertNotIn("rows_per_orientation", metadata)
        self.assertNotIn("rows_per_image_orientation_cell", metadata)
        summary = sampler.global_batch_summary(0)
        self.assertEqual(len(summary["image_counts"]), 1)
        self.assertEqual(sum(summary["orientations"].values()), 32)

    def test_nonbalanced_strict_retains_all_rows_and_per_count_frequencies(self):
        rows = _count_rows((64, 96))
        sampler = self.sampler(rows, balance_image_counts=False, strict_full_epoch=True, remainder_policy="strict")
        batches = self.batches(sampler, 0)
        flat = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flat), list(range(len(rows))))
        self.assertEqual(Counter(rows[batch[0]]["reference_image_count"] for batch in batches), {1: 2, 2: 3})
        self.assertEqual(Counter(rows[index]["target_orientation"] for index in flat), Counter(row["target_orientation"] for row in rows))

    def test_nonbalanced_rotating_drop_is_per_count_and_has_no_mixed_tail(self):
        rows = _count_rows((65, 98))
        sampler = self.sampler(rows, balance_image_counts=False)
        self.assertEqual(sampler.rows_per_image_count, {1: 64, 2: 96})
        self.assertEqual(sampler.dropped_per_image_count, {1: 1, 2: 2})
        for batch in self.batches(sampler, 0):
            self.assertEqual(len({rows[index]["reference_image_count"] for index in batch}), 1)
        covered = {index for epoch in range(2) for batch in self.batches(sampler, epoch) for index in batch}
        self.assertEqual(covered, set(range(len(rows))))

    def test_observed_counts_need_not_cover_configured_range_when_explicitly_allowed(self):
        rows = [row for row in _count_rows() if row["reference_image_count"] in (1, 3, 6)]
        sampler = self.sampler(rows, image_counts=range(1, 7), require_all_image_counts=False)
        self.assertEqual(sampler.image_counts, (1, 3, 6))
        self.assertEqual(sampler.rows_per_image_count, {1: 64, 3: 64, 6: 64})

    def test_rejects_small_balanced_bucket_and_incompatible_policies(self):
        with self.assertRaisesRegex(ValueError, "every observed image-count bucket.*dp_world_size=32"):
            self.sampler(_count_rows((31, 96)))
        with self.assertRaisesRegex(ValueError, "balance_orientation=false"):
            self.sampler(balance_orientation=True)
        with self.assertRaisesRegex(ValueError, "balance_image_counts=true requires"):
            self.sampler(strict_full_epoch=True, remainder_policy="strict")
        with self.assertRaisesRegex(ValueError, "every image-count bucket.*divisible"):
            self.sampler(_count_rows((65, 98)), balance_image_counts=False, strict_full_epoch=True, remainder_policy="strict")

    def test_old_modes_keep_exact_prechange_batch_streams_and_metadata_versions(self):
        cases = (
            (_rows(), {}, 1, ((8, 13, 12, 9), (10, 15, 11, 14), (0, 1, 4, 5), (6, 3, 7, 2))),
            (_uneven_rows(), {"strict_full_epoch": False, "remainder_policy": "rotating_drop"}, 2, ((19, 20, 13, 14), (15, 22, 16, 23), (18, 17, 12, 11), (8, 2, 6, 0), (10, 3, 4, 9))),
            (_uneven_rows(), {"strict_full_epoch": False, "remainder_policy": "rotating_drop", "balance_image_counts": True}, 2, ((11, 20, 17, 12), (14, 23, 16, 22), (0, 1, 5, 6), (8, 4, 10, 2))),
            (
                _uneven_rows(),
                {"batch_mode": "count_coverage", "balance_orientation": False, "strict_full_epoch": False, "remainder_policy": "rotating_drop"},
                3,
                ((7, 20, 1, 19), (23, 10, 16, 4), (5, 18, 17, 11), (22, 9, 3, 21), (8, 14, 15, 2), (6, 13, 0, 12)),
            ),
        )
        for rows, options, version, expected in cases:
            with self.subTest(options=options):
                sampler = MiniMaxH3ReferenceCostSampler(_MetadataDataset(rows), num_replicas=4, rank=0, seed=42, image_counts=(1, 2), **options)
                self.assertEqual(sampler._batches_for_epoch(0), expected)
                metadata = sampler.checkpoint_metadata(gradient_accumulation_iters=1, fake_update_ratio=1)
                self.assertEqual(metadata["schema_version"], version)
                self.assertEqual(metadata["route_mode"], "ref_cost_bucket")
                if version < 3:
                    self.assertNotIn("batch_mode", metadata)
                self.assertNotIn("rows_per_image_count", metadata)


if __name__ == "__main__":
    unittest.main()
