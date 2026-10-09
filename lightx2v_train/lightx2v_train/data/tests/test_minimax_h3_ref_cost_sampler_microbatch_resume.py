"""Absolute data-cursor resume for single-role PDMD optimizer updates (CPU)."""

import unittest
from unittest.mock import patch

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler


class _MetadataDataset:
    def __init__(self):
        self.samples = []
        # Unequal non-divisible buckets exercise rotating drops and multiple
        # data epochs; natural orientations remain intentionally skewed.
        for image_count, size in enumerate((9, 14, 17), 1):
            for offset in range(size):
                self.samples.append(
                    {
                        "type": "metadata",
                        "row": {
                            "condition_path": f"/unused/count{image_count}_{offset}.pt",
                            "target_orientation": "portrait" if offset < image_count else "landscape",
                            "reference_image_count": image_count,
                            "reference_video_count": 0,
                            "reference_audio_count": 0,
                            "packed_sequence_tokens_124": image_count * 1000 + offset,
                        },
                    }
                )


class ReferenceCostSamplerMicrobatchResumeTest(unittest.TestCase):
    def setUp(self):
        # Hundreds of tiny sampler configurations should not flood test logs.
        logging = patch("lightx2v_train.data.minimax_h3_cache_dataset.logger.info")
        logging.start()
        self.addCleanup(logging.stop)

    def sampler(self, *, rank=0, balance_image_counts=True):
        return MiniMaxH3ReferenceCostSampler(
            _MetadataDataset(),
            num_replicas=4,
            rank=rank,
            seed=42,
            batch_mode="count_random",
            balance_orientation=False,
            balance_image_counts=balance_image_counts,
            strict_full_epoch=False,
            remainder_policy="rotating_drop",
        )

    @staticmethod
    def collect_loader_epochs(sampler, epochs):
        result = []
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            result.extend(sampler)
        return result

    def test_iteration_still_requires_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "configure"):
            list(self.sampler())

    def test_legacy_configure_retains_outer_iteration_and_epoch_offsets(self):
        for rank in range(4):
            sampler = self.sampler(rank=rank)
            sampler.configure(start_iteration=3, gradient_accumulation_iters=2, fake_update_ratio=5)
            self.assertIsNone(sampler.start_microbatch)
            self.assertEqual(sampler.samples_per_outer_iteration, 12)
            for epoch in (0, 1, 3):
                sampler.set_epoch(epoch)
                start = 36 + epoch * sampler.num_global_batches
                self.assertEqual(list(sampler), [sampler.sample_index(index) for index in range(start, start + len(sampler))])
            self.assertNotIn("data_consumption_mode", sampler.checkpoint_metadata())

    def test_arbitrary_absolute_offsets_and_set_epoch_match_direct_indexing(self):
        for balanced in (False, True):
            for rank in range(4):
                sampler = self.sampler(rank=rank, balance_image_counts=balanced)
                batches = sampler.num_global_batches
                for offset in (0, 1, batches - 1, batches + 1, 2 * batches + 3):
                    sampler.configure_from_microbatch_offset(start_microbatch=offset, gradient_accumulation_iters=3, fake_update_ratio=5)
                    self.assertEqual(sampler.start_microbatch, offset)
                    self.assertIsNone(sampler.samples_per_outer_iteration)
                    for epoch in (0, 1, 3):
                        with self.subTest(balanced=balanced, rank=rank, offset=offset, epoch=epoch):
                            sampler.set_epoch(epoch)
                            start = offset + epoch * batches
                            expected = [sampler.sample_index(index) for index in range(start, start + batches)]
                            self.assertEqual(list(sampler), expected)

    def test_resumed_stream_is_full_uninterrupted_suffix_across_epochs(self):
        for balanced in (False, True):
            for rank in range(4):
                uninterrupted = self.sampler(rank=rank, balance_image_counts=balanced)
                uninterrupted.configure(start_iteration=0, gradient_accumulation_iters=2, fake_update_ratio=5)
                full = self.collect_loader_epochs(uninterrupted, 12)
                for offset in (1, len(uninterrupted) - 1, len(uninterrupted) + 1, 17):
                    resumed = self.sampler(rank=rank, balance_image_counts=balanced)
                    resumed.configure_from_microbatch_offset(start_microbatch=offset, gradient_accumulation_iters=2, fake_update_ratio=5)
                    actual = self.collect_loader_epochs(resumed, 3)
                    self.assertEqual(actual, full[offset : offset + 3 * len(resumed)])

    def test_official_five_critic_one_student_updates_consume_only_critic_data(self):
        for accumulation in (1, 2, 3):
            for rank in range(4):
                baseline = self.sampler(rank=rank)
                consumed = 0
                for completed_updates in range(25):
                    expected_cursor = (completed_updates - completed_updates // 6) * accumulation
                    self.assertEqual(consumed, expected_cursor)
                    resumed = self.sampler(rank=rank)
                    resumed.configure_from_microbatch_offset(
                        start_microbatch=expected_cursor,
                        gradient_accumulation_iters=accumulation,
                        fake_update_ratio=5,
                    )
                    actual = self.collect_loader_epochs(resumed, 2)
                    expected = [baseline.sample_index(index) for index in range(consumed, consumed + 2 * len(resumed))]
                    self.assertEqual(actual, expected)
                    # Updates 6, 12, ... reuse the last critic trajectory.
                    if (completed_updates + 1) % 6:
                        consumed += accumulation
                self.assertEqual((5 - 5 // 6) * accumulation, (6 - 6 // 6) * accumulation)

    def test_new_configuration_does_not_change_count_random_batches_or_quotas(self):
        legacy, explicit = self.sampler(), self.sampler()
        legacy.configure(start_iteration=2, gradient_accumulation_iters=2, fake_update_ratio=5)
        explicit.configure_from_microbatch_offset(start_microbatch=24, gradient_accumulation_iters=2, fake_update_ratio=5)
        self.assertEqual(list(legacy), list(explicit))
        self.assertEqual(legacy.rows_per_image_count, explicit.rows_per_image_count)
        self.assertEqual(legacy.dropped_per_image_count, explicit.dropped_per_image_count)
        for data_epoch in range(4):
            self.assertEqual(legacy._batches_for_epoch(data_epoch), explicit._batches_for_epoch(data_epoch))
            batches = explicit._batches_for_epoch(data_epoch)
            indices = [index for batch in batches for index in batch]
            self.assertEqual(len(indices), len(set(indices)))
            for batch in batches:
                counts = {explicit.records[index]["image_count"] for index in batch}
                self.assertEqual(len(counts), 1)

    def test_metadata_distinguishes_consumption_mode_but_not_resume_progress(self):
        legacy, explicit, resumed = self.sampler(), self.sampler(), self.sampler()
        legacy.configure(start_iteration=0, gradient_accumulation_iters=2, fake_update_ratio=5)
        explicit.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=2, fake_update_ratio=5)
        resumed.configure_from_microbatch_offset(start_microbatch=47, gradient_accumulation_iters=2, fake_update_ratio=5)
        legacy_metadata, explicit_metadata = legacy.checkpoint_metadata(), explicit.checkpoint_metadata()
        self.assertNotEqual(legacy_metadata, explicit_metadata)
        self.assertEqual(explicit_metadata, resumed.checkpoint_metadata())
        new_fields = {
            "data_consumption_mode": "optimizer_update_critic_only",
            "student_data_source": "reuse_preceding_critic_trajectory",
            "resume_cursor_unit": "absolute_global_microbatch",
        }
        self.assertEqual(explicit_metadata, {**legacy_metadata, **new_fields})
        self.assertEqual(explicit_metadata["schema_version"], 4)
        self.assertEqual(explicit_metadata["batch_mode"], "count_random")

    def test_reconfiguration_explicitly_restores_legacy_mode(self):
        sampler = self.sampler()
        sampler.configure_from_microbatch_offset(start_microbatch=47, gradient_accumulation_iters=3, fake_update_ratio=5)
        sampler.configure(start_iteration=2, gradient_accumulation_iters=2, fake_update_ratio=5)
        self.assertIsNone(sampler.start_microbatch)
        self.assertNotIn("data_consumption_mode", sampler.checkpoint_metadata())
        self.assertEqual(list(sampler), [sampler.sample_index(index) for index in range(24, 24 + len(sampler))])

    def test_invalid_explicit_configuration_is_rejected(self):
        for changes, message in (({"start_microbatch": -1}, "start_microbatch"), ({"gradient_accumulation_iters": 0}, "gradient_accumulation_iters"), ({"fake_update_ratio": 0}, "fake_update_ratio")):
            sampler = self.sampler()
            options = {"start_microbatch": 0, "gradient_accumulation_iters": 2, "fake_update_ratio": 5, **changes}
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                sampler.configure_from_microbatch_offset(**options)


if __name__ == "__main__":
    unittest.main()
