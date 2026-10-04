import unittest
from collections import Counter

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler


class _MetadataDataset:
    def __init__(self, rows):
        self.samples = [{"type": "metadata", "row": row, "base_dir": "/unused"} for row in rows]


def _rows():
    rows = []
    index = 0
    for image_count in (1, 2):
        for orientation in ("landscape", "portrait"):
            for offset in range(4):
                rows.append(
                    {
                        "condition_path": f"condition_{index:04d}.pt",
                        "target_height": 768 if orientation == "landscape" else 1344,
                        "target_width": 1344 if orientation == "landscape" else 768,
                        "num_frames": 107 + 17 * offset,
                        "target_orientation": orientation,
                        "reference_image_count": image_count,
                        "reference_video_count": 0,
                        "reference_audio_count": 0,
                        "packed_sequence_tokens_124": image_count * 10_000 + offset,
                    }
                )
                index += 1
    return rows


def _uneven_rows():
    """Three cost buckets with unequal, non-divisible orientation totals."""

    rows = []
    index = 0
    cell_sizes = {
        (1, "landscape"): 5,
        (1, "portrait"): 6,
        (2, "landscape"): 6,
        (2, "portrait"): 7,
    }
    for (image_count, orientation), count in cell_sizes.items():
        for offset in range(count):
            rows.append(
                {
                    "condition_path": f"uneven_condition_{index:04d}.pt",
                    "target_orientation": orientation,
                    "reference_image_count": image_count,
                    "reference_video_count": 0,
                    "reference_audio_count": 0,
                    "packed_sequence_tokens_124": image_count * 10_000 + offset,
                }
            )
            index += 1
    return rows


def _image_audio_uneven_rows():
    rows = []
    index = 0
    for image_count, per_orientation in ((1, 9), (2, 7), (3, 5)):
        for orientation in ("landscape", "portrait"):
            for offset in range(per_orientation):
                rows.append(
                    {
                        "condition_path": f"mixed_condition_{index:04d}.pt",
                        "target_orientation": orientation,
                        "reference_image_count": image_count,
                        "reference_video_count": 0,
                        "reference_audio_count": 1 if offset % 2 else 0,
                        "packed_sequence_tokens_124": image_count * 10_000 + offset,
                    }
                )
                index += 1
    return rows


class MiniMaxH3ReferenceCostSamplerTest(unittest.TestCase):
    def _sampler(self, rank=0):
        sampler = MiniMaxH3ReferenceCostSampler(
            _MetadataDataset(_rows()),
            num_replicas=4,
            rank=rank,
            seed=42,
            image_counts=(1, 2),
        )
        sampler.configure(
            start_iteration=0,
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
        )
        return sampler

    def test_global_batches_are_unique_balanced_and_cost_local(self):
        samplers = [self._sampler(rank) for rank in range(4)]
        rank_indices = [list(iter(sampler)) for sampler in samplers]
        self.assertEqual({len(indices) for indices in rank_indices}, {4})

        rows = _rows()
        all_indices = []
        for global_batch in zip(*rank_indices):
            self.assertEqual(len(set(global_batch)), 4)
            all_indices.extend(global_batch)
            batch_rows = [rows[index] for index in global_batch]
            self.assertEqual(
                sorted(row["target_orientation"] for row in batch_rows),
                ["landscape", "landscape", "portrait", "portrait"],
            )
            self.assertEqual(
                len({row["reference_image_count"] for row in batch_rows}),
                1,
            )
        self.assertEqual(set(all_indices), set(range(len(rows))))
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_resume_uses_absolute_outer_iteration_offset(self):
        uninterrupted = self._sampler(rank=0)
        expected = [uninterrupted.sample_index(index) for index in range(2, 6)]

        resumed = MiniMaxH3ReferenceCostSampler(
            _MetadataDataset(_rows()),
            num_replicas=4,
            rank=0,
            seed=42,
            image_counts=(1, 2),
        )
        resumed.configure(
            start_iteration=1,
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
        )
        self.assertEqual(list(iter(resumed)), expected)

    def test_rejects_non_image_reference_rows(self):
        rows = _rows()
        rows[0]["reference_video_count"] = 1
        with self.assertRaisesRegex(ValueError, "image-only"):
            MiniMaxH3ReferenceCostSampler(
                _MetadataDataset(rows),
                num_replicas=4,
                rank=0,
            )

    def test_strict_mode_still_rejects_unequal_orientation_counts(self):
        with self.assertRaisesRegex(ValueError, "equal landscape/portrait"):
            MiniMaxH3ReferenceCostSampler(
                _MetadataDataset(_uneven_rows()),
                num_replicas=4,
                rank=0,
                image_counts=(1, 2),
            )

    def test_rotating_drop_is_balanced_unique_and_eventually_covers_all_rows(self):
        rows = _uneven_rows()
        samplers = []
        for rank in range(4):
            sampler = MiniMaxH3ReferenceCostSampler(
                _MetadataDataset(rows),
                num_replicas=4,
                rank=rank,
                seed=42,
                image_counts=(1, 2),
                strict_full_epoch=False,
                remainder_policy="rotating_drop",
            )
            sampler.configure(
                start_iteration=0,
                gradient_accumulation_iters=1,
                fake_update_ratio=1,
            )
            samplers.append(sampler)

        self.assertEqual(samplers[0].rows_per_orientation, 10)
        self.assertEqual(samplers[0].epoch_rows, 20)
        self.assertEqual(samplers[0].num_global_batches, 5)
        self.assertEqual(
            samplers[0].dropped_per_orientation,
            {"landscape": 1, "portrait": 3},
        )

        covered = set()
        epoch_selections = []
        for data_epoch in range(2):
            for sampler in samplers:
                sampler.set_epoch(data_epoch)
            rank_indices = [list(iter(sampler)) for sampler in samplers]
            self.assertEqual({len(indices) for indices in rank_indices}, {5})

            epoch_indices = []
            for global_batch in zip(*rank_indices):
                self.assertEqual(len(set(global_batch)), 4)
                batch_rows = [rows[index] for index in global_batch]
                self.assertEqual(
                    sorted(row["target_orientation"] for row in batch_rows),
                    ["landscape", "landscape", "portrait", "portrait"],
                )
                epoch_indices.extend(global_batch)
            self.assertEqual(len(epoch_indices), 20)
            self.assertEqual(len(epoch_indices), len(set(epoch_indices)))
            epoch_selections.append(set(epoch_indices))
            covered.update(epoch_indices)

        self.assertNotEqual(epoch_selections[0], epoch_selections[1])
        self.assertEqual(covered, set(range(len(rows))))

        metadata = samplers[0].checkpoint_metadata(
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
        )
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["remainder_policy"], "rotating_drop")
        self.assertEqual(metadata["epoch_rows"], 20)

    def test_non_strict_mode_requires_explicit_remainder_policy(self):
        with self.assertRaisesRegex(ValueError, "explicit remainder_policy"):
            MiniMaxH3ReferenceCostSampler(
                _MetadataDataset(_uneven_rows()),
                num_replicas=4,
                rank=0,
                strict_full_epoch=False,
            )

    def test_rotating_drop_resume_follows_the_same_absolute_batch_stream(self):
        kwargs = {
            "num_replicas": 4,
            "rank": 0,
            "seed": 42,
            "image_counts": (1, 2),
            "strict_full_epoch": False,
            "remainder_policy": "rotating_drop",
        }
        uninterrupted = MiniMaxH3ReferenceCostSampler(
            _MetadataDataset(_uneven_rows()),
            **kwargs,
        )
        uninterrupted.configure(
            start_iteration=0,
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
        )
        expected = [uninterrupted.sample_index(index) for index in range(4, 9)]

        resumed = MiniMaxH3ReferenceCostSampler(
            _MetadataDataset(_uneven_rows()),
            **kwargs,
        )
        resumed.configure(
            start_iteration=2,
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
        )
        self.assertEqual(list(iter(resumed)), expected)

    def test_balanced_image_counts_accept_audio_and_rotate_omitted_rows(self):
        rows = _image_audio_uneven_rows()
        samplers = []
        for rank in range(4):
            sampler = MiniMaxH3ReferenceCostSampler(
                _MetadataDataset(rows),
                num_replicas=4,
                rank=rank,
                seed=42,
                require_image_only=False,
                image_counts=(1, 2, 3),
                balance_image_counts=True,
                strict_full_epoch=False,
                remainder_policy="rotating_drop",
            )
            sampler.configure(
                start_iteration=0,
                gradient_accumulation_iters=1,
                fake_update_ratio=1,
            )
            samplers.append(sampler)

        self.assertEqual(samplers[0].rows_per_image_orientation_cell, 4)
        self.assertEqual(samplers[0].epoch_rows, 24)
        self.assertEqual(samplers[0].num_global_batches, 6)
        covered = set()
        for epoch in range(3):
            epoch_indices = []
            for batch_index in range(samplers[0].num_global_batches):
                batch = [sampler.sample_index(epoch * sampler.num_global_batches + batch_index) for sampler in samplers]
                batch_rows = [rows[index] for index in batch]
                self.assertEqual(len({row["reference_image_count"] for row in batch_rows}), 1)
                self.assertEqual(
                    sorted(row["target_orientation"] for row in batch_rows),
                    ["landscape", "landscape", "portrait", "portrait"],
                )
                epoch_indices.extend(batch)
            self.assertEqual(
                dict(sorted(Counter(rows[index]["reference_image_count"] for index in epoch_indices).items())),
                {1: 8, 2: 8, 3: 8},
            )
            covered.update(epoch_indices)
        self.assertEqual(covered, set(range(len(rows))))


if __name__ == "__main__":
    unittest.main()
