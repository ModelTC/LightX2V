import json
import tempfile
import unittest
from itertools import islice
from pathlib import Path

from lightx2v_train.data.minimax_h3_cache_dataset import (
    MiniMaxH3TaskCycleSampler,
    _discover_minimax_h3_cache_metadata,
)


class _SamplerDataset:
    def __init__(self, rows_per_cell=24):
        self.samples = []
        for bucket in ("landscape", "portrait"):
            height, width = (768, 1344) if bucket == "landscape" else (1344, 768)
            for task in ("t2av", "i2av", "l2av", "fl2av"):
                for row_id in range(rows_per_cell):
                    self.samples.append(
                        {
                            "type": "metadata",
                            "row": {
                                "id": f"{task}-{bucket}-{row_id}",
                                "task": task,
                                "aspect_bucket": bucket,
                                "target_height": height,
                                "target_width": width,
                            },
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class MiniMaxH3CacheDatasetTest(unittest.TestCase):
    @staticmethod
    def _write_manifest(root, task, bucket, shard=None, *, fingerprint=None, condition=None):
        directory = Path(root) / task / bucket
        if shard is not None:
            directory /= shard
        directory.mkdir(parents=True, exist_ok=True)
        condition = condition or f"conditions/{task}_{bucket}.pt"
        height, width = (768, 1344) if bucket == "landscape" else (1344, 768)
        row = {
            "task": task,
            "caption": f"{task} {bucket}",
            "prompt_variant": "original",
            "aspect_bucket": bucket,
            "target_height": height,
            "target_width": width,
            "condition_path": condition,
        }
        if fingerprint is not None:
            row["cache_fingerprint"] = fingerprint
        path = directory / "metadata.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def test_discovers_disjoint_legacy_bucket_and_shard_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = root / "t2av"
            direct.mkdir(parents=True)
            (direct / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "task": "t2av",
                        "caption": "text",
                        "prompt_variant": "original",
                        "aspect_bucket": "landscape",
                        "target_height": 768,
                        "target_width": 1344,
                        "condition_path": "conditions/t2.pt",
                        "cache_fingerprint": "t2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bucket = self._write_manifest(root, "i2av", "landscape", fingerprint="i2-base")
            shard = self._write_manifest(
                root,
                "i2av",
                "portrait",
                "expansion_v1",
                fingerprint="i2-expansion",
            )
            manifests = _discover_minimax_h3_cache_metadata(
                root,
                tasks=("t2av", "i2av"),
                buckets=("landscape", "portrait"),
                require_all_tasks=True,
                require_all_buckets=False,
            )
            self.assertEqual(set(manifests), {(direct / "metadata.jsonl").resolve(), bucket.resolve(), shard.resolve()})

    def test_rejects_duplicate_condition_path_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_manifest(
                root,
                "i2av",
                "landscape",
                fingerprint="same",
                condition="/tmp/shared_condition.pt",
            )
            self._write_manifest(
                root,
                "i2av",
                "portrait",
                "expansion_v1",
                fingerprint="same",
                condition="/tmp/shared_condition.pt",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate MiniMax-H3 condition sample"):
                _discover_minimax_h3_cache_metadata(
                    root,
                    tasks=("i2av",),
                    buckets=("landscape", "portrait"),
                )

    def test_rejects_task_bucket_and_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, "i2av", "portrait")
            row = json.loads(manifest.read_text(encoding="utf-8"))
            row["target_height"], row["target_width"] = 768, 1344
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "portrait cache must use"):
                _discover_minimax_h3_cache_metadata(
                    root,
                    tasks=("i2av",),
                    buckets=("portrait",),
                )

    def test_task_specific_discovery_ignores_inactive_portrait_manifests(self):
        task_bucket_order = {
            "t2av": ("landscape", "portrait"),
            "i2av": ("landscape",),
            "l2av": ("landscape",),
            "fl2av": ("landscape",),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = set()
            for task, task_buckets in task_bucket_order.items():
                for bucket in task_buckets:
                    active.add(
                        self._write_manifest(
                            root,
                            task,
                            bucket,
                            fingerprint=f"active-{task}-{bucket}",
                        ).resolve()
                    )
            inactive = self._write_manifest(
                root,
                "i2av",
                "portrait",
                fingerprint="stale-i2av-portrait",
            ).resolve()
            manifests = _discover_minimax_h3_cache_metadata(
                root,
                tasks=("t2av", "i2av", "l2av", "fl2av"),
                buckets=("landscape", "portrait"),
                require_all_tasks=True,
                require_all_buckets=True,
                task_bucket_order=task_bucket_order,
            )
            self.assertEqual(set(manifests), active)
            self.assertNotIn(inactive, manifests)


class MiniMaxH3TaskCycleSamplerTest(unittest.TestCase):
    task_order = ("t2av", "i2av", "l2av", "fl2av")
    bucket_order = ("landscape", "portrait")

    def _sampler(self, dataset, rank, start=0, block=3):
        sampler = MiniMaxH3TaskCycleSampler(
            dataset,
            task_order=self.task_order,
            bucket_order=self.bucket_order,
            num_replicas=2,
            rank=rank,
            seed=123,
            drop_last=True,
        )
        sampler.configure(
            start_iteration=start,
            samples_per_outer_iteration=block,
        )
        return sampler

    def _stratified_sampler(
        self,
        dataset,
        rank,
        *,
        num_replicas=4,
        grad_accum_iters=8,
        fake_update_ratio=5,
        start=0,
    ):
        sampler = MiniMaxH3TaskCycleSampler(
            dataset,
            task_order=("t2av", "i2av", "fl2av"),
            bucket_order=self.bucket_order,
            num_replicas=num_replicas,
            rank=rank,
            seed=123,
            drop_last=True,
            route_mode="stratified",
            balance_route_loss=True,
            require_all_routes_per_optimizer_step=True,
        )
        sampler.configure(
            start_iteration=start,
            gradient_accumulation_iters=grad_accum_iters,
            fake_update_ratio=fake_update_ratio,
        )
        return sampler

    @staticmethod
    def _groups(dataset, indices):
        return [
            (
                dataset.samples[index]["row"]["task"],
                dataset.samples[index]["row"]["aspect_bucket"],
            )
            for index in indices
        ]

    def test_outer_iteration_blocks_are_homogeneous_and_rank_disjoint(self):
        dataset = _SamplerDataset()
        rank_zero_sampler = self._sampler(dataset, 0)
        self.assertEqual(rank_zero_sampler.route_mode, "homogeneous")
        rank_zero = list(rank_zero_sampler)
        rank_one = list(self._sampler(dataset, 1))
        expected = [
            ("t2av", "landscape"),
            ("i2av", "landscape"),
            ("l2av", "landscape"),
            ("fl2av", "landscape"),
            ("t2av", "portrait"),
            ("i2av", "portrait"),
            ("l2av", "portrait"),
            ("fl2av", "portrait"),
        ]
        block = 3
        for outer_iteration, expected_group in enumerate(expected):
            start = outer_iteration * block
            rank_zero_block = rank_zero[start : start + block]
            rank_one_block = rank_one[start : start + block]
            self.assertEqual(
                self._groups(dataset, rank_zero_block),
                [expected_group] * block,
            )
            self.assertEqual(
                self._groups(dataset, rank_one_block),
                [expected_group] * block,
            )
            self.assertTrue(set(rank_zero_block).isdisjoint(rank_one_block))

    def test_resume_continues_schedule_and_exact_shuffle_position(self):
        dataset = _SamplerDataset()
        uninterrupted = list(self._sampler(dataset, 0, start=0))
        block = 3
        for start, expected_group in (
            (2, ("l2av", "landscape")),
            (5, ("i2av", "portrait")),
            (250, ("l2av", "landscape")),
        ):
            resumed = list(self._sampler(dataset, 0, start=start))
            self.assertEqual(self._groups(dataset, resumed[:block]), [expected_group] * block)
            if start < 8:
                self.assertEqual(
                    resumed[:block],
                    uninterrupted[start * block : (start + 1) * block],
                )

    def test_stratified_dp4_ga8_balances_six_routes_over_three_updates(self):
        dataset = _SamplerDataset(rows_per_cell=128)
        sampler = self._stratified_sampler(dataset, rank=0)
        expected_routes = {(task, bucket) for task in ("t2av", "i2av", "fl2av") for bucket in self.bucket_order}

        per_update_counts = [sampler.optimizer_step_route_counts(role_update_index=role_update_index) for role_update_index in range(3)]
        for counts in per_update_counts:
            self.assertEqual(set(counts), expected_routes)
            self.assertEqual(sum(counts.values()), 4 * 8)
            self.assertEqual(sorted(counts.values()), [5, 5, 5, 5, 6, 6])

        # 3 * 32 global slots divide exactly across six routes. Rotating which
        # routes receive the two extra slots prevents a persistent task bias.
        three_update_totals = {route: sum(counts[route] for counts in per_update_counts) for route in expected_routes}
        self.assertEqual(set(three_update_totals.values()), {16})

        # Per-route scaling makes one optimizer update an equal mean over the
        # six routes even when 32 is not divisible by six. Its global average
        # remains one, so it does not change the configured learning-rate scale.
        weighted_slot_sum = 0.0
        for rank in range(4):
            for micro_step in range(8):
                route = sampler.expected_micro_group(
                    outer_iteration=0,
                    stage="student",
                    micro_step=micro_step,
                    rank=rank,
                )
                scale = sampler.microbatch_loss_scale(
                    outer_iteration=0,
                    stage="student",
                    micro_step=micro_step,
                    rank=rank,
                )
                expected_scale = 32.0 / (6.0 * per_update_counts[0][route])
                self.assertAlmostEqual(scale, expected_scale)
                weighted_slot_sum += scale
        self.assertAlmostEqual(weighted_slot_sum / 32.0, 1.0)

    def test_stratified_dp16_ga1_covers_every_route_in_one_update(self):
        dataset = _SamplerDataset(rows_per_cell=128)
        sampler = self._stratified_sampler(
            dataset,
            rank=0,
            num_replicas=16,
            grad_accum_iters=1,
        )
        expected_routes = {(task, bucket) for task in ("t2av", "i2av", "fl2av") for bucket in self.bucket_order}
        counts = sampler.optimizer_step_route_counts(role_update_index=0)
        self.assertEqual(set(counts), expected_routes)
        self.assertEqual(sum(counts.values()), 16)
        self.assertEqual(sorted(counts.values()), [2, 2, 3, 3, 3, 3])

        # This is the logical topology of 64 physical GPUs with SP=4: all SP
        # peers share one logical DP rank/sample, while the 16 DP ranks cover
        # all routes. Supplying rank explicitly must match a sampler owned by
        # that logical DP rank.
        rank_three = self._stratified_sampler(
            dataset,
            rank=3,
            num_replicas=16,
            grad_accum_iters=1,
        )
        self.assertEqual(
            rank_three.expected_micro_group(
                outer_iteration=0,
                stage="student",
                micro_step=0,
            ),
            sampler.expected_micro_group(
                outer_iteration=0,
                stage="student",
                micro_step=0,
                rank=3,
            ),
        )
        duplicate_sp_peer = self._stratified_sampler(
            dataset,
            rank=3,
            num_replicas=16,
            grad_accum_iters=1,
        )
        self.assertEqual(
            list(islice(iter(rank_three), 6)),
            list(islice(iter(duplicate_sp_peer), 6)),
        )

    def test_stratified_resume_is_exact_and_student_fake_cursors_are_independent(self):
        dataset = _SamplerDataset(rows_per_cell=256)
        grad_accum_iters = 8
        fake_update_ratio = 5
        local_samples_per_outer = grad_accum_iters * (1 + fake_update_ratio)
        uninterrupted_sampler = self._stratified_sampler(
            dataset,
            rank=2,
            grad_accum_iters=grad_accum_iters,
            fake_update_ratio=fake_update_ratio,
            start=0,
        )
        uninterrupted = list(islice(iter(uninterrupted_sampler), 3 * local_samples_per_outer))
        resumed_sampler = self._stratified_sampler(
            dataset,
            rank=2,
            grad_accum_iters=grad_accum_iters,
            fake_update_ratio=fake_update_ratio,
            start=2,
        )
        resumed = list(islice(iter(resumed_sampler), local_samples_per_outer))
        self.assertEqual(
            resumed,
            uninterrupted[2 * local_samples_per_outer : 3 * local_samples_per_outer],
        )

        # Student and fake stages have separate role-update cursors. Therefore
        # student update 0 and fake update 0 share the same balanced layout,
        # while each role advances independently to layout 1.
        for rank in range(4):
            for micro_step in range(grad_accum_iters):
                self.assertEqual(
                    uninterrupted_sampler.expected_micro_group(
                        outer_iteration=0,
                        stage="student",
                        micro_step=micro_step,
                        rank=rank,
                    ),
                    uninterrupted_sampler.expected_micro_group(
                        outer_iteration=0,
                        stage="fake",
                        fake_update_index=0,
                        micro_step=micro_step,
                        rank=rank,
                    ),
                )
                self.assertEqual(
                    uninterrupted_sampler.expected_micro_group(
                        outer_iteration=1,
                        stage="student",
                        micro_step=micro_step,
                        rank=rank,
                    ),
                    uninterrupted_sampler.expected_micro_group(
                        outer_iteration=0,
                        stage="fake",
                        fake_update_index=1,
                        micro_step=micro_step,
                        rank=rank,
                    ),
                )

    def test_stratified_optimizer_windows_stay_rank_disjoint_at_shuffle_boundaries(self):
        # DP4 x GA8 uses at most six rows from one route per optimizer update.
        # Twelve rows per route forces several pool wraps during this outer
        # iteration and exercises the cross-shuffle boundary guard.
        dataset = _SamplerDataset(rows_per_cell=12)
        iterators = [iter(self._stratified_sampler(dataset, rank=rank)) for rank in range(4)]
        for _stage_update in range(1 + 5):
            global_indices = []
            for _micro_step in range(8):
                global_indices.extend(next(iterator) for iterator in iterators)
            self.assertEqual(len(global_indices), 32)
            self.assertEqual(len(set(global_indices)), 32)

    def test_task_specific_buckets_keep_tasks_uniform_and_only_t2av_alternates(self):
        dataset = _SamplerDataset()
        dataset.samples = [sample for sample in dataset.samples if sample["row"]["task"] == "t2av" or sample["row"]["aspect_bucket"] == "landscape"]
        task_bucket_order = {
            "t2av": ("landscape", "portrait"),
            "i2av": ("landscape",),
            "l2av": ("landscape",),
            "fl2av": ("landscape",),
        }

        def sampler(rank, start=0):
            result = MiniMaxH3TaskCycleSampler(
                dataset,
                task_order=self.task_order,
                bucket_order=self.bucket_order,
                task_bucket_order=task_bucket_order,
                num_replicas=2,
                rank=rank,
                seed=123,
                drop_last=True,
            )
            result.configure(start_iteration=start, samples_per_outer_iteration=3)
            return result

        expected = [
            ("t2av", "landscape"),
            ("i2av", "landscape"),
            ("l2av", "landscape"),
            ("fl2av", "landscape"),
            ("t2av", "portrait"),
            ("i2av", "landscape"),
            ("l2av", "landscape"),
            ("fl2av", "landscape"),
        ]
        rank_zero_sampler = sampler(0)
        rank_zero = list(rank_zero_sampler)
        rank_one = list(sampler(1))
        self.assertEqual(rank_zero_sampler.schedule_period, 8)
        self.assertEqual(rank_zero_sampler.num_cells, 5)
        block = 3
        for outer_iteration, expected_group in enumerate(expected):
            start = outer_iteration * block
            rank_zero_block = rank_zero[start : start + block]
            rank_one_block = rank_one[start : start + block]
            self.assertEqual(self._groups(dataset, rank_zero_block), [expected_group] * block)
            self.assertEqual(self._groups(dataset, rank_one_block), [expected_group] * block)
            self.assertTrue(set(rank_zero_block).isdisjoint(rank_one_block))

        # Resume uses the absolute task occurrence, including T2AV's bucket
        # phase and the exact per-cell shuffled block position.
        resume_iteration = 13
        resumed = list(sampler(0, start=resume_iteration))
        self.assertEqual(
            resumed[:block],
            rank_zero[resume_iteration * block : (resume_iteration + 1) * block],
        )
        self.assertEqual(
            self._groups(dataset, resumed[:block]),
            [("i2av", "landscape")] * block,
        )

        first_40_tasks = [rank_zero_sampler.expected_group(index)[0] for index in range(40)]
        self.assertEqual(
            {task: first_40_tasks.count(task) for task in self.task_order},
            {task: 10 for task in self.task_order},
        )


if __name__ == "__main__":
    unittest.main()
