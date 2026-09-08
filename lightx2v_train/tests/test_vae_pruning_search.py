import itertools
import unittest

import torch

from lightx2v_train.model_zoo.native.pruning_search import build_residual_search_masks


class ResidualSearchMasksTest(unittest.TestCase):
    def test_global_candidates_preserve_legacy_order(self):
        for depth, keep in ((12, 3), (10, 3), (14, 5)):
            masks = build_residual_search_masks(depth, {}, [], default_keep=keep)
            expected = list(itertools.combinations(range(depth), keep))
            self.assertEqual([tuple(row.nonzero().flatten().tolist()) for row in masks], expected)
            explicit = build_residual_search_masks(depth, {"grouping": "global", "keep_residuals": keep}, [], default_keep=1)
            torch.testing.assert_close(explicit, masks, rtol=0, atol=0)

    def test_stage_defaults_and_model_candidate_counts(self):
        for groups, expected_count in (
            ([tuple(range(i, i + 2)) for i in range(0, 12, 2)], 64),
            ([tuple(range(i, i + 2)) for i in range(0, 10, 2)], 32),
            ([(0, 1), (2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13)], 162),
        ):
            depth = sum(map(len, groups))
            masks = build_residual_search_masks(depth, {"grouping": "stage"}, groups, default_keep=3)
            self.assertEqual(tuple(masks.shape), (expected_count, depth))
            self.assertEqual(len(masks.unique(dim=0)), expected_count)
            self.assertTrue(torch.all((masks == 0) | (masks == 1)))
            for group in groups:
                torch.testing.assert_close(masks[:, list(group)].sum(-1), torch.ones(expected_count))

    def test_per_stage_budgets_allow_zero_without_changing_global_budget(self):
        groups = [(0, 1), (2, 3, 4), (5, 6)]
        masks = build_residual_search_masks(
            7, {"grouping": "stage", "keep_per_group": [1, 2, 0], "keep_residuals": 3}, groups, default_keep=1,
        )
        self.assertEqual(tuple(masks.shape), (6, 7))
        for group, budget in zip(groups, (1, 2, 0)):
            torch.testing.assert_close(masks[:, list(group)].sum(-1), torch.full((6,), float(budget)))

    def test_conflicting_or_impossible_budgets_fail(self):
        invalid_searches = [
            {"grouping": "unknown"},
            {"keep_residuals": 0},
            {"keep_residuals": 5},
            {"keep_per_group": 1},
            {"grouping": "stage", "keep_per_group": 3},
            {"grouping": "stage", "keep_per_group": [1]},
            {"grouping": "stage", "keep_per_group": [-1, 1]},
            {"grouping": "stage", "keep_per_group": [0, 0]},
            {"grouping": "stage", "keep_residuals": 3},
        ]
        for search in invalid_searches:
            with self.subTest(search=search), self.assertRaises(ValueError):
                build_residual_search_masks(4, search, [(0, 1), (2, 3)], default_keep=2)

    def test_stage_groups_must_cover_branches_once_in_order(self):
        for groups in ([], [(0, 1), ()], [(0, 1), (1, 2, 3)], [(0, 1), (3,)], [(2, 3), (0, 1)]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                build_residual_search_masks(4, {"grouping": "stage"}, groups, default_keep=2)


if __name__ == "__main__":
    unittest.main()
