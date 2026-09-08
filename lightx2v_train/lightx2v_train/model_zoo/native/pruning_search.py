"""Legal residual-branch masks for global or stage-constrained VAE search."""

import itertools

import torch


def build_residual_search_masks(depth, search, stage_groups, *, default_keep):
    grouping = search.get("grouping", "global")
    if grouping == "global":
        if "keep_per_group" in search:
            raise ValueError("keep_per_group requires grouping='stage'.")
        keep = int(search.get("keep_residuals", default_keep))
        if not 1 <= keep <= depth:
            raise ValueError("keep_residuals must be within the original residual depth.")
        choices = list(itertools.combinations(range(depth), keep))
    elif grouping == "stage":
        groups = [tuple(group) for group in stage_groups]
        if not groups or any(not group for group in groups) or [i for group in groups for i in group] != list(range(depth)):
            raise ValueError("Stage groups must partition all residual branches in network order.")
        budget = search.get("keep_per_group", 1)
        budgets = [budget] * len(groups) if isinstance(budget, int) else list(budget)
        if len(budgets) != len(groups) or any(not 0 <= keep <= len(group) for group, keep in zip(groups, budgets)):
            raise ValueError("keep_per_group must fit every stage and have one entry per stage.")
        keep = sum(budgets)
        if keep == 0:
            raise ValueError("Search must retain at least one residual branch.")
        if "keep_residuals" in search and int(search["keep_residuals"]) != keep:
            raise ValueError("keep_residuals conflicts with the sum of keep_per_group.")
        options = [itertools.combinations(group, budget) for group, budget in zip(groups, budgets)]
        choices = [tuple(i for selected in combination for i in selected) for combination in itertools.product(*options)]
    else:
        raise ValueError("Search grouping must be 'global' or 'stage'.")
    masks = torch.zeros(len(choices), depth, dtype=torch.float32)
    for index, choice in enumerate(choices):
        masks[index, list(choice)] = 1
    return masks
