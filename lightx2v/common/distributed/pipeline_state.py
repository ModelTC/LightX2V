"""Pipeline parallel runtime state and stage helpers for LightX2V.

Manages patch metadata (how the latent image is split across pipeline patches)
and provides stage-identification utilities (is_pipeline_first_stage, etc.).
"""

from typing import List, Optional

import torch.distributed as dist

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_pp_group: Optional[dist.ProcessGroup] = None
_runtime_state: Optional["PipelineRuntimeState"] = None


def init_pipeline_parallel_state(pp_group: dist.ProcessGroup):
    """Register the pipeline-parallel process group.  Called once during
    ``set_parallel_config`` when ``pp_size > 1``."""
    global _pp_group, _runtime_state
    _pp_group = pp_group
    _runtime_state = PipelineRuntimeState()


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def get_pp_group() -> dist.ProcessGroup:
    assert _pp_group is not None, "pipeline parallel group is not initialised"
    return _pp_group


def get_pipeline_parallel_rank() -> int:
    if _pp_group is None:
        return 0
    return dist.get_rank(_pp_group)


def get_pipeline_parallel_world_size() -> int:
    if _pp_group is None:
        return 1
    return dist.get_world_size(_pp_group)


def is_pipeline_first_stage() -> bool:
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    return get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1


def get_pipeline_runtime_state() -> "PipelineRuntimeState":
    assert _runtime_state is not None, "PipelineRuntimeState not initialised"
    return _runtime_state


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class PipelineRuntimeState:
    """Runtime metadata for patch-level pipeline parallelism (PipeFusion).

    Computes how the latent token sequence is split into *patches* so that
    each pipeline stage processes a subset of patches in async mode.
    """

    def __init__(self):
        self.num_pipeline_patch: int = 1
        self.pipeline_patch_idx: int = 0
        self.patch_mode: bool = False  # True = async, False = sync
        self.warmup_steps: int = 1

        # Patch metadata (along the latent token / sequence dimension)
        self.pp_patches_token_num: List[int] = [0]
        self.pp_patches_token_start_end_idx_global: List[List[int]] = [[0, 0]]

        # Input parameters
        self.height: int = 0
        self.width: int = 0
        self.batch_size: int = 1
        self.packed_h: int = 0
        self.packed_w: int = 0
        self.vae_scale_factor: int = 16
        self.patch_size: int = 1

    # -- configuration -------------------------------------------------------

    def set_input_parameters(
        self,
        height: int,
        width: int,
        batch_size: int = 1,
        num_pipeline_patch: Optional[int] = None,
        warmup_steps: int = 1,
        vae_scale_factor: int = 16,
        patch_size: int = 1,
        total_tokens: Optional[int] = None,
    ):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.vae_scale_factor = vae_scale_factor
        self.patch_size = patch_size
        self.warmup_steps = warmup_steps
        if num_pipeline_patch is not None:
            self.num_pipeline_patch = num_pipeline_patch

        if total_tokens is not None:
            self.packed_h = 0
            self.packed_w = 0
            tok_count = total_tokens
        else:
            # Compute packed dimensions
            multiple_of = vae_scale_factor * 2
            self.packed_h = height // multiple_of
            self.packed_w = width // multiple_of
            tok_count = self.packed_h * self.packed_w

        # Split tokens evenly across patches
        base = tok_count // self.num_pipeline_patch
        remainder = tok_count % self.num_pipeline_patch
        self.pp_patches_token_num = []
        self.pp_patches_token_start_end_idx_global = []
        start = 0
        for i in range(self.num_pipeline_patch):
            n = base + (1 if i < remainder else 0)
            self.pp_patches_token_num.append(n)
            self.pp_patches_token_start_end_idx_global.append([start, start + n])
            start += n

    # -- patch mode ----------------------------------------------------------

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx >= self.num_pipeline_patch:
                self.pipeline_patch_idx = 0

    @property
    def current_patch_token_start_end(self) -> List[int]:
        return self.pp_patches_token_start_end_idx_global[self.pipeline_patch_idx]
