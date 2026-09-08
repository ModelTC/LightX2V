from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_h3_qk_rope():
    from torch_musa.utils.musa_extension import load

    source_dir = Path(__file__).with_name("csrc")
    return load(
        name="lightx2v_h3_qk_rope",
        sources=[
            str(source_dir / "h3_qk_rope.cpp"),
            str(source_dir / "h3_qk_rope.mu"),
        ],
        extra_cflags=["-O3"],
        extra_musa_cflags=["-O3", "-ffp-contract=off"],
        verbose=False,
    )


def apply_h3_qk_rope(query, key, cos, sin):
    return _load_h3_qk_rope().h3_qk_rope(query, key, cos, sin)


def apply_h3_qk_rope_fp32(query, key, cos, sin):
    return _load_h3_qk_rope().h3_qk_rope_fp32(query, key, cos, sin)
