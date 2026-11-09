"""Exercise the actual training loader against the prompt encoder's output."""

from pathlib import Path
from unittest.mock import patch

import test_h3_pdmd_gan_launch as fixtures

from lightx2v_train.data import minimax_h3_cache_dataset as cache


def test_encoded_prompt_manifest_is_discoverable_by_fsdp32_training(tmp_path):
    source = fixtures.make_input(tmp_path, count=64)
    output = tmp_path / "encoded"
    fixtures.encode_shards(source, output)
    receipt = fixtures.merge(source, output)
    assert receipt["training_manifest_sha256"]
    config = {
        "data_path": str(output),
        "tasks": ["t2av"],
        "buckets": ["landscape"],
        "task_order": ["t2av"],
        "task_bucket_order": {"t2av": ["landscape"]},
        "route_mode": "stratified",
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "defer_latent_loading": True,
        "balance_orientation": False,
        "require_all_tasks": True,
        "require_all_buckets": True,
    }
    with patch.object(cache, "get_data_parallel_world_size", return_value=32), patch.object(cache, "get_data_parallel_rank", return_value=0):
        loader = cache.build_minimax_h3_cache_dataset(config)
    loader.sampler.configure_from_microbatch_offset(start_microbatch=0, fake_update_ratio=5)
    assert len(loader.dataset) == 64
    batch = next(iter(loader))
    assert batch["conditioning"]["positive"]["task"] == ["t2av"]
    assert batch["inputs"] == {}
    condition_path = Path(batch["meta"]["condition_path"][0])
    assert condition_path.is_absolute() and condition_path.is_file()
    assert condition_path.parent == output / "conditions"
