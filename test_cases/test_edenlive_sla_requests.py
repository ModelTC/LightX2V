import pytest

from lightx2v.models.runners.wan.edenlive_runner import Wan22EdenLiveARRunner


@pytest.mark.parametrize("mask,allowed", [("sla_mask_generator", True), ("svg_mask_generator", False), ("nbhd_mask_generator", False)])
def test_only_sla_accepts_per_request_frame_count(mask, allowed):
    runner = Wan22EdenLiveARRunner.__new__(Wan22EdenLiveARRunner)
    runner.config = {
        "enable_cfg": False,
        "self_attn_1_type": "general_sparse_attn",
        "general_sparse_attn_setting": {"sparse_mask_generator": mask},
    }
    fields = runner.get_supported_request_fields("i2v")
    assert ("num_frames" in fields) is allowed
    assert "video_path" in fields
