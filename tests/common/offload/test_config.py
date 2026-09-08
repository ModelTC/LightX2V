from lightx2v.common.offload.config import get_offload_granularity, normalize_offload_plan, use_event_offload
from lightx2v.pipeline import LightX2VPipeline


def test_legacy_offload_keys_are_normalized_into_one_plan():
    config = {
        "offload_granularity": "model",
        "use_event_offload": True,
    }

    normalize_offload_plan(config)

    assert config["offload_plan"] == {
        "offload_granularity": "model",
        "resident_blocks": {},
        "use_event_offload": True,
    }


def test_explicit_offload_plan_takes_precedence():
    config = {
        "offload_granularity": "model",
        "use_event_offload": False,
        "offload_plan": {
            "offload_granularity": "block",
            "resident_blocks": {"blocks": 8},
            "use_event_offload": True,
        },
    }

    normalize_offload_plan(config)

    assert get_offload_granularity(config) == "block"
    assert use_event_offload(config) is True
    assert config["offload_plan"]["resident_blocks"] == {"blocks": 8}


def test_offload_plan_defaults_are_added_without_changing_model_specific_keys():
    config = {"offload_plan": {"use_block_slab": True}}

    normalize_offload_plan(config)

    assert config["offload_plan"] == {
        "offload_granularity": "block",
        "resident_blocks": {},
        "use_event_offload": False,
        "use_block_slab": True,
    }


def test_pipeline_accepts_model_specific_offload_plan():
    pipeline = LightX2VPipeline.__new__(LightX2VPipeline)
    pipeline.model_cls = "flux2_dev"

    pipeline.enable_offload(
        cpu_offload=True,
        offload_plan={
            "offload_granularity": "block",
            "resident_blocks": {"double_blocks": 4, "single_blocks": 8},
            "use_event_offload": True,
            "use_block_slab": True,
        },
    )

    assert pipeline.offload_plan == {
        "offload_granularity": "block",
        "resident_blocks": {"double_blocks": 4, "single_blocks": 8},
        "use_event_offload": True,
        "use_block_slab": True,
    }
