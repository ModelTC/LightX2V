"""External serving registration, imported in each vLLM-Omni process."""

from lightx2v_rl.models.qwen_image.serving import (
    CPSQwenImage21Serving,
    FlowGRPOQwenImage21Serving,
    GeoQwenImage21Serving,
)

__all__ = ["CPSQwenImage21Serving", "FlowGRPOQwenImage21Serving", "GeoQwenImage21Serving"]
