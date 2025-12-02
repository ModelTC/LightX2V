"""
PYTHONPATH=/path-to-LightX2V PLATFORM=cuda python test_device.py
PYTHONPATH=/path-to-LightX2V PLATFORM=mlu python test_device.py
PYTHONPATH=/path-to-LightX2V PLATFORM=metax python test_device.py
"""

# This import will initialize the AI device
import lightx2v_platform.set_ai_device  # noqa: F401
from lightx2v_platform.base.global_var import AI_DEVICE

print(f"AI_DEVICE: {AI_DEVICE}")
