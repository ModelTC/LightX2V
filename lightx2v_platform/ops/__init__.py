import os

from lightx2v_platform.base.global_var import AI_DEVICE

if AI_DEVICE == "mlu":
    from .attn.cambricon_mlu import *
    from .mm.cambricon_mlu import *
elif AI_DEVICE == "cuda":
    # Check if running on Hygon DCU platform
    if os.getenv("PLATFORM") == "hygon_dcu":
        from .attn.hygon_dcu import *
elif AI_DEVICE == "npu":
    from .attn.ascend_npu import *
    from .mm.ascend_npu import *
