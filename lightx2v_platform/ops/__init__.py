import os

from lightx2v_platform.base.global_var import AI_DEVICE

if AI_DEVICE == "mlu":
    from .attn.cambricon_mlu import *
    from .mm.cambricon_mlu import *
elif AI_DEVICE == "cuda":
    # Check if running on DCU platform
    if os.getenv("PLATFORM") == "dcu":
        from .attn.dcu import *
        from .mm.dcu import *
