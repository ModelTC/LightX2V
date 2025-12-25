import os

from lightx2v_platform.base.global_var import AI_DEVICE

if AI_DEVICE == "mlu":
    from .attn.cambricon_mlu import *
    from .mm.cambricon_mlu import *
elif AI_DEVICE == "cuda":
    platform = os.getenv("PLATFORM")
    if platform == "hygon_dcu":
        from .attn.hygon_dcu import *
    elif platform == "amd_rocm":
        from .attn.amd_rocm import *
