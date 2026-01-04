import os

from lightx2v_platform.base.global_var import AI_DEVICE

PLATFORM = os.getenv("PLATFORM")
if PLATFORM == "mlu":
    from .attn.cambricon_mlu import *
    from .mm.cambricon_mlu import *
elif PLATFORM == "hygon_dcu":
    from .attn.hygon_dcu import *
elif PLATFORM == "amd_rocm":
    from .attn.amd_rocm import *
elif PLATFORM == "ascend_npu":
    from .attn.ascend_npu import *
    from .mm.ascend_npu import *
elif PLATFORM == "metax_cuda":
    from .attn.metax_cuda import *
