import os

from lightx2v_platform.base.global_var import AI_DEVICE

PLATFORM = os.getenv("PLATFORM")
if PLATFORM == "cambricon_mlu":
    from .attn.cambricon_mlu import *
    from .mm.cambricon_mlu import *
    from .norm.cambricon_mlu import *
elif PLATFORM == "hygon_dcu":
    from .attn.hygon_dcu import *
    from .mm.hygon_dcu import *
elif PLATFORM == "amd_rocm":
    from .attn.amd_rocm import *
elif PLATFORM == "ascend_npu":
    # Keep MoE before MM: MM imports common utilities that snapshot the
    # platform registries during lightx2v initialization.
    # isort: off
    from .attn.ascend_npu import *
    from .moe.ascend_npu import *
    from .mm.ascend_npu import *
    from .norm.ascend_npu import *
    from .rope.ascend_npu import *
    # isort: on
elif PLATFORM == "metax_cuda":
    # MetaX attention imports the common registry, so platform MoE backends
    # must be registered before that one-time registry merge.
    # isort: off
    from .moe.metax_cuda import *
    from .attn.metax_cuda import *
    # isort: on
elif PLATFORM == "enflame_gcu":
    from .attn.enflame_gcu import *
    from .norm.enflame_gcu import *
    from .rope.enflame_gcu import *
elif PLATFORM == "intel_xpu":
    # Register platform RMSNorm before the attention modules import the
    # framework registry and snapshot all PLATFORM_* registries.
    from .attn.intel_xpu import *
    from .mm.intel_xpu import *
    from .norm.intel_xpu import *
elif PLATFORM == "iluvatar_cuda":
    from .attn.iluvatar_cuda import *
    from .mm.iluvatar_cuda import *
    from .norm.iluvatar_cuda import *
    from .rope.iluvatar_cuda import *
