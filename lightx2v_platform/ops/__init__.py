from lightx2v_platform.base.global_var import AI_DEVICE

if AI_DEVICE == "mlu":
    from .attn.cambricon_mlu import *
