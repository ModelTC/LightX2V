import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
import torch.nn.functional as F
from loguru import logger

try:
    from spas_sage_attn.autotune import SparseAttentionMeansim
except ImportError:
    logger.info("SparseAttentionMeansim not found, please install sparge first")
    SparseAttentionMeansim = None

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn import flash_attn_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

if torch.cuda.get_device_capability(0) == (8, 9):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None


from lightx2v.attentions.common.radial_attn import radial_attn


class AttnWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name):
        self.weight_name = weight_name
        self.config = {}

    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        return destination


@ATTN_WEIGHT_REGISTER("naive")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, softmax_scale, is_causal, model_cls=None):
        dtype = q.dtype
        q = q * softmax_scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        if is_causal:
            causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
            causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
            attn += causal_mask
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        x = attn @ v
        x = x.transpose(1, 2).contiguous()
        return x


@ATTN_WEIGHT_REGISTER("flash_attn2_base")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, dropout_p, softmax_scale, is_causal, model_cls=None):
        x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        return x


@ATTN_WEIGHT_REGISTER("xformers")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, dropout_p, softmax_scale, attn_bias, model_cls=None):
        x = xformers.ops.memory_efficient_attention(
            q, 
            k, 
            v, 
            p=dropout_p,
            attn_bias=attn_bias, 
            scale=softmax_scale
        )
        return x


@ATTN_WEIGHT_REGISTER("flash_attn2")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        mask_map=None,
    ):
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("flash_attn3")
class FlashAttn3Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        mask_map=None,
    ):
        x = flash_attn_varlen_func_v3(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )[0].reshape(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("radial_attn")
class RadialAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        mask_map=None,
        sparsity_type="radial",
        block_size=128,
        decay_factor=1,
        model_cls="wan",
    ):
        assert len(q.shape) == 3

        x = radial_attn(
            q,
            k,
            v,
            mask_map=mask_map,
            sparsity_type=sparsity_type,
            block_size=block_size,
            model_cls=model_cls[:3],  # Use first 3 characters to match "wan", "wan2", etc.
            decay_factor=decay_factor,
        )
        x = x.view(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        mask_map=None,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if model_cls == "hunyuan":
            x1 = sageattn(
                q[: cu_seqlens_q[1]].unsqueeze(0),
                k[: cu_seqlens_kv[1]].unsqueeze(0),
                v[: cu_seqlens_kv[1]].unsqueeze(0),
                tensor_layout="NHD",
            )
            x2 = sageattn(
                q[cu_seqlens_q[1] :].unsqueeze(0),
                k[cu_seqlens_kv[1] :].unsqueeze(0),
                v[cu_seqlens_kv[1] :].unsqueeze(0),
                tensor_layout="NHD",
            )
            x = torch.cat((x1, x2), dim=1)
            x = x.view(max_seqlen_q, -1)
        elif model_cls in ["wan2.1", "wan2.1_distill", "wan2.1_causvid", "wan2.1_df"]:
            x = sageattn(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                tensor_layout="NHD",
            )
            x = x.view(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        mask_map=None,
    ):
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        x = x.transpose(1, 2)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out.squeeze(0)


@ATTN_WEIGHT_REGISTER("Sparge")
class SpargeAttnWeight(AttnWeightTemplate):
    def __init__(
        self,
        weight_name,
        verbose=False,
        l1=0.07,
        pv_l1=0.08,
        tune_pv=True,
        inner_attn_type="flash_attn3",
    ):
        self.verbose = (verbose,)
        self.l1 = (l1,)
        self.pv_l1 = (pv_l1,)
        self.tune_pv = (tune_pv,)
        self.inner_attn_type = inner_attn_type
        self.inner_cls = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, tune_pv=tune_pv)
        super().__init__(weight_name)

    def load(self, weight_dict):
        # match all key with prefix weight_name
        for key in weight_dict.keys():
            if key.startswith(self.weight_name):
                sub_name = key.split(".")[-1]
                setattr(
                    self.inner_cls,
                    sub_name,
                    nn.Parameter(weight_dict[key], requires_grad=False),
                )

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        x = self.inner_cls(q, k, v, tensor_layout="NHD")
        x = x.flatten(2)
        x = x.squeeze(0)

        return x
