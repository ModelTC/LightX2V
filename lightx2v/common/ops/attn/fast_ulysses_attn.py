import os
import socket

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .ulysses_attn import UlyssesAttnWeight


def _to_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _cu_value(cu_seqlens, index):
    return _to_int(cu_seqlens[index])


def _env_use_tma():
    value = os.environ.get("LIGHTX2V_FAST_ULYSSES_USE_TMA", "auto").lower()
    if value == "auto":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("LIGHTX2V_FAST_ULYSSES_USE_TMA must be auto, 1, or 0")


@ATTN_WEIGHT_REGISTER("fast_ulysses")
class FastUlyssesAttnWeight(AttnWeightTemplate):
    _groups = {}
    _single_node_cache = {}
    _fallback_warnings = set()
    _fast_path_logs = set()
    _fallback_calls = 0
    _fast_path_calls = 0

    def __init__(self):
        self.config = {}
        self._fallback_backend = UlyssesAttnWeight()

    @classmethod
    def _cache_key(cls, seq_p_group, device=None):
        group = seq_p_group if seq_p_group is not None else dist.group.WORLD
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        if device is None:
            device_index = torch.cuda.current_device()
        else:
            device = torch.device(device)
            device_index = device.index if device.index is not None else torch.cuda.current_device()
        return ranks, device_index

    def _get_group(self, seq_p_group, device=None):
        from lightx2v_fast_ulysses import UlyssesGroup

        key = self._cache_key(seq_p_group, device)
        if key not in type(self)._groups:
            pool_bytes = int(os.environ.get("LIGHTX2V_FAST_ULYSSES_POOL_BYTES", str(2 << 30)))
            type(self)._groups[key] = UlyssesGroup(process_group=seq_p_group, device=device, initial_pool_bytes=pool_bytes)
        return type(self)._groups[key]

    @classmethod
    def destroy_cached_groups(cls):
        for group in list(cls._groups.values()):
            group.destroy()
        cls._groups.clear()
        cls._single_node_cache.clear()
        cls.reset_runtime_stats()

    @classmethod
    def reset_runtime_stats(cls):
        cls._fallback_warnings.clear()
        cls._fast_path_logs.clear()
        cls._fallback_calls = 0
        cls._fast_path_calls = 0

    @classmethod
    def runtime_stats(cls):
        return {
            "fallback_calls": cls._fallback_calls,
            "fast_path_calls": cls._fast_path_calls,
        }

    @classmethod
    def _rank_key(cls, seq_p_group):
        group = seq_p_group if seq_p_group is not None else dist.group.WORLD
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))

    def _is_single_node(self, seq_p_group):
        try:
            key = self._rank_key(seq_p_group)
            if len(key) <= 1:
                return True
            if key not in type(self)._single_node_cache:
                host = socket.gethostname()
                hosts = [None for _ in key]
                dist.all_gather_object(hosts, host, group=seq_p_group)
                type(self)._single_node_cache[key] = len(set(hosts)) == 1
            return type(self)._single_node_cache[key]
        except Exception as exc:
            self._log_fallback_once(f"single-node check failed: {exc}")
            return False

    @classmethod
    def _log_fallback_once(cls, reason, seq_p_group=None):
        cls._fallback_calls += 1
        if reason in cls._fallback_warnings:
            return
        try:
            rank = dist.get_rank(seq_p_group)
        except Exception:
            rank = 0
        if rank == 0:
            logger.info("fast_ulysses fallback to ulysses: {}", reason)
        cls._fallback_warnings.add(reason)

    def _log_fast_path_once(self, seq_p_group):
        try:
            key = self._rank_key(seq_p_group)
        except Exception:
            key = ("unknown",)
        if key in type(self)._fast_path_logs:
            return
        try:
            rank = dist.get_rank(seq_p_group)
        except Exception:
            rank = 0
        if rank == 0:
            logger.info("fast_ulysses native A2A path active for ranks {}", key)
        type(self)._fast_path_logs.add(key)

    def _fallback(
        self,
        reason,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        use_tensor_fusion=False,
        enable_head_parallel=False,
        img_first=True,
        q_only_img=False,
        **kwargs,
    ):
        self._log_fallback_once(reason, seq_p_group)
        return self._fallback_backend.apply(
            q=q,
            k=k,
            v=v,
            slice_qkv_len=slice_qkv_len,
            cu_seqlens_qkv=cu_seqlens_qkv,
            attention_module=attention_module,
            seq_p_group=seq_p_group,
            use_fp8_comm=use_fp8_comm,
            use_fp4_comm=use_fp4_comm,
            use_tensor_fusion=use_tensor_fusion,
            enable_head_parallel=enable_head_parallel,
            img_first=img_first,
            q_only_img=q_only_img,
            **kwargs,
        )

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        use_tensor_fusion=False,
        enable_head_parallel=False,
        img_first=True,
        q_only_img=False,
        **kwargs,
    ):
        if attention_module is None:
            raise ValueError("fast_ulysses requires attention_module")
        if use_fp8_comm:
            return self._fallback(
                "use_fp8_comm is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if use_fp4_comm:
            return self._fallback(
                "use_fp4_comm is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if use_tensor_fusion:
            return self._fallback(
                "use_tensor_fusion is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if enable_head_parallel:
            return self._fallback(
                "enable_head_parallel is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if q_only_img:
            return self._fallback(
                "q_only_img is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if not img_first:
            return self._fallback(
                "img_first=False is not supported by the fast A2A path",
                q,
                k,
                v,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )

        q_orig, k_orig, v_orig = q, k, v

        if q.dim() == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        if q.shape != k.shape or q.shape != v.shape:
            return self._fallback(
                "GQA or mismatched q/k/v shapes are not supported by the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
            return self._fallback(
                "non-CUDA tensors are not supported by the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if q.device != k.device or q.device != v.device:
            return self._fallback(
                "q/k/v must be on the same CUDA device for the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
            return self._fallback(
                "CUDA tensors must be fp16/bf16 for the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )

        s_local, heads, head_dim = q.shape
        if (head_dim * q.element_size()) % 16 != 0:
            return self._fallback(
                "head_dim * elem_size must be 16-byte aligned for the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        world_size = dist.get_world_size(seq_p_group)
        if world_size < 1 or world_size > 8:
            return self._fallback(
                "fast A2A path only supports single-node world_size in [1, 8]",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if not self._is_single_node(seq_p_group):
            return self._fallback(
                "sequence-parallel group spans multiple nodes",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        if heads % world_size != 0:
            return self._fallback(
                "head count is not divisible by sequence-parallel world_size",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )

        if _to_int(slice_qkv_len) != s_local or len(cu_seqlens_qkv) != 2 or _cu_value(cu_seqlens_qkv, 0) != 0 or _cu_value(cu_seqlens_qkv, 1) != s_local:
            return self._fallback(
                "mixed text/image split is not supported by the fast A2A path",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )

        use_tma = _env_use_tma()
        if use_tma is True:
            device_major = torch.cuda.get_device_capability(q.device)[0]
            if device_major < 9:
                return self._fallback(
                    "LIGHTX2V_FAST_ULYSSES_USE_TMA=1 requires sm90+",
                    q_orig,
                    k_orig,
                    v_orig,
                    slice_qkv_len,
                    cu_seqlens_qkv,
                    attention_module,
                    seq_p_group,
                    use_fp8_comm,
                    use_fp4_comm,
                    use_tensor_fusion,
                    enable_head_parallel,
                    img_first,
                    q_only_img,
                    **kwargs,
                )

        try:
            group = self._get_group(seq_p_group, q.device)
        except Exception as exc:
            return self._fallback(
                f"native fast-ulysses group is unavailable: {exc}",
                q_orig,
                k_orig,
                v_orig,
                slice_qkv_len,
                cu_seqlens_qkv,
                attention_module,
                seq_p_group,
                use_fp8_comm,
                use_fp4_comm,
                use_tensor_fusion,
                enable_head_parallel,
                img_first,
                q_only_img,
                **kwargs,
            )
        global_seqlen = s_local * world_size
        shard_heads = heads // world_size
        self._log_fast_path_once(seq_p_group)

        qh = group.all_to_all_single_4d(q.reshape(1, s_local, heads, head_dim), mode=0, tag="lightx2v_q", use_tma=use_tma)
        kh = group.all_to_all_single_4d(k.reshape(1, s_local, heads, head_dim), mode=0, tag="lightx2v_k", use_tma=use_tma)
        vh = group.all_to_all_single_4d(v.reshape(1, s_local, heads, head_dim), mode=0, tag="lightx2v_v", use_tma=use_tma)

        attn = attention_module.apply(
            q=qh[0],
            k=kh[0],
            v=vh[0],
            max_seqlen_q=global_seqlen,
            max_seqlen_kv=global_seqlen,
            **kwargs,
        )
        attn = attn.reshape(1, global_seqlen, shard_heads, head_dim)
        out = group.all_to_all_single_4d(attn, mode=1, tag="lightx2v_out", use_tma=use_tma)
        type(self)._fast_path_calls += 1
        return out[0].reshape(s_local, heads * head_dim)
