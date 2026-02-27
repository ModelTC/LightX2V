import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.utils import _triple

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.seedvr.utils import na
from lightx2v.models.networks.seedvr.utils.attention import FlashAttentionVarlen
from lightx2v.models.networks.seedvr.utils.ops import gather_heads_scatter_seq, gather_seq_scatter_heads_qkv, safe_pad_operation
from lightx2v.models.networks.seedvr.utils.rope import get_na_rope
from lightx2v.models.networks.seedvr.utils.window import get_window_op

from .utils import apply_adaln_single, norm_no_weight


class SeedVRTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.num_layers = config["num_layers"]
        self.heads = config["heads"]
        self.head_dim = config["head_dim"]
        self.norm_type = config.get("norm", "fusedrms")
        self.qk_norm_type = config.get("qk_norm", "fusedrms")
        self.norm_eps = config.get("norm_eps", 1.0e-5)
        self.rope_type = config.get("rope_type", None)
        self.rope_dim = config.get("rope_dim", None)
        self.mlp_type = config.get("mlp_type", "swiglu")

        self.rope = get_na_rope(rope_type=self.rope_type, dim=self.rope_dim) if self.rope_type else None
        self.attn = FlashAttentionVarlen()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _get_branch(self, block_weight, name: str, branch: str):
        if block_weight.shared_weights:
            return getattr(block_weight, f"{name}_all")
        return getattr(block_weight, f"{name}_{branch}")

    def _attn_forward(self, block_weight, vid, txt, vid_shape, txt_shape, cache):
        qkv_vid = self._get_branch(block_weight, "attn_qkv", "vid").apply(vid)
        qkv_txt = self._get_branch(block_weight, "attn_qkv", "txt").apply(txt)

        qkv_vid = gather_seq_scatter_heads_qkv(qkv_vid, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid"))
        qkv_txt = gather_seq_scatter_heads_qkv(qkv_txt, seq_dim=0, qkv_shape=txt_shape, cache=cache.namespace("txt"))

        qkv_vid = rearrange(qkv_vid, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        qkv_txt = rearrange(qkv_txt, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = qkv_vid.unbind(1)
        txt_q, txt_k, txt_v = qkv_txt.unbind(1)

        norm_q_vid = self._get_branch(block_weight, "attn_norm_q", "vid")
        norm_q_txt = self._get_branch(block_weight, "attn_norm_q", "txt")
        norm_k_vid = self._get_branch(block_weight, "attn_norm_k", "vid")
        norm_k_txt = self._get_branch(block_weight, "attn_norm_k", "txt")

        vid_q = norm_q_vid.apply(vid_q)
        txt_q = norm_q_txt.apply(txt_q)
        vid_k = norm_k_vid.apply(vid_k)
        txt_k = norm_k_txt.apply(txt_k)

        if self.rope is not None:
            if self.rope.mm:
                vid_q, vid_k, txt_q, txt_k = self.rope(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache)
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, vid_shape, cache)

        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        all_len = cache("all_len", lambda: vid_len + txt_len)
        concat, unconcat = cache("mm_pnp", lambda: na.concat_idx(vid_len, txt_len))

        attn = self.attn(
            q=concat(vid_q, txt_q).bfloat16(),
            k=concat(vid_k, txt_k).bfloat16(),
            v=concat(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache("mm_seqlens", lambda: safe_pad_operation(all_len.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache("mm_maxlen", lambda: all_len.max().item()),
            max_seqlen_k=cache("mm_maxlen", lambda: all_len.max().item()),
        ).type_as(vid_q)

        attn = rearrange(attn, "l h d -> l (h d)")
        vid_out, txt_out = unconcat(attn)
        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        proj_out_vid = self._get_branch(block_weight, "attn_out", "vid")
        proj_out_txt = self._get_branch(block_weight, "attn_out", "txt")
        vid_out = proj_out_vid.apply(vid_out)
        txt_out = proj_out_txt.apply(txt_out)

        return vid_out, txt_out

    def _attn_window_forward(self, block_weight, vid, txt, vid_shape, txt_shape, cache):
        qkv_vid = self._get_branch(block_weight, "attn_qkv", "vid").apply(vid)
        qkv_txt = self._get_branch(block_weight, "attn_qkv", "txt").apply(txt)

        qkv_vid = gather_seq_scatter_heads_qkv(qkv_vid, seq_dim=0, qkv_shape=vid_shape, cache=cache.namespace("vid"))
        qkv_txt = gather_seq_scatter_heads_qkv(qkv_txt, seq_dim=0, qkv_shape=txt_shape, cache=cache.namespace("txt"))

        window = _triple(block_weight.window)
        window_method = block_weight.window_method
        window_op = get_window_op(window_method)
        cache_win = cache.namespace(f"{window_method}_{window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = window_op((t, h, w), window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        qkv_vid_win = window_partition(qkv_vid)

        qkv_vid_win = rearrange(qkv_vid_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        qkv_txt = rearrange(qkv_txt, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = qkv_vid_win.unbind(1)
        txt_q, txt_k, txt_v = qkv_txt.unbind(1)

        norm_q_vid = self._get_branch(block_weight, "attn_norm_q", "vid")
        norm_q_txt = self._get_branch(block_weight, "attn_norm_q", "txt")
        norm_k_vid = self._get_branch(block_weight, "attn_norm_k", "vid")
        norm_k_txt = self._get_branch(block_weight, "attn_norm_k", "txt")

        vid_q = norm_q_vid.apply(vid_q)
        txt_q = norm_q_txt.apply(txt_q)
        vid_k = norm_k_vid.apply(vid_k)
        txt_k = norm_k_txt.apply(txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win("mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count))

        if self.rope is not None:
            if self.rope.mm:
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = na.unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = [t for sub in txt_q_repeat for t in sub]
                txt_q_repeat, txt_shape_repeat = na.flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = na.unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = [t for sub in txt_k_repeat for t in sub]
                txt_k_repeat, _ = na.flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q,
                    vid_k,
                    window_shape,
                    txt_q_repeat,
                    txt_k_repeat,
                    txt_shape_repeat,
                    cache_win,
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

        out = self.attn(
            q=concat_win(vid_q, txt_q).bfloat16(),
            k=concat_win(vid_k, txt_k).bfloat16(),
            v=concat_win(vid_v, txt_v).bfloat16(),
            cu_seqlens_q=cache_win("vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()),
            cu_seqlens_k=cache_win("vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()),
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max().item()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max().item()),
        ).type_as(vid_q)

        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out = gather_heads_scatter_seq(vid_out, head_dim=1, seq_dim=0)
        txt_out = gather_heads_scatter_seq(txt_out, head_dim=1, seq_dim=0)

        proj_out_vid = self._get_branch(block_weight, "attn_out", "vid")
        proj_out_txt = self._get_branch(block_weight, "attn_out", "txt")
        vid_out = proj_out_vid.apply(vid_out)
        txt_out = proj_out_txt.apply(txt_out)

        return vid_out, txt_out

    def _infer_block(self, block_weight, vid, txt, vid_shape, txt_shape, emb, cache):
        vid_len = cache("vid_len", lambda: vid_shape.prod(-1))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        # Attention norm (no affine)
        vid_attn = norm_no_weight(vid, self.norm_type, self.norm_eps)
        txt_attn = norm_no_weight(txt, self.norm_type, self.norm_eps)

        # Ada (attn in)
        vid_attn = apply_adaln_single(
            vid_attn,
            emb,
            layer_idx=0,
            num_layers=2,
            mode="in",
            cache=cache,
            hid_len=vid_len,
            branch_tag="vid",
            shift=self._get_branch(block_weight, "ada_attn_shift", "vid").tensor,
            scale=self._get_branch(block_weight, "ada_attn_scale", "vid").tensor,
            gate=self._get_branch(block_weight, "ada_attn_gate", "vid").tensor,
        )
        txt_attn = apply_adaln_single(
            txt_attn,
            emb,
            layer_idx=0,
            num_layers=2,
            mode="in",
            cache=cache,
            hid_len=txt_len,
            branch_tag="txt",
            shift=self._get_branch(block_weight, "ada_attn_shift", "txt").tensor,
            scale=self._get_branch(block_weight, "ada_attn_scale", "txt").tensor,
            gate=self._get_branch(block_weight, "ada_attn_gate", "txt").tensor,
        )

        # Attention
        if block_weight.window is not None and block_weight.window_method is not None:
            vid_attn, txt_attn = self._attn_window_forward(block_weight, vid_attn, txt_attn, vid_shape, txt_shape, cache)
        else:
            vid_attn, txt_attn = self._attn_forward(block_weight, vid_attn, txt_attn, vid_shape, txt_shape, cache)

        # Ada (attn out)
        vid_attn = apply_adaln_single(
            vid_attn,
            emb,
            layer_idx=0,
            num_layers=2,
            mode="out",
            cache=cache,
            hid_len=vid_len,
            branch_tag="vid",
            shift=self._get_branch(block_weight, "ada_attn_shift", "vid").tensor,
            scale=self._get_branch(block_weight, "ada_attn_scale", "vid").tensor,
            gate=self._get_branch(block_weight, "ada_attn_gate", "vid").tensor,
        )
        txt_attn = apply_adaln_single(
            txt_attn,
            emb,
            layer_idx=0,
            num_layers=2,
            mode="out",
            cache=cache,
            hid_len=txt_len,
            branch_tag="txt",
            shift=self._get_branch(block_weight, "ada_attn_shift", "txt").tensor,
            scale=self._get_branch(block_weight, "ada_attn_scale", "txt").tensor,
            gate=self._get_branch(block_weight, "ada_attn_gate", "txt").tensor,
        )

        vid_attn = vid_attn + vid
        txt_attn = txt_attn + txt

        # MLP norm (no affine)
        vid_mlp = norm_no_weight(vid_attn, self.norm_type, self.norm_eps)
        if not block_weight.vid_only:
            txt_mlp = norm_no_weight(txt_attn, self.norm_type, self.norm_eps)
        else:
            txt_mlp = txt_attn

        # Ada (mlp in)
        vid_mlp = apply_adaln_single(
            vid_mlp,
            emb,
            layer_idx=1,
            num_layers=2,
            mode="in",
            cache=cache,
            hid_len=vid_len,
            branch_tag="vid",
            shift=self._get_branch(block_weight, "ada_mlp_shift", "vid").tensor,
            scale=self._get_branch(block_weight, "ada_mlp_scale", "vid").tensor,
            gate=self._get_branch(block_weight, "ada_mlp_gate", "vid").tensor,
        )
        if not block_weight.vid_only:
            txt_mlp = apply_adaln_single(
                txt_mlp,
                emb,
                layer_idx=1,
                num_layers=2,
                mode="in",
                cache=cache,
                hid_len=txt_len,
                branch_tag="txt",
                shift=self._get_branch(block_weight, "ada_mlp_shift", "txt").tensor,
                scale=self._get_branch(block_weight, "ada_mlp_scale", "txt").tensor,
                gate=self._get_branch(block_weight, "ada_mlp_gate", "txt").tensor,
            )

        # MLP
        if self.mlp_type == "swiglu":
            vid_mlp = self._get_branch(block_weight, "mlp_proj_out", "vid").apply(
                F.silu(self._get_branch(block_weight, "mlp_proj_in_gate", "vid").apply(vid_mlp)) * self._get_branch(block_weight, "mlp_proj_in", "vid").apply(vid_mlp)
            )
            if not block_weight.vid_only:
                txt_mlp = self._get_branch(block_weight, "mlp_proj_out", "txt").apply(
                    F.silu(self._get_branch(block_weight, "mlp_proj_in_gate", "txt").apply(txt_mlp)) * self._get_branch(block_weight, "mlp_proj_in", "txt").apply(txt_mlp)
                )
        else:
            vid_mlp = self._get_branch(block_weight, "mlp_proj_out", "vid").apply(F.gelu(self._get_branch(block_weight, "mlp_proj_in", "vid").apply(vid_mlp), approximate="tanh"))
            if not block_weight.vid_only:
                txt_mlp = self._get_branch(block_weight, "mlp_proj_out", "txt").apply(F.gelu(self._get_branch(block_weight, "mlp_proj_in", "txt").apply(txt_mlp), approximate="tanh"))

        # Ada (mlp out)
        vid_mlp = apply_adaln_single(
            vid_mlp,
            emb,
            layer_idx=1,
            num_layers=2,
            mode="out",
            cache=cache,
            hid_len=vid_len,
            branch_tag="vid",
            shift=self._get_branch(block_weight, "ada_mlp_shift", "vid").tensor,
            scale=self._get_branch(block_weight, "ada_mlp_scale", "vid").tensor,
            gate=self._get_branch(block_weight, "ada_mlp_gate", "vid").tensor,
        )
        if not block_weight.vid_only:
            txt_mlp = apply_adaln_single(
                txt_mlp,
                emb,
                layer_idx=1,
                num_layers=2,
                mode="out",
                cache=cache,
                hid_len=txt_len,
                branch_tag="txt",
                shift=self._get_branch(block_weight, "ada_mlp_shift", "txt").tensor,
                scale=self._get_branch(block_weight, "ada_mlp_scale", "txt").tensor,
                gate=self._get_branch(block_weight, "ada_mlp_gate", "txt").tensor,
            )

        vid_mlp = vid_mlp + vid_attn
        if not block_weight.vid_only:
            txt_mlp = txt_mlp + txt_attn
        else:
            txt_mlp = txt_attn

        return vid_mlp, txt_mlp, vid_shape, txt_shape

    @torch.no_grad()
    def infer(self, block_weights, pre_infer_out):
        vid = pre_infer_out.vid
        txt = pre_infer_out.txt
        vid_shape = pre_infer_out.vid_shape
        txt_shape = pre_infer_out.txt_shape
        emb = pre_infer_out.emb
        cache = pre_infer_out.cache

        for block_weight in block_weights:
            vid, txt, vid_shape, txt_shape = self._infer_block(block_weight, vid, txt, vid_shape, txt_shape, emb, cache)

        return vid, txt, vid_shape, txt_shape
