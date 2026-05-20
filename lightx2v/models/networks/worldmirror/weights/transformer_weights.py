"""WeightModule-based weight containers for the WorldMirror ViT backbone
and the CameraHead refinement trunk.

Scope (alignment checklist, decision 1 = A):
  * ``visual_geometry_transformer.frame_blocks.{0..depth-1}`` (ViT trunk,
    alternating-attention local branch)
  * ``visual_geometry_transformer.global_blocks.{0..depth-1}`` (ViT trunk,
    alternating-attention global branch)
  * ``cam_head.refine_net.{0..cam_trunk_depth-1}`` (CameraHead internal
    transformer stack)

Out of scope (kept as native ``nn.Module`` in :class:`WorldMirror`):
  * ``patch_embed`` (DinoV2 ViT wrapper, Conv2d + its own sub-transformer)
  * DPT heads (depth / norm / pts / gs_head)
  * ``gs_renderer`` (gsplat CUDA rasterization)
  * CameraHead ancillary parts (token_norm / out_norm / adapt_norm_gen /
    param_embed / init_token / param_predictor)

Each block registers:
  * ``attn_qkv`` (MM with bias)
  * ``attn_q_norm`` / ``attn_k_norm`` (LN, only when the underlying Block
    was built with ``qk_norm=True``, which is the VGT default)
  * ``attn_proj`` (MM with bias)
  * ``norm1`` / ``norm2`` (LN)
  * ``mlp_fc1`` (MM with bias)
  * ``mlp_fc2`` (MM with bias; ``Default-ForceFp32`` when ``use_fp32_fc2``)

Key-naming: hard-coded to match the HY-WorldMirror-2.0 checkpoint layout
(alignment decision 5 = "do not ship drift protection yet").
"""

from __future__ import annotations

import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.global_paras import CALIB
from lightx2v.utils.registry_factory import (
    CONV2D_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)

_DEFAULT_LN_EPS = 1e-6
# q_norm / k_norm in the ViT blocks operate on head_dim rather than embed_dim
# and use a slightly larger epsilon in the source impl (``nn.LayerNorm`` default
# = 1e-5) — keeping them separate avoids a silent numerical drift.
_QK_LN_EPS = 1e-5


class VitBlockWeights(WeightModule):
    """WeightModule container for one ViT-style Block (attn + mlp + norms)."""

    def __init__(
        self,
        block_name: str,
        *,
        mm_type: str,
        ln_type: str,
        qk_norm: bool,
        use_fp32_fc2: bool,
        use_fp32_attn_proj: bool = False,
        use_fp32_attn_qkv: bool = False,
        use_fp32_fc1: bool = False,
    ):
        super().__init__()
        self.block_name = block_name
        # --- Attention ---
        qkv_scheme = "Default-ForceFp32" if use_fp32_attn_qkv else mm_type
        self.add_module(
            "attn_qkv",
            MM_WEIGHT_REGISTER[qkv_scheme](
                f"{block_name}.attn.qkv.weight",
                f"{block_name}.attn.qkv.bias",
            ),
        )
        if qk_norm:
            self.add_module(
                "attn_q_norm",
                LN_WEIGHT_REGISTER[ln_type](
                    f"{block_name}.attn.q_norm.weight",
                    f"{block_name}.attn.q_norm.bias",
                    eps=_QK_LN_EPS,
                ),
            )
            self.add_module(
                "attn_k_norm",
                LN_WEIGHT_REGISTER[ln_type](
                    f"{block_name}.attn.k_norm.weight",
                    f"{block_name}.attn.k_norm.bias",
                    eps=_QK_LN_EPS,
                ),
            )
        # ``attn.proj`` is the heads-combiner and empirically sensitive to
        # per-tensor activation quantization — an optional fp32 override
        # mirrors the "keep the shortcut side clean" trick common in fp8
        # transformer recipes.
        proj_scheme = "Default-ForceFp32" if use_fp32_attn_proj else mm_type
        self.add_module(
            "attn_proj",
            MM_WEIGHT_REGISTER[proj_scheme](
                f"{block_name}.attn.proj.weight",
                f"{block_name}.attn.proj.bias",
            ),
        )
        # --- Pre-norms ---
        self.add_module(
            "norm1",
            LN_WEIGHT_REGISTER[ln_type](
                f"{block_name}.norm1.weight",
                f"{block_name}.norm1.bias",
                eps=_DEFAULT_LN_EPS,
            ),
        )
        self.add_module(
            "norm2",
            LN_WEIGHT_REGISTER[ln_type](
                f"{block_name}.norm2.weight",
                f"{block_name}.norm2.bias",
                eps=_DEFAULT_LN_EPS,
            ),
        )
        # --- MLP ---
        fc1_scheme = "Default-ForceFp32" if use_fp32_fc1 else mm_type
        self.add_module(
            "mlp_fc1",
            MM_WEIGHT_REGISTER[fc1_scheme](
                f"{block_name}.mlp.fc1.weight",
                f"{block_name}.mlp.fc1.bias",
            ),
        )
        fc2_scheme = "Default-ForceFp32" if use_fp32_fc2 else mm_type
        self.add_module(
            "mlp_fc2",
            MM_WEIGHT_REGISTER[fc2_scheme](
                f"{block_name}.mlp.fc2.weight",
                f"{block_name}.mlp.fc2.bias",
            ),
        )


class CameraHeadParamPredictorWeights(WeightModule):
    """WM container for ``cam_head.param_predictor`` (the MlpFP32 head).

    The upstream pipeline explicitly calls ``fc2`` on an ``x.float()`` tensor
    inside :meth:`MlpFP32.forward_infer`, so fc2 is always fp32. fc1 follows
    the block-level ``mm_type`` (can be quantized) unless the caller opts in
    to ``use_fp32_fc1=True``. Keeping the split mirrors ``VitBlockWeights``
    so upstream callers can reason about the quant scope the same way.
    """

    def __init__(self, *, mm_type: str, use_fp32_fc1: bool = False, use_fp32_fc2: bool = True):
        super().__init__()
        fc1_scheme = "Default-ForceFp32" if use_fp32_fc1 else mm_type
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER[fc1_scheme](
                "cam_head.param_predictor.fc1.weight",
                "cam_head.param_predictor.fc1.bias",
            ),
        )
        fc2_scheme = "Default-ForceFp32" if use_fp32_fc2 else mm_type
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER[fc2_scheme](
                "cam_head.param_predictor.fc2.weight",
                "cam_head.param_predictor.fc2.bias",
            ),
        )


class DPTOutputConv2Weights(WeightModule):
    """WM container for one DPT head's ``scratch.output_conv2`` sub-sequence.

    The source ``output_conv2`` is ``nn.Sequential(Conv2d[3x3], ReLU, Conv2d
    [1x1])`` — indices 0 and 2 are the conv leaves (index 1 is ReLU, no
    weights). ``dense_head.py`` calls this sequence on ``fused.float()``, so
    the layer is fp32-by-convention even when the rest of the model is bf16.
    Pin both convs to ``Default-ForceFp32`` so the WM path stays faithful.
    """

    def __init__(self, head_name: str, *, conv_scheme: str = "Default-ForceFp32"):
        super().__init__()
        # Conv2d(conv2_in_channels=features//2, 32, k=3, s=1, p=1)
        self.add_module(
            "conv_0",
            CONV2D_WEIGHT_REGISTER[conv_scheme](
                f"{head_name}.scratch.output_conv2.0.weight",
                f"{head_name}.scratch.output_conv2.0.bias",
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            ),
        )
        # Conv2d(32, output_dim, k=1, s=1, p=0)
        self.add_module(
            "conv_2",
            CONV2D_WEIGHT_REGISTER[conv_scheme](
                f"{head_name}.scratch.output_conv2.2.weight",
                f"{head_name}.scratch.output_conv2.2.bias",
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )


class WorldMirrorTransformerWeights(WeightModule):
    """Top-level WeightModule: VGT frame+global blocks, plus cam_head trunk.

    Parameters
    ----------
    config : dict-like
        Reads ``dit_quant_scheme`` (default ``"Default"``) and
        ``ln_type`` (default ``"torch"``).
    depth : int
        Number of frame / global blocks in ``visual_geometry_transformer``.
    cam_trunk_depth : int
        Number of blocks in ``cam_head.refine_net``.
    qk_norm : bool
        Mirrors ``Block(qk_norm=...)``; VGT uses True, cam_head uses False.
    """

    def __init__(
        self,
        config,
        *,
        depth: int = 24,
        cam_trunk_depth: int = 4,
        vgt_qk_norm: bool = True,
        cam_qk_norm: bool = False,
    ):
        super().__init__()
        self.config = config
        self.depth = depth
        self.cam_trunk_depth = cam_trunk_depth
        # run_calib forces every MM leaf to a calibration scheme so a
        # forward-pass sweep fills ``CALIB["absmax"]`` with per-layer input
        # absmax. Inference accuracy is intentionally identical to Default
        # (Calib variants do standard fp32 mm and only piggyback on the
        # forward to record stats), so the collected scales correspond to
        # the reference numerical path.
        #
        # Default is ``CalibMax`` (running-max absmax) because the fp8
        # activation quantizer benefits from a no-clip scale — WorldMirror
        # has occasional high-magnitude attention outliers that EMA would
        # smooth past. The legacy ``Calib`` scheme (EMA, designed for
        # NVFP4) is still reachable via ``run_calib_mode="ema"``.
        run_calib = bool(config.get("run_calib", False)) if hasattr(config, "get") else False
        calib_mode = config.get("run_calib_mode", "max") if hasattr(config, "get") else "max"
        if run_calib:
            mm_type = {"max": "CalibMax", "ema": "Calib"}.get(calib_mode, "CalibMax")
        else:
            mm_type = config.get("dit_quant_scheme", "Default")
        self._mm_type = mm_type
        ln_type = config.get("ln_norm_type", "torch")

        # Block-level fp32 protection: forcibly keep the first
        # ``fp32_first_n_blocks`` and last ``fp32_last_n_blocks`` of each
        # VGT branch in fp32 regardless of ``dit_quant_scheme``. The first
        # few blocks shape patch features directly from the Dino backbone
        # and the last few feed DPT heads; empirically both regions are
        # disproportionately sensitive to fp8 activation quantization.
        fp32_first = int(config.get("fp32_first_n_blocks", 0) or 0) if hasattr(config, "get") else 0
        fp32_last = int(config.get("fp32_last_n_blocks", 0) or 0) if hasattr(config, "get") else 0

        def _block_mm_type(i: int) -> str:
            if i < fp32_first or i >= depth - fp32_last:
                return "Default-ForceFp32"
            return mm_type

        # Per-sublayer edge protection (task ζ): while ``use_fp32_<sublayer>``
        # forces that layer fp32 across every block (most conservative),
        # ``fp32_<sublayer>_first_n_blocks`` / ``fp32_<sublayer>_last_n_blocks``
        # constrain the fp32 protection to the first/last N blocks of each
        # branch — leaves the middle blocks quantized. Useful for ``attn.proj``
        # which is empirically sensitive at the top and bottom of the stack
        # (feeding DPT heads and rendering) but tolerates fp8 in the middle
        # of the ViT trunk. Leaving a knob per sublayer means we can push
        # proj into fp8 in 70%+ of blocks while still protecting the edges.
        def _sublayer_edge(key: str):
            first = int(config.get(f"fp32_{key}_first_n_blocks", 0) or 0) if hasattr(config, "get") else 0
            last = int(config.get(f"fp32_{key}_last_n_blocks", 0) or 0) if hasattr(config, "get") else 0
            return first, last

        _qkv_first, _qkv_last = _sublayer_edge("attn_qkv")
        _proj_first, _proj_last = _sublayer_edge("attn_proj")
        _fc1_first, _fc1_last = _sublayer_edge("fc1")
        _fc2_first, _fc2_last = _sublayer_edge("fc2")
        _sublayer_edges = {
            "use_fp32_attn_qkv": (_qkv_first, _qkv_last),
            "use_fp32_attn_proj": (_proj_first, _proj_last),
            "use_fp32_fc1": (_fc1_first, _fc1_last),
            "use_fp32_fc2": (_fc2_first, _fc2_last),
        }

        def _block_use_fp32(i: int, key: str) -> bool:
            if i < fp32_first or i >= depth - fp32_last:
                return True
            # Per-sublayer edge protection overrides the global flag only
            # at the protected edges — middle blocks stay with the global
            # flag behaviour.
            edge_first, edge_last = _sublayer_edges.get(key, (0, 0))
            if edge_first and i < edge_first:
                return True
            if edge_last and i >= depth - edge_last:
                return True
            return bool(config.get(key, False)) if hasattr(config, "get") else False

        self.add_module(
            "frame_blocks",
            WeightModuleList(
                VitBlockWeights(
                    f"visual_geometry_transformer.frame_blocks.{i}",
                    mm_type=_block_mm_type(i),
                    ln_type=ln_type,
                    qk_norm=vgt_qk_norm,
                    use_fp32_fc2=_block_use_fp32(i, "use_fp32_fc2"),
                    use_fp32_attn_proj=_block_use_fp32(i, "use_fp32_attn_proj"),
                    use_fp32_attn_qkv=_block_use_fp32(i, "use_fp32_attn_qkv"),
                    use_fp32_fc1=_block_use_fp32(i, "use_fp32_fc1"),
                )
                for i in range(depth)
            ),
        )
        self.add_module(
            "global_blocks",
            WeightModuleList(
                VitBlockWeights(
                    f"visual_geometry_transformer.global_blocks.{i}",
                    mm_type=_block_mm_type(i),
                    ln_type=ln_type,
                    qk_norm=vgt_qk_norm,
                    use_fp32_fc2=_block_use_fp32(i, "use_fp32_fc2"),
                    use_fp32_attn_proj=_block_use_fp32(i, "use_fp32_attn_proj"),
                    use_fp32_attn_qkv=_block_use_fp32(i, "use_fp32_attn_qkv"),
                    use_fp32_fc1=_block_use_fp32(i, "use_fp32_fc1"),
                )
                for i in range(depth)
            ),
        )
        # cam_head.refine_net processes camera features — perturbations here
        # shift per-view camera poses, which in turn shift every splat's
        # world-space position. Keeping it fp32 is cheap (4 blocks, ~15 MB)
        # and recovers the last bit of gaussian-count agreement. Opt in via
        # ``cam_refine_fp32`` in the runtime config.
        cam_refine_fp32 = bool(config.get("cam_refine_fp32", False)) if hasattr(config, "get") else False
        cam_mm_type = "Default-ForceFp32" if cam_refine_fp32 else mm_type
        self.add_module(
            "cam_refine_blocks",
            WeightModuleList(
                VitBlockWeights(
                    f"cam_head.refine_net.{i}",
                    mm_type=cam_mm_type,
                    ln_type=ln_type,
                    qk_norm=cam_qk_norm,
                    use_fp32_fc2=cam_refine_fp32 or (bool(config.get("use_fp32_fc2", False)) if hasattr(config, "get") else False),
                    use_fp32_attn_proj=cam_refine_fp32 or (bool(config.get("use_fp32_attn_proj", False)) if hasattr(config, "get") else False),
                    use_fp32_attn_qkv=cam_refine_fp32 or (bool(config.get("use_fp32_attn_qkv", False)) if hasattr(config, "get") else False),
                    use_fp32_fc1=cam_refine_fp32 or (bool(config.get("use_fp32_fc1", False)) if hasattr(config, "get") else False),
                )
                for i in range(cam_trunk_depth)
            ),
        )

        # Extended scope (task δ): ``cam_head.param_predictor`` (MlpFP32,
        # fc1/fc2 Linear) and every DPT head's ``scratch.output_conv2[0/2]``
        # (Conv2d), opt-in via ``wm_extended_scope=true``. These layers are
        # physically tiny (<~10 MB total) and the upstream pipeline already
        # runs them at fp32 — putting them in the WM tree just formalises
        # that in the Default-ForceFp32 scheme, cleaning up the "identify
        # fp32 critical by isinstance" heuristic in bf16 cast.
        self._extended_scope = bool(config.get("wm_extended_scope", False)) if hasattr(config, "get") else False
        if self._extended_scope:
            # cam_head.param_predictor: fc1 follows quant scheme; fc2 is
            # always fp32 because MlpFP32.forward_infer casts the input.
            self.add_module(
                "cam_param_predictor",
                CameraHeadParamPredictorWeights(
                    mm_type=mm_type,
                    use_fp32_fc1=bool(config.get("use_fp32_param_predictor_fc1", False)) if hasattr(config, "get") else False,
                    use_fp32_fc2=True,
                ),
            )
            # DPT heads: depth / norm / pts / gs. The model-level enable
            # flags live on ``inner_model`` (not reachable from here), but
            # ``disable_heads`` does pass through the runtime config, so
            # consult it here and skip registering the corresponding WM
            # output_conv2 container — registering it would just immediately
            # KeyError at load time because the disabled head's keys are
            # not in the safetensors. Mapping mirrors model.py:_HEAD_MAPPING.
            disable_heads = config.get("disable_heads", None) if hasattr(config, "get") else None
            disabled = set(disable_heads or [])
            head_disable_map = {
                "depth": "depth_head",
                "normal": "norm_head",
                "points": "pts_head",
                "gs": "gs_head",
            }
            disabled_attrs = {head_disable_map[n] for n in disabled if n in head_disable_map}
            for head_name in ("depth_head", "norm_head", "pts_head", "gs_head"):
                if head_name in disabled_attrs:
                    continue
                self.add_module(
                    f"{head_name}_output_conv2",
                    DPTOutputConv2Weights(
                        head_name,
                        conv_scheme="Default-ForceFp32",
                    ),
                )
            # ``cam_head.param_predictor`` is also a head-conditional leaf:
            # if the user disables the camera head we must not register it
            # either (the inner_model won't even build cam_head, and the
            # safetensors keys won't exist).
            if "camera" in disabled and hasattr(self, "cam_param_predictor"):
                # ``add_module`` already happened above; remove it.
                del self._modules["cam_param_predictor"]

    # ------------------------------------------------------------------
    # Auto-quantize hook: for quant schemes that carry a ``load_func`` but
    # whose base ``load`` doesn't dispatch to it, we invoke the auto-quant
    # path explicitly when the user opts in via ``weight_auto_quant=True``.
    # This is how we can feed plain fp32 safetensors into fp8/int8 schemes
    # without first running an offline calibration pass.
    # ------------------------------------------------------------------
    def load(self, weight_dict):
        cfg = self.config
        auto_quant = bool(cfg.get("weight_auto_quant", False)) if hasattr(cfg, "get") else False
        if not auto_quant:
            super().load(weight_dict)
            return
        for _, leaf in self._walk_leaves():
            load_func = getattr(leaf, "load_func", None)
            if load_func is None or getattr(leaf, "weight_scale_name", None) is None:
                leaf.load(weight_dict)
                continue
            # Make sure the leaf's ``self.config`` has ``weight_auto_quant``
            # so its ``load_<scheme>`` picks the auto-quant branch.
            leaf.set_config(dict(cfg))
            load_func(weight_dict)
            # load_<scheme> populates weight+weight_scale only; bring bias
            # over separately (auto-quant paths assume the caller does this).
            bias_name = getattr(leaf, "bias_name", None)
            if bias_name is not None and bias_name in weight_dict:
                leaf.bias = weight_dict[bias_name]
            else:
                leaf.bias = None
            # fp8-pertensor auto-quant additionally needs a pre-calibrated
            # ``input_scale`` from the calibration pass. The runner merges
            # the calibration safetensors into ``weight_dict`` before load,
            # so it shows up under ``<name>.input_scale`` exactly where the
            # pre-quantized load_quantized path would find it.
            input_scale_name = getattr(leaf, "input_scale_name", None)
            if input_scale_name is not None and input_scale_name in weight_dict:
                leaf.input_scale = weight_dict[input_scale_name]
            if hasattr(leaf, "post_process"):
                leaf.post_process()

    # ------------------------------------------------------------------
    # Calibration export: after running inference under ``run_calib=True``
    # (which swaps every MM leaf to the ``Calib`` scheme), the global
    # ``CALIB["absmax"]`` dict carries the per-weight-name absmax for
    # activations. Convert each entry to an fp8-pertensor ``input_scale =
    # absmax / 448`` and save to a safetensors so subsequent runs can load
    # it and feed it into the quantized load path.
    # ------------------------------------------------------------------
    def export_calibration(self, path: str, fp8_e4m3_max: float = 448.0):
        """Write per-layer input_scale safetensors for the WM leaves.

        Called after one or more inference runs performed with
        ``run_calib=True``. The resulting file contains keys of the form
        ``<weight_name stripped of .weight>.input_scale`` holding a 1-dim
        fp32 tensor equal to ``absmax_input / 448`` (the fp8 e4m3 max).

        Returns the number of scale entries written.
        """
        from safetensors.torch import save_file

        out: dict[str, torch.Tensor] = {}
        for _, leaf in self._walk_leaves():
            wn = getattr(leaf, "weight_name", None)
            if wn is None:
                continue
            absmax = CALIB.get("absmax", {}).get(wn)
            if absmax is None:
                continue
            scale = (absmax.float() / fp8_e4m3_max).to(torch.float32).reshape(1)
            scale_key = wn.removesuffix(".weight") + ".input_scale"
            out[scale_key] = scale.clone()
        save_file(out, path)
        return len(out)

    # ------------------------------------------------------------------
    # Helpers for bf16 / device cast in tandem with the nn.Module wrapper.
    # ------------------------------------------------------------------
    def cast_dtype(self, dtype):
        """Cast every loaded MM/LN tensor to ``dtype`` in-place.

        Skip ``Default-ForceFp32`` leaves — those are defined as sticky fp32.
        Called from :class:`WorldMirrorWeightModel` when switching to bf16
        after load. Handles both the default (``.weight`` on GPU) and
        ``cpu_offload`` (``.pin_weight`` on pinned CPU) variants — the
        latter also needs re-pinning because ``torch.empty(..., pin_memory=
        True).to(dtype)`` would drop the pin_memory flag.
        """
        import torch

        for _, leaf in self._walk_leaves():
            is_force_fp32 = leaf.__class__.__name__ in (
                "MMWeightForceFp32",
                "Conv2dWeightForceFp32",
            )
            if is_force_fp32:
                continue
            # Quantized leaves own their storage dtype (fp8/int8/nvfp4/...).
            # The MM kernel signatures forbid bf16 weight tensors, so a
            # blanket cast_dtype would corrupt them — e.g. fp8-pertensor's
            # ``torch._scaled_mm`` requires ``mat2`` to be Float8.
            # ``apply()`` already down-casts bias to the active mm_dtype
            # per call, so we don't need to cast bias here either.
            is_quant = getattr(leaf, "weight_scale_name", None) is not None
            if is_quant:
                continue
            for attr in ("weight", "bias"):
                t = getattr(leaf, attr, None)
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    setattr(leaf, attr, t.to(dtype))
            # cpu_offload path: cast pinned buffers. ``.to(dtype)`` on a
            # pinned tensor does not preserve pin_memory on the destination,
            # so allocate a fresh pinned buffer and copy into it. The
            # weight's pin_tensor is typically a ``.t()`` view on a
            # contiguous [out, in] storage; preserve that view shape so
            # downstream ``.to(AI_DEVICE)`` yields the same logical layout.
            for attr in ("pin_weight", "pin_bias"):
                t = getattr(leaf, attr, None)
                if not (isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype):
                    continue
                transposed = not t.is_contiguous()
                base = t.t().contiguous() if transposed else t
                new_pin = torch.empty(base.shape, pin_memory=True, dtype=dtype)
                new_pin.copy_(base)
                if transposed:
                    new_pin = new_pin.t()
                setattr(leaf, attr, new_pin)

    def init_lazy(self):
        """Pre-set block-swap leaf live attributes to ``None`` so the adapter
        forward / hooks don't ``AttributeError`` before the first pre-hook
        has a chance to populate a block.

        Used by the true-lazy load path (``cpu_offload=true, lazy_load=true``)
        where neither ``pin_weight`` nor ``weight`` is populated at load
        time — the per-block pre-hook reads straight from the safetensors
        mmap into a GPU tensor and the post-hook drops it again.

        Only the block-swap leaves (VGT frame/global + cam_refine) go
        through this pre-None step. Extended-scope leaves (param_predictor,
        DPT output_conv2) are resident-GPU and get loaded eagerly by the
        model's :meth:`_eager_load_extended_from_file`, so resetting them
        here would just blow their weights away.
        """
        block_swap_prefixes = ("frame_blocks.", "global_blocks.", "cam_refine_blocks.")
        for leaf_name, leaf in self._walk_leaves():
            if not leaf_name.startswith(block_swap_prefixes):
                continue
            for attr in ("weight", "bias", "weight_scale", "input_scale"):
                if not hasattr(leaf, attr):
                    setattr(leaf, attr, None)

    def _walk_leaves(self, prefix: str = ""):
        """Yield (name, leaf) for every MM/LN leaf in the tree."""

        def _recurse(mod, pfx):
            for name, sub in mod._modules.items():
                if sub is None:
                    continue
                if isinstance(sub, (WeightModule, WeightModuleList)):
                    yield from _recurse(sub, pfx + name + ".")
                else:
                    yield pfx + name, sub

        yield from _recurse(self, prefix)
