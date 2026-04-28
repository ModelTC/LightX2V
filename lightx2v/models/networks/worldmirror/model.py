"""Top-level WorldMirror model for LightX2V.

Wraps the raw ``hyworldmirror.WorldMirror`` nn.Module and a side-car
:class:`WorldMirrorTransformerWeights` WeightModule. At load time the
ViT-backbone and CameraHead-trunk ``nn.Linear`` / ``nn.LayerNorm`` leaves
are swapped for thin adapters that delegate the compute to the registered
MMWeight / LNWeight objects, so those layers can participate in LightX2V
quantization / cpu_offload / lazy_load schemes.

This intentionally does **not** inherit from ``BaseTransformerModel`` — the
WorldMirror inference pipeline is single-step multi-head + gsplat
rasterization, not diffusion; the three-phase ``pre_infer / transformer_infer
/ post_infer`` abstraction doesn't fit and would only add ceremony.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from loguru import logger

from lightx2v.common.ops.utils import move_attr_to_cpu, move_attr_to_cuda
from lightx2v_platform.base.global_var import AI_DEVICE

from .models.layers.mlp import MlpFP32
from .models.models.worldmirror import WorldMirror
from .weights.transformer_weights import (
    VitBlockWeights,
    WorldMirrorTransformerWeights,
)


# ---------------------------------------------------------------------------
# Leaf adapters — pretend to be nn.Linear / nn.LayerNorm, but delegate to
# a registered MM / LN weight. Keeping them as nn.Module lets ``.eval()`` /
# ``.train()`` / parent ``named_modules()`` keep working.
# ---------------------------------------------------------------------------
class _MMLinearAdapter(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that routes through an MMWeight."""

    def __init__(self, mm_weight, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.mm_weight = mm_weight
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        # The ``Default`` / ``Default-ForceFp32`` scheme stores the weight
        # pre-transposed (shape [in, out]) and provides no dedicated kernel;
        # for those we go through ``F.linear`` to match ``nn.Linear``'s
        # autocast behaviour *exactly* (empirically, ``torch.addmm`` picks
        # a different cublas algo and compounds into MAE > 1e-3 after 48
        # blocks). All other schemes (fp8/int8/nvfp4/…) have their own
        # apply() kernel and must use that instead of F.linear.
        self._is_default = mm_weight.__class__.__name__ in (
            "MMWeight",
            "MMWeightForceFp32",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MMWeight.apply expects 2D; flatten leading dims.
        flat = x.reshape(-1, x.shape[-1])
        w = self.mm_weight.weight
        if w is None:
            raise RuntimeError(f"MMWeight for {self.mm_weight.weight_name!r} has no loaded weight; did the state_dict load miss this key?")

        if self._is_default:
            b = getattr(self.mm_weight, "bias", None)
            # Match autocast semantics: bf16/fp16 activation meeting an
            # fp32 weight ⇒ cast the weight down to the activation dtype,
            # as ``nn.Linear`` does under ``torch.amp.autocast``.
            if flat.dtype != w.dtype:
                if flat.is_floating_point() and flat.dtype in (torch.float16, torch.bfloat16):
                    w_use = w.to(flat.dtype)
                    b_use = b.to(flat.dtype) if b is not None else None
                else:
                    flat = flat.to(w.dtype)
                    w_use = w
                    b_use = b
            else:
                w_use = w
                b_use = b
            flat_c = flat.contiguous()
            # MMWeight stores ``w`` pre-transposed (shape [in, out]); give
            # F.linear the canonical (out, in) layout via ``.t()``.
            out = torch.nn.functional.linear(flat_c, w_use.t(), b_use)
        else:
            # Quantization path — each scheme handles its own activation
            # quantization + GEMM inside ``apply``. Leave dtype shenanigans
            # to the kernel; it knows what it wants.
            out = self.mm_weight.apply(flat.contiguous())
        return out.reshape(*x.shape[:-1], self.out_features)

    # nn.Module.to() would recurse into self.mm_weight (which isn't an
    # nn.Module and has no params() entry), but any dtype/device change
    # on the container should NOT silently re-cast the MMWeight tensor —
    # quant schemes own their own storage. Make it a no-op.
    def _apply(self, fn, *args, **kwargs):  # noqa: D401
        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, has_bias={self.has_bias}, weight_name={self.mm_weight.weight_name!r}"


class _LNAdapter(nn.Module):
    """Drop-in replacement for ``nn.LayerNorm`` that routes through an LNWeight."""

    def __init__(self, ln_weight, normalized_shape, eps: float):
        super().__init__()
        self.ln_weight = ln_weight
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.ln_weight._get_actual_weight()
        b = self.ln_weight._get_actual_bias()
        # Match autocast semantics for a bf16/fp16 activation meeting an
        # fp32 LN weight — cast the weight down so the LN runs in the
        # activation dtype (what ``nn.LayerNorm`` does under autocast).
        if w is not None and x.is_floating_point() and w.dtype != x.dtype and x.dtype in (torch.float16, torch.bfloat16):
            w = w.to(x.dtype)
            if b is not None:
                b = b.to(x.dtype)
        return torch.nn.functional.layer_norm(
            x,
            self.normalized_shape,
            w,
            b,
            self.eps,
        )

    def _apply(self, fn, *args, **kwargs):  # noqa: D401
        return self

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, weight_name={self.ln_weight.weight_name!r}"


class _Conv2dAdapter(nn.Module):
    """Drop-in replacement for ``nn.Conv2d`` that routes through a Conv2dWeight."""

    def __init__(self, conv_weight, *, in_channels: int, out_channels: int, kernel_size, stride, padding, has_bias: bool):
        super().__init__()
        self.conv_weight = conv_weight
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = has_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_weight.apply(x)

    # Mirror the MMLinearAdapter behaviour: don't let parent ``.to(dtype)``
    # silently re-cast the Conv2dWeight's stored tensor (it owns its own
    # storage lifecycle; ``Default-ForceFp32`` in particular is sticky fp32
    # and must not follow a parent module's bf16 cast).
    def _apply(self, fn, *args, **kwargs):  # noqa: D401
        return self

    def extra_repr(self) -> str:
        return f"in={self.in_channels}, out={self.out_channels}, k={self.kernel_size}, s={self.stride}, p={self.padding}, has_bias={self.has_bias}, weight_name={self.conv_weight.weight_name!r}"


# ---------------------------------------------------------------------------
# Head disable mapping — lives on the model now (decision 4 = B): runner
# calls ``model.disable_heads(names)``, doesn't mutate model sub-modules.
# ---------------------------------------------------------------------------
_HEAD_MAPPING = {
    "camera": ("enable_cam", ["cam_head"]),
    "depth": ("enable_depth", ["depth_head"]),
    "normal": ("enable_norm", ["norm_head"]),
    "points": ("enable_pts", ["pts_head"]),
    "gs": ("enable_gs", ["gs_head", "gs_renderer"]),
}
HEAD_NAMES = tuple(_HEAD_MAPPING)


def collect_fp32_critical_modules(model: nn.Module):
    """Find modules that must stay fp32 under mixed precision.

    Mirrors the HY-World-2.0 pipeline rule: keep ``MlpFP32.fc2`` and
    ``scratch.output_conv2`` (dense head tail conv) in fp32. Used by the
    bf16 cast path in :class:`WorldMirrorWeightModel`.
    """
    critical = set()
    for _, module in model.named_modules():
        if isinstance(module, MlpFP32) and hasattr(module, "fc2"):
            if any(p.dtype == torch.float32 for p in module.fc2.parameters()):
                critical.add(module.fc2)
        if hasattr(module, "scratch") and hasattr(module.scratch, "output_conv2"):
            oc2 = module.scratch.output_conv2
            if any(p.dtype == torch.float32 for p in oc2.parameters()):
                critical.add(oc2)
    return critical


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------
class WorldMirrorWeightModel:
    """High-level WorldMirror runner handle.

    Layout
    ------
    * ``self.inner_model`` — the raw ``WorldMirror`` ``nn.Module``. Houses the
      DPT heads, gs_renderer, patch_embed (DinoV2 Conv2d + sub-ViT),
      learnable tokens, rope buffers, and (post-swap) adapter-wrapped
      ViT/CameraHead-trunk linear/norm leaves.
    * ``self.transformer_weights`` — the :class:`WorldMirrorTransformerWeights`
      WeightModule owning registered MM/LN weights for the ViT backbone
      (48 blocks) and cam_head trunk (4 blocks).

    Not an ``nn.Module`` — exposes passthrough properties for the few
    attributes the runner needs (``enable_bf16``, ``gs_renderer``,
    ``config``, ``sp_size``).
    """

    def __init__(self, model_cfg: dict, runtime_cfg: dict = None):
        """
        Parameters
        ----------
        model_cfg : dict
            Architecture config for the raw ``WorldMirror`` nn.Module
            (img_size, patch_size, depth, enable_* flags, ...). Loaded from
            ``{model_path}/{subfolder}/config.{json,yaml}``.
        runtime_cfg : dict, optional
            LightX2V-side runtime config (``dit_quant_scheme`` etc.). The
            WeightModule tree picks its MM/LN scheme off this dict at
            construction — not off ``model_cfg`` — because the architecture
            config has no concept of quantization.
        """
        self.model_cfg = dict(model_cfg)
        self.runtime_cfg = dict(runtime_cfg) if runtime_cfg else {}
        self.inner_model: WorldMirror = WorldMirror(**model_cfg)
        # Peek at actual depths so WeightModule names track WorldMirror.
        depth = self.inner_model.depth
        cam_trunk_depth = getattr(self.inner_model.cam_head, "depth", 4) if getattr(self.inner_model, "enable_cam", False) else 4
        # Footgun guard: ``wm_extended_scope=true`` brings
        # ``cam_head.param_predictor.fc1`` into the WM tree under the
        # global ``dit_quant_scheme``. fp8-pertensor wants a per-layer
        # ``input_scale`` from calibration; the shipped calibration
        # safetensors only covers the 208 block-swap leaves, so an unguarded
        # config crashes inside ``MMWeightWfp8tensorAfp8tensordynamic.apply``
        # with ``AttributeError: ... has no attribute 'input_scale'``. Detect
        # the missing key and auto-fallback fc1 to fp32, with a warning, so
        # users who simply flip ``wm_extended_scope=true`` on the production
        # fp8 config don't have to know about ``use_fp32_param_predictor_fc1``.
        self._guard_extended_scope_fp8_pp_fc1()
        # Build empty WeightModule container; state is loaded in load_state_dict.
        self.transformer_weights = WorldMirrorTransformerWeights(
            self.runtime_cfg,
            depth=depth,
            cam_trunk_depth=cam_trunk_depth,
            vgt_qk_norm=True,  # VGT blocks are built with qk_norm=True
            cam_qk_norm=False,  # CameraHead refine_net uses default qk_norm=False
        )
        self._adapters_installed = False
        # Block-level CPU <-> GPU swap (decision T3, Stage 3). When true the
        # WeightModule leaves are held in pinned CPU memory and each block
        # is moved to GPU by a forward_pre_hook, then offloaded by a
        # forward_hook.
        self._cpu_offload = bool(self.runtime_cfg.get("cpu_offload", False))
        self._offload_hooks_installed = False
        # lazy_load (decision T3 part 2): when combined with cpu_offload,
        # the per-block WeightModule leaves read their weight tensors from
        # a safetensors file on demand (per block forward) and release
        # both the GPU *and* pinned-CPU copies after the forward returns.
        # Unused when cpu_offload is False. Reduces resident CPU RAM for
        # the pin_weight buffers at the cost of a safetensors mmap read
        # per block per forward (cheap in practice for single-scene
        # inference).
        self._lazy_load = bool(self.runtime_cfg.get("lazy_load", False)) and self._cpu_offload
        self._lazy_load_file = None  # populated by load_from_safetensors
        self._lazy_file_handle = None  # persistent safe_open handle, lazy_load only
        self._lazy_file_keys = None  # cached key set for the handle
        # Under lazy_load, ``cam_head.refine_net`` fires 16× per scene
        # (4 refinement iterations × 4 blocks). Re-reading the same
        # ~200 MB of weights that many times dominates the wall-clock.
        # Keeping the 4 cam_refine blocks permanently GPU-resident costs
        # ~200 MB extra GPU peak but shaves ~300 ms off inference. Opt
        # out via ``lazy_cam_resident=false`` to recover the GPU memory
        # at the cost of latency.
        self._lazy_cam_resident = bool(self.runtime_cfg.get("lazy_cam_resident", True)) if self._lazy_load else False
        # Extended WM scope (task δ/ε): also wrap ``cam_head.param_predictor``
        # and every DPT head's ``scratch.output_conv2[0/2]`` with adapters
        # that delegate to WM leaves. These layers are always fp32 in the
        # upstream pipeline (MlpFP32.forward_infer casts input to float;
        # DPT ``output_conv2(fused.float())``) — bringing them into WM
        # formalises that under the ``Default-ForceFp32`` scheme and
        # removes the ``isinstance(MlpFP32)`` / ``hasattr(scratch)``
        # heuristic in bf16 cast. Opt in via ``wm_extended_scope=true``
        # so existing configs stay untouched.
        self._extended_scope = bool(self.runtime_cfg.get("wm_extended_scope", False))

    # ------------------------------------------------------------------
    # Footgun guards (run before transformer_weights construction so the
    # mutated runtime_cfg picks up the override).
    # ------------------------------------------------------------------
    _PP_FC1_INPUT_SCALE_KEY = "cam_head.param_predictor.fc1.input_scale"

    def _guard_extended_scope_fp8_pp_fc1(self):
        """Auto-fallback ``cam_head.param_predictor.fc1`` to fp32 when the
        configured fp8 quant scheme needs an ``input_scale`` that the
        calibration safetensors does not provide.

        Triggered for ``wm_extended_scope=true`` + a fp8-pertensor-family
        scheme. Other quant schemes use dynamic per-token activation
        scaling and do not consume a pre-baked ``input_scale``, so they
        do not crash on the missing fc1 key — leave them alone.
        """
        cfg = self.runtime_cfg
        if not bool(cfg.get("wm_extended_scope", False)):
            return
        scheme = cfg.get("dit_quant_scheme", "Default") or "Default"
        # Only fp8-pertensor (per-layer pre-baked input_scale) is hit by the
        # missing-key footgun; dynamic fp8/int8 schemes self-quantize.
        if scheme != "fp8-pertensor":
            return
        if bool(cfg.get("use_fp32_param_predictor_fc1", False)):
            return
        # Look up the calibration safetensors. A missing file or a file
        # that lacks the fc1 input_scale both fall through to the fp32
        # override; otherwise the config is genuinely fine.
        scale_file = cfg.get("input_scale_file", None)
        has_key = False
        if scale_file:
            try:
                from safetensors import safe_open

                with safe_open(scale_file, framework="pt", device="cpu") as f:
                    has_key = self._PP_FC1_INPUT_SCALE_KEY in set(f.keys())
            except Exception as exc:
                logger.warning(f"[WorldMirror] could not inspect input_scale_file {scale_file!r} ({exc}); assuming missing fc1 input_scale and forcing use_fp32_param_predictor_fc1=true.")
        if has_key:
            return
        logger.warning(
            "[WorldMirror] wm_extended_scope=true + dit_quant_scheme=fp8-pertensor "
            f"but {self._PP_FC1_INPUT_SCALE_KEY!r} is not in the calibration "
            "file — auto-setting use_fp32_param_predictor_fc1=true to keep "
            "cam_head.param_predictor.fc1 in fp32. To silence this warning, "
            "set the flag explicitly in your config or rerun calibration "
            "with this layer covered."
        )
        self.runtime_cfg["use_fp32_param_predictor_fc1"] = True

    # ------------------------------------------------------------------
    # Passthroughs
    # ------------------------------------------------------------------
    @property
    def enable_bf16(self) -> bool:
        return self.inner_model.enable_bf16

    @property
    def enable_cam(self) -> bool:
        return getattr(self.inner_model, "enable_cam", False)

    @property
    def gs_renderer(self):
        return getattr(self.inner_model, "gs_renderer", None)

    @property
    def config(self):
        return self.inner_model.config

    @property
    def sp_size(self) -> int:
        return self.inner_model.sp_size

    def eval(self):
        self.inner_model.eval()
        return self

    def to_device(self, device):
        self.inner_model.to(device)
        return self

    def parameters(self):
        return self.inner_model.parameters()

    def state_dict(self):  # used by selective loader
        return self.inner_model.state_dict()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_from_safetensors(self, state: Dict[str, torch.Tensor]):
        """Load a safetensors-style state_dict into both containers.

        The ``state`` is expected to live on CPU (as returned by
        ``safetensors.torch.load_file``). For the WeightModule branch:

          * Default path (``cpu_offload=False``): tensors are moved to the
            model's device first so :func:`create_default_tensors` takes the
            GPU branch, storing the transposed weight directly in
            ``.weight`` — no pinned-CPU indirection.
          * cpu_offload path (``cpu_offload=True``): tensors remain on CPU
            so :func:`create_default_tensors` takes the CPU branch, writing
            them to ``pin_weight`` only (``weight`` stays ``None``). The
            ``pin_weight → weight`` promotion happens once per block-forward
            inside :meth:`_install_cpu_offload_hooks`.
        """
        # 1. nn.Module path: selective (shape-matched) strict load, matching
        #    the HY-World-2.0 behaviour of ignoring keys like
        #    ``attn.rope.periods`` that are not tensors.
        self._selective_load_nn(state)

        # 2. WeightModule path: assemble a device-resident view of just the
        #    keys the WeightModule leaves will consume, then hand it off.
        device = next(self.inner_model.parameters()).device
        wm_state = self._device_view_for_wm(state, device)
        self.transformer_weights.load(wm_state)

        # 3. Wire adapters: replace the target nn.Linear/LayerNorm leaves
        #    in inner_model with adapters that call through the WM leaves.
        self._install_adapters()

        # 4. When cpu_offload is enabled, register per-block forward hooks
        #    that swap the block's WM leaves in/out of GPU memory on demand.
        if self._cpu_offload:
            self._install_cpu_offload_hooks()
        return self

    def _device_view_for_wm(self, state, device):
        """Return a shallow ``{name: tensor}`` for WM keys only.

        When ``cpu_offload`` is enabled, block-swap leaves (in VGT +
        cam_refine branches) keep tensors on CPU so the Default-scheme load
        path takes the CPU branch (fills ``pin_weight``, leaves ``weight``
        as ``None``). Extended-scope leaves (``cam_param_predictor``,
        ``*_output_conv2``) are always loaded straight to GPU — they are
        tiny and stay resident so the per-forward swap machinery doesn't
        have to touch them.

        Also collects any auxiliary quantization keys the scheme carries
        (e.g. ``.input_scale`` for fp8-pertensor) — those arrive in the
        state via the calibration-file merge performed by the runner.
        """
        wanted = set()  # keys that must go to GPU
        wanted_cpu = set()  # keys that stay on CPU under cpu_offload
        scale_attrs = ("weight_name", "bias_name", "input_scale_name", "weight_scale_name")
        block_swap_prefixes = ("frame_blocks.", "global_blocks.", "cam_refine_blocks.")
        for leaf_name, leaf in self.transformer_weights._walk_leaves():
            is_block_swap = leaf_name.startswith(block_swap_prefixes)
            for attr in scale_attrs:
                name = getattr(leaf, attr, None)
                if name is None:
                    continue
                if self._cpu_offload and is_block_swap:
                    wanted_cpu.add(name)
                else:
                    wanted.add(name)
        out = {}
        for k in wanted_cpu:
            t = state.get(k)
            if t is None:
                continue
            out[k] = t if t.device.type == "cpu" else t.cpu()
        for k in wanted:
            t = state.get(k)
            if t is None:
                continue
            out[k] = t.to(device) if t.device != device else t
        return out

    def _selective_load_nn(self, ckpt_state: Dict[str, torch.Tensor]):
        current = self.inner_model.state_dict()
        matched = 0
        for key, tensor in current.items():
            src = ckpt_state.get(key)
            if src is not None and src.shape == tensor.shape:
                current[key] = src
                matched += 1
        self.inner_model.load_state_dict(current, strict=True)
        logger.info(f"[WorldMirror] Loaded {matched}/{len(current)} keys into nn.Module")

    # ------------------------------------------------------------------
    # True-lazy load: never materialize the full safetensors into Python
    # memory. Only keys consumed by the ``nn.Module`` half (DPT heads,
    # patch_embed, gs_renderer, cam_head ancillary) are read; the
    # WeightModule-owned ViT + cam_head-trunk leaves stay unloaded and
    # their per-block weights are pulled from disk on demand inside the
    # forward pre-hook.
    #
    # Slashes resident CPU RAM: the previous path held ~5 GB of raw
    # safetensors state for the whole inference; this one holds only the
    # ~1.5 GB going into nn.Module.
    # ------------------------------------------------------------------
    def load_from_safetensors_lazy(self, safetensors_path: str):
        if not self._lazy_load:
            raise RuntimeError("load_from_safetensors_lazy called without cpu_offload + lazy_load; use load_from_safetensors(state) for the default path.")
        from safetensors import safe_open

        self._lazy_load_file = safetensors_path

        # 1. Selective nn.Module load, one key at a time. safetensors mmap
        #    gives us a key-indexed read that stays off the Python heap
        #    except for the tensor we're pulling.
        current = self.inner_model.state_dict()
        matched = 0
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            file_keys = set(f.keys())
            for key, tensor in current.items():
                if key not in file_keys:
                    continue
                src = f.get_tensor(key)
                if src.shape == tensor.shape:
                    current[key] = src
                    matched += 1
                else:
                    # Non-matching shape means the checkpoint has a leaf we
                    # don't use at this model size — drop it.
                    del src
        self.inner_model.load_state_dict(current, strict=True)
        del current
        logger.info(f"[WorldMirror] Lazy-loaded {matched} nn.Module keys (skipped WM keys)")

        # 2. WM side: don't populate block-swap weights. Pre-set the
        #    attribute names so the adapter forward doesn't AttributeError
        #    before the pre-hook fires.
        self.transformer_weights.init_lazy()

        # 2a. Extended-scope leaves are resident-GPU (no block swap), so
        #     the lazy path still needs to load them eagerly. Re-open the
        #     safetensors mmap and pull just those keys.
        if self._extended_scope:
            self._eager_load_extended_from_file(safetensors_path)

        # 3. Wire adapters + per-block swap hooks. The hook installer
        #    sees ``self._lazy_load == True`` and registers the
        #    mmap-read pre-hook / GPU-free post-hook pair.
        self._install_adapters()
        self._install_cpu_offload_hooks()
        return self

    def _eager_load_extended_from_file(self, safetensors_path: str):
        """Read extended-scope WM keys from safetensors directly.

        Used by the true-lazy load path where the runner never builds a
        flat state dict. Only pulls keys consumed by WM leaves outside the
        block-swap branches — those stay resident on GPU throughout
        inference instead of participating in per-block pin_weight ↔
        weight swap.
        """
        from safetensors import safe_open

        tw = self.transformer_weights
        block_swap_prefixes = ("frame_blocks.", "global_blocks.", "cam_refine_blocks.")
        device = next(self.inner_model.parameters()).device
        # Build a map from weight-name → leaf so we can walk the
        # safetensors keys once rather than per-leaf.
        ext_leaves = []
        for leaf_name, leaf in tw._walk_leaves():
            if leaf_name.startswith(block_swap_prefixes):
                continue
            ext_leaves.append(leaf)
        if not ext_leaves:
            return
        wanted = {}
        scale_attrs = ("weight_name", "bias_name")
        for leaf in ext_leaves:
            for attr in scale_attrs:
                name = getattr(leaf, attr, None)
                if name is not None:
                    wanted[name] = None
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            file_keys = set(f.keys())
            for k in list(wanted.keys()):
                if k in file_keys:
                    wanted[k] = f.get_tensor(k).to(device)
        # Now hand off to each leaf's ``load`` — safe because the Default
        # / Conv2d loaders are simple and do their own .to(device).
        for leaf in ext_leaves:
            leaf.load(wanted)

    # ------------------------------------------------------------------
    # Adapter swap
    # ------------------------------------------------------------------
    def _install_adapters(self):
        """Replace nn.Linear/LayerNorm leaves in VGT + cam_head trunk
        blocks with adapter wrappers delegating to WorldMirrorTransformerWeights.

        Invariants:
          * Only swap leaves whose paths match the WeightModule's key names.
          * DPT / gs_renderer / patch_embed / cam_head ancillary parts are
            untouched.
        """
        if self._adapters_installed:
            return
        im = self.inner_model
        depth = self.transformer_weights.depth
        cam_trunk_depth = self.transformer_weights.cam_trunk_depth

        vgt = im.visual_geometry_transformer
        for i in range(depth):
            self._swap_block_leaves(
                vgt.frame_blocks[i],
                self.transformer_weights.frame_blocks[i],
            )
            self._swap_block_leaves(
                vgt.global_blocks[i],
                self.transformer_weights.global_blocks[i],
            )
        if self.enable_cam:
            for i in range(cam_trunk_depth):
                # refine_net is an nn.Sequential, index by int.
                self._swap_block_leaves(
                    im.cam_head.refine_net[i],
                    self.transformer_weights.cam_refine_blocks[i],
                )

        # Extended scope (task ε): param_predictor + DPT output_conv2. Only
        # fires when ``wm_extended_scope=true`` — otherwise ``transformer_weights``
        # doesn't carry the corresponding WM containers and there is nothing
        # to wire.
        if self._extended_scope:
            self._install_extended_adapters()

        self._adapters_installed = True

    def _install_extended_adapters(self):
        """Swap param_predictor.{fc1,fc2} and DPT scratch.output_conv2[0/2]."""
        im = self.inner_model
        tw = self.transformer_weights

        # --- cam_head.param_predictor ---
        if self.enable_cam and hasattr(tw, "cam_param_predictor"):
            pp = im.cam_head.param_predictor
            fc1_old, fc2_old = pp.fc1, pp.fc2
            pp.fc1 = _MMLinearAdapter(
                tw.cam_param_predictor.fc1,
                fc1_old.in_features,
                fc1_old.out_features,
                has_bias=fc1_old.bias is not None,
            )
            pp.fc2 = _MMLinearAdapter(
                tw.cam_param_predictor.fc2,
                fc2_old.in_features,
                fc2_old.out_features,
                has_bias=fc2_old.bias is not None,
            )

        # --- DPT heads' scratch.output_conv2 ---
        for head_attr in ("depth_head", "norm_head", "pts_head", "gs_head"):
            head = getattr(im, head_attr, None)
            if head is None or not hasattr(head, "scratch"):
                continue
            wm = getattr(tw, f"{head_attr}_output_conv2", None)
            if wm is None:
                continue
            seq = head.scratch.output_conv2  # Sequential(Conv2d, ReLU, Conv2d)
            conv0_old = seq[0]
            conv2_old = seq[2]
            seq[0] = _Conv2dAdapter(
                wm.conv_0,
                in_channels=conv0_old.in_channels,
                out_channels=conv0_old.out_channels,
                kernel_size=conv0_old.kernel_size,
                stride=conv0_old.stride,
                padding=conv0_old.padding,
                has_bias=conv0_old.bias is not None,
            )
            seq[2] = _Conv2dAdapter(
                wm.conv_2,
                in_channels=conv2_old.in_channels,
                out_channels=conv2_old.out_channels,
                kernel_size=conv2_old.kernel_size,
                stride=conv2_old.stride,
                padding=conv2_old.padding,
                has_bias=conv2_old.bias is not None,
            )

    @staticmethod
    def _swap_block_leaves(block: nn.Module, bw: VitBlockWeights):
        """Swap a single ``Block``'s attn/norm/mlp leaves with adapters."""
        # Attention qkv / proj
        attn = block.attn
        qkv_old = attn.qkv
        attn.qkv = _MMLinearAdapter(
            bw.attn_qkv,
            qkv_old.in_features,
            qkv_old.out_features,
            has_bias=qkv_old.bias is not None,
        )
        proj_old = attn.proj
        attn.proj = _MMLinearAdapter(
            bw.attn_proj,
            proj_old.in_features,
            proj_old.out_features,
            has_bias=proj_old.bias is not None,
        )
        # q_norm / k_norm — only present (as LayerNorm) when qk_norm=True,
        # otherwise they are nn.Identity and the block has no q/k LN weight.
        if hasattr(bw, "attn_q_norm") and isinstance(attn.q_norm, nn.LayerNorm):
            attn.q_norm = _LNAdapter(
                bw.attn_q_norm,
                attn.q_norm.normalized_shape,
                attn.q_norm.eps,
            )
            attn.k_norm = _LNAdapter(
                bw.attn_k_norm,
                attn.k_norm.normalized_shape,
                attn.k_norm.eps,
            )
        # norm1 / norm2
        n1 = block.norm1
        block.norm1 = _LNAdapter(bw.norm1, n1.normalized_shape, n1.eps)
        n2 = block.norm2
        block.norm2 = _LNAdapter(bw.norm2, n2.normalized_shape, n2.eps)
        # mlp.fc1 / fc2
        mlp = block.mlp
        fc1_old = mlp.fc1
        mlp.fc1 = _MMLinearAdapter(
            bw.mlp_fc1,
            fc1_old.in_features,
            fc1_old.out_features,
            has_bias=fc1_old.bias is not None,
        )
        fc2_old = mlp.fc2
        mlp.fc2 = _MMLinearAdapter(
            bw.mlp_fc2,
            fc2_old.in_features,
            fc2_old.out_features,
            has_bias=fc2_old.bias is not None,
        )

    # ------------------------------------------------------------------
    # cpu_offload: per-block forward hooks swap the WeightModule leaves
    # in and out of GPU memory. Only the currently-executing block keeps
    # its MM/LN weights resident on the device.
    # ------------------------------------------------------------------
    def _install_cpu_offload_hooks(self):
        """Register forward hooks on each VGT / cam_head block for swap.

        Expects ``cpu_offload`` to have left the WeightModule leaves with
        ``pin_weight`` populated and ``weight`` as ``None``. The pre-hook
        promotes ``pin_weight → weight`` on the device; the post-hook
        copies the possibly-updated weight back into pinned memory and
        frees the device-side copy.

        ``cam_head.refine_net`` is an ``nn.Sequential`` whose blocks are
        re-entered on every camera-refinement iteration (4 iterations of
        4 sub-blocks = 16 forward calls per scene). The hook fires on
        every one of those, so each iteration keeps only one cam_head
        block resident at a time.
        """
        if self._offload_hooks_installed:
            return
        im = self.inner_model
        depth = self.transformer_weights.depth
        cam_trunk_depth = self.transformer_weights.cam_trunk_depth

        # If lazy_load is on, release the pre-populated pin buffers now
        # so the only time we hold CPU memory for a block's weights is
        # between the pre-hook's disk read and the post-hook's release.
        if self._lazy_load:
            if self._lazy_load_file is None:
                raise RuntimeError("lazy_load=true but no safetensors path was exposed on the model; runner must set model._lazy_load_file before load_from_safetensors().")
            self._release_pin_buffers()
            # Open the safetensors file once and keep the handle alive
            # for the whole model lifetime. Reopening in every block's
            # pre-hook was the dominant cost of the lazy_load path
            # (header-deserialise fires on every open). Persistent handle
            # amortises that work across all 52 blocks per inference.
            self._ensure_lazy_handle()

        vgt = im.visual_geometry_transformer
        for i in range(depth):
            self._register_swap_hooks(
                vgt.frame_blocks[i],
                self.transformer_weights.frame_blocks[i],
                lazy=self._lazy_load,
                lazy_load_file=self._lazy_load_file,
            )
            self._register_swap_hooks(
                vgt.global_blocks[i],
                self.transformer_weights.global_blocks[i],
                lazy=self._lazy_load,
                lazy_load_file=self._lazy_load_file,
            )
        if self.enable_cam:
            for i in range(cam_trunk_depth):
                # cam_refine blocks fire 4× per scene (the CameraHead
                # iterates refine_net 4 times). Under lazy_load that
                # means re-reading the same ~200 MB of weights 16 times
                # per inference, which dominates wall-clock. Opt-in
                # ``lazy_cam_resident`` (on by default) keeps them GPU-
                # resident under lazy: +200 MB peak, -300 ms wall-clock.
                if self._lazy_cam_resident:
                    self._eager_load_block_to_gpu(self.transformer_weights.cam_refine_blocks[i])
                    continue
                self._register_swap_hooks(
                    im.cam_head.refine_net[i],
                    self.transformer_weights.cam_refine_blocks[i],
                    lazy=self._lazy_load,
                    lazy_load_file=self._lazy_load_file,
                )
        self._offload_hooks_installed = True
        logger.info(
            f"[WorldMirror] cpu_offload hooks installed on {depth} frame + {depth} global + {cam_trunk_depth if self.enable_cam else 0} cam_head blocks{' (lazy_load=on)' if self._lazy_load else ''}"
        )

    def _release_pin_buffers(self):
        """Free per-leaf pinned CPU buffers (used by lazy_load path)."""
        for _, leaf in self.transformer_weights._walk_leaves():
            for attr in ("pin_weight", "pin_bias"):
                if hasattr(leaf, attr):
                    setattr(leaf, attr, None)
        import gc as _gc

        _gc.collect()

    def _eager_load_block_to_gpu(self, wm_block: "VitBlockWeights"):
        """Populate a WM block's leaves with resident-GPU weights.

        Used for ``cam_refine_blocks`` under lazy_load, which fire 4× per
        inference inside CameraHead's iterative refinement. Keeping them
        swapping in/out of GPU is pure overhead since the whole trunk
        fits in ~60 MB — eager-load once, hold on GPU for the life of
        the model.
        """
        if getattr(self, "_lazy_file_handle", None) is None:
            self._ensure_lazy_handle()
        f = self._lazy_file_handle
        keys = self._lazy_file_keys
        for leaf in wm_block._modules.values():
            if leaf is None or not hasattr(leaf, "base_attrs"):
                continue
            for name, attr_name, transpose in leaf.base_attrs:
                if name not in keys:
                    setattr(leaf, attr_name, None)
                    continue
                t = f.get_tensor(name)
                if transpose:
                    t = t.t()
                setattr(leaf, attr_name, t.to(AI_DEVICE))

    def close(self):
        """Release any persistent file handles (lazy_load).

        For one-shot CLI invocations the OS reclaims handles at process
        exit, but long-running services that hot-reload models will leak
        the safetensors mmap and pin its pages in RSS until restart.
        Call from runner teardown or before dropping the last reference
        to the model. Idempotent.
        """
        f = getattr(self, "_lazy_file_handle", None)
        if f is None:
            return
        try:
            f.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover — handle was already closed
            logger.warning(f"[WorldMirror] safe_open close raised: {exc}")
        self._lazy_file_handle = None
        self._lazy_file_keys = None

    def __del__(self):
        # Best-effort cleanup. ``__del__`` is unreliable (interpreter
        # shutdown, exception during init, etc.) but it's a useful
        # backstop so a forgotten ``close()`` doesn't leak the mmap for
        # the lifetime of the process.
        try:
            self.close()
        except Exception:
            pass

    def _ensure_lazy_handle(self):
        """Open (and cache) a persistent safetensors handle for lazy_load.

        Reopening the file on every block's pre-hook was adding ~500ms to
        each block-forward (the safetensors C++ layer deserialises the
        header each time the context is entered). Sharing one handle
        across all block hooks collapses that to a ~10μs lookup per
        ``get_tensor``. We never need to close this during inference;
        explicit cleanup happens only if/when the model is torn down.

        We open with ``device="cpu"``: counter-intuitively this is
        ~10× faster than ``device="cuda"`` for small-tensor bursts
        because the GPU-direct path in current safetensors spins up
        per-tensor CUDA launches while the CPU-then-``.to(cuda)`` path
        re-uses a single PCIe DMA stream (~0.8ms/tensor vs ~15ms/tensor).
        """
        if getattr(self, "_lazy_file_handle", None) is not None:
            return
        from safetensors import safe_open

        f = safe_open(self._lazy_load_file, framework="pt", device="cpu").__enter__()
        self._lazy_file_handle = f
        self._lazy_file_keys = set(f.keys())

    def _register_swap_hooks(
        self,
        nn_block: nn.Module,
        wm_block: VitBlockWeights,
        *,
        lazy: bool = False,
        lazy_load_file: str = None,
    ):
        self_model = self  # bind for closure readability
        """Install forward_pre_hook / forward_hook on ``nn_block``.

        Two modes:

        * **Default (lazy=False)** — ``pin_weight`` holds weights in pinned
          CPU memory for the whole inference. pre-hook copies pinned→GPU
          (non_blocking), post-hook copies GPU→pinned and drops GPU refs.
        * **lazy=True** — no pinned residency. pre-hook mmap-reads the
          weight from ``lazy_load_file`` into a freshly-allocated pinned
          buffer, copies to GPU, and drops the pinned buffer. post-hook
          drops the GPU tensor. Trades a safetensors read per block
          forward for a big CPU-RAM save.
        """
        if lazy:
            # Capture a per-leaf copy of (weight_name, bias_name, transpose)
            # at registration time. We need transpose info because the
            # Default/fp8 MM weights store weight pre-transposed.
            leaf_infos = []
            for leaf in wm_block._modules.values():
                if leaf is None or not hasattr(leaf, "base_attrs"):
                    continue
                leaf_infos.append((leaf, list(leaf.base_attrs)))

            def pre(module, args):
                # The model instance holds a persistent safe_open handle
                # plus a key set (see WorldMirrorWeightModel._ensure_lazy_handle).
                # Reopening the context per block was the dominant cost
                # (~500ms each vs ~1ms for a persistent handle). Sharing
                # the handle across all 52 block hooks cuts lazy_load
                # latency from +119% to roughly +70% vs eager.
                f = self_model._lazy_file_handle
                keys = self_model._lazy_file_keys
                for leaf, base_attrs in leaf_infos:
                    for name, attr_name, transpose in base_attrs:
                        if name not in keys:
                            setattr(leaf, attr_name, None)
                            continue
                        t = f.get_tensor(name)
                        if transpose:
                            # Match the default cpu_offload path: the
                            # adapter expects ``weight`` to be a
                            # (non-contiguous) transposed view of
                            # [out, in] contiguous memory. ``.t()``
                            # on the CPU-side tensor gives exactly that
                            # shape; ``.to(device)`` preserves the
                            # non-contig stride layout.
                            t = t.t()
                        setattr(leaf, attr_name, t.to(AI_DEVICE, non_blocking=True))

            def post(module, args, output):
                for leaf, base_attrs in leaf_infos:
                    for _, attr_name, _ in base_attrs:
                        setattr(leaf, attr_name, None)

            nn_block.register_forward_pre_hook(pre)
            nn_block.register_forward_hook(post)
            return

        # Default cpu_offload path: swap via existing pin_weight.
        def pre(module, args):
            for leaf in wm_block._modules.values():
                if leaf is None:
                    continue
                if hasattr(leaf, "base_attrs"):
                    move_attr_to_cuda(leaf, leaf.base_attrs, leaf.lora_attrs, non_blocking=True)

        def post(module, args, output):
            for leaf in wm_block._modules.values():
                if leaf is None:
                    continue
                if hasattr(leaf, "base_attrs"):
                    move_attr_to_cpu(leaf, leaf.base_attrs, leaf.lora_attrs, non_blocking=True)

        nn_block.register_forward_pre_hook(pre)
        nn_block.register_forward_hook(post)

    # ------------------------------------------------------------------
    # Head control (decision 4 = B: model exposes, runner calls)
    # ------------------------------------------------------------------
    def disable_heads(self, head_names: Sequence[str]):
        """Disable and free the given output heads."""
        unknown = [n for n in head_names if n not in _HEAD_MAPPING]
        if unknown:
            raise ValueError(f"Unknown head name(s): {unknown}. Valid heads are {list(HEAD_NAMES)}.")
        freed = 0
        for name in head_names:
            attr, modules = _HEAD_MAPPING[name]
            setattr(self.inner_model, attr, False)
            for mod_name in modules:
                if hasattr(self.inner_model, mod_name):
                    mod = getattr(self.inner_model, mod_name)
                    freed += sum(p.numel() for p in mod.parameters())
                    mod.cpu()
                    delattr(self.inner_model, mod_name)
                    del mod
        if freed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[WorldMirror] Disabled heads: {list(head_names)}, freed ~{freed / 1e6:.1f}M params")

    # ------------------------------------------------------------------
    # BF16 cast — kept here so the runner doesn't need to reach into
    # internals (decision 4 = B).
    # ------------------------------------------------------------------
    def apply_bf16_cast(self):
        """Cast non-critical fp32 params/buffers to bf16 on both containers."""
        crit = collect_fp32_critical_modules(self.inner_model)
        self.inner_model.to(torch.bfloat16)
        for mod in crit:
            mod.to(torch.float32)
        # WM tensors were loaded fresh and are currently fp32 — cast now.
        self.transformer_weights.cast_dtype(torch.bfloat16)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        views: Dict[str, torch.Tensor],
        cond_flags: List[int],
        *,
        is_inference: bool = True,
        sp_size: int = 1,
        sp_group=None,
    ):
        """Single-step reconstruction: ViT → heads → gsplat → predictions dict."""
        fwd_kw = dict(
            views=views,
            cond_flags=cond_flags,
            is_inference=is_inference,
        )
        if sp_size > 1:
            fwd_kw["sp_size"] = sp_size
            fwd_kw["sp_group"] = sp_group
        return self.inner_model(**fwd_kw)

    # Allow `model(...)` for minimal changes to callers that used to pass
    # the raw nn.Module.
    def __call__(self, *args, **kwargs):
        if args:
            # Keep positional behaviour consistent with nn.Module.forward.
            return self.inner_model(*args, **kwargs)
        return self.infer(**kwargs)
