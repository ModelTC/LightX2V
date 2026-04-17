"""Runner for HY-WorldMirror-2.0 (3D world reconstruction) model.

Wraps the WorldMirror model (migrated from HY-World-2.0) and exposes it through
the LightX2V runner interface. Unlike the other LightX2V runners, this is a
single-pass reconstruction pipeline (not diffusion): the model ingests a set
of input images and produces depth / normal / point cloud / gaussian splat
predictions that are saved to disk.

This runner mirrors the reconstruction feature set of
``hyworld2/worldrecon/pipeline.py:WorldMirrorPipeline``, including:

- Single-GPU and multi-GPU inference via WorldMirror's own sequence-parallel
  path (``DistBlock`` + ``_Allgather``). FSDP is intentionally not used:
  LightX2V provides ``cpu_offload`` / ``lazy_load`` / quantization instead.
- ``config_path`` + ``ckpt_path`` (training-output) load path as an
  alternative to the exported ``{model_path}/{subfolder}/`` layout.
- bf16 mixed precision with fp32-critical modules kept at full precision.
- Optional disable of any output head to save memory/compute.
- Optional camera-interpolated Gaussian splat video rendering via
  :func:`render_interpolated_video`.
- Prior camera / prior depth inputs.
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import load_file as load_safetensors

from lightx2v.models.networks.worldmirror.model import WorldMirrorWeightModel
from lightx2v.models.networks.worldmirror.utils.inference_utils import (
    compute_adaptive_target_size,
    compute_filter_mask,
    compute_preprocessing_transform,
    compute_sky_mask,
    load_prior_camera,
    load_prior_depth,
    prepare_images_to_tensor,
    prepare_input,
    print_and_save_timings,
    save_results,
)
from lightx2v.models.networks.worldmirror.utils.render_utils import render_interpolated_video
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


# ---------------------------------------------------------------------------
# Helpers (inlined from hyworld2.worldrecon.pipeline so that LightX2V has no
# runtime dependency on the HY-World-2.0 repository).
#
# NOTE: ``_collect_fp32_critical_modules`` / bf16 cast / ``_disable_heads``
# have moved into :class:`WorldMirrorWeightModel` (alignment decision 4 = B:
# runner no longer mutates model submodules).
# ---------------------------------------------------------------------------
def _load_checkpoint_state_dict(ckpt_path: str) -> dict:
    """Load a training .ckpt or a .safetensors into a flat state_dict."""
    if ckpt_path.endswith(".safetensors"):
        return load_safetensors(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "state_dict" in ckpt:
        # Only strip the leading "model." prefix; ``str.replace`` would also
        # substitute occurrences elsewhere in the key (e.g. "some_model.x").
        prefix = "model."
        state = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state.items()}
    return state


def _get_model_config_from_yaml(cfg) -> dict:
    """Extract a flat model kwargs dict from an OmegaConf YAML config."""
    from omegaconf import OmegaConf

    if hasattr(cfg, "wrapper") and hasattr(cfg.wrapper, "model"):
        model_cfg = cfg.wrapper.model
    elif hasattr(cfg, "model"):
        model_cfg = cfg.model
    else:
        raise ValueError("No model config found (expect wrapper.model or model).")
    out = OmegaConf.to_container(model_cfg, resolve=True)
    out.pop("_target_", None)
    return out


def _has_model_files(path: str) -> bool:
    has_weights = os.path.isfile(os.path.join(path, "model.safetensors"))
    has_config = os.path.isfile(os.path.join(path, "config.yaml")) or os.path.isfile(os.path.join(path, "config.json"))
    return has_weights and has_config


def _load_model_config(model_dir: str) -> dict:
    """Load model.config from a model dir, handling yaml or json."""
    from omegaconf import OmegaConf

    yaml_path = os.path.join(model_dir, "config.yaml")
    json_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(yaml_path):
        cfg = OmegaConf.load(yaml_path)
        return _get_model_config_from_yaml(cfg)
    if os.path.isfile(json_path):
        with open(json_path) as f:
            return json.load(f)
    raise FileNotFoundError(f"No config.yaml or config.json in {model_dir}")


def _broadcast_string(s, rank, src=0):
    """Broadcast a Python string across ranks via a uint8 tensor."""
    device = torch.device("cuda")
    if rank == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(length, src=src)
    n = length.item()
    if rank == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device=device)
    dist.broadcast(tensor, src=src)
    return tensor.cpu().numpy().tobytes().decode("utf-8")


@RUNNER_REGISTER("worldmirror")
class WorldMirrorRunner(BaseRunner):
    """Runner for HY-WorldMirror-2.0 3D reconstruction model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self._inner_model = None  # cached unwrapped model (see _cache_inner_model)
        self.scheduler = None  # no diffusion scheduler

        # Distributed / SP state. Populated lazily in ``init_modules``
        # when ``WORLD_SIZE > 1`` is set by ``torchrun`` — constructing a
        # runner must not trigger a distributed collective.
        self.sp_size = 1
        self.sp_group = None
        self.rank = 0
        self.is_distributed = False
        self.device = None

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------
    def _init_distributed(self) -> bool:
        """Set up SP process group if running under torchrun.

        Idempotent: safe to call twice (returns True both times once the
        group is established). Keeps the state-mutating side-effects
        (process-group creation, ``cuda.set_device``) out of ``__init__``.
        """
        if self.is_distributed:
            return True
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size <= 1:
            return False
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(local_rank)
        self.sp_size = world_size
        # Sequence-parallel group = all ranks. This matches HY-World-2.0.
        self.sp_group = dist.new_group(ranks=list(range(self.sp_size)))
        self.is_distributed = True
        if self.rank == 0:
            logger.info(f"[WorldMirror] Multi-GPU: world_size={world_size}, sp_size={self.sp_size}")
        return True

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _resolve_model_dir(self):
        """Resolve the directory containing config.json + model.safetensors.

        Preference order (mirrors HY-World-2.0):
          1. ``{model_path}/{subfolder}``
          2. ``{model_path}``
        HuggingFace auto-download is intentionally *not* replicated here —
        LightX2V users are expected to point ``model_path`` at a local dir.
        """
        model_path = self.config["model_path"]
        subfolder = self.config.get("subfolder", "HY-WorldMirror-2.0")

        candidate = os.path.join(model_path, subfolder)
        if os.path.isdir(candidate) and _has_model_files(candidate):
            return candidate
        if os.path.isdir(model_path) and _has_model_files(model_path):
            return model_path

        raise FileNotFoundError(f"Could not locate WorldMirror weights. Tried '{candidate}' and '{model_path}'. Make sure model.safetensors + config.{{json,yaml}} live under one of these paths.")

    def load_transformer(self):
        """Load the WorldMirror model.

        Two code paths, matching HY-World-2.0 ``from_pretrained``:
          1. Exported model dir: ``{model_path}/{subfolder}/{model.safetensors, config.{json,yaml}}``.
          2. Training artifacts: ``config_path`` (YAML) + ``ckpt_path``
             (.ckpt or .safetensors).
        """
        # Upstream bug guard (T19): gs_renderer.prepare_cameras unconditionally
        # reads predictions['camera_poses'], which the disabled camera head
        # never populates. Disabling camera without also disabling gs crashes
        # with KeyError deep in rasterization — fail fast with an actionable
        # error here.
        disable_heads = self.config.get("disable_heads", None) or []
        if "camera" in disable_heads and "gs" not in disable_heads:
            raise ValueError(
                "disable_heads=['camera'] without also disabling 'gs' is unsupported: "
                "gs_renderer.prepare_cameras unconditionally reads predictions['camera_poses'], "
                "which the disabled camera head never populates. "
                "Set disable_heads=['camera', 'gs'] to disable both together, "
                "or keep camera enabled."
            )

        config_path = self.config.get("config_path", None)
        ckpt_path = self.config.get("ckpt_path", None)

        enable_bf16_runtime = bool(self.config.get("enable_bf16", False))

        t0 = time.perf_counter()
        weights_file = None  # safetensors path for lazy_load (None if unsupported)
        # Eager-load state up front only when the caller isn't opting into
        # the true-lazy path (cpu_offload + lazy_load + a safetensors file).
        # For that path we skip the bulk safetensors read entirely — it is
        # the single biggest CPU-RAM cost in the legacy flow.
        want_true_lazy = bool(self.config.get("cpu_offload", False)) and bool(self.config.get("lazy_load", False))
        state = None
        if config_path and ckpt_path:
            logger.info(f"[WorldMirror] Loading config={config_path}, ckpt={ckpt_path}")
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_path)
            model_cfg = _get_model_config_from_yaml(cfg)
            if self.sp_size > 1:
                model_cfg["sp_size"] = self.sp_size
            if enable_bf16_runtime:
                model_cfg["enable_bf16"] = True
            source_name = ckpt_path
            if ckpt_path.endswith(".safetensors"):
                weights_file = ckpt_path
            # Training-ckpt paths rely on torch.load even in lazy mode —
            # safetensors mmap only works for the exported-model path.
            if not (want_true_lazy and weights_file is not None):
                state = _load_checkpoint_state_dict(ckpt_path)
        else:
            model_dir = self._resolve_model_dir()
            logger.info(f"[WorldMirror] Loading model from {model_dir}")
            model_cfg = _load_model_config(model_dir)
            if self.sp_size > 1:
                model_cfg["sp_size"] = self.sp_size
            if enable_bf16_runtime:
                model_cfg["enable_bf16"] = True
            weights_file = os.path.join(model_dir, "model.safetensors")
            if not want_true_lazy:
                state = load_safetensors(weights_file)
            source_name = model_dir

        # Build the wrapped model (nn.Module + side-car WeightModule) and
        # load state into both. The side-car owns the ViT-backbone +
        # cam_head trunk leaves for quantization / cpu_offload / lazy_load.
        # ``runtime_cfg`` is the LightX2V config (carries
        # ``dit_quant_scheme`` / ``ln_norm_type`` / ...), separate from the
        # HY-World architecture config.
        model = WorldMirrorWeightModel(model_cfg, runtime_cfg=dict(self.config))
        model.inner_model.to(self.device)

        # True-lazy path (cpu_offload=true + lazy_load=true + safetensors
        # file available): the runner deliberately *never* materializes the
        # full checkpoint into a Python dict — only nn.Module keys are
        # read, one by one, through the safetensors mmap. WM keys stay on
        # disk and are pulled per-block-forward by the cpu_offload hook.
        # Saves ~3 GB of resident RSS during inference.
        use_true_lazy = bool(self.config.get("cpu_offload", False)) and bool(self.config.get("lazy_load", False)) and weights_file is not None and weights_file.endswith(".safetensors")
        if use_true_lazy:
            if config_path and ckpt_path:
                # The training-ckpt path already pulled state via torch.load;
                # we have no safetensors we can mmap in that case, so fall
                # through to the legacy path.
                use_true_lazy = False

        if use_true_lazy:
            # In true-lazy mode the calibration input_scale merge has no
            # natural path (we never build the flat state dict). Reject
            # the combination loudly — the auto-quant paths need the
            # calibration state before post_process anyway, so silently
            # dropping it would produce wrong numbers.
            if self.config.get("input_scale_file", None):
                raise RuntimeError("lazy_load=true is not currently compatible with input_scale_file (fp8-pertensor calibration). Disable one of the two.")
            if self.config.get("weight_auto_quant", False):
                raise RuntimeError("lazy_load=true is not currently compatible with weight_auto_quant (quant schemes materialize weights at load-time). Disable one of the two.")
            model.load_from_safetensors_lazy(weights_file)
            logger.info(f"[WorldMirror] Lazy-loaded weights from {source_name}")
            # No ``state`` dict was built in this branch — skip the del.
        else:
            # Expose the safetensors path on the model so lazy_load can
            # re-read block weights from disk on demand inside hooks.
            model._lazy_load_file = weights_file

            # Optional calibration merge: ``input_scale_file`` points at a
            # safetensors produced by ``scripts/worldmirror/run_calibration.py``
            # and holds pre-collected ``<name>.input_scale`` tensors that the
            # fp8-pertensor auto-quant path consumes. Keys just pass through to
            # ``_device_view_for_wm`` and end up on the leaves before
            # ``post_process`` runs.
            input_scale_file = self.config.get("input_scale_file", None)
            if input_scale_file:
                if not os.path.isfile(input_scale_file):
                    raise FileNotFoundError(f"input_scale_file '{input_scale_file}' not found — run scripts/worldmirror/run_calibration.py first.")
                logger.info(f"[WorldMirror] Merging calibration from {input_scale_file}")
                calib_state = load_safetensors(input_scale_file)
                collision = [k for k in calib_state if k in state]
                if collision:
                    logger.warning(f"[WorldMirror] {len(collision)} calibration keys already in state (first few: {collision[:3]}) — overwriting.")
                state.update(calib_state)
            model.load_from_safetensors(state)
            del state
            logger.info(f"[WorldMirror] Loaded weights from {source_name}")
        gc.collect()
        torch.cuda.empty_cache()

        enable_bf16 = bool(model_cfg.get("enable_bf16", False))
        if enable_bf16:
            model.apply_bf16_cast()

        model.eval()

        disable_heads = self.config.get("disable_heads", None)
        if disable_heads:
            model.disable_heads(disable_heads)

        if self.rank == 0:
            logger.info(f"[WorldMirror] Model ready in {time.perf_counter() - t0:.1f}s")
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
                logger.info(f"[WorldMirror] memory allocated={alloc:.2f}GB")

        return model

    # ------------------------------------------------------------------
    # LightX2V lifecycle hooks
    # ------------------------------------------------------------------
    def init_modules(self):
        logger.info("Initializing WorldMirror runner modules...")
        # Distributed init happens lazily here rather than in __init__ so
        # that merely constructing a runner doesn't join a collective.
        if self._init_distributed():
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", self.rank)))
        else:
            self.device = torch.device(AI_DEVICE if torch.cuda.is_available() else "cpu")
        self.model = self.load_transformer()
        # ``_inner_model`` is kept as the ``nn.Module`` handle for code
        # (e.g. the rendering block) that reaches into ``gs_renderer`` or
        # bf16 flags. ``self.model`` is the :class:`WorldMirrorWeightModel`
        # wrapper exposing ``.infer(...)``.
        self._inner_model = self.model.inner_model
        self.config.lock()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference(self, img_paths, target_size, prior_cam_path, prior_depth_path):
        imgs = prepare_images_to_tensor(img_paths, target_size=target_size, resize_strategy="crop").to(self.device)
        views = {"img": imgs}
        B, S, C, H, W = imgs.shape

        if self.sp_size > 1 and S < self.sp_size:
            raise ValueError(f"Number of input images ({S}) must be >= number of GPUs ({self.sp_size}) in multi-GPU mode. Please provide at least {self.sp_size} images, or use fewer GPUs.")

        if self.rank == 0:
            logger.info(f"[WorldMirror] {S} images, shape={imgs.shape}, sp_size={self.sp_size}")

        pp_xform = compute_preprocessing_transform(img_paths, target_size)
        cond_flags = [0, 0, 0]

        if prior_cam_path and os.path.isfile(prior_cam_path):
            extr, intr = load_prior_camera(prior_cam_path, img_paths, preprocess_transform=pp_xform)
            if extr is not None:
                first = extr[0, 0]
                extr = torch.linalg.inv(first.float()).to(first.dtype).unsqueeze(0).unsqueeze(0) @ extr
                views["camera_poses"] = extr.to(self.device)
                cond_flags[0] = 1
            if intr is not None:
                views["camera_intrs"] = intr.to(self.device)
                cond_flags[2] = 1

        if prior_depth_path and os.path.isdir(prior_depth_path):
            depth = load_prior_depth(prior_depth_path, img_paths, H, W, preprocess_transform=pp_xform)
            if depth is not None:
                views["depthmap"] = depth.to(self.device)
                cond_flags[1] = 1

        model_bf16 = self.model.enable_bf16
        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(not model_bf16 and use_amp), dtype=torch.bfloat16):
            predictions = self.model.infer(
                views=views,
                cond_flags=cond_flags,
                is_inference=True,
                sp_size=self.sp_size,
                sp_group=self.sp_group,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        if self.rank == 0:
            logger.info(f"[WorldMirror] Inference done in {infer_time:.2f}s")
        return predictions, imgs, infer_time

    @torch.no_grad()
    def run_pipeline(self, input_info):
        """Run WorldMirror reconstruction on a directory/video/image path."""
        self.input_info = input_info
        input_path = input_info.input_path
        if not input_path:
            raise ValueError("input_info.input_path must be set")

        cfg = self.config
        output_path = input_info.save_result_path or cfg.get("output_path", "inference_output")
        strict_output_path = input_info.strict_output_path or cfg.get("strict_output_path", None)

        target_size = cfg.get("target_size", 952)
        fps = cfg.get("fps", 1)
        video_strategy = cfg.get("video_strategy", "new")
        video_min_frames = cfg.get("video_min_frames", 1)
        video_max_frames = cfg.get("video_max_frames", 32)

        save_depth = cfg.get("save_depth", True)
        save_normal = cfg.get("save_normal", True)
        save_gs = cfg.get("save_gs", True)
        save_camera = cfg.get("save_camera", True)
        save_points = cfg.get("save_points", True)
        save_colmap = cfg.get("save_colmap", False)
        save_conf = cfg.get("save_conf", False)
        save_sky_mask = cfg.get("save_sky_mask", False)

        apply_sky_mask = cfg.get("apply_sky_mask", True)
        apply_edge_mask = cfg.get("apply_edge_mask", True)
        apply_confidence_mask = cfg.get("apply_confidence_mask", False)
        sky_mask_source = cfg.get("sky_mask_source", "auto")
        model_sky_threshold = cfg.get("model_sky_threshold", 0.45)
        confidence_percentile = cfg.get("confidence_percentile", 10.0)
        edge_normal_threshold = cfg.get("edge_normal_threshold", 1.0)
        edge_depth_threshold = cfg.get("edge_depth_threshold", 0.03)

        compress_pts = cfg.get("compress_pts", True)
        compress_pts_max_points = cfg.get("compress_pts_max_points", 2_000_000)
        compress_pts_voxel_size = cfg.get("compress_pts_voxel_size", 0.002)
        max_resolution = cfg.get("max_resolution", 1920)
        compress_gs_max_points = cfg.get("compress_gs_max_points", 5_000_000)

        save_rendered = cfg.get("save_rendered", False)
        render_interp_per_pair = cfg.get("render_interp_per_pair", 15)
        render_depth = cfg.get("render_depth", False)

        prior_cam_path = input_info.prior_cam_path or cfg.get("prior_cam_path", None)
        prior_depth_path = input_info.prior_depth_path or cfg.get("prior_depth_path", None)
        log_time = cfg.get("log_time", True)

        case_t0 = time.perf_counter()
        timings = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Prepare input
        t0 = time.perf_counter()
        img_paths, subdir_name = prepare_input(
            input_path,
            target_size=target_size,
            fps=fps,
            video_strategy=video_strategy,
            min_frames=video_min_frames,
            max_frames=video_max_frames,
        )
        if log_time:
            timings["data_loading"] = time.perf_counter() - t0

        if strict_output_path is not None:
            outdir = Path(strict_output_path)
        else:
            outdir = Path(output_path) / subdir_name / timestamp

        # 2. Adaptive resolution
        effective = compute_adaptive_target_size(img_paths, target_size)
        if self.rank == 0 and effective != target_size:
            logger.info(f"[WorldMirror] Adaptive resolution: {effective} (max={target_size})")

        # 3. Inference
        # Only synchronize when we need accurate preprocess/infer timings;
        # the waits are otherwise pure latency added to every case.
        if log_time and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        t0_all = time.perf_counter()
        try:
            predictions, imgs, infer_time = self._run_inference(
                img_paths,
                effective,
                prior_cam_path,
                prior_depth_path,
            )
        except ValueError as e:
            if self.rank == 0:
                logger.info(f"[WorldMirror] Skipping '{input_path}': {e}")
            return None

        if log_time:
            timings["inference"] = infer_time
            timings["inference_preprocess"] = time.perf_counter() - t0_all - infer_time

        if log_time and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if self.is_distributed:
                peak_t = torch.tensor([peak], dtype=torch.float64, device=self.device)
                gathered = [torch.zeros(1, dtype=torch.float64, device=self.device) for _ in range(self.sp_size)]
                dist.all_gather(gathered, peak_t, group=self.sp_group)
                timings["gpu_mem_peak_per_rank_gb"] = [t.item() for t in gathered]
                timings["gpu_mem_peak_avg_gb"] = sum(timings["gpu_mem_peak_per_rank_gb"]) / self.sp_size
            else:
                timings["gpu_mem_peak_gb"] = peak

        # 4. Post-processing and saving — rank 0 only so files aren't duplicated.
        if self.rank == 0:
            B, S, C, H, W = imgs.shape
            t0 = time.perf_counter()

            sky_mask = (
                compute_sky_mask(
                    img_paths,
                    H,
                    W,
                    S,
                    predictions=predictions,
                    source=sky_mask_source,
                    model_threshold=model_sky_threshold,
                    processed_aspect_ratio=W / H,
                )
                if apply_sky_mask
                else None
            )

            filter_mask, gs_filter_mask = None, None
            if apply_confidence_mask or apply_edge_mask or apply_sky_mask:
                filter_mask, gs_filter_mask = compute_filter_mask(
                    predictions,
                    imgs,
                    img_paths,
                    H,
                    W,
                    S,
                    apply_confidence_mask=apply_confidence_mask,
                    apply_edge_mask=apply_edge_mask,
                    apply_sky_mask=apply_sky_mask,
                    confidence_percentile=confidence_percentile,
                    edge_normal_threshold=edge_normal_threshold,
                    edge_depth_threshold=edge_depth_threshold,
                    sky_mask=sky_mask,
                    use_gs_depth=save_gs,
                )

            if log_time:
                timings["compute_mask"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            save_timings = save_results(
                predictions,
                imgs,
                img_paths,
                outdir,
                save_depth=save_depth,
                save_normal=save_normal,
                save_gs=save_gs,
                save_camera=save_camera,
                save_points=save_points,
                save_colmap=save_colmap,
                save_sky_mask=save_sky_mask,
                save_conf=save_conf,
                log_time=log_time,
                max_resolution=max_resolution,
                filter_mask=filter_mask,
                gs_filter_mask=gs_filter_mask,
                sky_mask=sky_mask,
                compress_pts=compress_pts,
                compress_pts_max_points=compress_pts_max_points,
                compress_pts_voxel_size=compress_pts_voxel_size,
                compress_gs_max_points=compress_gs_max_points,
            )
            if log_time:
                timings.update(save_timings or {})
                timings["save_total_wall"] = time.perf_counter() - t0

            # Optional: interpolated flythrough video rendered from Gaussian splats.
            if save_rendered and "splats" in predictions and hasattr(self._inner_model, "gs_renderer"):
                t0_render = time.perf_counter()
                try:
                    splats_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions["splats"].items()}
                    camera_poses_f32 = predictions["camera_poses"].float()
                    camera_intrs_f32 = predictions["camera_intrs"].float()
                    # Original bf16 splat refs are no longer needed; dropping
                    # them halves peak GPU memory during rendering for scenes
                    # with many gaussians. save_results already finished above.
                    predictions.pop("splats", None)
                    torch.cuda.empty_cache()
                    render_interpolated_video(
                        self._inner_model.gs_renderer,
                        splats_f32,
                        camera_poses_f32,
                        camera_intrs_f32,
                        (H, W),
                        outdir / "rendered",
                        interp_per_pair=render_interp_per_pair,
                        loop_reverse=(S <= 2),
                        render_depth=render_depth,
                    )
                    if log_time:
                        timings["render_video"] = time.perf_counter() - t0_render
                except Exception as e:
                    logger.warning(f"[WorldMirror] video rendering failed: {e}")
                    if log_time:
                        timings["render_video"] = -1.0

            if not self.is_distributed:
                del predictions
                torch.cuda.empty_cache()

            timings["case_total"] = time.perf_counter() - case_t0
            if log_time:
                print_and_save_timings(timings, outdir)

            logger.info(f"[WorldMirror] Results saved to: {outdir}")

        if self.is_distributed:
            # Free local tensors and resync state across ranks before the
            # next request comes in.
            if "predictions" in locals():
                del predictions
            del imgs
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

        if input_info.return_result_tensor:
            return {"output_dir": str(outdir), "timings": timings if self.rank == 0 else None}
        return {"output_dir": str(outdir)}

    def end_run(self):
        self.input_info = None
        # Release the persistent safetensors mmap handle opened by
        # lazy_load (T19). end_run is session teardown for WorldMirror
        # (run_pipeline never calls it, only stop/pause paths in
        # base_runner do), so this does not re-pay the ~500ms reopen
        # cost per request that T18 worried about. Idempotent / no-op
        # when lazy_load was never activated.
        if self.model is not None and hasattr(self.model, "close"):
            self.model.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
