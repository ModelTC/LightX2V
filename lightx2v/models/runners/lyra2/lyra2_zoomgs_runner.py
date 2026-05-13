"""Runner for Lyra-2 ZoomGS (zoom-in/zoom-out spatial video from a single image).

This runner integrates the Lyra-2 inference pipeline into LightX2V's
runner-based architecture with full lifecycle separation:

  init_modules()
    └── load_model()
          ├── load_transformer()      # Lyra-2 DiT + LoRA (incl. DMD LoRA)
          ├── load_depth_model()      # DA3 depth estimation
          ├── load_moge_model()       # MoGe depth scale alignment (optional)
          └── _load_negative_prompt() # negative T5 embedding

  run_pipeline(input_info)
    ├── run_input_encoder()           # image → depth → T5 embeddings
    └── run_main()
          ├── _run_one_direction("zoom_in")
          ├── _run_one_direction("zoom_out")
          └── _save_results(result_in, result_out, ...)

Config JSON mapping:
  model_path   → directory containing Lyra-2 checkpoints
                 (e.g. /data/nvme1/wushuo/hf_models/nvidia/Lyra-2.0/checkpoints)
  lyra_repo    → Lyra-2 source-code repository root; added to sys.path and used
                 as working directory via os.chdir (e.g. /data/nvme1/wushuo/lyra_proj/Lyra-2)
  checkpoint_dir → path to the DiT weights directory.  If relative, resolved
                   against lyra_repo (symlink makes it work).  If omitted,
                   defaults to <model_path>/model (absolute).
  All other keys map 1-to-1 to the corresponding Lyra-2 argument defaults.

InputInfo mapping (uses I2VInputInfo via --task i2v):
  image_path          → input image to generate from
  prompt              → text caption for the video
  save_result_path    → output directory for videos
  seed                → random seed
"""

from __future__ import annotations

import gc
import os
import sys
import types

import cv2
import torch
from loguru import logger

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER

# DMD LoRA is a relative path resolved against the Lyra-2 repo working directory.
_DMD_LORA_PATH = "checkpoints/lora/dmd_distillation.safetensors"
_DMD_LORA_WEIGHT = 1.0


def _ensure_lyra2_on_path(lyra_repo: str) -> None:
    """Insert the Lyra-2 repository root into sys.path if not already present."""
    if not os.path.isdir(lyra_repo):
        raise FileNotFoundError(f"[Lyra2ZoomGS] Lyra-2 repo not found: '{lyra_repo}'. Set 'lyra_repo' in the config JSON to the absolute path of the Lyra-2 repo.")
    if lyra_repo not in sys.path:
        sys.path.insert(0, lyra_repo)
        logger.info(f"[Lyra2ZoomGS] Added to sys.path: {lyra_repo}")


@RUNNER_REGISTER("lyra2_zoomgs")
class Lyra2ZoomGSRunner(BaseRunner):
    """LightX2V runner for the Lyra-2 ZoomGS inference pipeline."""

    def __init__(self, config):
        super().__init__(config)
        # Ensure Lyra-2 imports are available as early as possible so that
        # subsequent imports (e.g. from infer.py module-level code) can resolve.
        _ensure_lyra2_on_path(config["lyra_repo"])

        # Instance state populated during init_modules / run_pipeline.
        self.model = None
        self.da3_model = None
        self.moge_model = None
        self.negative_prompt_data = None
        self.desired_device = None
        self.desired_dtype = None
        self._prev_cwd = None
        self._process_group = None
        self.inputs = None  # populated by run_input_encoder

    # ------------------------------------------------------------------
    # Init / Model Loading
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Init modules")
    def init_modules(self):
        """Set working directory, initialise distributed groups, load all sub-models."""
        lyra_repo = self.config["lyra_repo"]
        model_path = self.config["model_path"]

        _ensure_lyra2_on_path(lyra_repo)

        # Validate checkpoint directory early before committing to os.chdir.
        checkpoint_dir = self.config.get("checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir_abs = os.path.join(model_path, "model")
        elif os.path.isabs(checkpoint_dir):
            checkpoint_dir_abs = checkpoint_dir
        else:
            checkpoint_dir_abs = os.path.join(lyra_repo, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir_abs):
            raise FileNotFoundError(f"[Lyra2ZoomGS] checkpoint directory not found: '{checkpoint_dir_abs}'. Check 'checkpoint_dir' (or 'model_path') in your config JSON.")
        logger.info(f"[Lyra2ZoomGS] checkpoint_dir OK: {checkpoint_dir_abs}")

        # chdir to lyra_repo once; relative paths in Lyra-2 code resolve from here
        # (checkpoints/, lyra_2/_src/configs/, etc. are all relative to lyra_repo).
        self._prev_cwd = os.getcwd()
        os.chdir(lyra_repo)
        logger.info(f"[Lyra2ZoomGS] Working directory set to: {lyra_repo}")

        with ProfilingContext4DebugL1("Init process group"):
            self._init_process_group()
        with ProfilingContext4DebugL1("Load model"):
            self.load_model()
        self.config.lock()

    def _init_process_group(self):
        """Initialise context-parallel process group if requested."""
        cp_size = self.config.get("context_parallel_size", 1)
        if cp_size > 1:
            import imaginaire
            from megatron.core import parallel_state

            imaginaire.utils.distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=cp_size)
            self._process_group = parallel_state.get_context_parallel_group()

    def load_model(self):
        """Load all sub-models and store them as instance attributes."""
        with ProfilingContext4DebugL1("Load Lyra2 DiT"):
            self.model = self.load_transformer()
        with ProfilingContext4DebugL1("Load DA3 depth model"):
            self.da3_model = self.load_depth_model()
        with ProfilingContext4DebugL1("Load MoGe model"):
            self.moge_model = self.load_moge_model()
        with ProfilingContext4DebugL1("Load negative prompt"):
            self.negative_prompt_data = self._load_negative_prompt()

    def load_transformer(self):
        """Load Lyra-2 DiT, apply LoRA weights (including DMD LoRA when enabled)."""
        from lightx2v.models.networks.lyra2.model_loader import load_model_from_checkpoint

        cfg = self.config

        # Normalise lora_paths / lora_weights (may arrive as lists or comma-strings).
        lora_paths = cfg.get("lora_paths", None) or []
        if isinstance(lora_paths, str):
            lora_paths = [p.strip() for p in lora_paths.split(",") if p.strip()]
        else:
            lora_paths = list(lora_paths)

        lora_weights = cfg.get("lora_weights", None) or []
        if isinstance(lora_weights, str):
            lora_weights = [float(w) for w in lora_weights.split(",") if w.strip()]
        else:
            lora_weights = list(lora_weights)

        # Inject DMD LoRA if enabled (equivalent to _apply_dmd_defaults).
        if cfg.get("use_dmd", False):
            lora_paths.append(_DMD_LORA_PATH)
            lora_weights.append(_DMD_LORA_WEIGHT)
            logger.info(f"[Lyra2ZoomGS] DMD enabled – appended LoRA: {_DMD_LORA_PATH}")

        experiment_opts = [
            "model.config.use_mp_policy_fsdp=False",
            "model.config.keep_original_net_dtype=False",
        ]
        if lora_paths:
            experiment_opts.append("model.config.net.postpone_checkpoint=True")

        # checkpoint_dir: use absolute path so it is independent of cwd.
        model_path = cfg["model_path"]
        checkpoint_dir = cfg.get("checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(model_path, "model")
        # If relative, it resolves against the cwd which is now lyra_repo.

        model, _ = load_model_from_checkpoint(
            config_file="lyra_2/_src/configs/config.py",
            experiment_name=cfg.get("experiment", "lyra2"),
            checkpoint_path=checkpoint_dir,
            enable_fsdp=False,
            instantiate_ema=False,
            load_ema_to_reg=False,
            experiment_opts=experiment_opts,
        )

        if lora_paths:
            # Lyra2WanDiT is not a PEFT-compatible nn.Module; LoRA cannot be injected.
            logger.warning("[Lyra2ZoomGS] LoRA loading skipped: Lyra2WanDiT is not a PEFT-compatible nn.Module. Base weights only.")

        # Cache dtype/device so downstream methods don't need to re-query.
        self.desired_dtype = model.tensor_kwargs.get("dtype", None)
        self.desired_device = model.tensor_kwargs.get("device", None)

        # Move Lyra2WanDiT weights from CPU pinned memory to GPU.
        logger.info("[Lyra2ZoomGS] Moving Lyra2WanDiT weights to CUDA …")
        model.net.to_cuda()
        logger.info("[Lyra2ZoomGS] Lyra2WanDiT weights on CUDA.")

        assert getattr(model.config, "important_start", True) is True
        assert getattr(model.config, "encode_video_from_start", True) is True
        assert not getattr(model.config, "use_hd_map_cond", False)

        model.eval()

        if cfg.get("context_parallel_size", 1) > 1:
            model.net.enable_context_parallel(self._process_group)

        warp_chunk_size = cfg.get("warp_chunk_size", None)
        if warp_chunk_size is not None:
            model.config.warp_chunk_size = warp_chunk_size
            model.warp_chunk_size = warp_chunk_size

        logger.info("[Lyra2ZoomGS] Lyra-2 DiT loaded.")
        return model

    def load_depth_model(self):
        """Load DA3 depth estimation model."""
        from lightx2v.models.input_encoders.hf.lyra2.depth_utils import load_da3_model

        cfg = self.config
        device = self.desired_device or ("cuda" if torch.cuda.is_available() else "cpu")
        da3_model = load_da3_model(
            da3_model_name=cfg.get("da3_model_name", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
            da3_model_path_custom=cfg.get("da3_model_path_custom", "checkpoints/recon/model.pt"),
            device=device,
        )
        da3_model.eval()
        logger.info("[Lyra2ZoomGS] DA3 depth model loaded.")
        return da3_model

    def load_moge_model(self):
        """Load MoGe depth scale alignment model (returns None if disabled)."""
        if not self.config.get("use_moge_scale", True):
            return None
        from lightx2v.models.input_encoders.hf.lyra2.depth_utils import load_moge_model

        device = self.desired_device or ("cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(device)
        moge_model.eval()
        logger.info("[Lyra2ZoomGS] MoGe model loaded for depth scale alignment.")
        return moge_model

    def _load_negative_prompt(self):
        """Load negative T5 embedding tensor from the .pt file."""
        neg_pt = self.config.get("negative_prompt_pt", "checkpoints/text_encoder/negative_prompt.pt")
        if not os.path.isabs(neg_pt):
            neg_pt = os.path.join(os.getcwd(), neg_pt)
        data = torch.load(neg_pt, map_location="cpu", weights_only=False)
        logger.info(f"[Lyra2ZoomGS] Loaded negative prompt from: {neg_pt}")
        return data

    # ------------------------------------------------------------------
    # Input Encoding
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run input encoder")
    def run_input_encoder(self):
        """Load image, estimate depth, optionally align MoGe scale, compute T5.

        Returns a dict with keys:
            image      – img_bchw in [-1, 1] on desired_device
            depth_hw   – depth map tensor
            mask_hw    – validity mask tensor
            K_33       – intrinsics matrix
            t5         – positive T5 embedding
            neg_t5     – negative T5 embedding
            ground_normal – optional ground-plane normal (or None)
        """
        from lightx2v.models.input_encoders.hf.lyra2.depth_encode import (
            _da3_infer_depth_intrinsics_single,
        )
        from lightx2v.models.input_encoders.hf.lyra2.get_t5_emb import (
            get_umt5_embedding,
            get_umt5_embedding_offloaded,
        )
        from lightx2v.models.networks.lyra2.lyra2_utils import to as misc_to

        cfg = self.config
        img_path = self.input_info.image_path
        target_h, target_w = [int(x) for x in cfg.get("resolution", "480,832").split(",")]

        with ProfilingContext4DebugL1("Read image"):
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise RuntimeError(f"[Lyra2ZoomGS] Cannot read image: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb)  # H,W,3 uint8

        with ProfilingContext4DebugL1("Run DA3 depth"):
            logger.info("Running DA3 single-image depth...")
            image_chw01, depth_hw, K_33, mask_hw = _da3_infer_depth_intrinsics_single(
                da3_model=self.da3_model,
                img_rgb_uint8=rgb_t,
                target_hw=(target_h, target_w),
            )

        if self.moge_model is not None:
            from lightx2v.models.input_encoders.hf.lyra2.depth_utils import moge_infer_depth_intrinsics

            with ProfilingContext4DebugL1("Align MoGe scale"):
                logger.info("Aligning DA3 depth to MoGe scale...")
                self.moge_model.to(self.desired_device)
                with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                    _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                        self.moge_model,
                        rgb_t,
                        depth_pred_hw=(target_h, target_w),
                        target_hw=(target_h, target_w),
                    )

                da3_d = depth_hw.to(moge_depth_hw.device)
                da3_m = mask_hw.to(moge_mask_hw.device)
                valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)

                if valid_mask.sum() > 10:
                    inv_da3 = 1.0 / (da3_d[valid_mask] + 1e-6)
                    inv_moge = 1.0 / (moge_depth_hw[valid_mask] + 1e-6)
                    numerator = (inv_da3 * inv_moge).sum()
                    denominator = (inv_da3 * inv_da3).sum()
                    if denominator > 1e-8:
                        scale = numerator / denominator
                        logger.info(f"Global inverse-depth scale factor: {scale.item()}")
                        if scale > 1e-6:
                            depth_hw = depth_hw / scale.to(depth_hw.device)
                        else:
                            logger.warning(f"Scale too small ({scale.item()}), skipping.")
                    else:
                        logger.warning("Denominator too small for LS scale alignment.")
                else:
                    logger.warning("Not enough valid pixels for scale alignment.")

                self.moge_model.cpu()
                del moge_depth_hw, moge_mask_hw, da3_d, da3_m
                torch.cuda.empty_cache()
                gc.collect()

        img_bchw = image_chw01.to(device=self.desired_device) * 2.0 - 1.0  # [-1, 1]

        with ProfilingContext4DebugL1("Run T5 encoder"):
            caption = getattr(self.input_info, "prompt", "") or cfg.get("prompt", "")
            if not caption:
                raise RuntimeError("[Lyra2ZoomGS] No prompt provided. Pass --prompt or add 'prompt' to the config.")
            prompt_suffix = cfg.get("prompt_suffix", "")
            if prompt_suffix:
                caption = caption.rstrip() + " " + prompt_suffix

            if cfg.get("offload_when_prompt", False):
                t5 = get_umt5_embedding_offloaded(caption, device=self.desired_device)
            else:
                t5 = get_umt5_embedding(caption, device=self.desired_device)

            if self.desired_dtype is not None:
                t5 = t5.to(dtype=self.desired_dtype)
            if t5.dim() == 2:
                t5 = t5.unsqueeze(0)
            elif t5.dim() == 3 and t5.shape[0] != 1:
                t5 = t5[:1]

            neg_t5 = misc_to(self.negative_prompt_data["t5_text_embeddings"], **self.model.tensor_kwargs)

        ground_normal = None
        if cfg.get("ground_plane_align", False):
            from lightx2v.models.input_encoders.hf.lyra2.depth_encode import _fit_ground_normal_from_depth

            with ProfilingContext4DebugL1("Fit ground plane"):
                ground_normal = _fit_ground_normal_from_depth(
                    depth_hw,
                    K_33,
                    mask_hw,
                    bottom_frac=cfg.get("ground_plane_bottom_frac", 0.4),
                )
                if ground_normal is None:
                    logger.warning("Ground plane fitting failed; using original trajectory.")

        return {
            "image": img_bchw,
            "depth_hw": depth_hw,
            "mask_hw": mask_hw,
            "K_33": K_33,
            "t5": t5,
            "neg_t5": neg_t5,
            "ground_normal": ground_normal,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _make_gen_config(self) -> types.SimpleNamespace:
        """Build a SimpleNamespace from self.config for Lyra-2 generation functions.

        run_lyra2_sample / Lyra2InferencePipeline use attribute-style access on
        this object, so we mirror all relevant Lyra-2 argument defaults here.
        """
        cfg = self.config
        seed = getattr(self.input_info, "seed", None) or cfg.get("seed", 1)
        return types.SimpleNamespace(
            # sampling
            num_sampling_step=cfg.get("num_sampling_step", 20),
            guidance=cfg.get("guidance", 5.0),
            shift=cfg.get("shift", 5.0),
            seed=seed,
            fps=cfg.get("fps", 24),
            num_frames=cfg.get("num_frames", 161),
            # DMD / scheduler
            use_dmd_scheduler=cfg.get("use_dmd_scheduler", False) or cfg.get("use_dmd", False),
            # offload
            offload=cfg.get("offload", False),
            offload_da3_diffusion=cfg.get("offload_da3_diffusion", False),
            # DA3 AR settings
            da3_model_name=cfg.get("da3_model_name", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
            da3_model_path_custom=cfg.get("da3_model_path_custom", "checkpoints/recon/model.pt"),
            da3_frame_interval=cfg.get("da3_frame_interval", 8),
            da3_max_history_frames=cfg.get("da3_max_history_frames", 10),
            da3_include_ar_chunk_last_frames=cfg.get("da3_include_ar_chunk_last_frames", False),
            da3_use_predicted_pose=cfg.get("da3_use_predicted_pose", False),
            da3_predicted_pose_continuation=cfg.get("da3_predicted_pose_continuation", False),
            # context parallel / misc
            context_parallel_size=cfg.get("context_parallel_size", 1),
            num_retrieval_views=cfg.get("num_retrieval_views", 1),
            disable_cache_update=cfg.get("disable_cache_update", False),
            multiview_ids=cfg.get("multiview_ids", None),
            ablate_same_t5=cfg.get("ablate_same_t5", False),
        )

    @ProfilingContext4DebugL1("Run main")
    def run_main(self):
        """Run zoom-in and zoom-out generation, save videos, return output dict."""
        cfg = self.config
        img_path = self.input_info.image_path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = getattr(self.input_info, "save_result_path", None) or cfg.get("output_path", "outputs/lyra2_zoomgs")

        os.makedirs(output_path, exist_ok=True)
        per_image_dir = os.path.join(output_path, base_name)
        os.makedirs(per_image_dir, exist_ok=True)
        videos_dir = os.path.join(output_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        combined_video_path = os.path.join(videos_dir, f"{base_name}.mp4")

        N_in = int(cfg.get("num_frames_zoom_in") or cfg.get("num_frames", 161))
        N_out = int(cfg.get("num_frames_zoom_out") or cfg.get("num_frames", 161))

        logger.info(f"=== Generating ZOOM-IN ({cfg.get('zoom_in_trajectory')} {cfg.get('zoom_in_direction')} str={cfg.get('zoom_in_strength')}, N={N_in}) ===")
        with ProfilingContext4DebugL1("Generate zoom-in"):
            result_in = self._run_one_direction(
                trajectory=cfg.get("zoom_in_trajectory", "horizontal_zoom"),
                direction=cfg.get("zoom_in_direction", "right"),
                strength=cfg.get("zoom_in_strength", 0.5),
                N=N_in,
                log_prefix=f"{base_name}_zoom_in",
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"=== Generating ZOOM-OUT ({cfg.get('zoom_out_trajectory')} {cfg.get('zoom_out_direction')} str={cfg.get('zoom_out_strength')}, N={N_out}) ===")
        with ProfilingContext4DebugL1("Generate zoom-out"):
            result_out = self._run_one_direction(
                trajectory=cfg.get("zoom_out_trajectory", "horizontal_zoom"),
                direction=cfg.get("zoom_out_direction", "left"),
                strength=cfg.get("zoom_out_strength", 1.5),
                N=N_out,
                log_prefix=f"{base_name}_zoom_out",
                upward_shift=cfg.get("zoom_out_upward_shift", 0.05),
                zoom_out_upward_ratio=cfg.get("zoom_out_upward_ratio", 0.15),
            )

        if result_in is None and result_out is None:
            raise RuntimeError(f"[Lyra2ZoomGS] Both zoom-in and zoom-out generation failed for: {img_path}")

        with ProfilingContext4DebugL1("Save results"):
            self._save_results(
                result_in=result_in,
                result_out=result_out,
                per_image_dir=per_image_dir,
                combined_video_path=combined_video_path,
            )

        return {"output_dir": output_path}

    def _run_one_direction(
        self,
        trajectory: str,
        direction: str,
        strength: float,
        N: int,
        log_prefix: str,
        upward_shift: float = 0.0,
        zoom_out_upward_ratio: float = 0.0,
    ):
        """Call Lyra-2's _generate_one_direction for one zoom direction."""
        from lightx2v.models.networks.lyra2.generate import _generate_one_direction

        enc = self.inputs
        return _generate_one_direction(
            model=self.model,
            args=self._make_gen_config(),
            img_bchw=enc["image"],
            depth_hw=enc["depth_hw"],
            mask_hw=enc["mask_hw"],
            K_33=enc["K_33"],
            t5_embeddings=enc["t5"],
            neg_t5_embeddings=enc["neg_t5"],
            trajectory=trajectory,
            direction=direction,
            strength=strength,
            N=N,
            da3_model=self.da3_model,
            process_group=self._process_group,
            log_prefix=log_prefix,
            ground_normal_cam=enc.get("ground_normal"),
            upward_shift=upward_shift,
            zoom_out_upward_ratio=zoom_out_upward_ratio,
        )

    def _save_results(self, result_in, result_out, per_image_dir: str, combined_video_path: str):
        """Save individual direction videos and the combined zoom-out+in video."""
        from lightx2v.models.networks.lyra2.lyra2_ar_inference import save_output
        from lightx2v.models.video_encoders.lyra2.video import save_img_or_video

        fps = self.config.get("fps", 24)

        for tag, res in [("zoom_in", result_in), ("zoom_out", result_out)]:
            if res is None:
                continue
            vid_stem = os.path.join(per_image_dir, tag)
            to_show = []
            if res.get("warp_video") is not None:
                to_show.append(res["warp_video"])
            to_show.append(res["video"])
            save_output(to_show, vid_stem + ".mp4")
            logger.info(f"Saved {tag} video: {vid_stem}.mp4")

        videos_to_combine = []
        if result_out is not None:
            videos_to_combine.append(result_out["video"].flip(dims=[2]))
        if result_in is not None:
            videos_to_combine.append(result_in["video"])

        combined_video = torch.cat(videos_to_combine, dim=2)  # [B, C, T_total, H, W]
        logger.info(f"Combined video: {combined_video.shape[2]} frames from both directions")
        combined_01 = (combined_video[0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()

        os.makedirs(os.path.dirname(combined_video_path), exist_ok=True)
        save_img_or_video(combined_01, combined_video_path.replace(".mp4", ""), fps=fps)
        logger.info(f"Saved combined video: {combined_video_path}")

        per_image_combined = os.path.join(per_image_dir, "combined")
        save_img_or_video(combined_01, per_image_combined, fps=fps)

        del combined_video, combined_01
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["Lyra2ZoomGSRunner"],
    )
    def run_pipeline(self, input_info):
        """Run the full Lyra-2 ZoomGS inference for one image."""
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info

        from lightx2v.models.networks.lyra2.lyra2_utils import set_random_seed

        seed = getattr(input_info, "seed", None) or self.config.get("seed", 1)
        set_random_seed(seed=seed, by_rank=True)

        logger.info(f"[Lyra2ZoomGS] Starting inference  image={input_info.image_path}")
        logger.info(f"[Lyra2ZoomGS] Output → {getattr(input_info, 'save_result_path', None) or self.config.get('output_path')}")

        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"[Lyra2ZoomGS] Done. Results saved to: {result['output_dir']}")
        return result

    def end_run(self):
        """Restore working directory; optionally release GPU memory."""
        self.inputs = None
        self.input_info = None

        if self._prev_cwd and os.path.isdir(self._prev_cwd):
            os.chdir(self._prev_cwd)
            logger.info(f"[Lyra2ZoomGS] Restored working directory to: {self._prev_cwd}")

        if self.config.get("offload", False):
            for attr in ("model", "da3_model", "moge_model"):
                m = getattr(self, attr, None)
                if m is not None:
                    try:
                        m.cpu()
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
