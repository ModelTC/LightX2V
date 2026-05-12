"""Runner for Lyra-2 custom camera trajectory (image → video).

Pipeline:
  init_modules()
    └── load_model()
          ├── load_transformer()      # Lyra-2 DiT (Lyra2WanDiT)
          ├── load_depth_model()      # DA3 depth estimation
          ├── load_moge_model()       # MoGe depth scale alignment (optional)
          └── _load_negative_prompt() # negative T5 embedding

  run_pipeline(input_info)
    ├── run_input_encoder()   # image → depth → trajectory loading → T5 embeddings
    └── run_main()            # assemble data_batch → run_lyra2_sample → save video

Config JSON mapping:
  model_path     → directory containing Lyra-2 checkpoints
  lyra_repo      → Lyra-2 source-code repository root (added to sys.path / os.chdir)
  checkpoint_dir → path to the DiT weights directory (relative to lyra_repo or absolute)
  num_frames     → number of frames to generate (must satisfy (N-1) % framepack_stride == 0)
  pose_scale     → scale factor applied to w2c translation vectors (default 1.1)

InputInfo mapping (uses Lyra2CustomTrajInputInfo via --task lyra2_custom_traj):
  image_path        → input image
  trajectory_path   → .npz trajectory file or directory of per-image .npz files
  prompt            → text caption string  OR  path to a per-chunk captions .json file
  save_result_path  → output directory
  seed              → random seed
"""

from __future__ import annotations

import gc
import json
import os
import sys
import types

import cv2
import numpy as np
import torch

from loguru import logger

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER

# DMD LoRA relative path (resolved against lyra_repo cwd).
_DMD_LORA_PATH = "checkpoints/lora/dmd_distillation.safetensors"
_DMD_LORA_WEIGHT = 1.0


def _ensure_lyra2_on_path(lyra_repo: str) -> None:
    """Insert the Lyra-2 repository root into sys.path if not already present."""
    if not os.path.isdir(lyra_repo):
        raise FileNotFoundError(
            f"[Lyra2CustomTraj] Lyra-2 repo not found: '{lyra_repo}'. "
            "Set 'lyra_repo' in the config JSON to the absolute path of the Lyra-2 repo."
        )
    if lyra_repo not in sys.path:
        sys.path.insert(0, lyra_repo)
        logger.info(f"[Lyra2CustomTraj] Added to sys.path: {lyra_repo}")


def _load_trajectory(path: str, num_frames: int, target_hw=None, pose_scale: float = 1.0):
    """Load camera trajectory from a .npz file.

    Expected keys:
        w2c        – (N, 4, 4) world-to-camera matrices
        intrinsics – (N, 3, 3) camera intrinsic matrices
        image_height, image_width – reference resolution for rescaling intrinsics

    Returns (w2c, intrinsics) tensors truncated to ``num_frames``.
    """
    data = np.load(path)
    w2c = torch.from_numpy(data["w2c"][:num_frames].astype(np.float32))
    intrinsics = torch.from_numpy(data["intrinsics"][:num_frames].astype(np.float32))

    if pose_scale != 1.0:
        w2c[:, :3, 3] *= pose_scale

    if target_hw is not None and "image_height" in data and "image_width" in data:
        orig_h, orig_w = int(data["image_height"]), int(data["image_width"])
        tgt_h, tgt_w = target_hw
        if (orig_h, orig_w) != (tgt_h, tgt_w):
            sx = tgt_w / orig_w
            sy = tgt_h / orig_h
            intrinsics[:, 0, 0] *= sx
            intrinsics[:, 0, 2] *= sx
            intrinsics[:, 1, 1] *= sy
            intrinsics[:, 1, 2] *= sy

    return w2c, intrinsics


@RUNNER_REGISTER("lyra2_custom_traj")
class Lyra2CustomTrajRunner(BaseRunner):
    """LightX2V runner for Lyra-2 custom camera trajectory inference."""

    def __init__(self, config):
        super().__init__(config)
        _ensure_lyra2_on_path(config["lyra_repo"])

        self.model = None
        self.da3_model = None
        self.moge_model = None
        self.negative_prompt_data = None
        self.desired_device = None
        self.desired_dtype = None
        self._prev_cwd = None
        self._process_group = None
        self.inputs = None

    # ------------------------------------------------------------------
    # Init / Model Loading
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Init modules")
    def init_modules(self):
        """Set working directory, initialise distributed groups, load all sub-models."""
        lyra_repo = self.config["lyra_repo"]
        model_path = self.config["model_path"]

        _ensure_lyra2_on_path(lyra_repo)

        checkpoint_dir = self.config.get("checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir_abs = os.path.join(model_path, "model")
        elif os.path.isabs(checkpoint_dir):
            checkpoint_dir_abs = checkpoint_dir
        else:
            checkpoint_dir_abs = os.path.join(lyra_repo, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir_abs):
            raise FileNotFoundError(
                f"[Lyra2CustomTraj] checkpoint directory not found: '{checkpoint_dir_abs}'. "
                "Check 'checkpoint_dir' (or 'model_path') in your config JSON."
            )
        logger.info(f"[Lyra2CustomTraj] checkpoint_dir OK: {checkpoint_dir_abs}")

        self._prev_cwd = os.getcwd()
        os.chdir(lyra_repo)
        logger.info(f"[Lyra2CustomTraj] Working directory set to: {lyra_repo}")

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

    @ProfilingContext4DebugL1("Load transformer")
    def load_transformer(self):
        """Load Lyra-2 DiT (Lyra2WanDiT), apply DMD LoRA when enabled."""
        from lightx2v.models.networks.lyra2.model_loader import load_model_from_checkpoint

        cfg = self.config

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

        if cfg.get("use_dmd", False):
            lora_paths.append(_DMD_LORA_PATH)
            lora_weights.append(_DMD_LORA_WEIGHT)
            logger.info(f"[Lyra2CustomTraj] DMD enabled – appended LoRA: {_DMD_LORA_PATH}")

        experiment_opts = [
            "model.config.use_mp_policy_fsdp=False",
            "model.config.keep_original_net_dtype=False",
        ]
        if lora_paths:
            experiment_opts.append("model.config.net.postpone_checkpoint=True")

        model_path = cfg["model_path"]
        checkpoint_dir = cfg.get("checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(model_path, "model")

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
            logger.warning(
                "[Lyra2CustomTraj] LoRA loading skipped: Lyra2WanDiT is not a PEFT-compatible "
                "nn.Module. Base weights only."
            )

        self.desired_dtype = model.tensor_kwargs.get("dtype", None)
        self.desired_device = model.tensor_kwargs.get("device", None)

        logger.info("[Lyra2CustomTraj] Moving Lyra2WanDiT weights to CUDA …")
        model.net.to_cuda()
        logger.info("[Lyra2CustomTraj] Lyra2WanDiT weights on CUDA.")

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

        logger.info("[Lyra2CustomTraj] Lyra-2 DiT loaded.")
        return model

    @ProfilingContext4DebugL1("Load depth model")
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
        logger.info("[Lyra2CustomTraj] DA3 depth model loaded.")
        return da3_model

    @ProfilingContext4DebugL1("Load MoGe model")
    def load_moge_model(self):
        """Load MoGe depth scale alignment model (returns None if disabled)."""
        if not self.config.get("use_moge_scale", True):
            return None
        from lightx2v.models.input_encoders.hf.lyra2.depth_utils import load_moge_model

        device = self.desired_device or ("cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(device)
        moge_model.eval()
        logger.info("[Lyra2CustomTraj] MoGe model loaded.")
        return moge_model

    def _load_negative_prompt(self):
        """Load negative T5 embedding tensor from the .pt file."""
        neg_pt = self.config.get(
            "negative_prompt_pt", "checkpoints/text_encoder/negative_prompt.pt"
        )
        if not os.path.isabs(neg_pt):
            neg_pt = os.path.join(os.getcwd(), neg_pt)
        data = torch.load(neg_pt, map_location="cpu", weights_only=False)
        logger.info(f"[Lyra2CustomTraj] Loaded negative prompt from: {neg_pt}")
        return data

    # ------------------------------------------------------------------
    # Input Encoding
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1("Run input encoder")
    def run_input_encoder(self):
        """Load image, run depth, load trajectory, encode captions.

        Returns a dict with keys:
            image      – img_bchw in [-1, 1] on desired_device
            depth_hw   – depth map tensor [H, W]
            mask_hw    – validity mask tensor [H, W]
            H, W       – image spatial dimensions after resize
            t5         – positive T5 embedding
            neg_t5     – negative T5 embedding
            w2cs       – [1, N, 4, 4] world-to-camera tensors
            Ks         – [1, N, 3, 3] intrinsics tensors
            t5_chunk_embeddings / t5_chunk_mask / t5_chunk_keys / sample_frame_indices
                       – per-chunk T5 tensors (present only when prompt is a .json file)
        """
        from lightx2v.models.networks.lyra2.lyra2_utils import to as misc_to
        from lightx2v.models.input_encoders.hf.lyra2.depth_encode import (
            _da3_infer_depth_intrinsics_single,
        )
        from lightx2v.models.input_encoders.hf.lyra2.get_t5_emb import (
            get_umt5_embedding,
            get_umt5_embedding_offloaded,
        )

        cfg = self.config
        img_path = self.input_info.image_path
        target_h, target_w = [int(x) for x in cfg.get("resolution", "480,832").split(",")]
        N = int(cfg.get("num_frames", 81))
        pose_scale = float(cfg.get("pose_scale", 1.1))

        # ---- Read image ----
        with ProfilingContext4DebugL1("Read image"):
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise RuntimeError(f"[Lyra2CustomTraj] Cannot read image: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb)

        # ---- Load trajectory ----
        with ProfilingContext4DebugL1("Load trajectory"):
            traj_path = self.input_info.trajectory_path
            if not traj_path:
                raise RuntimeError(
                    "[Lyra2CustomTraj] trajectory_path is required. "
                    "Pass --trajectory_path to the launch script."
                )
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if os.path.isdir(traj_path):
                traj_file = os.path.join(traj_path, f"{base_name}.npz")
            else:
                traj_file = traj_path
            if not os.path.isfile(traj_file):
                raise FileNotFoundError(
                    f"[Lyra2CustomTraj] Trajectory file not found: {traj_file}"
                )
            w2cs_T_44, Ks_T_33 = _load_trajectory(
                traj_file, N, target_hw=(target_h, target_w), pose_scale=pose_scale
            )
            logger.info(
                f"[Lyra2CustomTraj] Loaded trajectory: {w2cs_T_44.shape[0]} frames from {traj_file}"
            )

        # ---- DA3 depth estimation ----
        with ProfilingContext4DebugL1("Run DA3 depth"):
            logger.info("Running DA3 single-image depth...")
            image_chw01, depth_hw, _K_33_da3, mask_hw = _da3_infer_depth_intrinsics_single(
                da3_model=self.da3_model,
                img_rgb_uint8=rgb_t,
                target_hw=(target_h, target_w),
            )
        H, W = image_chw01.shape[-2:]

        # ---- Optionally align DA3 depth to MoGe scale ----
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

        img_bchw = image_chw01.to(device=self.desired_device) * 2.0 - 1.0

        # ---- Captions / T5 encoding ----
        with ProfilingContext4DebugL1("Run T5 encoder"):
            t5, chunk_data = self._encode_captions(
                get_umt5_embedding=get_umt5_embedding,
                get_umt5_embedding_offloaded=get_umt5_embedding_offloaded,
                N=N,
            )
            neg_t5 = misc_to(
                self.negative_prompt_data["t5_text_embeddings"], **self.model.tensor_kwargs
            )

        result = {
            "image": img_bchw,
            "depth_hw": depth_hw,
            "mask_hw": mask_hw,
            "H": H,
            "W": W,
            "t5": t5,
            "neg_t5": neg_t5,
            "w2cs": w2cs_T_44,
            "Ks": Ks_T_33,
        }
        result.update(chunk_data)
        return result

    def _encode_captions(self, get_umt5_embedding, get_umt5_embedding_offloaded, N: int):
        """Encode captions.

        If input_info.prompt points to a .json file, loads per-chunk captions and
        encodes each chunk separately (producing t5_chunk_embeddings etc.).
        Otherwise, encodes as a single T5 embedding.

        Returns (t5_base, chunk_data_dict).
        chunk_data_dict is empty when a single caption is used.
        """
        cfg = self.config
        prompt_or_path = getattr(self.input_info, "prompt", "") or cfg.get("prompt", "")
        prompt_suffix = cfg.get("prompt_suffix", "")
        offload = cfg.get("offload_when_prompt", False)

        def _encode(text):
            if offload:
                emb = get_umt5_embedding_offloaded(text, device=self.desired_device)
            else:
                emb = get_umt5_embedding(text, device=self.desired_device)
            if self.desired_dtype is not None:
                emb = emb.to(dtype=self.desired_dtype)
            return emb

        # Detect if prompt is a .json captions file
        captions_dict = None
        if prompt_or_path.endswith(".json") and os.path.isfile(prompt_or_path):
            with open(prompt_or_path, "r") as f:
                captions_dict = json.load(f)
            logger.info(
                f"[Lyra2CustomTraj] Loaded captions JSON: {prompt_or_path} "
                f"({len(captions_dict)} entries)"
            )

        # ---- Per-chunk captions path ----
        if captions_dict is not None:
            chunk_keys_int = sorted(int(k) for k in captions_dict if int(k) < N)
            if len(chunk_keys_int) > 1:
                logger.info(
                    f"[Lyra2CustomTraj] Using {len(chunk_keys_int)} per-chunk captions"
                )
                chunk_keys = torch.tensor(
                    chunk_keys_int, dtype=torch.long, device=self.desired_device
                )
                chunk_embs, chunk_masks = [], []
                for ck in chunk_keys_int:
                    cap = captions_dict[str(ck)]
                    if prompt_suffix:
                        cap = cap.rstrip() + " " + prompt_suffix
                    emb = _encode(cap)
                    if emb.dim() == 3:
                        emb = emb[0]
                    S, D = emb.shape
                    S = min(S, 512)
                    D = min(D, 4096)
                    padded_emb = torch.zeros(
                        512, 4096, dtype=self.desired_dtype, device=self.desired_device
                    )
                    padded_emb[:S, :D] = emb[:S, :D]
                    padded_mask = torch.zeros(
                        512, dtype=self.desired_dtype, device=self.desired_device
                    )
                    padded_mask[:S] = 1.0
                    chunk_embs.append(padded_emb)
                    chunk_masks.append(padded_mask)

                t5_chunk_embeddings = torch.stack(chunk_embs).unsqueeze(0)   # [1, K, 512, 4096]
                t5_chunk_mask = torch.stack(chunk_masks).unsqueeze(0)        # [1, K, 512]
                t5_chunk_keys = chunk_keys.unsqueeze(0)                       # [1, K]
                sample_frame_indices = torch.arange(
                    N, dtype=torch.long, device=self.desired_device
                ).unsqueeze(0)                                                 # [1, N]
                t5_base = t5_chunk_embeddings[:, 0, :, :]                     # first chunk as base
                if t5_base.dim() == 2:
                    t5_base = t5_base.unsqueeze(0)
                return t5_base, {
                    "t5_chunk_embeddings": t5_chunk_embeddings,
                    "t5_chunk_mask": t5_chunk_mask,
                    "t5_chunk_keys": t5_chunk_keys,
                    "sample_frame_indices": sample_frame_indices,
                }

            # Only one entry → fall through to single-caption path with that text
            if chunk_keys_int:
                prompt_or_path = captions_dict.get(str(chunk_keys_int[0]), "")
            else:
                prompt_or_path = ""

        # ---- Single-caption path ----
        if not prompt_or_path:
            raise RuntimeError(
                "[Lyra2CustomTraj] No caption provided. Pass --prompt with a text string "
                "or the path to a per-chunk captions .json file."
            )
        caption = prompt_or_path
        if prompt_suffix:
            caption = caption.rstrip() + " " + prompt_suffix
        t5 = _encode(caption)
        if t5.dim() == 2:
            t5 = t5.unsqueeze(0)
        elif t5.dim() == 3 and t5.shape[0] != 1:
            t5 = t5[:1]
        return t5, {}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _make_gen_config(self) -> types.SimpleNamespace:
        """Build a SimpleNamespace from self.config for Lyra-2 generation functions."""
        cfg = self.config
        seed = getattr(self.input_info, "seed", None) or cfg.get("seed", 1)
        return types.SimpleNamespace(
            num_sampling_step=cfg.get("num_sampling_step", 35),
            guidance=cfg.get("guidance", 5.0),
            shift=cfg.get("shift", 5.0),
            seed=seed,
            fps=cfg.get("fps", 16),
            num_frames=cfg.get("num_frames", 81),
            use_dmd_scheduler=cfg.get("use_dmd_scheduler", False) or cfg.get("use_dmd", False),
            offload=cfg.get("offload", False),
            offload_da3_diffusion=cfg.get("offload_da3_diffusion", False),
            da3_model_name=cfg.get("da3_model_name", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
            da3_model_path_custom=cfg.get("da3_model_path_custom", "checkpoints/recon/model.pt"),
            da3_frame_interval=cfg.get("da3_frame_interval", 8),
            da3_max_history_frames=cfg.get("da3_max_history_frames", 10),
            da3_include_ar_chunk_last_frames=cfg.get("da3_include_ar_chunk_last_frames", False),
            da3_use_predicted_pose=cfg.get("da3_use_predicted_pose", False),
            da3_predicted_pose_continuation=cfg.get("da3_predicted_pose_continuation", False),
            context_parallel_size=cfg.get("context_parallel_size", 1),
            num_retrieval_views=cfg.get("num_retrieval_views", 1),
            disable_cache_update=cfg.get("disable_cache_update", False),
            multiview_ids=cfg.get("multiview_ids", None),
            ablate_same_t5=cfg.get("ablate_same_t5", False),
        )

    @ProfilingContext4DebugL1("Run main")
    def run_main(self):
        """Assemble data_batch, run AR inference, save output video."""
        from lightx2v.models.networks.lyra2.lyra2_ar_inference import run_lyra2_sample
        from lightx2v.models.video_encoders.lyra2.video import save_img_or_video

        cfg = self.config
        enc = self.inputs
        img_path = self.input_info.image_path
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        output_path = (
            getattr(self.input_info, "save_result_path", None)
            or cfg.get("output_path", "outputs/lyra2_custom_traj")
        )
        os.makedirs(output_path, exist_ok=True)

        N = int(cfg.get("num_frames", 81))
        fps = cfg.get("fps", 16)
        H, W = enc["H"], enc["W"]

        # ---- Assemble data_batch ----
        with ProfilingContext4DebugL1("Assemble data batch"):
            w2cs_b_t_44 = enc["w2cs"].unsqueeze(0).to(
                dtype=torch.float32, device=self.desired_device
            )
            Ks_b_t_33 = enc["Ks"].unsqueeze(0).to(
                dtype=torch.float32, device=self.desired_device
            )
            depth_b_thw = (
                enc["depth_hw"]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, N, 1, 1)
                .to(device=self.desired_device)
            )

            data_batch = {
                "video": enc["image"].unsqueeze(2),             # [1, C, 1, H, W]
                "t5_text_embeddings": enc["t5"],
                "neg_t5_text_embeddings": enc["neg_t5"],
                "fps": torch.tensor([fps], dtype=torch.int32, device=self.desired_device),
                "padding_mask": torch.zeros(
                    (1, 1, H, W),
                    dtype=self.model.tensor_kwargs["dtype"],
                    device=self.desired_device,
                ),
                "is_preprocessed": torch.tensor(
                    [True], dtype=torch.bool, device=self.desired_device
                ),
                "camera_w2c": w2cs_b_t_44,
                "intrinsics": Ks_b_t_33,
                "depth": depth_b_thw,
            }
            # Inject per-chunk caption tensors when present
            for key in ("t5_chunk_keys", "t5_chunk_embeddings", "t5_chunk_mask", "sample_frame_indices"):
                if key in enc:
                    data_batch[key] = enc[key]

            # safe_to: cast float tensors to model dtype, keep camera/depth as float32
            from lightx2v.models.networks.lyra2.lyra2_ar_inference import safe_to
            skip_keys = {"camera_w2c", "intrinsics", "depth", "t5_chunk_keys", "sample_frame_indices"}
            data_batch = safe_to(
                data_batch,
                device=self.model.tensor_kwargs.get("device", None),
                dtype=self.model.tensor_kwargs.get("dtype", None),
                skip_keys=skip_keys,
            )

        # ---- AR inference ----
        logger.info(f"=== Generating video ({N} frames) for {base_name} ===")
        with ProfilingContext4DebugL1("Run AR inference"):
            result = run_lyra2_sample(
                self.model,
                data_batch,
                self._make_gen_config(),
                process_group=self._process_group,
                da3_model=self.da3_model,
                show_progress=True,
                log_prefix=f"{base_name}_custom_traj",
            )

        if result is None:
            raise RuntimeError(
                f"[Lyra2CustomTraj] Generation failed for image: {img_path}"
            )

        # ---- Save output ----
        with ProfilingContext4DebugL1("Save results"):
            video_path = os.path.join(output_path, f"{base_name}.mp4")
            video_01 = (result["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()
            save_img_or_video(video_01, video_path.replace(".mp4", ""), fps=fps)
            logger.info(f"[Lyra2CustomTraj] Saved video: {video_path}")

        del result, data_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"output_dir": output_path}

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["Lyra2CustomTrajRunner"],
    )
    def run_pipeline(self, input_info):
        """Run the full Lyra-2 custom trajectory inference for one image."""
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info

        from lightx2v.models.networks.lyra2.lyra2_utils import set_random_seed

        seed = getattr(input_info, "seed", None) or self.config.get("seed", 1)
        set_random_seed(seed=seed, by_rank=True)

        logger.info(f"[Lyra2CustomTraj] Starting inference  image={input_info.image_path}")
        logger.info(f"[Lyra2CustomTraj] Trajectory: {input_info.trajectory_path}")
        logger.info(
            f"[Lyra2CustomTraj] Output → "
            f"{getattr(input_info, 'save_result_path', None) or self.config.get('output_path')}"
        )

        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"[Lyra2CustomTraj] Done. Results saved to: {result['output_dir']}")
        return result

    def end_run(self):
        """Restore working directory; optionally release GPU memory."""
        self.inputs = None
        self.input_info = None

        if self._prev_cwd and os.path.isdir(self._prev_cwd):
            os.chdir(self._prev_cwd)
            logger.info(f"[Lyra2CustomTraj] Restored working directory to: {self._prev_cwd}")

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
