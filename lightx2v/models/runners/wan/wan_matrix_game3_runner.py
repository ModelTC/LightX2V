import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, build_wan_model_with_lora
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import best_output_size
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MATRIX_GAME3_OFFICIAL_ROOT_RELATIVE_CANDIDATES = (
    Path("Matrix-Game-3") / "Matrix-Game-3",
    Path("Matrix-Game-3"),
)
_MATRIX_GAME3_CONFIG_ROOT_RELATIVE = Path("Matrix-Game-3.0")
_MATRIX_GAME3_OFFICIAL_PACKAGE = "_lightx2v_matrix_game3_official"


@dataclass
class MatrixGame3SegmentState:
    """Precomputed inputs and bookkeeping for one Matrix-Game-3 segment.

    The runner generates video in overlapping chunks. For each chunk we cache:
    - the absolute frame window covered by this segment;
    - the latent tensor shape the scheduler should sample;
    - how many latent frames are fixed by conditioning instead of sampled;
    - the condition tensors that will be forwarded through `dit_cond_dict`;
    - how many decoded RGB frames should be trimmed before concatenation.
    """

    segment_idx: int
    first_clip: bool
    current_start_frame_idx: int
    current_end_frame_idx: int
    frame_count: int
    fixed_latent_frames: int
    latent_shape: list[int]
    decode_trim_frames: int
    append_latent_start: int
    keyboard_cond: torch.Tensor
    mouse_cond: torch.Tensor
    vae_encoder_out: torch.Tensor
    dit_cond_dict: dict[str, Any]


def _load_module_from_path(module_name: str, file_path: Path):
    """Import an official Matrix-Game-3 helper module by filesystem path once."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_namespace_package(package_name: str, package_path: Path):
    """Register a synthetic namespace package so relative imports inside official code work."""
    if package_name in sys.modules:
        return sys.modules[package_name]
    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module
    return module


def _expand_path_candidates(path_value: Any) -> list[Path]:
    """Resolve a user-provided path against cwd and the project root when needed."""
    raw_path = Path(str(path_value)).expanduser()
    if raw_path.is_absolute():
        return [raw_path]
    candidates = [Path.cwd() / raw_path]
    project_relative = _PROJECT_ROOT / raw_path
    if project_relative != candidates[0]:
        candidates.append(project_relative)
    return candidates


@RUNNER_REGISTER("wan2.2_matrix_game3")
class WanMatrixGame3Runner(Wan22DenseRunner):
    """Runner-only Matrix-Game-3 adapter on top of the existing Wan2.2 lifecycle.

    Official provenance:
    - CLI / mode semantics: Matrix-Game-3/generate.py
    - Segment lengths / condition assembly: pipeline/inference_pipeline.py
    - Interactive action refresh: pipeline/inference_interactive_pipeline.py
    - Keyboard / mouse dimensions: utils/conditions.py
    - Pose / plucker helpers: utils/cam_utils.py and utils/utils.py
    - Structural config truth: Matrix-Game-3.0/*/config.json

    Execution model:
    - Reuse Wan2.2 text encoder / scheduler / VAE lifecycle from `Wan22DenseRunner`.
    - Replace the normal i2v input path with a first-frame-only conditioning scheme.
    - Convert keyboard, mouse, and camera trajectories into per-segment DiT conditions.
    - Roll latent history across overlapping segments, then trim duplicated decoded frames.
    """

    def __init__(self, config):
        with config.temporarily_unlocked():
            # The public pipeline still instantiates us as "wan2.2_matrix_game3", but
            # the shared Wan2.2 runner expects `model_cls == "wan2.2"` for common setup.
            original_model_cls = str(config.get("model_cls", "wan2.2_matrix_game3"))
            config["runner_model_cls"] = original_model_cls
            config["model_cls"] = "wan2.2"
            config["mode"] = "matrix_game3"
            config["use_image_encoder"] = False
            config["use_base_model"] = bool(config.get("use_base_model", False))
            if "sub_model_folder" not in config:
                config["sub_model_folder"] = "base_model" if config["use_base_model"] else "base_distilled_model"
            config["num_channels_latents"] = int(config.get("num_channels_latents", 48))
            config["vae_stride"] = tuple(config.get("vae_stride", (4, 16, 16)))
            config["patch_size"] = tuple(config.get("patch_size", (1, 2, 2)))
            # Load the official MG3 sub-model config before the parent runner
            # constructs the scheduler. The shared Wan scheduler expects fields
            # like `dim` and `num_heads` to already exist in `self.config`.
            self.config = config
            self.matrix_game3_model_cls = original_model_cls
            self._load_matrix_game3_model_config()
        super().__init__(config)

        self.matrix_game3_model_cls = original_model_cls
        # Official MG3 timeline convention:
        # - first segment predicts 57 frames from the input image;
        # - later segments operate on a 56-frame window;
        # - every new segment contributes 40 new frames and reuses 16 historical frames.
        action_config = self.config.get("action_config", {})
        self.first_clip_frame = int(self.config.get("first_clip_frame", 57))
        self.clip_frame = int(self.config.get("clip_frame", 56))
        self.incremental_segment_frames = int(self.config.get("incremental_segment_frames", 40))
        self.past_frame = int(self.config.get("past_frame", 16))
        self.conditioning_latent_frames = int(self.config.get("conditioning_latent_frames", 4))
        self.mouse_dim_in = int(self.config.get("mouse_dim_in", action_config.get("mouse_dim_in", 2)))
        self.keyboard_dim_in = int(self.config.get("keyboard_dim_in", action_config.get("keyboard_dim_in", 6)))

        # Session-scoped caches filled by `_prepare_matrix_game3_session()` and then
        # consumed incrementally as each segment is initialized and decoded.
        self._segment_states: dict[int, MatrixGame3SegmentState] = {}
        self._official_modules: Optional[dict[str, Any]] = None
        self._mg3_lat_h: Optional[int] = None
        self._mg3_lat_w: Optional[int] = None
        self._mg3_target_h: Optional[int] = None
        self._mg3_target_w: Optional[int] = None
        self._mg3_base_intrinsics: Optional[torch.Tensor] = None
        self._mg3_intrinsics_all: Optional[torch.Tensor] = None
        self._mg3_keyboard_all: Optional[torch.Tensor] = None
        self._mg3_mouse_all: Optional[torch.Tensor] = None
        self._mg3_extrinsics_all: Optional[torch.Tensor] = None
        self._mg3_num_iterations: int = 1
        self._mg3_expected_total_frames: int = self.first_clip_frame
        self._mg3_interactive = bool(self.config.get("interactive", False))
        self._mg3_last_pose = np.zeros(5, dtype=np.float32)
        self._mg3_current_segment_state: Optional[MatrixGame3SegmentState] = None
        self._mg3_current_segment_full_latents: Optional[torch.Tensor] = None
        self._mg3_generated_latent_history: list[torch.Tensor] = []
        self._mg3_tail_latents: Optional[torch.Tensor] = None
        self._mg3_noise_generator: Optional[torch.Generator] = None

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        # Some callers still use `pose`, others use `action_path`. Mirror both so the
        # runner remains compatible with older LightX2V entry points.
        if "action_path" in self.input_info.__dataclass_fields__:
            self.input_info.action_path = inputs.get("action_path", inputs.get("pose", ""))
        if "pose" in self.input_info.__dataclass_fields__:
            self.input_info.pose = inputs.get("pose", inputs.get("action_path", ""))

    def load_transformer(self):
        from lightx2v.models.networks.wan.matrix_game3_model import WanMtxg3Model

        # The backbone is still a Wan2.2 DiT, but Matrix-Game-3 swaps in a dedicated
        # network wrapper that understands keyboard / mouse / camera conditions.
        model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            return WanMtxg3Model(**model_kwargs)
        return build_wan_model_with_lora(WanMtxg3Model, self.config, model_kwargs, lora_configs, model_type="wan2.2")

    def _get_sub_model_folder(self) -> str:
        """Resolve which MG3 sub-model folder should be used for config lookup."""
        return str(self.config.get("sub_model_folder", "base_model" if self.config.get("use_base_model", False) else "base_distilled_model"))

    def _resolve_official_root_candidate(self, candidate: Path) -> Optional[Path]:
        """Accept either the inner package root or its parent repository directory."""
        direct_root = candidate.expanduser()
        if (direct_root / "generate.py").is_file() and (direct_root / "pipeline").is_dir() and (direct_root / "utils").is_dir():
            return direct_root

        nested_root = direct_root / "Matrix-Game-3"
        if (nested_root / "generate.py").is_file() and (nested_root / "pipeline").is_dir() and (nested_root / "utils").is_dir():
            return nested_root
        return None

    def resolve_official_root(self) -> Path:
        """Resolve the official Matrix-Game-3 source root using config-first priority."""
        configured_root = self.config.get("matrix_game3_official_root")
        if configured_root:
            for candidate in _expand_path_candidates(configured_root):
                resolved = self._resolve_official_root_candidate(candidate)
                if resolved is not None:
                    return resolved
            raise FileNotFoundError(
                "Matrix-Game-3 official source root is missing or invalid for "
                f"matrix_game3_official_root={configured_root!r}. "
                "The runner needs the official utils/pipeline files to build camera and action conditions. "
                "Please set config['matrix_game3_official_root'] to the official source root directory."
            )

        for relative_path in _MATRIX_GAME3_OFFICIAL_ROOT_RELATIVE_CANDIDATES:
            resolved = self._resolve_official_root_candidate(_PROJECT_ROOT / relative_path)
            if resolved is not None:
                return resolved

        raise FileNotFoundError(
            "Matrix-Game-3 official source root could not be resolved from the project layout. "
            "The runner needs it to import official utils/conditions.py, utils/cam_utils.py, utils/utils.py, "
            "and pipeline helpers. Please set config['matrix_game3_official_root'] explicitly."
        )

    def resolve_model_config_path(self) -> Path:
        """Resolve the MG3 base/distilled config.json with explicit override support."""
        configured_path = self.config.get("matrix_game3_config_path")
        if configured_path:
            for candidate in _expand_path_candidates(configured_path):
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                "Matrix-Game-3 config.json is missing for "
                f"matrix_game3_config_path={configured_path!r}. "
                "The runner needs this file to align latent channels, patch size, and action_config with the checkpoint. "
                "Please set config['matrix_game3_config_path'] to a valid config.json path."
            )

        sub_model_folder = self._get_sub_model_folder()
        candidates: list[Path] = []
        model_path = self.config.get("model_path")
        if model_path:
            for candidate_root in _expand_path_candidates(model_path):
                candidate = candidate_root / sub_model_folder / "config.json"
                if candidate not in candidates:
                    candidates.append(candidate)
        candidates.append(_PROJECT_ROOT / _MATRIX_GAME3_CONFIG_ROOT_RELATIVE / sub_model_folder / "config.json")

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        checked_locations = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Matrix-Game-3 sub-model config.json could not be resolved. "
            f"Checked: {checked_locations}. "
            "The runner needs this file to determine the official base/distilled structure. "
            "Please set config['matrix_game3_config_path'], or provide a valid config['model_path'] and "
            "config['sub_model_folder'] (defaulted from config['use_base_model'])."
        )

    def _load_matrix_game3_model_config(self):
        """Merge the official MG3 config so latent/channel sizes match the checkpoint."""
        config_path = self.resolve_model_config_path()
        with config_path.open("r") as f:
            model_config = json.load(f)

        with self.config.temporarily_unlocked():
            self.config.update(model_config)
            self.config["num_channels_latents"] = int(model_config.get("in_dim", self.config.get("num_channels_latents", 48)))
            self.config["vae_stride"] = tuple(self.config.get("vae_stride", (4, 16, 16)))
            self.config["patch_size"] = tuple(model_config.get("patch_size", self.config.get("patch_size", (1, 2, 2))))

        action_config = self.config.get("action_config", {})
        self.keyboard_dim_in = int(self.config.get("keyboard_dim_in", action_config.get("keyboard_dim_in", 6)))
        self.mouse_dim_in = int(self.config.get("mouse_dim_in", action_config.get("mouse_dim_in", 2)))

    def _get_official_modules(self) -> dict[str, Any]:
        """Lazy-load helper code from the official Matrix-Game-3 repository.

        We intentionally reuse the official camera/action utilities instead of
        re-implementing pose math in the LightX2V runner.
        """
        if self._official_modules is not None:
            return self._official_modules

        official_root = self.resolve_official_root()
        utils_root = official_root / "utils"
        if not utils_root.is_dir():
            raise FileNotFoundError(
                f"Matrix-Game-3 utils directory is missing under {official_root}. "
                "The runner needs the official utils modules to construct action and camera conditions. "
                "Please set config['matrix_game3_official_root'] to the official source root directory."
            )

        required_utils = {
            "conditions": utils_root / "conditions.py",
            "cam_utils": utils_root / "cam_utils.py",
            "transform": utils_root / "transform.py",
            "utils": utils_root / "utils.py",
        }
        missing_utils = [str(path) for path in required_utils.values() if not path.is_file()]
        if missing_utils:
            raise FileNotFoundError(
                "Matrix-Game-3 official utility files are incomplete. "
                f"Missing: {missing_utils}. "
                "The runner needs these files to reuse the official action/camera preprocessing. "
                "Please set config['matrix_game3_official_root'] to a complete official source checkout."
            )

        _ensure_namespace_package(_MATRIX_GAME3_OFFICIAL_PACKAGE, official_root)
        utils_pkg = f"{_MATRIX_GAME3_OFFICIAL_PACKAGE}.utils"
        _ensure_namespace_package(utils_pkg, utils_root)

        modules = {
            "conditions": _load_module_from_path(f"{utils_pkg}.conditions", required_utils["conditions"]),
            "cam_utils": _load_module_from_path(f"{utils_pkg}.cam_utils", required_utils["cam_utils"]),
            "transform": _load_module_from_path(f"{utils_pkg}.transform", required_utils["transform"]),
            "utils": _load_module_from_path(f"{utils_pkg}.utils", required_utils["utils"]),
        }
        self._official_modules = modules
        return modules

    def _get_expected_total_frames(self, raw_total_frames: Optional[int] = None) -> tuple[int, int]:
        """Resolve how many segments to run.

        Matrix-Game-3 only supports lengths of `57 + 40 * k`. If a control sequence
        does not align exactly, the tail is ignored so the segment schedule stays valid.
        """
        num_iterations = self.config.get("num_iterations", None)
        if num_iterations is not None:
            num_iterations = max(int(num_iterations), 1)
            return num_iterations, self.first_clip_frame + (num_iterations - 1) * self.incremental_segment_frames

        if raw_total_frames is None:
            return 1, self.first_clip_frame

        if raw_total_frames <= self.first_clip_frame:
            return 1, self.first_clip_frame

        additional_frames = raw_total_frames - self.first_clip_frame
        num_iterations = 1 + max(additional_frames // self.incremental_segment_frames, 0)
        expected_total_frames = self.first_clip_frame + (num_iterations - 1) * self.incremental_segment_frames
        if additional_frames % self.incremental_segment_frames != 0:
            logger.warning(
                "[matrix-game-3] raw control sequence has {} frames; truncating tail to {} frames so it matches 57 + 40*k.",
                raw_total_frames,
                expected_total_frames,
            )
        return num_iterations, expected_total_frames

    def _segment_latent_shape(self, lat_h: int, lat_w: int, frame_count: int) -> list[int]:
        """Compute `[C, T, H, W]` latent shape for one segment window."""
        return [
            self.config.get("num_channels_latents", 48),
            (frame_count - 1) // self.config["vae_stride"][0] + 1,
            lat_h,
            lat_w,
        ]

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanMatrixGame3Runner"],
    )
    def run_vae_encoder(self, img):
        # Unlike the generic Wan2.2 i2v path, MG3 only encodes the first frame. The
        # remaining temporal slots are left zeroed and later mixed with scheduler noise.
        max_area = self.config.target_height * self.config.target_width
        ih, iw = img.height, img.width
        dh = self.config.patch_size[1] * self.config.vae_stride[1]
        dw = self.config.patch_size[2] * self.config.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))

        image_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(AI_DEVICE).unsqueeze(1)
        first_frame_latent = self.get_vae_encoder_output(image_tensor)
        lat_h = oh // self.config["vae_stride"][1]
        lat_w = ow // self.config["vae_stride"][2]
        latent_shape = self._segment_latent_shape(lat_h, lat_w, self.first_clip_frame)
        vae_encoder_out = torch.zeros(latent_shape, device=first_frame_latent.device, dtype=first_frame_latent.dtype)
        vae_encoder_out[:, : first_frame_latent.shape[1]] = first_frame_latent
        return vae_encoder_out, latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        # MG3 does not use the CLIP image encoder branch. The conditioning payload is:
        # - text encoder output from the normal Wan pipeline;
        # - a first-frame VAE latent;
        # - segment metadata prepared for later `init_run_segment()` calls.
        _, img_ori = self.read_image_input(self.input_info.image_path)
        vae_encoder_out, latent_shape = self.run_vae_encoder(img_ori)
        self.input_info.latent_shape = latent_shape
        text_encoder_output = self.run_text_encoder(self.input_info)
        self._prepare_matrix_game3_session(img_ori, latent_shape, vae_encoder_out)
        torch_device_module.empty_cache()
        return self.get_encoder_output_i2v(None, vae_encoder_out, text_encoder_output)

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
        # Keep the standard LightX2V output contract so downstream scheduler / model
        # code can stay unchanged. Segment-specific conditions are injected later.
        image_encoder_output = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encoder_out": vae_encoder_out,
            "dit_cond_dict": {},
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    def _prepare_matrix_game3_session(self, pil_image: Image.Image, latent_shape: list[int], vae_encoder_out: torch.Tensor):
        # Official source:
        # - Non-interactive path mirrors pipeline/inference_pipeline.py
        # - Interactive segment refreshing mirrors pipeline/inference_interactive_pipeline.py
        # - Camera/action fallback semantics follow the user's requested runner contract
        #
        # This method performs all once-per-request setup:
        # - resolve spatial sizes used by camera/plucker helpers;
        # - reset cached segment state and latent history;
        # - pre-load the entire control sequence for offline mode; or
        # - defer control acquisition to segment time for interactive mode.
        self._get_official_modules()
        self._segment_states.clear()
        self._mg3_generated_latent_history = []
        self._mg3_tail_latents = None
        self._mg3_current_segment_state = None
        self._mg3_current_segment_full_latents = None
        self._mg3_interactive = bool(self.config.get("interactive", False))
        self._mg3_last_pose = np.zeros(5, dtype=np.float32)
        self._mg3_lat_h = int(latent_shape[-2])
        self._mg3_lat_w = int(latent_shape[-1])
        self._mg3_target_h = self._mg3_lat_h * self.config["vae_stride"][1]
        self._mg3_target_w = self._mg3_lat_w * self.config["vae_stride"][2]
        self._mg3_base_intrinsics = self._default_intrinsics().to(dtype=torch.float32)

        if self._mg3_interactive:
            num_iterations = self.config.get("num_iterations", 1)
            self._mg3_num_iterations = max(int(num_iterations), 1)
            self._mg3_expected_total_frames = self.first_clip_frame + (self._mg3_num_iterations - 1) * self.incremental_segment_frames
            self._mg3_keyboard_all = None
            self._mg3_mouse_all = None
            self._mg3_extrinsics_all = None
            self._mg3_intrinsics_all = None
            return

        action_path = self.input_info.action_path or self.input_info.pose or ""
        raw_controls = self._load_control_payload(action_path)
        raw_total_frames = self._infer_raw_total_frames(raw_controls)
        self._mg3_num_iterations, self._mg3_expected_total_frames = self._get_expected_total_frames(raw_total_frames)
        self._mg3_keyboard_all, self._mg3_mouse_all, self._mg3_extrinsics_all, self._mg3_intrinsics_all = self._build_noninteractive_controls(raw_controls)

    def _infer_raw_total_frames(self, payload: dict[str, Any]) -> Optional[int]:
        """Infer sequence length from whichever control tensor is present."""
        lengths = []
        for value in payload.values():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                if value.ndim >= 1:
                    lengths.append(int(value.shape[0]))
            elif isinstance(value, torch.Tensor):
                if value.ndim >= 1:
                    lengths.append(int(value.shape[0]))
            elif isinstance(value, list):
                lengths.append(len(value))
        return max(lengths) if lengths else None

    def _load_control_payload(self, action_path: str) -> dict[str, Any]:
        """Load keyboard/mouse/pose/intrinsics controls from a file or a directory."""
        if not action_path:
            logger.warning("[matrix-game-3] action_path missing, fallback to zero keyboard/mouse and identity poses.")
            return {}

        path = Path(action_path)
        if not path.exists():
            logger.warning("[matrix-game-3] action_path not found: {}. Fallback to zero keyboard/mouse and identity poses.", action_path)
            return {}

        if path.is_dir():
            return self._load_control_payload_from_dir(path)
        return self._load_control_payload_from_file(path)

    def _load_control_payload_from_dir(self, path: Path) -> dict[str, Any]:
        """Best-effort directory loader that accepts several common file names."""
        payload: dict[str, Any] = {}
        name_groups = {
            "keyboard_cond": ["keyboard_cond.npy", "keyboard_condition.npy", "keyboard_cond.pt", "keyboard_condition.pt", "keyboard_cond.json", "keyboard_condition.json"],
            "mouse_cond": ["mouse_cond.npy", "mouse_condition.npy", "mouse_cond.pt", "mouse_condition.pt", "mouse_cond.json", "mouse_condition.json"],
            "poses": ["poses.npy", "pose.npy", "poses.pt", "pose.pt", "poses.json", "pose.json", "c2ws.npy", "c2w.npy"],
            "intrinsics": ["intrinsics.npy", "intrinsics.pt", "intrinsics.json", "Ks.npy", "K.npy"],
        }
        for key, names in name_groups.items():
            for file_name in names:
                candidate = path / file_name
                if not candidate.exists():
                    continue
                payload[key] = self._load_control_payload_from_file(candidate).get(key)
                break
        return payload

    def _load_control_payload_from_file(self, path: Path) -> dict[str, Any]:
        """Parse one control file and map it onto the normalized payload schema."""
        suffix = path.suffix.lower()
        stem = path.stem.lower()
        if suffix == ".npz":
            data = dict(np.load(path, allow_pickle=True))
            return self._normalize_payload_keys(data)
        if suffix == ".json":
            with path.open("r") as f:
                data = json.load(f)
            return self._normalize_payload_keys(data)
        if suffix == ".npy":
            array = np.load(path, allow_pickle=True)
        elif suffix in {".pt", ".pth"}:
            array = torch.load(path, map_location="cpu")
            if isinstance(array, dict):
                return self._normalize_payload_keys(array)
        else:
            raise ValueError(f"unsupported action_path format: {path}")

        if "keyboard" in stem:
            return {"keyboard_cond": array}
        if "mouse" in stem:
            return {"mouse_cond": array}
        if "intrinsic" in stem or stem in {"k", "ks"}:
            return {"intrinsics": array}
        if "pose" in stem or "c2w" in stem:
            return {"poses": array}
        raise ValueError(f"unsupported action_path file name: {path}")

    def _normalize_payload_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """Collapse different naming conventions into the runner's canonical keys."""
        payload: dict[str, Any] = {}
        key_aliases = {
            "keyboard_cond": {"keyboard_cond", "keyboard_condition"},
            "mouse_cond": {"mouse_cond", "mouse_condition"},
            "poses": {"poses", "pose", "c2ws", "c2w", "extrinsics"},
            "intrinsics": {"intrinsics", "k", "ks"},
        }
        for target_key, aliases in key_aliases.items():
            for key, value in data.items():
                if str(key).lower() in aliases:
                    payload[target_key] = value
                    break
        return payload

    def _default_intrinsics(self) -> torch.Tensor:
        """Generate the default camera intrinsics for the current output resolution."""
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        return modules["cam_utils"].get_intrinsics(self._mg3_target_h, self._mg3_target_w)

    def _to_tensor(self, value: Any, dtype=torch.float32) -> Optional[torch.Tensor]:
        """Convert numpy/list/scalar inputs into CPU tensors for normalization."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        if isinstance(value, list):
            return torch.tensor(value, dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _resize_time_axis(self, tensor: torch.Tensor, total_frames: int) -> torch.Tensor:
        # MG3 expects exact per-frame control lengths. To make the runner tolerant of
        # slightly malformed inputs, short sequences are padded by repeating the last
        # value and long sequences are truncated.
        if tensor.shape[0] == total_frames:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(total_frames, *([1] * (tensor.ndim - 1)))
        if tensor.shape[0] < total_frames:
            pad = tensor[-1:].repeat(total_frames - tensor.shape[0], *([1] * (tensor.ndim - 1)))
            logger.warning(
                "[matrix-game-3] control length {} shorter than expected {}, padding with the last value.",
                tensor.shape[0],
                total_frames,
            )
            return torch.cat([tensor, pad], dim=0)
        logger.warning(
            "[matrix-game-3] control length {} longer than expected {}, truncating the tail.",
            tensor.shape[0],
            total_frames,
        )
        return tensor[:total_frames]

    def _normalize_keyboard_cond(self, value: Any, total_frames: int) -> torch.Tensor:
        """Normalize keyboard controls into `[1, T, keyboard_dim_in]`."""
        if value is None:
            return torch.zeros((1, total_frames, self.keyboard_dim_in), dtype=torch.float32)
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.keyboard_dim_in:
            raise ValueError(f"keyboard_cond shape mismatch, expected [T,{self.keyboard_dim_in}], got {tuple(tensor.shape)}")
        tensor = self._resize_time_axis(tensor, total_frames)
        return tensor.unsqueeze(0)

    def _normalize_mouse_cond(self, value: Any, total_frames: int) -> torch.Tensor:
        """Normalize mouse controls into `[1, T, mouse_dim_in]`."""
        if value is None:
            return torch.zeros((1, total_frames, self.mouse_dim_in), dtype=torch.float32)
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.mouse_dim_in:
            raise ValueError(f"mouse_cond shape mismatch, expected [T,{self.mouse_dim_in}], got {tuple(tensor.shape)}")
        tensor = self._resize_time_axis(tensor, total_frames)
        return tensor.unsqueeze(0)

    def _normalize_intrinsics(self, value: Any, total_frames: int) -> Optional[torch.Tensor]:
        """Accept either flattened `[fx, fy, cx, cy]` or 3x3 intrinsics matrices."""
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if tensor.ndim == 1:
            if tensor.shape[0] == 4:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == 9:
                tensor = tensor.view(3, 3).unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[-2:] == (3, 3):
            tensor = torch.stack([tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 0, 2], tensor[..., 1, 2]], dim=-1)
        if tensor.ndim != 2 or tensor.shape[-1] != 4:
            raise ValueError(f"intrinsics shape mismatch, expected [T,4] or [T,3,3], got {tuple(tensor.shape)}")
        return self._resize_time_axis(tensor, total_frames)

    def _normalize_poses(self, value: Any, total_frames: int) -> Optional[torch.Tensor]:
        """Normalize poses into `[T, 4, 4]` camera-to-world extrinsics."""
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if tensor.ndim == 2 and tensor.shape[-1] == 5:
            # The official action pipeline also uses a compact 5D pose
            # `[x, y, z, pitch, yaw]`. Convert it here to full extrinsics.
            modules = self._get_official_modules()
            rotations = np.concatenate([np.zeros((tensor.shape[0], 1), dtype=np.float32), tensor[:, 3:5].numpy()], axis=1).tolist()
            positions = tensor[:, :3].numpy().tolist()
            tensor = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[-2:] == (4, 4):
            tensor = self._resize_time_axis(tensor, total_frames)
            return tensor
        raise ValueError(f"poses shape mismatch, expected [T,4,4] or [T,5], got {tuple(tensor.shape)}")

    def _build_noninteractive_controls(self, payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Official source:
        # - utils/conditions.py defines keyboard_dim_in=6 and mouse_dim_in=2 semantics
        # - utils/utils.py computes poses from actions when explicit poses are absent
        #
        # Offline mode materializes the whole control trajectory up front so later
        # segments only need cheap slicing instead of re-reading user inputs.
        total_frames = self._mg3_expected_total_frames
        keyboard_cond = self._normalize_keyboard_cond(payload.get("keyboard_cond"), total_frames)
        mouse_cond = self._normalize_mouse_cond(payload.get("mouse_cond"), total_frames)
        intrinsics_all = self._normalize_intrinsics(payload.get("intrinsics"), total_frames)

        poses = self._normalize_poses(payload.get("poses"), total_frames)
        if poses is None:
            modules = self._get_official_modules()
            if not payload:
                # No action file at all: keep the camera fixed at identity.
                identity_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(total_frames, 1, 1)
                poses = identity_pose
            else:
                # Action file exists but explicit poses do not: reconstruct camera motion
                # with the official action-to-pose integrator.
                first_pose = np.zeros(5, dtype=np.float32)
                all_poses = modules["utils"].compute_all_poses_from_actions(
                    keyboard_cond.squeeze(0).cpu(),
                    mouse_cond.squeeze(0).cpu(),
                    first_pose=first_pose,
                )
                positions = all_poses[:, :3].tolist()
                rotations = np.concatenate([np.zeros((all_poses.shape[0], 1), dtype=np.float32), all_poses[:, 3:5]], axis=1).tolist()
                poses = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
        return keyboard_cond, mouse_cond, poses, intrinsics_all

    def get_video_segment_num(self):
        self.video_segment_num = self._mg3_num_iterations

    def init_run(self):
        # This mostly mirrors `DefaultRunner.init_run()`, but we immediately override
        # the scheduler state with the first segment's custom latent/mask setup.
        self.gen_video_final = None
        self.get_video_segment_num()
        self._mg3_noise_generator = torch.Generator(device=AI_DEVICE).manual_seed(self.input_info.seed)
        self._mg3_generated_latent_history = []
        self._mg3_tail_latents = None
        self._mg3_current_segment_full_latents = None
        self._mg3_current_segment_state = None

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        self.model.scheduler.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape, image_encoder_output=self.inputs["image_encoder_output"])
        self._apply_segment_scheduler_state(self._build_or_get_segment_state(0))
        self.inputs["image_encoder_output"]["vae_encoder_out"] = None

    def _append_interactive_segment_controls(self, segment_idx: int):
        """Collect one segment worth of controls from stdin in interactive mode."""
        modules = self._get_official_modules()
        first_clip = segment_idx == 0
        action_frames = self.first_clip_frame if first_clip else self.incremental_segment_frames

        if not dist.is_initialized() or dist.get_rank() == 0:
            actions = self._prompt_current_action()
            # The prompt returns one action token; MG3 applies it uniformly across the
            # newly generated frame span for that segment.
            keyboard_curr = actions["keyboard"].repeat(action_frames, 1)
            mouse_curr = actions["mouse"].repeat(action_frames, 1)
            if first_clip:
                first_pose = np.zeros(5, dtype=np.float32)
            else:
                first_pose = self._mg3_last_pose
            all_poses, last_pose = modules["utils"].compute_all_poses_from_actions(
                keyboard_curr.cpu(),
                mouse_curr.cpu(),
                first_pose=first_pose,
                return_last_pose=True,
            )
            positions = all_poses[:, :3].tolist()
            rotations = np.concatenate([np.zeros((all_poses.shape[0], 1), dtype=np.float32), all_poses[:, 3:5]], axis=1).tolist()
            extrinsics_curr = modules["cam_utils"].get_extrinsics(rotations, positions).to(dtype=torch.float32)
            payload = [
                keyboard_curr.numpy(),
                mouse_curr.numpy(),
                extrinsics_curr.numpy(),
                last_pose.astype(np.float32),
            ]
        else:
            payload = [None, None, None, None]

        if dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)

        keyboard_curr = torch.from_numpy(payload[0]).to(dtype=torch.float32).unsqueeze(0)
        mouse_curr = torch.from_numpy(payload[1]).to(dtype=torch.float32).unsqueeze(0)
        extrinsics_curr = torch.from_numpy(payload[2]).to(dtype=torch.float32)
        self._mg3_last_pose = np.array(payload[3], dtype=np.float32)

        if self._mg3_keyboard_all is None:
            self._mg3_keyboard_all = keyboard_curr
            self._mg3_mouse_all = mouse_curr
            self._mg3_extrinsics_all = extrinsics_curr
        else:
            # Interactive mode grows the global control timeline as segments progress.
            self._mg3_keyboard_all = torch.cat([self._mg3_keyboard_all, keyboard_curr], dim=1)
            self._mg3_mouse_all = torch.cat([self._mg3_mouse_all, mouse_curr], dim=1)
            self._mg3_extrinsics_all = torch.cat([self._mg3_extrinsics_all, extrinsics_curr], dim=0)

    def _prompt_current_action(self) -> dict[str, torch.Tensor]:
        """Minimal CLI UX for interactive MG3 generation."""
        cam_value = 0.1
        print()
        print("-" * 30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM")
        print("(I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT")
        print("(W: forward, S: back, A: left, D: right, Q: no move)")
        print("-" * 30)

        camera_value_map = {
            "i": [cam_value, 0.0],
            "k": [-cam_value, 0.0],
            "j": [0.0, -cam_value],
            "l": [0.0, cam_value],
            "u": [0.0, 0.0],
        }
        keyboard_idx = {
            "w": [1, 0, 0, 0, 0, 0],
            "s": [0, 1, 0, 0, 0, 0],
            "a": [0, 0, 1, 0, 0, 0],
            "d": [0, 0, 0, 1, 0, 0],
            "q": [0, 0, 0, 0, 0, 0],
        }
        while True:
            idx_mouse = input("Please input the mouse action (e.g. `U`):\n").strip().lower()
            idx_keyboard = input("Please input the keyboard action (e.g. `W`):\n").strip().lower()
            if idx_mouse in camera_value_map and idx_keyboard in keyboard_idx:
                return {
                    "mouse": torch.tensor(camera_value_map[idx_mouse], dtype=torch.float32),
                    "keyboard": torch.tensor(keyboard_idx[idx_keyboard], dtype=torch.float32),
                }

    def _interpolate_intrinsics(self, intrinsics_seq: Optional[torch.Tensor], src_indices: np.ndarray, tgt_indices: np.ndarray) -> torch.Tensor:
        """Interpolate intrinsics onto the latent timeline used by the DiT."""
        assert self._mg3_base_intrinsics is not None
        if intrinsics_seq is None:
            return self._mg3_base_intrinsics.to(dtype=torch.float32).repeat(len(tgt_indices), 1)

        intrinsics_seq = intrinsics_seq.to(dtype=torch.float32)
        if intrinsics_seq.shape[0] == 1:
            return intrinsics_seq.repeat(len(tgt_indices), 1)

        src_indices = np.asarray(src_indices, dtype=np.float32)
        tgt_indices = np.asarray(tgt_indices, dtype=np.float32)
        src_indices = np.clip(np.round(src_indices).astype(np.int64), 0, intrinsics_seq.shape[0] - 1)
        src_intrinsics = intrinsics_seq[src_indices]
        out = []
        for column_idx in range(src_intrinsics.shape[-1]):
            column = np.interp(tgt_indices, src_indices.astype(np.float32), src_intrinsics[:, column_idx].cpu().numpy())
            out.append(torch.from_numpy(column).to(dtype=torch.float32))
        return torch.stack(out, dim=-1)

    def _build_plucker_from_c2ws(
        self,
        c2ws_seq: torch.Tensor,
        src_indices: np.ndarray,
        tgt_indices: np.ndarray,
        framewise: bool,
        intrinsics_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Official source:
        # - utils/cam_utils.py: interpolate poses, compute relative poses, build plucker rays
        # - utils/utils.py: build_plucker_from_c2ws reshaping convention
        #
        # The model consumes camera control as plucker ray embeddings aligned to latent
        # time and latent spatial resolution, not as raw pose matrices.
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        assert self._mg3_lat_h is not None and self._mg3_lat_w is not None
        c2ws_np = c2ws_seq.cpu().numpy()
        c2ws_infer = modules["cam_utils"]._interpolate_camera_poses_handedness(
            src_indices=src_indices,
            src_rot_mat=c2ws_np[:, :3, :3],
            src_trans_vec=c2ws_np[:, :3, 3],
            tgt_indices=tgt_indices,
        ).to(device=c2ws_seq.device)
        # `framewise=True` means each timestep is represented relative to its own local
        # frame history, which matches the official per-segment conditioning path.
        c2ws_infer = modules["cam_utils"].compute_relative_poses(c2ws_infer, framewise=framewise)
        Ks = self._interpolate_intrinsics(intrinsics_seq, src_indices, tgt_indices).to(device=c2ws_infer.device, dtype=c2ws_infer.dtype)
        plucker = modules["cam_utils"].get_plucker_embeddings(c2ws_infer, Ks, self._mg3_target_h, self._mg3_target_w)
        c1 = self._mg3_target_h // self._mg3_lat_h
        c2 = self._mg3_target_w // self._mg3_lat_w
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=c1,
            c2=c2,
        )
        plucker = plucker[None, ...]
        return rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=len(tgt_indices),
            h=self._mg3_lat_h,
            w=self._mg3_lat_w,
        )

    def _build_plucker_from_pose(self, c2ws_pose: torch.Tensor, intrinsics_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build plucker embeddings when poses are already on the target timeline."""
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        assert self._mg3_lat_h is not None and self._mg3_lat_w is not None
        if intrinsics_seq is None:
            Ks = self._mg3_base_intrinsics.to(device=c2ws_pose.device, dtype=c2ws_pose.dtype).repeat(c2ws_pose.shape[0], 1)
        else:
            Ks = intrinsics_seq.to(device=c2ws_pose.device, dtype=c2ws_pose.dtype)
        plucker = modules["cam_utils"].get_plucker_embeddings(c2ws_pose, Ks, self._mg3_target_h, self._mg3_target_w)
        c1 = self._mg3_target_h // self._mg3_lat_h
        c2 = self._mg3_target_w // self._mg3_lat_w
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=c1,
            c2=c2,
        )
        plucker = plucker[None, ...]
        return rearrange(
            plucker,
            "b (f h w) c -> b c f h w",
            f=c2ws_pose.shape[0],
            h=self._mg3_lat_h,
            w=self._mg3_lat_w,
        )

    def _build_memory_metadata(self, segment_idx: int, current_start_frame_idx: int, current_end_frame_idx: int) -> dict[str, Any]:
        # Official source: pipeline/inference_pipeline.py and utils/cam_utils.py.
        # Current downstream model code only requires c2ws_plucker_emb / keyboard_cond / mouse_cond,
        # but we still stage the memory-facing metadata here so the runner owns segment bookkeeping.
        #
        # Later segments can attend to a sparse set of previously generated latent
        # frames. This method selects those frames, prepares their latent indices, and
        # builds the matching plucker embeddings for the memory branch.
        if segment_idx == 0 or not self._mg3_generated_latent_history:
            return {
                "x_memory": None,
                "timestep_memory": None,
                "keyboard_cond_memory": None,
                "mouse_cond_memory": None,
                "memory_latent_idx": None,
                "plucker_emb_with_memory": None,
            }

        modules = self._get_official_modules()
        assert self._mg3_extrinsics_all is not None
        assert self._mg3_base_intrinsics is not None

        def align_frame_to_block(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

        def get_latent_idx(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 + 1 if frame_idx > 0 else 0

        selected_index_base = [current_end_frame_idx - offset for offset in range(1, 34, 8)]
        selected_index = modules["cam_utils"].select_memory_idx_fov(
            self._mg3_extrinsics_all,
            current_start_frame_idx,
            selected_index_base,
            use_gpu=torch.cuda.is_available(),
        )
        if selected_index:
            # The official code hard-pins the oldest memory anchor to frame 4.
            selected_index[-1] = 4

        memory_pluckers = []
        latent_idx = []
        for mem_idx, reference_idx in zip(selected_index, selected_index_base):
            latent_idx.append(get_latent_idx(mem_idx))
            mem_idx_aligned = align_frame_to_block(mem_idx)
            mem_block = self._mg3_extrinsics_all[mem_idx_aligned : mem_idx_aligned + 4]
            mem_src = np.linspace(mem_idx_aligned, mem_idx_aligned + 3, mem_block.shape[0])
            mem_tgt = np.array([mem_idx_aligned + 3], dtype=np.float32)
            mem_pose = modules["cam_utils"]._interpolate_camera_poses_handedness(
                src_indices=mem_src,
                src_rot_mat=mem_block[:, :3, :3].cpu().numpy(),
                src_trans_vec=mem_block[:, :3, 3].cpu().numpy(),
                tgt_indices=mem_tgt,
            )
            reference_pose = self._mg3_extrinsics_all[reference_idx : reference_idx + 1]
            rel_pair = torch.cat([reference_pose, mem_pose], dim=0)
            rel_pose = modules["cam_utils"].compute_relative_poses(rel_pair, framewise=False)[1:2]
            memory_pluckers.append(self._build_plucker_from_pose(rel_pose.to(device=AI_DEVICE)))

        current_plucker = self._build_or_get_segment_camera_only(segment_idx)
        plucker_with_memory = torch.cat(memory_pluckers + [current_plucker], dim=2) if memory_pluckers else current_plucker
        src = torch.cat(self._mg3_generated_latent_history, dim=1)
        valid_latent_idx = [idx for idx in latent_idx if 0 <= idx < src.shape[1]]
        if valid_latent_idx != latent_idx:
            logger.warning(
                "[matrix-game-3] memory latent index truncated from {} to {} because generated latent history is shorter.",
                latent_idx,
                valid_latent_idx,
            )
        x_memory = src[:, valid_latent_idx].unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE()) if valid_latent_idx else None
        if x_memory is None:
            timestep_memory = None
            keyboard_cond_memory = None
            mouse_cond_memory = None
        else:
            timestep_memory = x_memory.new_zeros((1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))
            keyboard_cond_memory = -torch.ones((1, len(valid_latent_idx), self.keyboard_dim_in), device=x_memory.device, dtype=x_memory.dtype)
            mouse_cond_memory = torch.ones((1, len(valid_latent_idx), self.mouse_dim_in), device=x_memory.device, dtype=x_memory.dtype)

        return {
            "x_memory": x_memory,
            "timestep_memory": timestep_memory,
            "keyboard_cond_memory": keyboard_cond_memory,
            "mouse_cond_memory": mouse_cond_memory,
            "memory_latent_idx": valid_latent_idx,
            "plucker_emb_with_memory": plucker_with_memory,
        }

    def _build_or_get_segment_camera_only(self, segment_idx: int) -> torch.Tensor:
        """Access just the camera plucker embedding without rebuilding other state."""
        state = self._segment_states.get(segment_idx)
        if state is not None and "c2ws_plucker_emb" in state.dit_cond_dict:
            return state.dit_cond_dict["c2ws_plucker_emb"]
        state = self._build_or_get_segment_state(segment_idx)
        return state.dit_cond_dict["c2ws_plucker_emb"]

    def _build_or_get_segment_state(self, segment_idx: int) -> MatrixGame3SegmentState:
        """Materialize one segment's complete conditioning package.

        This is the core of the adapter. It decides:
        - which absolute frames this segment covers;
        - which latent timesteps are fixed from prior context;
        - which camera/action conditions should be sliced for this window;
        - which overlap should be trimmed after decoding.
        """
        if segment_idx in self._segment_states:
            return self._segment_states[segment_idx]

        if self._mg3_interactive and (self._mg3_keyboard_all is None or self._mg3_keyboard_all.shape[1] < self.first_clip_frame + segment_idx * self.incremental_segment_frames):
            self._append_interactive_segment_controls(segment_idx)

        assert self._mg3_keyboard_all is not None
        assert self._mg3_mouse_all is not None
        assert self._mg3_extrinsics_all is not None
        first_clip = segment_idx == 0

        def get_latent_idx(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 + 1 if frame_idx > 0 else 0

        current_end_frame_idx = self.first_clip_frame if first_clip else self.first_clip_frame + segment_idx * self.incremental_segment_frames
        current_start_frame_idx = 0 if first_clip else current_end_frame_idx - self.clip_frame
        frame_count = self.first_clip_frame if first_clip else self.clip_frame
        latent_start_idx = get_latent_idx(current_start_frame_idx)
        latent_end_idx = get_latent_idx(current_end_frame_idx)
        fixed_latent_frames = 1 if first_clip else self.conditioning_latent_frames
        # After decoding, the first RGB frames of every later segment correspond to
        # history that was already emitted by the previous segment, so they are dropped.
        decode_trim_frames = 0 if first_clip else 1 + self.config["vae_stride"][0] * (fixed_latent_frames - 1)
        append_latent_start = 0 if first_clip else fixed_latent_frames

        c2ws_chunk = self._mg3_extrinsics_all[current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE)
        src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1, frame_count)

        intrinsics_chunk = None
        if self._mg3_intrinsics_all is not None:
            intrinsics_chunk = self._mg3_intrinsics_all[current_start_frame_idx:current_end_frame_idx]

        latent_shape = self._segment_latent_shape(self._mg3_lat_h, self._mg3_lat_w, frame_count)
        # The latent timeline is coarser than RGB time because Wan2.2 uses a temporal
        # VAE stride of 4. Later segments start interpolation at `start + 3` so the
        # first 4 latent slots line up with the carried-over conditioning tail.
        tgt_indices = np.linspace(0 if first_clip else current_start_frame_idx + 3, current_end_frame_idx - 1, latent_shape[1])

        camera_only = self._build_plucker_from_c2ws(
            c2ws_chunk,
            src_indices=src_indices,
            tgt_indices=tgt_indices,
            framewise=True,
            intrinsics_seq=intrinsics_chunk,
        ).to(device=AI_DEVICE, dtype=GET_DTYPE())

        keyboard_cond = self._mg3_keyboard_all[:, current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE, dtype=GET_DTYPE())
        mouse_cond = self._mg3_mouse_all[:, current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE, dtype=GET_DTYPE())

        vae_encoder_out = torch.zeros(latent_shape, device=AI_DEVICE, dtype=GET_DTYPE())
        if first_clip:
            # Segment 0 is anchored by the input image latent in the first temporal slot.
            vae_encoder_out[:, :1] = self.inputs["image_encoder_output"]["vae_encoder_out"][:, :1]
        else:
            if self._mg3_tail_latents is None:
                raise RuntimeError("matrix-game-3 segment requested without previous tail latents")
            # Later segments are conditioned on the last 4 latent frames produced by the
            # previous segment, which creates temporal continuity across chunk boundaries.
            vae_encoder_out[:, : self.conditioning_latent_frames] = self._mg3_tail_latents.to(device=AI_DEVICE, dtype=GET_DTYPE())

        # Fields below intentionally stay in the standard LightX2V image_encoder_output["dit_cond_dict"]
        # container so downstream model / infer / weights code can consume them without a new top-level protocol.
        dit_cond_dict: dict[str, Any] = {
            "keyboard_cond": keyboard_cond,
            "mouse_cond": mouse_cond,
            "c2ws_plucker_emb": camera_only,
            "predict_latent_idx": (latent_start_idx, latent_end_idx),
            "segment_frame_range": (current_start_frame_idx, current_end_frame_idx),
            "segment_idx": segment_idx,
            "first_clip": first_clip,
        }
        dit_cond_dict.update(self._build_memory_metadata(segment_idx, current_start_frame_idx, current_end_frame_idx))

        state = MatrixGame3SegmentState(
            segment_idx=segment_idx,
            first_clip=first_clip,
            current_start_frame_idx=current_start_frame_idx,
            current_end_frame_idx=current_end_frame_idx,
            frame_count=frame_count,
            fixed_latent_frames=fixed_latent_frames,
            latent_shape=latent_shape,
            decode_trim_frames=decode_trim_frames,
            append_latent_start=append_latent_start,
            keyboard_cond=keyboard_cond,
            mouse_cond=mouse_cond,
            vae_encoder_out=vae_encoder_out,
            dit_cond_dict=dit_cond_dict,
        )
        self._segment_states[segment_idx] = state
        return state

    def _apply_segment_scheduler_state(self, segment_state: MatrixGame3SegmentState):
        """Seed the scheduler latents and mask for the current segment window."""
        scheduler = self.model.scheduler
        latents = torch.randn(
            tuple(segment_state.latent_shape),
            device=AI_DEVICE,
            dtype=torch.float32,
            generator=self._mg3_noise_generator,
        )
        scheduler.vae_encoder_out = segment_state.vae_encoder_out.to(device=AI_DEVICE, dtype=torch.float32)
        scheduler.mask = torch.ones_like(latents)
        # Mask value 0 means "keep the provided latent conditioning", while 1 means
        # "sample this slot from noise through the diffusion process".
        scheduler.mask[:, : segment_state.fixed_latent_frames] = 0
        scheduler.latents = (1.0 - scheduler.mask) * scheduler.vae_encoder_out + scheduler.mask * latents

    @ProfilingContext4DebugL1(
        "Init run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_init_run_segment_duration,
        metrics_labels=["WanMatrixGame3Runner"],
    )
    def init_run_segment(self, segment_idx):
        # Official source: pipeline/inference_pipeline.py and inference_interactive_pipeline.py
        # refresh per-segment action / camera / latent-conditioning state here so the outer lifecycle
        # remains the standard LightX2V segment loop.
        #
        # The base runner calls this before every segment. We use that hook to swap in
        # the next segment's control tensors and, for later segments, reset the scheduler
        # so it samples against the new latent shape and conditioning mask.
        self.segment_idx = segment_idx
        segment_state = self._build_or_get_segment_state(segment_idx)
        self._mg3_current_segment_state = segment_state
        self.input_info.latent_shape = segment_state.latent_shape
        self.inputs["image_encoder_output"]["dit_cond_dict"] = segment_state.dit_cond_dict
        self.inputs["image_encoder_output"]["vae_encoder_out"] = segment_state.vae_encoder_out
        if segment_idx > 0:
            self.model.scheduler.reset(self.input_info.seed, segment_state.latent_shape)
            self._apply_segment_scheduler_state(segment_state)

    def run_segment(self, segment_idx=0):
        # Save the raw latent output before the VAE decoder trims or converts anything;
        # the next segment needs these latents for temporal conditioning and memory.
        latents = super().run_segment(segment_idx)
        self._mg3_current_segment_full_latents = latents.detach().clone()
        return latents

    def end_run_segment(self, segment_idx=None):
        """Carry segment outputs forward and remove overlap from decoded frames."""
        if self._mg3_current_segment_state is None or self._mg3_current_segment_full_latents is None:
            raise RuntimeError("matrix-game-3 end_run_segment called before the current segment state was prepared")

        full_latents = self._mg3_current_segment_full_latents
        # full_latents follows Wan2.2 runner convention: [C, T, H, W].
        # Keep only the tail that should condition the next segment.
        self._mg3_tail_latents = full_latents[:, -self.conditioning_latent_frames :].detach().clone()
        # Only append genuinely new latent timesteps to history; the carried-over prefix
        # belongs to the previous segment and would otherwise duplicate memory entries.
        new_latents = full_latents[:, self._mg3_current_segment_state.append_latent_start :].detach().clone()
        self._mg3_generated_latent_history.append(new_latents)

        segment_video = self.gen_video
        if self._mg3_current_segment_state.decode_trim_frames > 0:
            # Remove RGB frames that correspond to the reused latent prefix.
            segment_video = segment_video[:, :, self._mg3_current_segment_state.decode_trim_frames :]
        self.gen_video = segment_video
        self.gen_video_final = segment_video if self.gen_video_final is None else torch.cat([self.gen_video_final, segment_video], dim=2)
        self._mg3_current_segment_state = None
        self._mg3_current_segment_full_latents = None

    def process_images_after_vae_decoder(self):
        # `DefaultRunner.process_images_after_vae_decoder()` expects `gen_video_final`
        # to already contain the full stitched clip.
        if self.gen_video_final is None:
            self.gen_video_final = self.gen_video
        return super().process_images_after_vae_decoder()
