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
from lightx2v.utils.envs import GET_DTYPE, torch_device_module
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import best_output_size
from lightx2v_platform.base.global_var import AI_DEVICE


DEFAULT_MATRIX_GAME3_OFFICIAL_ROOT = Path("/home/michael/Project/LightX2V/Matrix-Game-3/Matrix-Game-3")
DEFAULT_MATRIX_GAME3_BASE_CONFIG = Path("/home/michael/Project/LightX2V/Matrix-Game-3.0/base_model/config.json")
DEFAULT_MATRIX_GAME3_DISTILLED_CONFIG = Path("/home/michael/Project/LightX2V/Matrix-Game-3.0/base_distilled_model/config.json")
_MATRIX_GAME3_OFFICIAL_PACKAGE = "_lightx2v_matrix_game3_official"


@dataclass
class MatrixGame3SegmentState:
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
    if package_name in sys.modules:
        return sys.modules[package_name]
    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module
    return module


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
    """

    def __init__(self, config):
        with config.temporarily_unlocked():
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
        super().__init__(config)

        self.matrix_game3_model_cls = original_model_cls
        self.first_clip_frame = 57
        self.clip_frame = 56
        self.incremental_segment_frames = 40
        self.past_frame = 16
        self.conditioning_latent_frames = 4
        self.mouse_dim_in = 2
        self.keyboard_dim_in = 6
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
        self._load_matrix_game3_model_config()

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        if "action_path" in self.input_info.__dataclass_fields__:
            self.input_info.action_path = inputs.get("action_path", inputs.get("pose", ""))
        if "pose" in self.input_info.__dataclass_fields__:
            self.input_info.pose = inputs.get("pose", inputs.get("action_path", ""))

    def load_transformer(self):
        from lightx2v.models.networks.wan.matrix_game3_model import WanMtxg3Model

        model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            return WanMtxg3Model(**model_kwargs)
        return build_wan_model_with_lora(WanMtxg3Model, self.config, model_kwargs, lora_configs, model_type="wan2.2")

    def _load_matrix_game3_model_config(self):
        config_path = Path(self.config["model_path"]) / self.config["sub_model_folder"] / "config.json"
        if not config_path.exists():
            config_path = DEFAULT_MATRIX_GAME3_BASE_CONFIG if self.config["use_base_model"] else DEFAULT_MATRIX_GAME3_DISTILLED_CONFIG
        if not config_path.exists():
            logger.warning("matrix-game-3 config.json not found at {}", config_path)
            return

        with config_path.open("r") as f:
            model_config = json.load(f)

        with self.config.temporarily_unlocked():
            self.config.update(model_config)
            self.config["num_channels_latents"] = int(model_config.get("in_dim", self.config.get("num_channels_latents", 48)))
            self.config["vae_stride"] = tuple(self.config.get("vae_stride", (4, 16, 16)))
            self.config["patch_size"] = tuple(model_config.get("patch_size", self.config.get("patch_size", (1, 2, 2))))

        action_config = self.config.get("action_config", {})
        self.keyboard_dim_in = int(action_config.get("keyboard_dim_in", 6))
        self.mouse_dim_in = int(action_config.get("mouse_dim_in", 2))

    def _get_official_modules(self) -> dict[str, Any]:
        if self._official_modules is not None:
            return self._official_modules

        official_root = Path(self.config.get("matrix_game3_official_root", DEFAULT_MATRIX_GAME3_OFFICIAL_ROOT))
        if not official_root.exists():
            raise FileNotFoundError(f"Matrix-Game-3 official root not found: {official_root}")

        _ensure_namespace_package(_MATRIX_GAME3_OFFICIAL_PACKAGE, official_root)
        utils_pkg = f"{_MATRIX_GAME3_OFFICIAL_PACKAGE}.utils"
        _ensure_namespace_package(utils_pkg, official_root / "utils")

        modules = {
            "conditions": _load_module_from_path(f"{utils_pkg}.conditions", official_root / "utils" / "conditions.py"),
            "cam_utils": _load_module_from_path(f"{utils_pkg}.cam_utils", official_root / "utils" / "cam_utils.py"),
            "transform": _load_module_from_path(f"{utils_pkg}.transform", official_root / "utils" / "transform.py"),
            "utils": _load_module_from_path(f"{utils_pkg}.utils", official_root / "utils" / "utils.py"),
        }
        self._official_modules = modules
        return modules

    def _get_expected_total_frames(self, raw_total_frames: Optional[int] = None) -> tuple[int, int]:
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
        _, img_ori = self.read_image_input(self.input_info.image_path)
        vae_encoder_out, latent_shape = self.run_vae_encoder(img_ori)
        self.input_info.latent_shape = latent_shape
        text_encoder_output = self.run_text_encoder(self.input_info)
        self._prepare_matrix_game3_session(img_ori, latent_shape, vae_encoder_out)
        torch_device_module.empty_cache()
        return self.get_encoder_output_i2v(None, vae_encoder_out, text_encoder_output)

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
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
        modules = self._get_official_modules()
        assert self._mg3_target_h is not None and self._mg3_target_w is not None
        return modules["cam_utils"].get_intrinsics(self._mg3_target_h, self._mg3_target_w)

    def _to_tensor(self, value: Any, dtype=torch.float32) -> Optional[torch.Tensor]:
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
        if value is None:
            return None
        tensor = self._to_tensor(value)
        if tensor.ndim == 2 and tensor.shape[-1] == 5:
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
        total_frames = self._mg3_expected_total_frames
        keyboard_cond = self._normalize_keyboard_cond(payload.get("keyboard_cond"), total_frames)
        mouse_cond = self._normalize_mouse_cond(payload.get("mouse_cond"), total_frames)
        intrinsics_all = self._normalize_intrinsics(payload.get("intrinsics"), total_frames)

        poses = self._normalize_poses(payload.get("poses"), total_frames)
        if poses is None:
            modules = self._get_official_modules()
            if not payload:
                identity_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(total_frames, 1, 1)
                poses = identity_pose
            else:
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
        modules = self._get_official_modules()
        first_clip = segment_idx == 0
        action_frames = self.first_clip_frame if first_clip else self.incremental_segment_frames

        if not dist.is_initialized() or dist.get_rank() == 0:
            actions = self._prompt_current_action()
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
            self._mg3_keyboard_all = torch.cat([self._mg3_keyboard_all, keyboard_curr], dim=1)
            self._mg3_mouse_all = torch.cat([self._mg3_mouse_all, mouse_curr], dim=1)
            self._mg3_extrinsics_all = torch.cat([self._mg3_extrinsics_all, extrinsics_curr], dim=0)

    def _prompt_current_action(self) -> dict[str, torch.Tensor]:
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
        state = self._segment_states.get(segment_idx)
        if state is not None and "c2ws_plucker_emb" in state.dit_cond_dict:
            return state.dit_cond_dict["c2ws_plucker_emb"]
        state = self._build_or_get_segment_state(segment_idx)
        return state.dit_cond_dict["c2ws_plucker_emb"]

    def _build_or_get_segment_state(self, segment_idx: int) -> MatrixGame3SegmentState:
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
        decode_trim_frames = 0 if first_clip else 1 + self.config["vae_stride"][0] * (fixed_latent_frames - 1)
        append_latent_start = 0 if first_clip else fixed_latent_frames

        c2ws_chunk = self._mg3_extrinsics_all[current_start_frame_idx:current_end_frame_idx].to(device=AI_DEVICE)
        src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1, frame_count)

        intrinsics_chunk = None
        if self._mg3_intrinsics_all is not None:
            intrinsics_chunk = self._mg3_intrinsics_all[current_start_frame_idx:current_end_frame_idx]

        latent_shape = self._segment_latent_shape(self._mg3_lat_h, self._mg3_lat_w, frame_count)
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
            vae_encoder_out[:, :1] = self.inputs["image_encoder_output"]["vae_encoder_out"][:, :1]
        else:
            if self._mg3_tail_latents is None:
                raise RuntimeError("matrix-game-3 segment requested without previous tail latents")
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
        scheduler = self.model.scheduler
        latents = torch.randn(
            tuple(segment_state.latent_shape),
            device=AI_DEVICE,
            dtype=torch.float32,
            generator=self._mg3_noise_generator,
        )
        scheduler.vae_encoder_out = segment_state.vae_encoder_out.to(device=AI_DEVICE, dtype=torch.float32)
        scheduler.mask = torch.ones_like(latents)
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
        latents = super().run_segment(segment_idx)
        self._mg3_current_segment_full_latents = latents.detach().clone()
        return latents

    def end_run_segment(self, segment_idx=None):
        if self._mg3_current_segment_state is None or self._mg3_current_segment_full_latents is None:
            raise RuntimeError("matrix-game-3 end_run_segment called before the current segment state was prepared")

        full_latents = self._mg3_current_segment_full_latents
        # full_latents follows Wan2.2 runner convention: [C, T, H, W].
        self._mg3_tail_latents = full_latents[:, -self.conditioning_latent_frames :].detach().clone()
        new_latents = full_latents[:, self._mg3_current_segment_state.append_latent_start :].detach().clone()
        self._mg3_generated_latent_history.append(new_latents)

        segment_video = self.gen_video
        if self._mg3_current_segment_state.decode_trim_frames > 0:
            segment_video = segment_video[:, :, self._mg3_current_segment_state.decode_trim_frames :]
        self.gen_video = segment_video
        self.gen_video_final = segment_video if self.gen_video_final is None else torch.cat([self.gen_video_final, segment_video], dim=2)
        self._mg3_current_segment_state = None
        self._mg3_current_segment_full_latents = None

    def process_images_after_vae_decoder(self):
        if self.gen_video_final is None:
            self.gen_video_final = self.gen_video
        return super().process_images_after_vae_decoder()
