import gc
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from safetensors import safe_open

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.wan.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.dreamzero_model import DreamZeroModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.dreamzero.scheduler import DreamZeroFlowUniPCScheduler
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("dreamzero")
class WanDreamZeroRunner(WanRunner):
    def __init__(self, config):
        config["enable_cfg"] = config.get("enable_cfg", config.get("sample_guide_scale", 1.0) > 1)
        config["cfg_parallel"] = False
        super().__init__(config)
        self.vae_cls = WanVAE
        self.tiny_vae_cls = WanVAE_tiny
        self.vae_name = config.get("vae_name", "Wan2.1_VAE.pth")
        self.tiny_vae_name = "taew2_1.pth"
        self.cache_name = "pos"

    def init_scheduler(self):
        self.scheduler = DreamZeroFlowUniPCScheduler(self.config, shift_key="sample_shift", infer_steps_key="infer_steps")
        self.action_scheduler = DreamZeroFlowUniPCScheduler(self.config, shift_key="action_sample_shift", infer_steps_key="action_infer_steps")

    def load_transformer(self):
        return DreamZeroModel(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
            model_type="dreamzero",
        )

    def _load_dreamzero_component_state_dict(self, prefix, target_dtype=None):
        ckpt_path = self.config.get("dit_original_ckpt") or self.config["model_path"]
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            logger.warning("DreamZero checkpoint index not found at {}, skip {} component override", index_path, prefix)
            return {}

        with open(index_path, "r") as f:
            index = json.load(f)

        file_to_keys = {}
        for key, filename in index["weight_map"].items():
            if key.startswith(prefix):
                file_to_keys.setdefault(os.path.join(ckpt_path, filename), []).append(key)

        state_dict = {}
        for file_path, keys in sorted(file_to_keys.items()):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    if target_dtype is not None and tensor.dtype.is_floating_point:
                        tensor = tensor.to(target_dtype)
                    state_dict[key[len(prefix) :]] = tensor
        return state_dict

    def _overlay_dreamzero_component(self, module, prefix, component_name, strict=False):
        if not self.config.get("load_dreamzero_component_weights", True):
            return

        try:
            target_dtype = next(module.parameters()).dtype
        except StopIteration:
            target_dtype = None

        state_dict = self._load_dreamzero_component_state_dict(prefix, target_dtype=target_dtype)
        if not state_dict:
            logger.info("No DreamZero {} component weights found for prefix {}", component_name, prefix)
            return

        logger.info("Overlaying DreamZero {} component weights ({} tensors)", component_name, len(state_dict))
        load_info = module.load_state_dict(state_dict, strict=strict)
        missing = getattr(load_info, "missing_keys", [])
        unexpected = getattr(load_info, "unexpected_keys", [])
        if missing or unexpected:
            logger.warning(
                "DreamZero {} component load_state_dict: missing={}, unexpected={}",
                component_name,
                len(missing),
                len(unexpected),
            )
        del state_dict
        gc.collect()

    def load_image_encoder(self):
        if self.config["task"] != "i2va" or not self.config.get("use_image_encoder", True):
            return super().load_image_encoder()
        clip_offload = self.config.get("clip_cpu_offload", self.config.get("cpu_offload", False))
        clip_device = torch.device("cpu") if clip_offload else torch.device(AI_DEVICE)
        clip_model_name = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        clip_original_ckpt = find_torch_model_path(self.config, "clip_original_ckpt", clip_model_name)
        image_encoder = CLIPModel(
            dtype=torch.bfloat16,
            device=clip_device,
            checkpoint_path=clip_original_ckpt,
            clip_quantized=False,
            clip_quantized_ckpt=None,
            quant_scheme=None,
            cpu_offload=clip_offload,
            use_31_block=self.config.get("use_31_block", True),
            load_from_rank0=self.config.get("load_from_rank0", False),
            dummy_model=self.config.get("dummy_model", False),
        )
        if not self.config.get("dummy_model", False):
            self._overlay_dreamzero_component(image_encoder.model, "action_head.image_encoder.model.", "image_encoder", strict=False)
        return image_encoder

    def load_text_encoder(self):
        t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload"))
        t5_device = torch.device("cpu") if t5_offload else torch.device(AI_DEVICE)
        t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
        t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name)
        tokenizer_path = self.config.get("t5_tokenizer_path") or os.path.join(os.path.dirname(t5_original_ckpt), "google/umt5-xxl")

        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=t5_device,
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=t5_offload,
            t5_quantized=False,
            t5_quantized_ckpt=None,
            quant_scheme=None,
            load_from_rank0=self.config.get("load_from_rank0", False),
            lazy_load=self.config.get("t5_lazy_load", False),
            dummy_model=self.config.get("dummy_model", False),
        )
        return [text_encoder]

    def load_vae_encoder(self):
        if self.config["task"] not in ["i2va"]:
            return super().load_vae_encoder()
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
            "dummy_model": self.config.get("dummy_model", False),
            "dtype": GET_DTYPE() if not self.config.get("vae_dtype", None) else self.config["vae_dtype"],
        }
        vae_encoder = self.vae_cls(**vae_config)
        if not self.config.get("dummy_model", False):
            self._overlay_dreamzero_component(vae_encoder.model, "action_head.vae.model.", "vae", strict=True)
        return vae_encoder

    def load_vae_decoder(self):
        if self.config["task"] not in ["i2va"]:
            return super().load_vae_decoder()
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)
        if self.config.get("use_tae", False):
            tae_path = find_torch_model_path(self.config, "tae_path", self.tiny_vae_name)
            return self.tiny_vae_cls(vae_path=tae_path, device=self.init_device, need_scaled=self.config.get("need_scaled", False)).to(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "use_lightvae": self.config.get("use_lightvae", False),
            "dtype": GET_DTYPE() if not self.config.get("vae_dtype", None) else self.config["vae_dtype"],
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "dummy_model": self.config.get("dummy_model", False),
        }
        vae_decoder = self.vae_cls(**vae_config)
        if not self.config.get("dummy_model", False):
            self._overlay_dreamzero_component(vae_decoder.model, "action_head.vae.model.", "vae", strict=True)
        return vae_decoder

    def init_modules(self):
        super().init_modules()
        if self.config["task"] == "i2va":
            self.run_input_encoder = self._run_input_encoder_local_i2va

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2va(self):
        original_prompt = self.input_info.prompt
        original_negative_prompt = self.input_info.negative_prompt
        self.input_info.prompt = self._format_droid_prompt(original_prompt)
        if not original_negative_prompt:
            self.input_info.negative_prompt = self.config.get("negative_prompt", "")
        text_encoder_output = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt
        self.input_info.negative_prompt = original_negative_prompt
        torch_device_module.empty_cache()
        gc.collect()
        return {"text_encoder_output": text_encoder_output, "image_encoder_output": None}

    @staticmethod
    def _format_droid_prompt(prompt):
        prompt = (prompt or "").strip().lower()
        return (
            "A multi-view video shows that a robot "
            + prompt
            + " The video is split into three views: The top view shows the camera view from the robot's wrist, "
            + "the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view "
            + "shows the camera view from the right exterior camera. During training, one of the two bottom exterior "
            + "views may be a black screen (dropped view). The robot "
            + prompt
        )

    def _q99_normalize(self, values, stat, dim):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] < dim:
            values = np.pad(values, (0, dim - values.shape[0]))
        values = values[:dim]
        q01 = np.asarray(stat["q01"], dtype=np.float32).reshape(-1)[:dim]
        q99 = np.asarray(stat["q99"], dtype=np.float32).reshape(-1)[:dim]
        mask = q01 != q99
        normalized = np.zeros_like(values)
        normalized[mask] = 2.0 * (values[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        normalized[~mask] = values[~mask]
        return np.clip(normalized, -1.0, 1.0)

    def _q99_denormalize(self, values, stat, dim):
        values = np.asarray(values, dtype=np.float32)[..., :dim]
        q01 = np.asarray(stat["q01"], dtype=np.float32).reshape(1, 1, -1)[..., :dim]
        q99 = np.asarray(stat["q99"], dtype=np.float32).reshape(1, 1, -1)[..., :dim]
        return (values + 1.0) / 2.0 * (q99 - q01) + q01

    def _load_state_array(self):
        state_path = getattr(self.input_info, "state_path", "") or ""
        if not state_path:
            return np.zeros(8, dtype=np.float32)
        state_path = os.path.expanduser(str(state_path))
        if os.path.isdir(state_path):
            for name in ("state.json", "state.npy"):
                candidate = os.path.join(state_path, name)
                if os.path.exists(candidate):
                    state_path = candidate
                    break

        if state_path.endswith(".json"):
            with open(state_path, "r") as f:
                payload = json.load(f)
        elif state_path.endswith(".npy") or state_path.endswith(".npz"):
            payload = np.load(state_path, allow_pickle=True)
            if isinstance(payload, np.lib.npyio.NpzFile):
                payload = {key: payload[key] for key in payload.files}
            elif payload.shape == () and isinstance(payload.item(), dict):
                payload = payload.item()
        else:
            payload = np.loadtxt(state_path, delimiter=",", dtype=np.float32)

        if isinstance(payload, dict):
            joint = payload.get("state.joint_position", payload.get("observation/joint_position", payload.get("joint_position")))
            gripper = payload.get("state.gripper_position", payload.get("observation/gripper_position", payload.get("gripper_position")))
            if joint is None:
                joint = np.zeros(7, dtype=np.float32)
            if gripper is None:
                gripper = np.zeros(1, dtype=np.float32)
            return np.concatenate([np.asarray(joint).reshape(-1)[:7], np.asarray(gripper).reshape(-1)[:1]]).astype(np.float32)
        return np.asarray(payload, dtype=np.float32).reshape(-1)[:8]

    def _prepare_state(self):
        raw_state = self._load_state_array()
        if raw_state.shape[0] < 8:
            raw_state = np.pad(raw_state, (0, 8 - raw_state.shape[0]))
        raw_state = raw_state[:8]
        state_stat = self.config["state_norm_stat"]
        normalized = self._q99_normalize(raw_state, state_stat, 8)
        padded = np.zeros(self.config.get("max_state_dim", 64), dtype=np.float32)
        padded[:8] = normalized
        return raw_state.astype(np.float32), torch.from_numpy(padded).view(1, 1, -1).to(AI_DEVICE).to(GET_DTYPE())

    @staticmethod
    def _read_image(path):
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _read_video(path):
        reader = imageio.get_reader(path)
        try:
            frames = [np.asarray(frame[..., :3], dtype=np.uint8) for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"No frames loaded from {path}")
        return np.stack(frames, axis=0)

    def _resolve_camera_files(self, image_path):
        image_path = os.path.expanduser(str(image_path))
        cam_keys = self.config["obs_cam_keys"]
        aliases = self.config.get("obs_cam_aliases", {})
        if os.path.isdir(image_path):
            files = []
            for key in cam_keys:
                stems = [key, key.replace("video.", ""), aliases.get(key, "")]
                found = None
                for stem in stems:
                    if not stem:
                        continue
                    for ext in (".mp4", ".mov", ".avi", ".png", ".jpg", ".jpeg"):
                        candidate = os.path.join(image_path, f"{stem}{ext}")
                        if os.path.exists(candidate):
                            found = candidate
                            break
                    if found:
                        break
                if found is None:
                    raise FileNotFoundError(f"Could not find camera input for {key} under {image_path}")
                files.append(found)
            return files
        files = [item.strip() for item in image_path.split(",") if item.strip()]
        if len(files) != len(cam_keys):
            raise ValueError(f"Expected {len(cam_keys)} camera files, got {len(files)} from image_path={image_path}")
        return files

    def _load_camera_sequences(self):
        image_path = getattr(self.input_info, "image_path", "")
        if not image_path:
            raise ValueError("DreamZero i2va requires image_path as a directory or comma-separated camera files.")
        sequences = []
        for file_path in self._resolve_camera_files(image_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                seq = self._read_video(file_path)
            else:
                seq = self._read_image(file_path)[None]
            sequences.append(seq)
        min_len = min(seq.shape[0] for seq in sequences)
        if min_len <= 0:
            raise RuntimeError("DreamZero camera inputs are empty.")
        return [seq[:min_len] for seq in sequences]

    def _resize_frames(self, frames, height, width):
        crop_scale = float(self.config.get("view_crop_scale", 1.0))
        if crop_scale < 1.0:
            in_h, in_w = frames.shape[1:3]
            crop_h = max(1, int(in_h * crop_scale))
            crop_w = max(1, int(in_w * crop_scale))
            top = max(0, (in_h - crop_h) // 2)
            left = max(0, (in_w - crop_w) // 2)
            frames = frames[:, top : top + crop_h, left : left + crop_w]
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
        return (tensor * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    def _compose_droid_video(self, camera_sequences, frame_indices):
        view_height = int(self.config.get("view_height", self.config["target_height"] // 2))
        view_width = int(self.config.get("view_width", self.config["target_width"] // 2))
        selected = []
        for seq in camera_sequences:
            idx = np.clip(np.asarray(frame_indices, dtype=np.int64), 0, seq.shape[0] - 1)
            selected.append(self._resize_frames(seq[idx], view_height, view_width))

        left_exterior, right_exterior, wrist = selected
        canvas = np.zeros((len(frame_indices), view_height * 2, view_width * 2, 3), dtype=np.uint8)
        canvas[:, :view_height, :] = np.repeat(wrist, 2, axis=2)
        canvas[:, view_height:, :view_width] = left_exterior
        canvas[:, view_height:, view_width:] = right_exterior
        return canvas

    def _pixels_to_tensor(self, frames):
        tensor = torch.from_numpy(frames).to(AI_DEVICE).permute(3, 0, 1, 2).unsqueeze(0).float()
        return (tensor / 255.0 * 2.0 - 1.0).to(GET_DTYPE())

    def _encode_first_frame_condition(self, videos):
        _, _, _, height, width = videos.shape
        frame_slice = videos[:, :, -1:] if videos.shape[2] in (4, 9) else videos[:, :, :1]
        first_frame = frame_slice[:, :, 0]
        clip_feature = self.run_image_encoder(first_frame)
        image_zeros = torch.zeros(
            videos.shape[0],
            3,
            self.config["target_video_length"] - 1,
            height,
            width,
            dtype=videos.dtype,
            device=videos.device,
        )
        first_frame_latent = self.vae_encoder.encode(torch.cat([frame_slice, image_zeros], dim=2).to(self.vae_encoder.dtype))
        if first_frame_latent.dim() == 4:
            first_frame_latent = first_frame_latent.unsqueeze(0)
        msk = torch.zeros(
            first_frame_latent.shape[0],
            4,
            first_frame_latent.shape[2],
            first_frame_latent.shape[3],
            first_frame_latent.shape[4],
            dtype=first_frame_latent.dtype,
            device=first_frame_latent.device,
        )
        msk[:, :, 0:1] = 1
        y = torch.cat([msk, first_frame_latent], dim=1)
        return clip_feature, y.to(GET_DTYPE()), first_frame_latent[:, :, 0:1].to(GET_DTYPE())

    def _encode_observed_latents(self, videos):
        if videos.shape[2] == 1:
            return None
        if videos.shape[2] < 4:
            repeat_factor = max(1, 4 // videos.shape[2])
            videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
        if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
            pass
        elif videos.shape[2] // 4 != self.num_frame_per_block:
            base = max(videos.shape[2] // 4, 1)
            repeat_factor = max(1, self.num_frame_per_block // base)
            videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
            videos = torch.cat([videos[:, :, 0:1], videos], dim=2)
        else:
            videos = torch.cat([videos[:, :, 0:1], videos], dim=2)

        latents = self.vae_encoder.encode(videos.to(self.vae_encoder.dtype))
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        return latents.to(AI_DEVICE).to(GET_DTYPE())

    def _slice_y(self, start_frame, length):
        if start_frame + length <= self.ys.shape[2]:
            return self.ys[:, :, start_frame : start_frame + length]
        return self.ys[:, :, -length:]

    def _run_cache_warmup(self, latents, start_frame, length):
        if latents is None:
            return
        inputs = {
            "video_latents": latents.to(AI_DEVICE).to(GET_DTYPE()),
            "timestep": torch.zeros([1, length], dtype=torch.int64, device=AI_DEVICE),
            "context": self.prompt_embeds,
            "negative_context": self.negative_prompt_embeds,
            "clip_feature": self.clip_feas,
            "y": self._slice_y(start_frame, length),
            "current_start_frame": start_frame,
            "update_cache": True,
            "cache_name": self.cache_name,
            "enable_cfg": self.use_cfg,
            "guide_scale": self.config["sample_guide_scale"],
        }
        self.model.infer(inputs)

    def _build_model_inputs(self, video_latents, action_latents, video_t, action_t):
        return {
            "video_latents": video_latents.to(AI_DEVICE).to(GET_DTYPE()),
            "timestep": torch.ones([1, video_latents.shape[2]], dtype=torch.int64, device=AI_DEVICE) * video_t,
            "action": action_latents.to(AI_DEVICE).to(GET_DTYPE()),
            "timestep_action": torch.ones([1, self.action_horizon], dtype=torch.int64, device=AI_DEVICE) * action_t,
            "state": self.state_tensor,
            "context": self.prompt_embeds,
            "negative_context": self.negative_prompt_embeds,
            "clip_feature": self.clip_feas,
            "y": self._slice_y(self.current_start_frame, self.num_frame_per_block),
            "current_start_frame": self.current_start_frame,
            "update_cache": False,
            "cache_name": self.cache_name,
            "enable_cfg": self.use_cfg,
            "guide_scale": self.config["sample_guide_scale"],
        }

    def _postprocess_action(self, action):
        action_np = action.detach().float().cpu().numpy()
        action_np = self._q99_denormalize(action_np, self.config["action_norm_stat"], self.action_raw_dim)[0]
        if self.config.get("relative_action", True):
            action_np[:, :7] += self.raw_state[:7]
        return action_np.astype(np.float32)

    def init_run(self):
        self.pred_latent_lst = []
        self.pred_latent_segments = []
        self._current_latent_segment = []
        self.pred_action_lst = []
        self.gen_video = None
        self.gen_video_final = None
        self.current_start_frame = 0
        self.last_chunk_started_new_segment = False
        self.use_cfg = self.config.get("enable_cfg", False)
        self.num_frame_per_block = int(self.config.get("num_frame_per_block", 2))
        self.action_horizon = int(self.config.get("action_horizon", 24))
        self.action_dim = int(self.config.get("action_dim", 32))
        self.action_raw_dim = int(self.config.get("action_raw_dim", 8))
        self.dit_step_mask = self.config.get("dit_step_mask", [True] * int(self.config["infer_steps"]))
        self.camera_sequences = self._load_camera_sequences()
        self.raw_state, self.state_tensor = self._prepare_state()
        self.noise_seed = int(self.config.get("dreamzero_seed", self.input_info.seed))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        self.model.clear_cache(self.cache_name)
        text_encoder_output = self.inputs["text_encoder_output"]
        self.prompt_embeds = text_encoder_output["context"].to(AI_DEVICE).to(GET_DTYPE())
        self.negative_prompt_embeds = text_encoder_output.get("context_null")
        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = self.negative_prompt_embeds.to(AI_DEVICE).to(GET_DTYPE())
        elif self.use_cfg:
            raise ValueError("DreamZero CFG is enabled but text_encoder_output does not include context_null.")

    def _frame_schedule(self):
        total_frames = min(seq.shape[0] for seq in self.camera_sequences)
        if total_frames <= 1:
            return [[0]]
        chunks = [[0]]
        current_frame = int(self.config.get("first_anchor_frame", 23))
        num_chunks = int(self.config.get("num_chunks", 1))
        offsets = self.config.get("relative_offsets", [-23, -16, -8, 0])
        action_horizon = int(self.config.get("action_horizon", 24))
        for _ in range(num_chunks):
            indices = [max(current_frame + int(offset), 0) for offset in offsets]
            if indices[-1] >= total_frames:
                break
            chunks.append(indices)
            current_frame += action_horizon
        return chunks

    def run_chunk(self, frame_indices):
        frames = self._compose_droid_video(self.camera_sequences, frame_indices)
        videos = self._pixels_to_tensor(frames)

        self.last_chunk_started_new_segment = len(frame_indices) == 1 or self.current_start_frame >= int(self.config.get("local_attn_size", 9))
        if self.last_chunk_started_new_segment:
            self.current_start_frame = 0
            self.model.clear_cache(self.cache_name)

        observed_latents = None
        if self.current_start_frame == 0:
            self.clip_feas, self.ys, image_latent = self._encode_first_frame_condition(videos)
            self._run_cache_warmup(image_latent, 0, 1)
            self.current_start_frame += 1
        else:
            observed_latents = self._encode_observed_latents(videos)

        if self.current_start_frame != 1 and observed_latents is not None:
            current_ref_latents = observed_latents[:, :, -self.num_frame_per_block :]
            self._run_cache_warmup(current_ref_latents, self.current_start_frame - self.num_frame_per_block, self.num_frame_per_block)

        latent_shape = (
            1,
            self.config["num_channels_latents"],
            self.num_frame_per_block,
            self.ys.shape[-2],
            self.ys.shape[-1],
        )
        action_shape = (1, self.action_horizon, self.action_dim)
        self.scheduler.generator = None
        self.scheduler.prepare_loop(
            infer_steps=self.config["infer_steps"],
            device=AI_DEVICE,
            latent_shape=latent_shape,
            seed=self.noise_seed,
            dtype=GET_DTYPE(),
        )
        self.action_scheduler.generator = None
        self.action_scheduler.prepare_loop(
            infer_steps=self.config["action_infer_steps"],
            device=AI_DEVICE,
            latent_shape=action_shape,
            seed=self.noise_seed,
            dtype=GET_DTYPE(),
        )

        prev_video_pred = None
        prev_action_pred = None
        for step_index in range(self.scheduler.infer_steps):
            video_t = self.scheduler.timesteps[step_index]
            action_t = self.action_scheduler.timesteps[step_index]
            should_run = bool(self.dit_step_mask[step_index]) if step_index < len(self.dit_step_mask) else True
            if should_run or prev_video_pred is None or prev_action_pred is None:
                inputs = self._build_model_inputs(self.scheduler.latents, self.action_scheduler.latents, video_t, action_t)
                pred = self.model.infer(inputs)
                prev_video_pred = pred["video"]
                prev_action_pred = pred["action"]

            self.scheduler.step_index = step_index
            self.scheduler.noise_pred = prev_video_pred
            self.scheduler.step_post()
            self.action_scheduler.step_index = step_index
            self.action_scheduler.noise_pred = prev_action_pred
            self.action_scheduler.step_post()

            if self.progress_callback:
                self.progress_callback((step_index + 1) / self.scheduler.infer_steps * 100, 100)

        latents = self.scheduler.latents
        if self.current_start_frame == 1:
            latents = torch.cat([image_latent.to(latents.device, latents.dtype), latents], dim=2)
        self.current_start_frame += self.num_frame_per_block
        return self._postprocess_action(self.action_scheduler.latents), latents

    def run_vae_decoder(self, pred_latent):
        decoded = self.vae_decoder.decode(pred_latent.squeeze(0).to(GET_DTYPE()))
        return decoded[0].permute(1, 2, 3, 0).add(1.0).mul(0.5).clamp(0, 1)

    def run_vae_decoder_segments(self, latent_segments):
        decoded_segments = []
        for segment in latent_segments:
            if not segment:
                continue
            pred_latent = torch.cat(segment, dim=2)
            decoded_segments.append(self.run_vae_decoder(pred_latent))
        if not decoded_segments:
            raise RuntimeError("DreamZero produced no latent segments to decode.")
        return torch.cat(decoded_segments, dim=0)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        schedule = self._frame_schedule()
        for chunk_id, frame_indices in enumerate(schedule):
            logger.info(f"DreamZero chunk {chunk_id + 1}/{len(schedule)} frames={frame_indices}")
            with ProfilingContext4DebugL1(f"DreamZero chunk {chunk_id + 1}/{len(schedule)}"):
                self.check_stop()
                actions, latents = self.run_chunk(frame_indices)
                self.pred_action_lst.append(torch.from_numpy(actions))
                self.pred_latent_lst.append(latents)
                if self.last_chunk_started_new_segment and self._current_latent_segment:
                    self.pred_latent_segments.append(self._current_latent_segment)
                    self._current_latent_segment = []
                self._current_latent_segment.append(latents)

        if self._current_latent_segment:
            self.pred_latent_segments.append(self._current_latent_segment)
            self._current_latent_segment = []
        self.pred_action = torch.cat(self.pred_action_lst, dim=0)
        if self.config.get("decode_segments", False):
            self.gen_video = self.run_vae_decoder_segments(self.pred_latent_segments)
        else:
            self.gen_video = self.run_vae_decoder(torch.cat(self.pred_latent_lst, dim=2))
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def process_images_after_vae_decoder(self):
        self.gen_video_final = self.gen_video
        video_path = getattr(self.input_info, "save_result_path", None)
        if not video_path:
            raise ValueError("DreamZero requires save_result_path from input_info.")
        video_path = str(video_path)
        action_path = getattr(self.input_info, "save_action_path", "") or str(Path(video_path).with_suffix(".actions.npy"))
        if not os.path.isabs(str(action_path)):
            action_path = os.path.join(os.path.dirname(video_path) or ".", str(action_path))
        save_to_video(self.gen_video_final, video_path, fps=self.config.get("target_fps", 10), method=self.config.get("save_video_method", "imageio"))
        os.makedirs(os.path.dirname(action_path) or ".", exist_ok=True)
        np.save(action_path, self.pred_action.numpy())
        logger.info("Saved DreamZero video to {}", video_path)
        logger.info("Saved DreamZero actions to {}", action_path)
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "actions": self.pred_action}
        return {"video": None}

    def end_run(self):
        self.model.clear_cache(self.cache_name)
        if self.scheduler is not None:
            self.scheduler.clear()
        if self.action_scheduler is not None:
            self.action_scheduler.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.model
        torch_device_module.empty_cache()
        gc.collect()

    def _run_pipeline_local(self):
        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()
        self.inputs = self.run_input_encoder()
        return self.run_main()

    @ProfilingContext4DebugL1("RUN pipeline", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_worker_request_duration, metrics_labels=["WanDreamZeroRunner"])
    @torch.no_grad()
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        gen_video_final = self._run_pipeline_local()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return gen_video_final
