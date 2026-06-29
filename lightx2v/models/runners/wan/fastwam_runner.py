import json
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER

AGENTVIEW_IMAGE_NAME = "agentview_image.png"
WRIST_IMAGE_NAME = "wrist_image.png"


def resize_rgb(image, width, height):
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


class MinMaxNormalizer:
    range_tol = 1e-4

    def __init__(self, stats):
        min_v = torch.as_tensor(stats["global_min"], dtype=torch.float32)
        max_v = torch.as_tensor(stats["global_max"], dtype=torch.float32)
        input_range = max_v - min_v
        ignore = input_range < self.range_tol
        input_range[ignore] = 2.0
        self.scale = 2.0 / input_range
        self.offset = -1.0 - self.scale * min_v
        self.offset[ignore] = -min_v[ignore]

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.clamp(x * self.scale + self.offset, -5.0, 5.0)

    def backward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x - self.offset) / self.scale


class FastWAMPolicy:
    def __init__(
        self,
        adapter_model_path=None,
        dataset_stats_path=None,
        device="cuda",
        model_path=None,
        action_chunk_size: Optional[int] = None,
        actions_per_plan=10,
        action_infer_steps=20,
        action_sample_shift=5.0,
        action_dim_hidden=1024,
        action_dim=7,
        robot_state_dim=8,
        seed: Optional[int] = 0,
        binarize_gripper=True,
        default_prompt=None,
        camera_size=224,
        config=None,
    ):
        self.device = torch.device(device)
        if not default_prompt:
            raise ValueError("FastWAM requires `default_prompt`.")
        if config is None:
            raise ValueError("FastWAM requires `config`.")

        paths = {
            "model_path": model_path,
            "adapter_model_path": adapter_model_path,
            "dataset_stats_path": dataset_stats_path,
        }
        for name, value in paths.items():
            if value is None or str(value).strip() == "":
                raise ValueError(f"FastWAM requires `{name}`.")
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"FastWAM requires absolute `{name}`, got: {path}")
            if not path.exists():
                raise FileNotFoundError(str(path))
            paths[name] = path.resolve()

        self.model_path = paths["model_path"]
        self.adapter_model_path = paths["adapter_model_path"]
        self.dataset_stats_path = paths["dataset_stats_path"]

        self.camera_size = int(camera_size)
        self.action_chunk_size = int(action_chunk_size) if action_chunk_size and int(action_chunk_size) > 0 else 32
        self.actions_per_plan = int(max(1, min(int(actions_per_plan), self.action_chunk_size)))
        self.action_infer_steps = int(action_infer_steps)
        self.action_sample_shift = float(action_sample_shift)
        self.action_dim_hidden = int(action_dim_hidden)
        self.action_dim = int(action_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.seed = None if seed is None or int(seed) < 0 else int(seed)
        self.binarize_gripper = bool(binarize_gripper)
        self.default_prompt = str(default_prompt)
        self.config = config
        self._check_model_config()
        self.pending_actions = deque()

        self.state_normalizer, self.action_normalizer = self._load_normalizers()
        self.text_encoder = self._load_text_encoder()
        self.vae = self._load_vae()
        self.model = FastWAMNativeModel(
            model_path=str(self.model_path),
            config=self.config,
            device=self.device,
        )

    @classmethod
    def from_config(cls, config):
        action_chunk_size = int(config.get("action_chunk_size", 0) or 0)
        return cls(
            adapter_model_path=config.get("adapter_model_path"),
            dataset_stats_path=config.get("dataset_stats_path"),
            device=config.get("device", "cuda"),
            model_path=config.get("model_path"),
            action_chunk_size=None if action_chunk_size <= 0 else action_chunk_size,
            actions_per_plan=config.get("actions_per_plan", 10),
            action_infer_steps=config.get("action_infer_steps", 20),
            action_sample_shift=config.get("action_sample_shift", 5.0),
            action_dim_hidden=config.get("action_dim_hidden", 1024),
            action_dim=config.get("action_dim", 7),
            robot_state_dim=config.get("robot_state_dim", 8),
            seed=config.get("seed", 0),
            binarize_gripper=config.get("binarize_gripper", True),
            default_prompt=config.get("default_prompt"),
            camera_size=config.get("camera_size", 224),
            config=config,
        )

    def _check_model_config(self):
        for key in ["dim", "num_layers", "num_heads", "freq_dim", "eps", "action_dim_hidden", "action_dim", "robot_state_dim"]:
            if key not in self.config:
                raise ValueError(f"FastWAM requires `{key}` in config.")

    def _load_normalizers(self):
        with open(self.dataset_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return (
            MinMaxNormalizer(stats["state"]["default"]),
            MinMaxNormalizer(stats["action"]["default"]),
        )

    def _load_text_encoder(self):
        t5_path = self._find_model_file("models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = self._find_model_dir("google/umt5-xxl")
        return T5EncoderModel(
            text_len=128,
            dtype=GET_DTYPE(),
            device=self.device,
            checkpoint_path=t5_path,
            tokenizer_path=str(tokenizer_path),
            cpu_offload=False,
        )

    def _load_vae(self):
        vae_path = self._find_model_file("Wan2.2_VAE.pth")
        return Wan2_2_VAE(
            vae_path=vae_path,
            device=self.device,
            dtype=GET_DTYPE(),
            vae_type="wan2.2",
            cpu_offload=False,
        )

    def _find_model_file(self, filename):
        path = self.model_path / filename
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Cannot find {filename} under {self.model_path}")

    def _find_model_dir(self, dirname):
        path = self.model_path / dirname
        if path.exists():
            return path
        raise FileNotFoundError(f"Cannot find {dirname} under {self.model_path}")

    def reset(self):
        self.pending_actions.clear()

    def next_action(self, agentview_rgb, wrist_rgb, state, task_description):
        if not self.pending_actions:
            action_chunk = self.predict_action_chunk(agentview_rgb, wrist_rgb, state, task_description)
            for action in action_chunk[: self.actions_per_plan]:
                self.pending_actions.append(np.asarray(action, dtype=np.float32))
        if not self.pending_actions:
            raise RuntimeError("FastWAM produced an empty action chunk")
        return self.pending_actions.popleft()

    def predict_action_chunk(self, agentview_rgb, wrist_rgb, state, task_description, seed=None):
        image = self.build_image_tensor(agentview_rgb, wrist_rgb)
        first_frame_latents = self.encode_image_latents(image)
        context, context_mask = self.encode_prompt(self.default_prompt.format(task_prompt=task_description))
        robot_state = self.state_normalizer.forward(np.asarray(state, dtype=np.float32))
        inputs, action_shape = self.model.prepare_action_inputs(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_chunk_size=self.action_chunk_size,
            robot_state=robot_state,
        )
        action = self._run_action_denoising(
            inputs=inputs,
            action_shape=action_shape,
            action_infer_steps=self.action_infer_steps,
            seed=self.seed if seed is None else seed,
        )
        action = self.action_normalizer.backward(action).numpy()
        action[..., -1] = action[..., -1] * 2 - 1
        action[..., -1] = action[..., -1] * -1.0
        if self.binarize_gripper:
            action[..., -1] = np.sign(action[..., -1])
        return action.astype(np.float32)

    def _run_action_denoising(self, inputs, action_shape, action_infer_steps, seed):
        scheduler = self.model.scheduler
        scheduler.prepare_loop(
            action_shape,
            seed=seed,
            device=self.device,
            dtype=GET_DTYPE(),
            infer_steps=action_infer_steps,
        )
        for step_index in range(scheduler.infer_steps):
            scheduler.step_pre(step_index)
            self.model.infer(inputs)
            scheduler.step_post()
        action = scheduler.latents[0].detach().to(device="cpu", dtype=torch.float32)
        scheduler.clear()
        return action

    def build_image_tensor(self, agentview_rgb, wrist_rgb):
        primary = resize_rgb(agentview_rgb, self.camera_size, self.camera_size)
        wrist = resize_rgb(wrist_rgb, self.camera_size, self.camera_size)
        rgb = np.concatenate([primary, wrist], axis=1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=self.device, dtype=GET_DTYPE())
        return tensor * (2.0 / 255.0) - 1.0

    def encode_image_latents(self, image):
        image = image.unsqueeze(1)
        latents = self.vae.encode(image.unsqueeze(0))
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        return latents.to(device=self.device, dtype=GET_DTYPE())

    def encode_prompt(self, prompt):
        ids, mask = self.text_encoder.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            context = self.text_encoder.model(ids, mask)
        for i, seq_len in enumerate(seq_lens):
            context[i, seq_len:] = 0
        mask = torch.ones_like(mask, dtype=torch.bool)
        return context[0].to(device=self.device, dtype=GET_DTYPE()), mask[0].to(device=self.device)

    def close(self):
        self.pending_actions.clear()


@RUNNER_REGISTER("fastwam")
class FastWAMRunner(BaseRunner):
    def init_modules(self):
        logger.info("Loading FastWAM policy...")
        self.policy = FastWAMPolicy.from_config(self.config)
        self.config.lock()
        logger.info("FastWAM policy loaded.")

    def _resolve_action_output_path(self):
        if getattr(self.input_info, "save_action_path", ""):
            return str(Path(self.input_info.save_action_path).expanduser().resolve())
        if getattr(self.input_info, "save_result_path", ""):
            return str(Path(self.input_info.save_result_path).expanduser().resolve().with_suffix(".actions.npy"))
        raise ValueError("FastWAM requires `save_action_path` or `save_result_path`.")

    def _load_image_pair(self):
        image_path = getattr(self.input_info, "image_path", "") or ""
        if not image_path:
            raise ValueError("FastWAM requires `image_path`.")

        image_path = os.path.expanduser(str(image_path))
        if os.path.isdir(image_path):
            agentview = os.path.join(image_path, AGENTVIEW_IMAGE_NAME)
            wrist = os.path.join(image_path, WRIST_IMAGE_NAME)
        else:
            items = [item.strip() for item in image_path.split(",") if item.strip()]
            if len(items) != 2:
                raise ValueError("FastWAM `image_path` must be a directory or two comma-separated image paths.")
            agentview, wrist = items

        return self._load_rgb(agentview), self._load_rgb(wrist)

    @staticmethod
    def _load_rgb(path):
        return np.asarray(Image.open(os.path.expanduser(str(path))).convert("RGB"))

    def _load_state(self):
        state_path = getattr(self.input_info, "state_path", "") or ""
        if not state_path:
            raise ValueError("FastWAM requires `state_path` with 8 floats.")
        state_path = str(Path(state_path).expanduser().resolve())
        suffix = Path(state_path).suffix.lower()

        if suffix == ".json":
            with open(state_path, "r") as f:
                payload = json.load(f)
        elif suffix in [".npy", ".npz"]:
            payload = np.load(state_path, allow_pickle=True)
            if isinstance(payload, np.lib.npyio.NpzFile):
                payload = {key: payload[key] for key in payload.files}
            elif payload.shape == () and isinstance(payload.item(), dict):
                payload = payload.item()
        else:
            payload = np.loadtxt(state_path, delimiter=",", dtype=np.float32)

        if isinstance(payload, dict):
            for key in ["state", "qpos", "robot_state", "observation.state"]:
                if key in payload:
                    payload = payload[key]
                    break
        state = np.asarray(payload, dtype=np.float32).reshape(-1)
        if state.size != 8:
            raise ValueError(f"FastWAM LIBERO state must contain 8 floats, got {state.size}.")
        return state

    def _save_actions(self, actions):
        action_path = self._resolve_action_output_path()
        os.makedirs(os.path.dirname(action_path) or ".", exist_ok=True)
        if action_path.endswith(".json"):
            with open(action_path, "w") as f:
                json.dump(actions.tolist(), f, ensure_ascii=False, indent=2)
        else:
            np.save(action_path, actions)
        logger.info("Saved FastWAM actions to {}", action_path)

    def run_pipeline(self, input_info):
        self.input_info = input_info
        if not self.input_info.prompt:
            raise ValueError("FastWAM requires `prompt` as the LIBERO task description.")

        agentview, wrist = self._load_image_pair()
        state = self._load_state()
        actions = self.policy.predict_action_chunk(
            agentview_rgb=agentview,
            wrist_rgb=wrist,
            state=state,
            task_description=self.input_info.prompt,
            seed=self.input_info.seed,
        )
        self._save_actions(actions)

        if self.input_info.return_result_tensor:
            return {"actions": actions}
        return {"actions": None}

    def end_run(self):
        if hasattr(self, "policy"):
            self.policy.close()
