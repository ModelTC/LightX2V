# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: UP035 - exported aliases used by the split infer modules

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import Tensor

from lightx2v.models.schedulers.hymotion.flow import odeint

from ..utils.body_model import WoodenMesh
from ..utils.geometry import matrix_to_quaternion, quaternion_fix_continuity, quaternion_to_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from ..utils.loaders import load_object
from ..utils.motion_process import smooth_rotation
from ..utils.type_converter import get_module_device


def length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """
        lengths: (B, 1)
        max_len: int
    Returns: (B, max_len)
    """
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    return mask


def start_end_frame_to_mask(start_frame: Tensor, end_frame: Tensor, max_len: int) -> Tensor:
    assert (start_frame >= 0).all() and (end_frame >= 0).all(), f"start_frame={start_frame}, end_frame={end_frame}"
    lengths = end_frame - start_frame + 1
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    batch_size = start_frame.shape[0]
    arange_ids = torch.arange(max_len, device=start_frame.device).unsqueeze(0).expand(batch_size, max_len)
    mask = (arange_ids >= start_frame.unsqueeze(1)) & (arange_ids <= end_frame.unsqueeze(1))
    return mask


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`.

    When passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the
    tensor is always created on the CPU.
    """
    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected. Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout) for i in range(batch_size)]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    return latents


class MotionGeneration(torch.nn.Module):
    def __init__(self, network_module: str, network_module_args: dict, text_encoder_module: str, text_encoder_cfg: dict, mean_std_dir: str, motion_type="auto", **kwargs):
        super().__init__()
        self._network_module_args = deepcopy(network_module_args)
        self.motion_transformer = load_object(network_module, network_module_args)
        self._text_encoder_module = text_encoder_module
        self._text_encoder_cfg = deepcopy(text_encoder_cfg)
        self.motion_type = motion_type
        self.null_vtxt_feat = torch.nn.Parameter(torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768)))
        self.null_ctxt_input = torch.nn.Parameter(torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096)))
        self.special_game_vtxt_feat = torch.nn.Parameter(torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768)))
        self.special_game_ctxt_feat = torch.nn.Parameter(torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096)))
        self.mean_std_dir = mean_std_dir
        self._parse_buffer(self.motion_type)
        self.output_mesh_fps = kwargs.get("output_mesh_fps", 30)
        self.train_frames = kwargs.get("train_frames", 360)
        self.uncondition_mode = kwargs.get("uncondition_mode", False)
        self.enable_ctxt_null_feat = kwargs.get("enable_ctxt_null_feat", False)
        self.enable_special_game_feat = kwargs.get("enable_special_game_feat", False)
        self.random_generator_on_gpu = kwargs.get("random_generator_on_gpu", True)

    def _parse_buffer(self, mode: str) -> None:
        self.body_model = WoodenMesh()
        self._find_motion_type(mode=mode)
        self._load_mean_std()

    def _load_mean_std(self, mean_std_name: str | None = None) -> None:
        mean_std_name = self.mean_std_dir if mean_std_name is None else mean_std_name
        if mean_std_name is not None and osp.isdir(mean_std_name):
            mean = torch.from_numpy(np.load(osp.join(mean_std_name, "Mean.npy"))).float()
            std = torch.from_numpy(np.load(osp.join(mean_std_name, "Std.npy"))).float()
            self._assert_motion_dimension(mean.unsqueeze(0), std.unsqueeze(0))
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        else:
            print(f"[{self.__class__.__name__}] No mean_std found, using blank mean_std, self.mean_std_dir={self.mean_std_dir}")
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))

    def _assert_motion_dimension(self, mean: Tensor, std: Tensor) -> None:
        assert mean.shape == std.shape, f"mean.shape={mean.shape} != std.shape={std.shape}"
        assert mean.ndim == 2, f"mean.ndim={mean.ndim} != 2"
        assert mean.shape == (1, 201), f"mean.shape={mean.shape} != (1, 201)"

    def _find_motion_type(self, mode: str) -> None:
        if mode == "auto":
            self.motion_type = "o6dp"
        else:
            self.motion_type = mode

    def decode_motion_from_latent(self, latent: Tensor, should_apply_smooothing: bool = True) -> dict[str, Tensor]:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration_decode_motion_from_latent

        return MotionGeneration_decode_motion_from_latent(self, latent, should_apply_smooothing)

    def _decode_o6dp(self, latent_denorm: torch.Tensor, num_joints: int, rel_trans: bool = False, should_apply_smooothing: bool = True) -> dict:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration__decode_o6dp

        return MotionGeneration__decode_o6dp(self, latent_denorm, num_joints, rel_trans, should_apply_smooothing)

    @staticmethod
    def smooth_with_savgol(input: torch.Tensor, window_length: int = 9, polyorder: int = 5) -> torch.Tensor:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration_smooth_with_savgol

        return MotionGeneration_smooth_with_savgol(input, window_length, polyorder)

    @staticmethod
    def smooth_with_slerp(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration_smooth_with_slerp

        return MotionGeneration_smooth_with_slerp(input, sigma)

    @staticmethod
    def noise_from_seeds(latent: Tensor, seeds: int | list[int], seed_start: int = 0, random_generator_on_gpu: bool = True) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration_noise_from_seeds

        return MotionGeneration_noise_from_seeds(latent, seeds, seed_start, random_generator_on_gpu)

    def _maybe_inject_source_token(
        self, vtxt_input: Tensor, ctxt_input: Tensor, ctxt_mask_temporal: Tensor, sources: list[str] | None, trigger_sources: set | None = None, prob: float = 0.5
    ) -> tuple[Tensor, Tensor, Tensor]:
        from lightx2v.models.networks.hymotion.infer.motion import MotionGeneration__maybe_inject_source_token

        return MotionGeneration__maybe_inject_source_token(self, vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources, prob)


class MotionFlowMatching(MotionGeneration):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        noise_scheduler_cfg: dict | None = None,
        infer_noise_scheduler_cfg: dict | None = None,
        mean_std_dir: str | None = None,
        losses_cfg: dict | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        **kwargs,
    ):
        if noise_scheduler_cfg is None:
            noise_scheduler_cfg = {"method": "euler"}
        if infer_noise_scheduler_cfg is None:
            infer_noise_scheduler_cfg = {"validation_steps": 50}
        super().__init__(
            network_module=network_module,
            network_module_args=network_module_args,
            text_encoder_module=text_encoder_module,
            text_encoder_cfg=text_encoder_cfg,
            losses_cfg=losses_cfg,
            mean_std_dir=mean_std_dir if mean_std_dir is not None else test_cfg.get("mean_std_dir", None),
            **kwargs,
        )
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg)
        self._infer_noise_scheduler_cfg = deepcopy(infer_noise_scheduler_cfg)
        self.train_cfg = deepcopy(train_cfg) if train_cfg else {}
        self.test_cfg = deepcopy(test_cfg) if test_cfg else {}
        self._parse_test_cfg()

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self._infer_noise_scheduler_cfg["validation_steps"]
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)

    @torch.no_grad()
    def generate(
        self, text: str | list[str], seed_input: list[int], duration_slider: int, cfg_scale: float | None = None, use_special_game_feat: bool = False, hidden_state_dict=None, length=None
    ) -> dict[str, Any]:
        from lightx2v.models.networks.hymotion.infer.motion import MotionFlowMatching_generate

        return MotionFlowMatching_generate(self, text, seed_input, duration_slider, cfg_scale, use_special_game_feat, hidden_state_dict, length)


__all__ = [
    "Any",
    "Dict",
    "List",
    "Optional",
    "Tensor",
    "Tuple",
    "Union",
    "get_module_device",
    "length_to_mask",
    "matrix_to_quaternion",
    "np",
    "odeint",
    "quaternion_fix_continuity",
    "quaternion_to_matrix",
    "randn_tensor",
    "rot6d_to_rotation_matrix",
    "rotation_matrix_to_rot6d",
    "savgol_filter",
    "smooth_rotation",
    "torch",
]
