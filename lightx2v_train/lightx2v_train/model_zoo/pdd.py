# PDD widening and parameter fusion follow NVlabs/FastGen's
# fastgen/networks/utils/pdd_mixin.py (NVIDIA, Apache-2.0).
"""Channel-major parallel decoding heads for H3 image and audio/video models."""

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from lightx2v_train.runtime.distributed import is_main_process


class ParallelDecodeLinear(torch.nn.Linear):
    """One parameter bank, with one native-sigma velocity head per interval.

    H3 uses channel-major patch rows, so the output dimension is [N,C,patch].
    A single widened weight also keeps FSDP's gradient reduction independent
    of which head indices are supervised on each rank.
    """

    def __init__(self, in_features, out_features, grid_size, *, bias=True, device=None, dtype=torch.float32):
        super().__init__(in_features, out_features * grid_size, bias=bias, device=device, dtype=dtype)
        self.grid_size = int(grid_size)
        self.single_out_features = int(out_features)
        self._fusion = None

    @classmethod
    def from_linear(cls, linear, grid_size):
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError("PDD output projections must be plain Linear layers; exclude them from LoRA target_modules.")
        result = cls(linear.in_features, linear.out_features, grid_size, bias=linear.bias is not None, device=linear.weight.device)
        if not linear.weight.is_meta:
            with torch.no_grad():
                result.weight.copy_(result.expand_teacher_parameter(linear.weight))
                if linear.bias is not None:
                    result.bias.copy_(result.expand_teacher_parameter(linear.bias))
        return result

    def expand_teacher_parameter(self, tensor):
        if tensor.shape[0] == self.out_features:
            return tensor
        if tensor.shape[0] != self.single_out_features:
            raise ValueError(f"PDD expected {self.single_out_features} teacher rows or {self.out_features} student rows, got {tuple(tensor.shape)}.")
        return tensor.repeat(self.grid_size, *([1] * (tensor.ndim - 1)))

    @contextmanager
    def fuse(self, start, end, weights):
        previous = self._fusion
        self._fusion = (int(start), int(end), weights)
        try:
            yield self
        finally:
            self._fusion = previous

    def forward(self, hidden):
        if self._fusion is None:
            return F.linear(hidden, self.weight, self.bias)
        start, end, weights = self._fusion
        # Fusion runs inside this module's FSDP forward, after all-gather.
        # The schedule weights are computed/normalized in scheduler precision;
        # accumulate parameters in FP32, matching FastGen precision_decoding.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            coefficients = weights[start:end].to(device=hidden.device, dtype=torch.float32)
            bank = self.weight.unflatten(0, (self.grid_size, self.single_out_features))[start:end].float()
            weight = torch.einsum("n,noi->oi", coefficients, bank).to(self.weight.dtype)
            bias = None
            if self.bias is not None:
                bank_bias = self.bias.unflatten(0, (self.grid_size, self.single_out_features))[start:end].float()
                bias = torch.einsum("n,no->o", coefficients, bank_bias).to(self.bias.dtype)
            return F.linear(hidden.to(weight.dtype), weight, bias)


class PDDModelMixin:
    """Train/save full output banks alongside the ordinary backbone LoRA."""

    def enable_pdd(self, grid_size):
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
            raise ValueError("PDD grid_size must be an integer >=2.")
        if self.is_fsdp2_wrapped():
            raise RuntimeError("Enable PDD after LoRA configuration and before FSDP sharding.")
        if hasattr(self, "pdd_steps"):
            raise RuntimeError("PDD output heads were already enabled.")
        self.pdd_steps = grid_size
        self._pdd_paths = self.pdd_output_layer_paths()
        for path in self._pdd_paths.values():
            layer = self.transformer.get_submodule(path)
            widened = ParallelDecodeLinear.from_linear(layer, grid_size)
            parent, _, name = path.rpartition(".")
            self.transformer.get_submodule(parent).add_module(name, widened)
        self.restore_pdd_trainable()

    def restore_pdd_trainable(self):
        for path in getattr(self, "_pdd_paths", {}).values():
            self.transformer.get_submodule(path).requires_grad_(True)

    def set_lora_trainable(self):
        super().set_lora_trainable()
        self.restore_pdd_trainable()

    def pdd_auxiliary_parameter_names(self):
        return tuple(f"{path}.{name}" for path in self._pdd_paths.values() for name, _ in self.transformer.get_submodule(path).named_parameters())

    def transform_pdd_pretrained_tensor(self, name, tensor):
        if not hasattr(self, "pdd_steps"):
            return tensor
        path, _, parameter_name = name.rpartition(".")
        if path in self._pdd_paths.values() and parameter_name in {"weight", "bias"}:
            return self.transformer.get_submodule(path).expand_teacher_parameter(tensor)
        return tensor

    def pdd_checkpoint_metadata(self):
        return {
            "schema_version": 1,
            "method": "pdd",
            "model": self.config["model"]["name"],
            "grid_size": self.pdd_steps,
            "channel_layout": "channel_major",
            "head_paths": self._pdd_paths,
            "velocity_convention": "noise_minus_clean",
        }

    def save_pdd_weights(self, save_dir):
        state = {name: self.transformer.get_parameter(name) for name in self.pdd_auxiliary_parameter_names()}
        state = self._gather_selected_state_dict_for_save(state)
        if is_main_process():
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            save_file(state, str(directory / "pdd_heads.safetensors"))
            (directory / "pdd_heads.json").write_text(json.dumps(self.pdd_checkpoint_metadata(), indent=2) + "\n")

    def load_pdd_weights(self, save_dir):
        directory = Path(save_dir)
        metadata = json.loads((directory / "pdd_heads.json").read_text())
        if metadata != self.pdd_checkpoint_metadata():
            raise ValueError(f"PDD head metadata differs from the current model: {directory}.")
        if self.is_fsdp2_wrapped():
            from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import _load_partial_full_state

            state = load_file(str(directory / "pdd_heads.safetensors")) if is_main_process() else {}
            _load_partial_full_state(self.transformer, state)
        else:
            state = load_file(str(directory / "pdd_heads.safetensors"))
            expected = set(self.pdd_auxiliary_parameter_names())
            if set(state) != expected:
                raise ValueError(f"PDD head tensor names do not match the current model: {directory}.")
            self.transformer.load_state_dict(state, strict=False)

    def predict_pdd_block(self, latents, start, end, condition, sigma_grid):
        """Return the block displacement, using fused per-modality weights."""
        if not 0 <= int(start) < int(end) <= self.pdd_steps or len(sigma_grid) != self.pdd_steps + 1:
            raise ValueError("PDD block requires 0 <= start < end <= grid_size and grid_size+1 sigma nodes.")
        schedules = self.pdd_sigma_schedule(sigma_grid)
        if torch.is_tensor(schedules):
            schedules = {next(iter(self._pdd_paths)): schedules}
        totals = {}
        with ExitStack() as stack:
            for modality, path in self._pdd_paths.items():
                differences = torch.diff(schedules[modality])
                total = differences[start:end].sum()
                totals[modality] = total
                stack.enter_context(self.transformer.get_submodule(path).fuse(start, end, differences / total))
            velocity = self.predict_pdd_teacher(latents, sigma_grid[start], condition)
        if torch.is_tensor(velocity):
            return velocity * next(iter(totals.values())).to(velocity)
        return type(velocity)(video=velocity.video * totals["video"].to(velocity.video), audio=velocity.audio * totals["audio"].to(velocity.audio), shape=latents.shape)
