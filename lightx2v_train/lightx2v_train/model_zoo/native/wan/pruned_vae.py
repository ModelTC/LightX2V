"""Shape-preserving residual pruning for the released Wan2.1 video VAE."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..pruning_search import build_residual_search_masks
from .modules.vae import CausalConv3d, Decoder3d, Encoder3d, Resample, ResidualBlock


WAN_VAE_CONFIG = {
    "dim": 96,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
}
WAN_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
WAN_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def posterior_stats(moments):
    mean, log_variance = moments.float().chunk(2, dim=1)
    return mean, torch.exp(0.5 * log_variance.clamp(-30.0, 20.0))


def normalized_posterior_stats(moments, *, latents_mean=WAN_LATENTS_MEAN, latents_std=WAN_LATENTS_STD):
    mean, std = posterior_stats(moments)
    offset = torch.as_tensor(latents_mean, device=mean.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    scale = torch.as_tensor(latents_std, device=mean.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    inverse_scale = scale.reciprocal()
    return (mean - offset) * inverse_scale, std * inverse_scale


def _spatial_resample(module, value):
    batch, channels, frames, height, width = value.shape
    value = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    value = module.resample(value)
    return value.reshape(batch, frames, *value.shape[1:]).permute(0, 2, 1, 3, 4)


def resample_full_sequence(module: Resample, value):
    """The native streaming protocol leaves the first temporal sample unfiltered."""
    if module.mode == "upsample3d" and value.shape[2] > 1:
        first = value[:, :, :1]
        rest = module.time_conv(value[:, :, 1:])
        batch, doubled_channels, frames, height, width = rest.shape
        rest = rest.reshape(batch, 2, doubled_channels // 2, frames, height, width)
        rest = rest.permute(0, 2, 3, 1, 4, 5).reshape(batch, doubled_channels // 2, 2 * frames, height, width)
        value = torch.cat((first, rest), dim=2)
    value = _spatial_resample(module, value)
    if module.mode == "downsample3d" and value.shape[2] > 1:
        value = torch.cat((value[:, :, :1], module.time_conv(value)), dim=2)
    return value


class SearchLoRAConv3d(CausalConv3d):
    """Temporary low-rank updates keep the original causal convolution state keys."""

    def __init__(self, convolution, rank, alpha):
        nn.Module.__init__(self)
        for name in (
            "in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation",
            "groups", "padding_mode", "transposed", "output_padding", "_padding",
            "_reversed_padding_repeated_twice",
        ):
            setattr(self, name, getattr(convolution, name))
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(convolution.weight.new_empty(rank, convolution.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(convolution.weight.new_zeros(convolution.out_channels, rank, 1, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and padding[4] > 0:
            value = torch.cat((cache_x.to(value.device), value), dim=2)
            padding[4] -= cache_x.shape[2]
        value = F.pad(value, padding)
        original = F.conv3d(value, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        update = F.conv3d(value, self.lora_A, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return original + F.conv3d(update, self.lora_B) * self.scale


class ResidualShortcut(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.shortcut = block.shortcut

    def forward(self, value):
        return self.shortcut(value)


class GatedResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.residual = block.residual
        self.shortcut = block.shortcut

    def forward(self, value, gate):
        residual = self.residual(value)
        return self.shortcut(value) + gate.to(residual.dtype).view(-1, 1, 1, 1, 1) * residual


class _WanPrunedComponent(nn.Module):
    component_name = ""
    default_keep = 0

    def __init__(
        self, teacher_config=None, *, kept_residual_indices=None, search=None,
        latents_mean=WAN_LATENTS_MEAN, latents_std=WAN_LATENTS_STD,
    ):
        super().__init__()
        self.teacher_config = dict(WAN_VAE_CONFIG if teacher_config is None else teacher_config)
        config = self.teacher_config
        self.latent_channels = config["z_dim"]
        if len(latents_mean) != self.latent_channels or len(latents_std) != self.latent_channels:
            raise ValueError("Wan latent statistics must match the latent channel count.")
        if any(value <= 0 for value in latents_std):
            raise ValueError("Wan latent standard deviations must be positive.")
        if 2 ** (len(config["dim_mult"]) - 1) != 8 or sum(config["temperal_downsample"]) != 2:
            raise ValueError("Wan2.1 pruning preserves spatial compression 8 and temporal compression 4.")
        options = {name: config[name] for name in ("dim", "dim_mult", "num_res_blocks", "attn_scales", "dropout")}
        if self.component_name == "encoder":
            self.encoder = Encoder3d(
                **options, z_dim=2 * self.latent_channels, temperal_downsample=config["temperal_downsample"],
            )
            self.conv1 = CausalConv3d(2 * self.latent_channels, 2 * self.latent_channels, 1)
        else:
            self.decoder = Decoder3d(
                **options, z_dim=self.latent_channels, temperal_upsample=config["temperal_downsample"][::-1],
            )
            self.conv2 = CausalConv3d(self.latent_channels, self.latent_channels, 1)

        self.register_buffer("latents_mean", torch.tensor(latents_mean, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(latents_std, dtype=torch.float32))
        self.gradient_checkpointing = False
        self.search_config = dict(search) if search is not None else None
        self._search_noise = None
        blocks = [(name, block) for name, block in self.network.named_modules() if isinstance(block, ResidualBlock)]
        self.depth = len(blocks)
        self.residual_groups = self._stage_residual_groups()
        self.search_grouping = search.get("grouping", "global") if search is not None else "global"
        if search is not None and kept_residual_indices is not None:
            raise ValueError("Search starts from every teacher residual branch.")
        kept = list(range(self.depth)) if kept_residual_indices is None else list(kept_residual_indices)
        if kept != sorted(set(kept)) or any(index < 0 or index >= self.depth for index in kept):
            raise ValueError("Retained residual indices must be unique, in range, and in original order.")
        self.kept_residual_indices = kept
        if search is not None:
            masks = build_residual_search_masks(
                self.depth, search, self.residual_groups, default_keep=self.default_keep,
            )
            self.keep_residuals = int(masks[0].sum().item())
            self.temperature = float(search.get("temperature", 4.0))
            self.gate_scale = float(search.get("gate_scale", 100.0))
            if self.temperature <= 0 or self.gate_scale <= 0:
                raise ValueError("Search temperature and gate_scale must be positive.")
            self.gate_logits = nn.Parameter(torch.zeros(self.depth))
            self.register_buffer("candidate_masks", masks, persistent=False)
        for index, (name, block) in enumerate(blocks):
            if search is not None:
                self.network.set_submodule(name, GatedResidualBlock(block))
            elif index not in kept:
                self.network.set_submodule(name, ResidualShortcut(block))
        self.architecture_config = {
            "teacher_config": self.teacher_config,
            "kept_residual_indices": kept if search is None else None,
            "search": self.search_config,
            "latents_mean": list(latents_mean),
            "latents_std": list(latents_std),
        }
        if search is not None:
            self.configure_search_trainable()

    @property
    def network(self):
        return getattr(self, self.component_name)

    def _stage_residual_groups(self):
        containers = (
            (self.encoder.downsamples, self.encoder.middle) if self.component_name == "encoder"
            else (self.decoder.middle, self.decoder.upsamples)
        )
        groups, index = [], 0
        for container in containers:
            group = []
            for layer in container:
                if isinstance(layer, ResidualBlock):
                    group.append(index)
                    index += 1
                elif isinstance(layer, Resample) and group:
                    groups.append(tuple(group))
                    group = []
            if group:
                groups.append(tuple(group))
        return tuple(groups)

    def configure_search_trainable(self):
        self.requires_grad_(False)
        rank = int(self.search_config.get("lora_rank", 16))
        if rank < 1:
            raise ValueError("Search lora_rank must be positive.")
        alpha = float(self.search_config.get("lora_alpha", 2 * rank))
        for block in self.network.modules():
            if isinstance(block, GatedResidualBlock):
                for index, layer in enumerate(block.residual):
                    if isinstance(layer, CausalConv3d):
                        if not isinstance(layer, SearchLoRAConv3d):
                            block.residual[index] = SearchLoRAConv3d(layer, rank, alpha)
                        block.residual[index].lora_A.requires_grad_(True)
                        block.residual[index].lora_B.requires_grad_(True)
        self.gate_logits.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def prepare_search_step(self, batch_size):
        if self.search_config is not None:
            self._search_noise = -torch.empty(
                batch_size, len(self.candidate_masks), device=self.gate_logits.device,
            ).exponential_().log()

    def clear_search_step(self):
        self._search_noise = None

    def selected_layers(self, logits=None):
        if self.search_config is None:
            return list(self.kept_residual_indices)
        values = self.gate_logits if logits is None else logits
        masks = self.candidate_masks.to(device=values.device, dtype=torch.float32)
        scores = (masks * values.detach().float().unsqueeze(0)).sum(-1)
        return masks[scores.argmax()].nonzero().flatten().cpu().tolist()

    def _search_mask(self, batch_size):
        masks = self.candidate_masks.float()
        scores = (masks * self.gate_logits.float().unsqueeze(0)).sum(-1) * self.gate_scale
        scores = scores.unsqueeze(0).expand(batch_size, -1)
        if not self.training:
            return masks[scores.argmax(-1)]
        noise = self._search_noise
        if noise is None:
            noise = -torch.empty_like(scores).exponential_().log()
        elif noise.shape[0] != batch_size:
            raise ValueError("Search noise batch size differs from the current video batch.")
        soft = ((scores + noise) / self.temperature).softmax(-1)
        hard = F.one_hot(soft.argmax(-1), len(masks)).to(soft.dtype)
        return ((hard - soft.detach() + soft).unsqueeze(-1) * masks.unsqueeze(0)).sum(-2)

    def _layers(self):
        yield self.network.conv1
        if self.component_name == "encoder":
            yield from self.encoder.downsamples
            yield from self.encoder.middle
        else:
            yield from self.decoder.middle
            yield from self.decoder.upsamples
        yield from self.network.head

    @staticmethod
    def _run(layer, value, *, gradient_checkpointing=False, gate=None):
        if gate is not None:
            arguments = (value, gate)
            operation = layer
        elif isinstance(layer, Resample):
            arguments = (value,)
            operation = lambda inputs: resample_full_sequence(layer, inputs)
        else:
            arguments = (value,)
            operation = layer
        if gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(operation, *arguments, use_reentrant=False)
        return operation(*arguments)

    def _forward_network(self, value, *, mask=None, return_features=False, auxiliary_feature_index=None,
                         start_after=None, gradient_checkpointing=False):
        features, auxiliary = [], None
        residual_index = -1
        active = start_after is None
        for layer in self._layers():
            residual = isinstance(layer, (ResidualBlock, GatedResidualBlock, ResidualShortcut))
            if residual:
                residual_index += 1
            if not active:
                if residual and residual_index == start_after:
                    active = True
                continue
            gate = mask[:, residual_index] if mask is not None and residual else None
            value = self._run(layer, value, gradient_checkpointing=gradient_checkpointing, gate=gate)
            if residual and return_features:
                features.append(value)
            if residual and residual_index == auxiliary_feature_index:
                auxiliary = value
        if self.component_name == "encoder":
            value = self._run(self.conv1, value, gradient_checkpointing=gradient_checkpointing)
        return value, tuple(features), auxiliary

    def forward(self, value, *, return_features=False, auxiliary_feature_index=None):
        if auxiliary_feature_index is not None and not 0 <= auxiliary_feature_index < self.depth:
            raise ValueError("Auxiliary index must refer to an original residual block.")
        if value.ndim != 5 or value.shape[2] < 1:
            raise ValueError("Wan VAE inputs must be nonempty BCTHW tensors.")
        if self.component_name == "encoder":
            if value.shape[1] != 3 or (value.shape[2] - 1) % 4 or value.shape[-2] % 8 or value.shape[-1] % 8:
                raise ValueError("Wan encoder expects RGB, 4*n+1 frames, and spatial dimensions divisible by 8.")
        else:
            if value.shape[1] != self.latent_channels:
                raise ValueError("Wan decoder latent channel count does not match the checkpoint.")
            value = self._run(self.conv2, self.denormalize_latents(value), gradient_checkpointing=self.gradient_checkpointing)
        mask = self._search_mask(value.shape[0]) if self.search_config is not None else None
        prediction, features, auxiliary = self._forward_network(
            value, mask=mask, return_features=return_features, auxiliary_feature_index=auxiliary_feature_index,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        if auxiliary_feature_index is not None:
            return prediction, features, auxiliary
        return (prediction, features) if return_features else prediction

    def forward_suffix(self, feature, residual_index, *, gradient_checkpointing=True):
        if not 0 <= residual_index < self.depth:
            raise ValueError("Teacher suffix index must refer to an original residual block.")
        return self._forward_network(
            feature, start_after=residual_index, gradient_checkpointing=gradient_checkpointing,
        )[0]

    def normalize_latents(self, value):
        mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.latents_std.float().view(1, -1, 1, 1, 1)
        return (value.float() - mean) * std.reciprocal()

    def denormalize_latents(self, value):
        mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.latents_std.float().view(1, -1, 1, 1, 1)
        return value.float() / std.reciprocal() + mean

    def posterior_mode(self, moments):
        return self.normalize_latents(moments[:, :self.latent_channels])

    def initialize_from_teacher_state(self, teacher_state: Mapping[str, torch.Tensor]):
        current = self.state_dict()
        copied = {
            name: teacher_state[name] for name in current
            if name not in {"latents_mean", "latents_std", "gate_logits"} and not name.endswith(("lora_A", "lora_B"))
        }
        self.load_state_dict({**current, **copied}, strict=True)

    def initialize_from_teacher(self, teacher):
        self.initialize_from_teacher_state(teacher.state_dict())

    def fsdp_modules(self):
        return tuple(layer for layer in self._layers() if next(layer.parameters(), None) is not None)

    def save_pretrained(self, save_directory, *, state_dict=None, safe_serialization=True):
        if not safe_serialization:
            raise ValueError("Pruned Wan VAE checkpoints use safetensors.")
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / self.config_name).open("w", encoding="utf-8") as handle:
            json.dump(self.architecture_config, handle, indent=2)
            handle.write("\n")
        weights = self.state_dict() if state_dict is None else state_dict
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in weights.items()},
            str(directory / self.weights_name), metadata={"format": "pt"},
        )

    @classmethod
    def from_pretrained(cls, model_path, *, device=None, dtype=None):
        path = Path(model_path)
        with (path / cls.config_name).open("r", encoding="utf-8") as handle:
            model = cls(**json.load(handle))
        model.load_state_dict(load_file(str(path / cls.weights_name), device="cpu"), strict=True)
        return model.to(device=device, dtype=dtype)


class WanPrunedEncoder(_WanPrunedComponent):
    component_name = "encoder"
    default_keep = 3
    config_name = "pruned_encoder_config.json"
    weights_name = "wan21_pruned_encoder.safetensors"

    def encode(self, video):
        return self.posterior_mode(self(video))


class WanPrunedDecoder(_WanPrunedComponent):
    component_name = "decoder"
    default_keep = 5
    config_name = "pruned_decoder_config.json"
    weights_name = "wan21_pruned_decoder.safetensors"

    def decode(self, normalized_latents):
        return self(normalized_latents).clamp(-1.0, 1.0)


def teacher_suffix(component, feature, residual_index, *, gradient_checkpointing=True):
    return component.forward_suffix(feature, residual_index, gradient_checkpointing=gradient_checkpointing)
