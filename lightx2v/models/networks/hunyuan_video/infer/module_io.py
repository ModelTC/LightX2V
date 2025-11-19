from dataclasses import dataclass

import torch


@dataclass
class HunyuanVideo15InferModuleOutput:
    img: torch.Tensor
    txt: torch.Tensor
    vec: torch.Tensor
    grid_sizes: tuple


@dataclass
class HunyuanVideo15ImgBranchOutput:
    img_mod1_gate: torch.Tensor
    img_mod2_shift: torch.Tensor
    img_mod2_scale: torch.Tensor
    img_mod2_gate: torch.Tensor


@dataclass
class HunyuanVideo15TxtBranchOutput:
    txt_mod1_gate: torch.Tensor
    txt_mod2_shift: torch.Tensor
    txt_mod2_scale: torch.Tensor
    txt_mod2_gate: torch.Tensor
