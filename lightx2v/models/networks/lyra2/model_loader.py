# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
model_loader.py  –  load Lyra2Model from a DCP checkpoint and swap its DiT (net)
with a LightX2V-style Lyra2WanDiT.

Flow:
  1. Instantiate Lyra2Model via lazy_config + DCP load.
  2. Extract net.* state dict → load into Lyra2WanDiT weight classes.
  3. Replace model.net with Lyra2WanDiT so all runner calls transparently
     use the new pre/transformer/post inference structure.
"""

import importlib
import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from lightx2v.models.networks.lyra2._imaginaire.lazy_config import instantiate
from lightx2v.models.networks.lyra2._imaginaire.utils.config_helper import get_config_module, override
from lightx2v.models.networks.lyra2.lyra2_utils import set_random_seed, timer
from loguru import logger

# ---- DCP planner -------------------------------------------------------
# Original comment: Prefer Lyra-2's custom DefaultLoadPlanner which handles
# _extra_state size mismatches (Flash-Attention operator state [4] vs [0]).
# Fall back to PyTorch's planner if lyra_2 is not importable.
try:
    from lyra_2._ext.imaginaire.checkpointer.dcp import DefaultLoadPlanner
except Exception:
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def load_model_from_checkpoint(
    experiment_name,
    checkpoint_path,
    config_file="lyra_2/_src/configs/t2v_wan/config.py",
    enable_fsdp=False,
    instantiate_ema=True,
    load_ema_to_reg=False,
    seed=0,
    experiment_opts: list[str] = [],
    strict=True,
):
    """
    Load Lyra2Model from a DCP (or .pth) checkpoint, then replace model.net
    with a LightX2V-style Lyra2WanDiT.

    Args:
        experiment_name  : experiment name string
        checkpoint_path  : path to DCP directory or .pth file
        config_file      : config file path relative to lyra_repo
        enable_fsdp      : enable FSDP sharding (inference: False)
        instantiate_ema  : whether to instantiate EMA model
        load_ema_to_reg  : load EMA weights into regular net
        seed             : random seed
        experiment_opts  : extra config overrides
        strict           : strict state_dict loading
    """
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if instantiate_ema is False and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    config.validate()
    config.freeze()
    set_random_seed(seed=seed, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    if not enable_fsdp:
        config.model.config.fsdp_shard_size = 1

    with timer("instantiate model"):
        model = instantiate(config.model).cuda()
        model.on_train_start()

    # ---- Load checkpoint weights ----
    if checkpoint_path.endswith(".pth"):
        logger.info(f"Loading model from consolidated checkpoint {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cuda", weights_only=False),
            strict=strict,
        )
    else:
        logger.info(f"Loading model from DCP checkpoint {checkpoint_path}")
        cur_key_ckpt_full_path = os.path.join(checkpoint_path, "model")
        storage_reader = FileSystemReader(cur_key_ckpt_full_path)
        _state_dict = _load_model_state_dict_for_dcp(model, load_ema_to_reg=load_ema_to_reg)
        dcp.load(
            _state_dict,
            storage_reader=storage_reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        _apply_model_state_dict(model, _state_dict, load_ema_to_reg=load_ema_to_reg)

    torch.cuda.empty_cache()

    # Replace model.net (Lyra2WanModel) with the LightX2V-style Lyra2WanDiT.
    # Lyra2WanDiT provides a named_modules() no-op stub so downstream LoRA
    # injection iterates over it without crashing (LoRA is simply not applied).
    _swap_net_with_lightx2v_dit(model)

    return model, config


def _swap_net_with_lightx2v_dit(model):
    """
    Replace model.net (Lyra2WanModel) with a Lyra2WanDiT that holds the same
    weights using LightX2V's weight/infer separation.

    Steps:
      1. Build a config dict from model.net's attributes.
      2. Extract net state dict to CPU.
      3. Free original net from GPU to avoid OOM.
      4. Instantiate Lyra2WanDiT from config + state dict.
      5. Assign model.net = lyra2_wan_dit via object.__setattr__ (bypassing
         nn.Module's type check since Lyra2WanDiT is not an nn.Module).

    After this, any call to model.net(x_B_C_T_H_W, timesteps, ...) will
    transparently route through Lyra2WanDiT.forward_dit().
    """
    from lightx2v.models.networks.lyra2.model import Lyra2WanDiT

    net = model.net
    logger.info("Swapping Lyra2WanModel → Lyra2WanDiT …")

    # 1. Build config dict from the original net's __init__ parameters
    net_config = _extract_net_config(model)

    # 2. Extract state dict (keys already relative to net, e.g. 'blocks.0.self_attn.q.weight')
    logger.info("  Extracting net state dict …")
    net_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    # 3. Free the original Lyra2WanModel from GPU memory BEFORE building Lyra2WanDiT.
    #    Without this step, the original net stays registered in model._modules["net"] and
    #    continues to occupy GPU memory, causing OOM when Lyra2WanDiT is moved to CUDA.
    logger.info("  Freeing original Lyra2WanModel from GPU …")
    original_net = model._modules.pop("net", None)
    if original_net is not None:
        original_net.cpu()
        del original_net
    import torch as _torch
    _torch.cuda.empty_cache()
    logger.info("  GPU memory freed.")

    # 4. Get kernel_sizes / kernel_types from the parent Lyra2Model
    kernel_sizes = list(getattr(model, "framepack_clean_latent_frame_kernel_sizes", []))
    kernel_types = list(getattr(model, "framepack_clean_latent_frame_kernel_types", []))
    if not kernel_sizes:
        logger.warning("  framepack_clean_latent_frame_kernel_sizes not found on model; "
                       "clean_patch_embeddings will not be registered.")

    # 5. Build and load Lyra2WanDiT (weights on CPU pinned memory)
    lyra2_wan_dit = Lyra2WanDiT.from_config_and_state(
        config=net_config,
        state_dict=net_state,
        kernel_sizes=kernel_sizes,
        kernel_types=kernel_types,
    )

    # 6. Replace model.net
    # model is an nn.Module, so direct assignment goes through torch.nn.Module.__setattr__
    # which only accepts nn.Module (or None) children.  Lyra2WanDiT is a plain Python class
    # (no nn.Module), so we bypass PyTorch's check via object.__setattr__.
    # PyTorch's parameter/buffer tracking won't see the new net, but Lyra2WanDiT manages
    # its own weights independently, so this is safe for inference.
    object.__setattr__(model, "net", lyra2_wan_dit)
    logger.info("  model.net swapped to Lyra2WanDiT ✓")


def _extract_net_config(model) -> dict:
    """
    Build a config dict for Lyra2WanDiT from the loaded Lyra2Model's net attributes.

    Original Lyra2WanModel attributes set in __init__ (wan2pt1_lyra2.py L257-287):
      model_type, patch_size, freq_dim, dim, ffn_dim, out_dim, num_heads, num_layers,
      eps, use_plucker_condition, use_correspondence, buffer_pixelshuffle,
      buffer_in_dim, buffer_sincos_multires, inject_kq_only, buffer_mlp_squeeze_dim
    """
    net = model.net
    cfg = dict(
        dim=net.dim,
        num_heads=net.num_heads,
        num_layers=net.num_layers,
        freq_dim=getattr(net, "freq_dim", 256),
        patch_size=list(net.patch_size),
        out_dim=getattr(net, "out_dim", 16),
        model_type=getattr(net, "model_type", "i2v"),
        eps=getattr(net, "eps", 1e-6),
        use_plucker_condition=getattr(net, "use_plucker_condition", True),
        use_correspondence=getattr(net, "use_correspondence", True),
        buffer_pixelshuffle=getattr(net, "buffer_pixelshuffle", True),
        buffer_in_dim=getattr(net, "buffer_in_dim", 0),
        buffer_sincos_multires=getattr(net, "buffer_sincos_multires", 2),
        inject_kq_only=getattr(net, "inject_kq_only", True),
        buffer_mlp_squeeze_dim=getattr(net, "buffer_mlp_squeeze_dim", 256),
        cross_attn_norm=getattr(net, "cross_attn_norm", False),
        rms_norm_type="torch",
    )
    return cfg


# ---------------------------------------------------------------------------
# Helpers replacing ModelWrapper from lyra_2._ext.imaginaire.checkpointer.dcp
# (unchanged from previous version – kept for the DCP load path)
# ---------------------------------------------------------------------------

def _load_model_state_dict_for_dcp(model, load_ema_to_reg: bool = False):
    """Return a state-dict that mirrors ModelWrapper.state_dict() for the DCP load path."""
    _state_dict = get_model_state_dict(model)
    if load_ema_to_reg:
        # EMA path: rename net_ema.* → net.*
        all_keys = list(_state_dict.keys())
        for k in all_keys:
            _state_dict[k.replace("net_ema.", "net.")] = _state_dict.pop(k)
    # LoRA key remapping: base_layer.weight → weight etc.
    if hasattr(model, "config") and hasattr(model.config, "lora_config") and model.config.lora_config.enabled:
        mapping = {"base_layer.": "", "base_model.model.": ""}
        keys_to_update = []
        for k in list(_state_dict.keys()):
            new_k = k
            for from_k, to_k in mapping.items():
                new_k = new_k.replace(from_k, to_k)
            if new_k != k:
                keys_to_update.append((k, new_k))
        for k, new_k in keys_to_update:
            _state_dict[new_k] = _state_dict.pop(k)
    return _state_dict


def _apply_model_state_dict(model, state_dict, load_ema_to_reg: bool = False):
    """Apply the DCP-loaded state_dict back, mirroring ModelWrapper.load_state_dict()."""
    if load_ema_to_reg and not getattr(getattr(model, "config", None), "ema", None):
        all_keys = list(state_dict.keys())
        for k in all_keys:
            if k.startswith("net_ema."):
                state_dict[k.replace("net_ema.", "net.")] = torch.clone(state_dict[k])
    set_model_state_dict(model, state_dict, options=StateDictOptions(strict=False))
