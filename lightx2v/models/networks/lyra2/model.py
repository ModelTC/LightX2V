"""
Lyra2WanDiT – LightX2V-style wrapper for the Lyra2 diffusion transformer.

Replaces the original Lyra2WanModel (lyra_2/_src/networks/wan2pt1_lyra2.py)
with LightX2V's weight/infer separation pattern, following the same structure as
QwenImageTransformerModel (lightx2v/models/networks/qwen_image/model.py).

Layout:
  pre_weight      : Lyra2PreWeights     (patch_embed, time/text embed, img_emb, clean_patch_embeds)
  transformer_weights : Lyra2TransformerWeights  (all Lyra2AttentionBlock weights)
  post_weight     : Lyra2PostWeights    (Head: modulation + head linear)

  pre_infer       : Lyra2PreInfer       (camera extract, patchify, time/text embed)
  transformer_infer : Lyra2TransformerInfer  (N attention blocks)
  post_infer      : Lyra2PostInfer      (head modulate + unpatch)

  forward_dit(...)  ← called with same signature as Lyra2WanModel.forward

Usage in model_loader.py:
  # After loading Lyra2Model via instantiate + DCP:
  net_state = extract_net_state_dict(lyra2_model)   # strips 'net.' prefix
  lyra2_wan_dit = Lyra2WanDiT.from_config_and_state(net_config, net_state, kernel_sizes, kernel_types)
  lyra2_model.net = lyra2_wan_dit   # swap the DiT; all other methods unchanged
"""

from typing import List, Optional, Tuple

import torch
from loguru import logger

from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import GET_RECORDER_MODE, ProfilingContext4DebugL1
from lightx2v.models.networks.lyra2.infer.post_infer import Lyra2PostInfer
from lightx2v.models.networks.lyra2.infer.pre_infer import Lyra2PreInfer
from lightx2v.models.networks.lyra2.infer.transformer_infer import Lyra2TransformerInfer
from lightx2v.models.networks.lyra2.weights.post_weights import Lyra2PostWeights
from lightx2v.models.networks.lyra2.weights.pre_weights import Lyra2PreWeights
from lightx2v.models.networks.lyra2.weights.transformer_weights import Lyra2TransformerWeights


def extract_net_state_dict(lyra2_model) -> dict:
    """
    Extract the DiT (net) state dict from a fully loaded Lyra2Model,
    stripping the 'net.' prefix so keys match our weight class expectations.

    Original state dict keys are like: 'net.blocks.0.self_attn.q.weight'
    After stripping: 'blocks.0.self_attn.q.weight'

    Args:
        lyra2_model: a loaded Lyra2Model instance (from model_loader.load_model_from_checkpoint)
    Returns:
        dict mapping stripped key → tensor (all on CPU)
    """
    net = lyra2_model.net
    raw = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return raw


class Lyra2WanDiT:
    """
    LightX2V-style Lyra2 DiT.  Equivalent to Lyra2WanModel but with weights and
    inference logic separated into weight-holder classes (following qwen_image).

    The `forward_dit` method is callable with the same signature as
    Lyra2WanModel.forward so it can be used as a drop-in via:
        lyra2_model.net = lyra2_wan_dit

    Config keys (subset of Lyra2WanModel.__init__ parameters):
        dim, num_heads, num_layers, freq_dim, patch_size, out_dim,
        model_type, eps, use_plucker_condition, use_correspondence,
        buffer_pixelshuffle, buffer_in_dim, buffer_sincos_multires,
        inject_kq_only, buffer_mlp_squeeze_dim
    """

    def __init__(self, config: dict):
        self.config = config
        self.dim = config["dim"]
        self.patch_size = tuple(config.get("patch_size", [1, 2, 2]))
        self.out_dim = config.get("out_dim", 16)

        # ---- Create weight holders ----
        self.pre_weight = Lyra2PreWeights(config)
        self.transformer_weights = Lyra2TransformerWeights(config)
        self.post_weight = Lyra2PostWeights(config)

        # ---- Create infer objects ----
        self.pre_infer = Lyra2PreInfer(config)
        self.transformer_infer = Lyra2TransformerInfer(config)
        self.post_infer = Lyra2PostInfer(config)
        self._dit_forward_count = 0
        self._profile_total_steps = 0

    def reset_dit_profile_step(self, total_steps: int = 0):
        """Reset per-forward DiT step counter (call at the start of each AR chunk)."""
        self._dit_forward_count = 0
        self._profile_total_steps = int(total_steps)

    # ------------------------------------------------------------------
    # nn.Module compatibility stubs
    # Lyra2WanDiT is NOT a real nn.Module, but some callers (e.g. LoRA
    # injection, model.named_modules) expect this interface.  We provide
    # no-op stubs so those callers skip this object gracefully.
    # ------------------------------------------------------------------

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        """Yield nothing – LoRA injection will find no Linear layers here."""
        return iter([])

    def parameters(self, recurse=True):
        """No nn.Parameters – weights are stored as plain tensors."""
        return iter([])

    def init_clean_patch_embeddings(
        self,
        kernel_sizes: List[int],
        kernel_types: List[str],
    ):
        """
        Mirror of Lyra2WanModel.init_clean_patch_embeddings.
        Registers clean patch embedding weight modules (one per kernel).
        Must be called before loading weights.

        Original (wan2pt1_lyra2.py L359-405):
          Creates nn.ModuleList of nn.Linear, one per kernel size/type.
        """
        self.pre_weight.init_clean_patch_embeddings(len(kernel_sizes))
        self.pre_infer.init_clean_kernels(kernel_sizes, kernel_types)

    def load_weights(self, state_dict: dict):
        """
        Load all weights from a state dict (keys without 'net.' prefix).

        Internally calls .load(state_dict) on each WeightModule, which
        dispatches to each registered weight class's own .load() method.

        Note: state_dict is mutated (tensors removed as loaded) to free memory.
        """
        logger.info("Loading Lyra2WanDiT weights …")
        self.pre_weight.load(state_dict)
        self.transformer_weights.load(state_dict)
        self.post_weight.load(state_dict)
        logger.info("Lyra2WanDiT weights loaded.")

    @classmethod
    def from_config_and_state(
        cls,
        config: dict,
        state_dict: dict,
        kernel_sizes: List[int],
        kernel_types: List[str],
    ) -> "Lyra2WanDiT":
        """
        Construct and initialise a Lyra2WanDiT from config + pre-loaded state dict.

        Args:
            config       : dict with Lyra2WanModel hyperparameters
            state_dict   : net.* state dict (with 'net.' prefix stripped)
            kernel_sizes : list of clean patch kernel sizes (e.g. [2, 4])
            kernel_types : list of kernel types per kernel ('k' or 's')
        """
        dit = cls(config)
        dit.init_clean_patch_embeddings(kernel_sizes, kernel_types)
        dit.load_weights(state_dict)
        return dit

    # ------------------------------------------------------------------
    # DiT forward  (replaces Lyra2WanModel.forward, framepack path only)
    # ------------------------------------------------------------------

    def forward_dit(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,          # [B, 1]
        crossattn_emb: torch.Tensor,
        frame_cond_crossattn_emb_B_L_D: Optional[torch.Tensor] = None,
        y_B_C_T_H_W: Optional[torch.Tensor] = None,
        y_buffer_B_C_T_H_W: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        framepack_indices: Optional[torch.Tensor] = None,
        framepack_splits: Optional[List[int]] = None,
        framepack_kernel_ids: Optional[List[int]] = None,
        framepack_kernel_types: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full DiT forward pass (framepack path only – matches ZoomGS inference).

        Corresponds to Lyra2WanModel.forward(use_framepack=True) L900-1018.

        Original signature (wan2pt1_lyra2.py L806-819):
          def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                      seq_len=None, frame_cond_crossattn_emb_B_L_D=None,
                      y_B_C_T_H_W=None, y_buffer_B_C_T_H_W=None,
                      padding_mask=None, is_uncond=False, slg_layers=None,
                      **kwargs)
        """
        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]   # [B]

        self._dit_forward_count += 1
        step_index = self._dit_forward_count
        total_steps = self._profile_total_steps or step_index

        # Optional latent conditioning concatenation (L827-828)
        # Original: if y_B_C_T_H_W is not None: x = cat([x, y], dim=1)
        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        with ProfilingContext4DebugL1(
            "Run Dit every step",
            recorder_mode=GET_RECORDER_MODE(),
            metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
            metrics_labels=[step_index, total_steps],
        ):
            logger.info(f"==> step_index: {step_index} / {total_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                pre_out = self.pre_infer.infer(
                    weights=self.pre_weight,
                    x_B_C_T_H_W=x_B_C_T_H_W,
                    t_B=t_B,
                    crossattn_emb=crossattn_emb,
                    framepack_indices=framepack_indices,
                    framepack_splits=framepack_splits,
                    framepack_kernel_ids=framepack_kernel_ids,
                    framepack_kernel_types=framepack_kernel_types,
                    y_buffer_B_C_T_H_W=y_buffer_B_C_T_H_W,
                    frame_cond_crossattn_emb=frame_cond_crossattn_emb_B_L_D,
                    padding_mask=padding_mask,
                )

            with ProfilingContext4DebugL1("🚀 infer_main"):
                x_B_L_D = self.transformer_infer.infer(
                    block_weights=self.transformer_weights,
                    pre_infer_out=pre_out,
                )

            with ProfilingContext4DebugL1("step_post"):
                x_B_C_T_H_W_out = self.post_infer.infer(
                    weights=self.post_weight,
                    x_B_L_D=x_B_L_D,
                    pre_infer_out=pre_out,
                )
        return x_B_C_T_H_W_out

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Allow using the DiT as a callable, forwarding to forward_dit."""
        return self.forward_dit(*args, **kwargs)

    # ------------------------------------------------------------------
    # CPU / GPU offloading helpers (mirrors WeightModule pattern)
    # ------------------------------------------------------------------

    def to_cuda(self, non_blocking: bool = True):
        self.pre_weight.to_cuda(non_blocking)
        self.transformer_weights.to_cuda(non_blocking)
        self.post_weight.to_cuda(non_blocking)

    def to_cpu(self, non_blocking: bool = True):
        self.pre_weight.to_cpu(non_blocking)
        self.transformer_weights.to_cpu(non_blocking)
        self.post_weight.to_cpu(non_blocking)
