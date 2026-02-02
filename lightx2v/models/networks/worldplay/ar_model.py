import glob
import os

import torch
from loguru import logger

from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.networks.worldplay.infer.ar_pre_infer import WorldPlayARPreInfer
from lightx2v.models.networks.worldplay.infer.ar_transformer_infer import WorldPlayARTransformerInfer
from lightx2v.models.networks.worldplay.infer.post_infer import WorldPlayPostInfer
from lightx2v.models.networks.worldplay.weights.post_weights import WorldPlayPostWeights
from lightx2v.models.networks.worldplay.weights.pre_weights import WorldPlayPreWeights
from lightx2v.models.networks.worldplay.weights.transformer_weights import WorldPlayTransformerWeights
from lightx2v.utils.envs import *


class WorldPlayARModel(HunyuanVideo15Model):
    """
    WorldPlay AR (Autoregressive) model with action conditioning and ProPE support.

    Extends HunyuanVideo15Model with:
    - Action conditioning via action_in embedder
    - ProPE (Projective Positional Encoding) for camera pose conditioning
    - KV Cache for autoregressive generation
    - Causal attention mechanism
    - Support for loading separate action model checkpoint

    Key differences from WorldPlayModel (Distill):
    - Uses causal attention instead of bidirectional
    - Implements KV cache for frame-by-frame generation
    - No guidance embedding required
    """

    def __init__(self, model_path, config, device, action_ckpt=None):
        self.action_ckpt = action_ckpt
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        """Initialize WorldPlay AR-specific inference classes."""
        self.pre_infer_class = WorldPlayARPreInfer
        self.post_infer_class = WorldPlayPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WorldPlayARTransformerInfer
        else:
            raise NotImplementedError(f"Feature caching {self.config['feature_caching']} not supported for AR model. AR model requires NoCaching due to KV cache management.")

    def _init_weights(self):
        """Initialize weights including action conditioning weights."""
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        sensitive_layer = {}

        if not self.dit_quantized:
            weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
        else:
            weight_dict = self._load_quant_ckpt(unified_dtype, sensitive_layer)

        # Load action model weights if provided
        if self.action_ckpt is not None:
            action_weight_dict = self._load_action_ckpt(unified_dtype, sensitive_layer)
            weight_dict.update(action_weight_dict)

        self.original_weight_dict = weight_dict
        self.pre_weight = WorldPlayPreWeights(self.config)
        self.transformer_weights = WorldPlayTransformerWeights(self.config)
        self.post_weight = WorldPlayPostWeights(self.config)
        self._apply_weights()

    def _load_action_ckpt(self, unified_dtype, sensitive_layer):
        """Load action model checkpoint."""
        action_ckpt = self.action_ckpt

        if os.path.isdir(action_ckpt):
            safetensors_files = glob.glob(os.path.join(action_ckpt, "*.safetensors"))
        else:
            safetensors_files = [action_ckpt]

        weight_dict = {}
        for file_path in safetensors_files:
            logger.info(f"Loading action weights from {file_path}")
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _init_infer(self):
        """Initialize inference modules and connect action weights."""
        super()._init_infer()

        # Connect action weights to transformer for ProPE projection
        if hasattr(self.pre_weight, "action_weights") and hasattr(self.transformer_infer, "set_action_weights"):
            self.transformer_infer.set_action_weights(self.pre_weight.action_weights)

    def set_scheduler(self, scheduler):
        """Set scheduler and connect to inference modules."""
        super().set_scheduler(scheduler)

    def init_kv_cache(self):
        """
        Initialize KV cache for autoregressive generation.
        Structure matches HY-WorldPlay original implementation.
        """
        if hasattr(self.transformer_infer, "init_kv_cache"):
            return self.transformer_infer.init_kv_cache()
        return None

    def clear_kv_cache(self):
        """Clear KV cache after generation."""
        if hasattr(self.transformer_infer, "clear_kv_cache"):
            self.transformer_infer.clear_kv_cache()

    def clear_vision_cache(self):
        """Clear only vision cache (keep text cache for next chunk)."""
        if hasattr(self.transformer_infer, "clear_vision_cache"):
            self.transformer_infer.clear_vision_cache()

    # ========== AR-specific inference methods ==========

    @torch.no_grad()
    def infer_txt(self, inputs, cache_txt=True):
        """
        Cache text KV, called once at the beginning of generation.
        Corresponds to original forward_txt().

        Args:
            inputs: Dict containing text_encoder_output, image_encoder_output
            cache_txt: Whether to cache text KV (default True)

        Returns:
            KV cache reference
        """
        # Initialize KV cache if not already done
        if not hasattr(self.transformer_infer, "_kv_cache") or self.transformer_infer._kv_cache is None:
            self.init_kv_cache()

        # Run text-only pre-processing
        infer_module_out = self.pre_infer.infer_txt_only(self.pre_weight, inputs)

        # Cache text KV
        return self.transformer_infer.infer_txt(self.transformer_weights, infer_module_out, cache_txt=cache_txt)

    @torch.no_grad()
    def infer_vision(self, inputs, cache_vision=False):
        """
        Vision inference using cached text KV.
        Corresponds to original forward_vision().

        Args:
            inputs: Dict containing encoder outputs and pose data
            cache_vision: Whether to cache vision KV for context frames

        Returns:
            If cache_vision=True: KV cache reference
            If cache_vision=False: Noise prediction output
        """
        # Store pose data in scheduler if provided
        if "pose_output" in inputs and inputs["pose_output"] is not None:
            pose_output = inputs["pose_output"]
            if "viewmats" in pose_output:
                self.scheduler.viewmats = pose_output["viewmats"]
            if "Ks" in pose_output:
                self.scheduler.Ks = pose_output["Ks"]
            if "action" in pose_output:
                self.scheduler.action = pose_output["action"]

        # Run pre-inference (full, including image)
        infer_module_out = self.pre_infer.infer(self.pre_weight, inputs)

        # Vision inference with KV cache
        output = self.transformer_infer.infer_vision(self.transformer_weights, infer_module_out, cache_vision=cache_vision)

        if cache_vision:
            return output  # Return KV cache
        else:
            # Run post-inference
            return self.post_infer.infer(self.post_weight, output)

    @torch.no_grad()
    def infer(self, inputs):
        """
        Run inference with action and camera pose conditioning.

        Args:
            inputs: Dict containing:
                - text_encoder_output: Text encoder outputs
                - image_encoder_output: Image encoder outputs
                - pose_output (optional): Dict with viewmats, Ks, action
        """
        # Store pose data in scheduler if provided
        if "pose_output" in inputs and inputs["pose_output"] is not None:
            pose_output = inputs["pose_output"]
            if "viewmats" in pose_output:
                self.scheduler.viewmats = pose_output["viewmats"]
            if "Ks" in pose_output:
                self.scheduler.Ks = pose_output["Ks"]
            if "action" in pose_output:
                self.scheduler.action = pose_output["action"]

        # Call parent inference
        super().infer(inputs)

    @torch.no_grad()
    def infer_chunk(self, inputs, chunk_idx, total_chunks):
        """
        Run inference for a single chunk in autoregressive generation.

        Args:
            inputs: Dict containing encoder outputs and pose data
            chunk_idx: Current chunk index (0-indexed)
            total_chunks: Total number of chunks

        Returns:
            Latent tensor for this chunk
        """
        # Store chunk info in scheduler
        self.scheduler.chunk_idx = chunk_idx
        self.scheduler.total_chunks = total_chunks

        # Store pose data if provided
        if "pose_output" in inputs and inputs["pose_output"] is not None:
            pose_output = inputs["pose_output"]
            if "viewmats" in pose_output:
                self.scheduler.viewmats = pose_output["viewmats"]
            if "Ks" in pose_output:
                self.scheduler.Ks = pose_output["Ks"]
            if "action" in pose_output:
                self.scheduler.action = pose_output["action"]

        # Run pre-inference
        infer_module_out = self.pre_infer.infer(self.pre_weight, inputs)

        # Run transformer with KV cache
        x = self.transformer_infer.infer(self.transformer_weights, infer_module_out)

        # Run post-inference
        output = self.post_infer.infer(self.post_weight, x)

        return output
