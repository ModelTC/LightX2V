import glob
import os

import torch
from loguru import logger

from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.networks.worldplay.infer.post_infer import WorldPlayPostInfer
from lightx2v.models.networks.worldplay.infer.pre_infer import WorldPlayPreInfer
from lightx2v.models.networks.worldplay.infer.transformer_infer import WorldPlayTransformerInfer
from lightx2v.models.networks.worldplay.weights.post_weights import WorldPlayPostWeights
from lightx2v.models.networks.worldplay.weights.pre_weights import WorldPlayPreWeights
from lightx2v.models.networks.worldplay.weights.transformer_weights import WorldPlayTransformerWeights
from lightx2v.utils.envs import *


class WorldPlayModel(HunyuanVideo15Model):
    """
    WorldPlay model with action conditioning and ProPE support.

    Extends HunyuanVideo15Model with:
    - Action conditioning via action_in embedder
    - ProPE (Projective Positional Encoding) for camera pose conditioning
    - Support for loading separate action model checkpoint
    """

    def __init__(self, model_path, config, device, action_ckpt=None):
        self.action_ckpt = action_ckpt
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        """Initialize WorldPlay-specific inference classes."""
        self.pre_infer_class = WorldPlayPreInfer
        self.post_infer_class = WorldPlayPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WorldPlayTransformerInfer
        else:
            # Fall back to base transformer for caching modes
            from lightx2v.models.networks.hunyuan_video.infer.feature_caching.transformer_infer import (
                HunyuanTransformerInferTeaCaching,
                HunyuanVideo15TransformerInferMagCaching,
            )

            if self.config["feature_caching"] == "Mag":
                self.transformer_infer_class = HunyuanVideo15TransformerInferMagCaching
            elif self.config["feature_caching"] == "Tea":
                self.transformer_infer_class = HunyuanTransformerInferTeaCaching
            else:
                raise NotImplementedError(f"Feature caching {self.config['feature_caching']} not supported")

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
