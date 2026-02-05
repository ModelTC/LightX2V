import glob
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.networks.worldplay.infer.bi_transformer_infer import WorldPlayBITransformerInfer
from lightx2v.models.networks.worldplay.infer.post_infer import WorldPlayPostInfer
from lightx2v.models.networks.worldplay.infer.pre_infer import WorldPlayPreInfer
from lightx2v.models.networks.worldplay.weights.post_weights import WorldPlayPostWeights
from lightx2v.models.networks.worldplay.weights.pre_weights import WorldPlayPreWeights
from lightx2v.models.networks.worldplay.weights.transformer_weights import WorldPlayTransformerWeights
from lightx2v.utils.envs import *


class WorldPlayBIModel(HunyuanVideo15Model):
    """
    WorldPlay BI (Bidirectional) model with action conditioning and ProPE support.

    Key differences from AR model:
    - Uses bidirectional attention (not causal)
    - No KV cache required
    - Standard 50-step inference
    - Supports classifier-free guidance

    Key differences from Distill model:
    - Standard 50-step inference (not 4-step)
    - Uses guidance_scale for CFG

    Extends HunyuanVideo15Model with:
    - Action conditioning via action_in embedder
    - ProPE (Projective Positional Encoding) for camera pose conditioning
    - Support for loading separate action model checkpoint
    """

    def __init__(self, model_path, config, device, action_ckpt=None):
        self.action_ckpt = action_ckpt
        # BI model needs byt5_in and vision_in weights, so don't remove them
        # This must be set before calling super().__init__() which sets remove_keys
        self._bi_model_keep_all_keys = config.get("use_bi_model_as_main", False)
        super().__init__(model_path, config, device)
        # Override remove_keys if using bi model as main (it has all weights)
        if self._bi_model_keep_all_keys:
            self.remove_keys = []

    def _init_infer_class(self):
        """Initialize WorldPlay BI-specific inference classes."""
        self.pre_infer_class = WorldPlayPreInfer
        self.post_infer_class = WorldPlayPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WorldPlayBITransformerInfer
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
        """Initialize weights including action conditioning weights.

        For BI model, the action_ckpt contains the COMPLETE model weights
        (not just action-related weights). When use_bi_model_as_main is True,
        we load directly from action_ckpt instead of merging with base model.
        """
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        sensitive_layer = {}

        use_bi_model_as_main = self.config.get("use_bi_model_as_main", False)

        if use_bi_model_as_main and self.action_ckpt is not None:
            # BI model: action_ckpt contains complete model weights
            logger.info("Loading BI model weights directly from action_ckpt (complete model)")
            weight_dict = self._load_action_ckpt(unified_dtype, sensitive_layer)
        else:
            # Legacy mode: load base model and merge action weights
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
        """Load action model checkpoint.

        For BI model with use_bi_model_as_main=True, this loads the complete model
        including byt5_in and vision_in weights.
        """
        action_ckpt = self.action_ckpt

        if os.path.isdir(action_ckpt):
            safetensors_files = glob.glob(os.path.join(action_ckpt, "*.safetensors"))
        else:
            safetensors_files = [action_ckpt]

        weight_dict = {}
        for file_path in safetensors_files:
            logger.info(f"Loading action weights from {file_path}")
            # Use _load_safetensor_to_dict_no_filter to keep all keys for BI model
            file_weights = self._load_safetensor_to_dict_no_filter(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _load_safetensor_to_dict_no_filter(self, file_path, unified_dtype, sensitive_layer):
        """Load safetensor without filtering any keys (for BI model complete weights)."""
        if self.device.type != "cpu" and dist.is_initialized():
            device = dist.get_rank()
        else:
            device = str(self.device)

        with safe_open(file_path, framework="pt", device=device) as f:
            return {key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE())) for key in f.keys()}

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

    @torch.no_grad()
    def infer_bi(self, inputs, context_inputs=None):
        """
        Run bidirectional inference for a chunk.

        This is the main inference method for BI model, supporting:
        - Context frame concatenation for non-first chunks
        - Per-token timestep for context vs current frames
        - Classifier-free guidance

        Args:
            inputs: Dict containing encoder outputs and current chunk data
            context_inputs: Optional dict with context frame data for non-first chunks

        Returns:
            Noise prediction output
        """
        # Note: pose data (viewmats, Ks, action) should already be set in scheduler
        # by the runner before calling this method. Don't override here.

        # Run pre-inference
        infer_module_out = self.pre_infer.infer(self.pre_weight, inputs)

        # If context inputs provided, merge them
        if context_inputs is not None:
            infer_module_out = self._merge_context(infer_module_out, context_inputs)

        # Run transformer inference
        x = self.transformer_infer.infer(self.transformer_weights, infer_module_out)

        # Run post-inference (needs weights and infer_module_out for grid_sizes)
        output = self.post_infer.infer(x, infer_module_out)

        return output

    def _merge_context(self, infer_module_out, context_inputs):
        """
        Merge context frame data with current chunk data.

        For BI model, context frames are concatenated with current frames
        and processed together with bidirectional attention.

        Args:
            infer_module_out: Current chunk inference module output
            context_inputs: Context frame data

        Returns:
            Merged inference module output
        """
        # This will be implemented in the transformer_infer to handle
        # the concatenation of context and current frames
        infer_module_out.context_inputs = context_inputs
        return infer_module_out
