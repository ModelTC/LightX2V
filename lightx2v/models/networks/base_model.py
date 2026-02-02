"""
Base model class for all transformer models in the networks module.

This base class provides common functionality for:
- Weight loading (original and quantized)
- CPU offloading
- Lazy loading
- Distributed weight loading
- Inference setup
"""

import gc
import glob
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.utils.custom_compiler import CompiledMethodsMixin, compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.ggml_tensor import load_gguf_sd_ckpt
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE


class BaseTransformerModel(CompiledMethodsMixin, ABC):
    """Base class for all transformer models.

    This class provides common functionality that can be shared across
    different model implementations (z_image, qwen_image, ltx2, etc.).

    Subclasses should define:
    - pre_weight_class: Class for pre-inference weights
    - transformer_weight_class: Class for transformer weights
    - post_weight_class: Class for post-inference weights (if applicable)
    """

    # These should be overridden by subclasses
    pre_weight_class = None
    transformer_weight_class = None
    post_weight_class = None

    def __init__(self, model_path, config, device, model_type=None, lora_path=None, lora_strength=1.0):
        """Initialize the base transformer model.

        Args:
            model_path: Path to model directory
            config: Configuration dictionary
            device: Device to use
            model_type: Model type (optional)
            lora_path: Path to LoRA weights (optional)
            lora_strength: LoRA strength (optional)
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_strength = lora_strength
        self.model_type = model_type
        self.remove_keys = []
        self.sensitive_layer = {}

        self.config = config
        self.cpu_offload = self.config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.lazy_load = self.config.get("lazy_load", False)
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.dit_quantized = self.config.get("dit_quantized", False)
        if self.dit_quantized:
            self._check_dit_quantized()

    def _check_dit_quantized(self):
        """Check if the model is quantized.

        Returns:
            bool: True if the model is quantized
        """
        assert self.config.get("dit_quant_scheme", "Default") in [
            "fp8-pertensor",
            "fp8-triton",
            "int8-triton",
            "fp8-vllm",
            "int8-vllm",
            "int8-vllm-hygon-dcu",
            "fp8-q8f",
            "int8-q8f",
            "fp8-b128-deepgemm",
            "fp8-sgl",
            "int8-sgl",
            "int8-torchao",
            "fp8-torchao",
            "nvfp4",
            "mxfp4",
            "mxfp6-mxfp8",
            "mxfp8",
            "int8-tmo",
            "gguf-Q8_0",
            "gguf-Q6_K",
            "gguf-Q5_K_S",
            "gguf-Q5_K_M",
            "gguf-Q5_0",
            "gguf-Q5_1",
            "gguf-Q4_K_S",
            "gguf-Q4_K_M",
            "gguf-Q4_0",
            "gguf-Q4_1",
            "gguf-Q3_K_S",
            "gguf-Q3_K_M",
            "int8-npu",
        ]

    @abstractmethod
    def _init_infer_class(self):
        """Initialize inference class references.

        Subclasses should override this to set their specific inference classes.
        """
        pass

    def _init_weights(self, weight_dict=None):
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        # Some layers run with float32 to achieve high accuracy
        sensitive_layer = self.sensitive_layer
        if weight_dict is None:
            is_weight_loader = self._should_load_weights()
            if is_weight_loader:
                if not self.dit_quantized:
                    # Load original weights
                    weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
                else:
                    # Load quantized weights
                    weight_dict = self._load_quant_ckpt(unified_dtype, sensitive_layer)

            if (self.config.get("device_mesh") is not None and self.config.get("load_from_rank0", False)) or (hasattr(self, "use_tp") and self.use_tp):
                weight_dict = self._load_weights_from_rank0(weight_dict, is_weight_loader)

            if hasattr(self, "_load_adapter_ckpt"):
                weight_dict.update(self._load_adapter_ckpt())

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        if self.lazy_load:
            self.transformer_weights = self.transformer_weight_class(self.config, self.lazy_load_path, self.lora_path)
        else:
            self.transformer_weights = self.transformer_weight_class(self.config)
        if hasattr(self, "post_weight_class") and self.post_weight_class is not None:
            self.post_weight = self.post_weight_class(self.config)

        if not self._should_init_empty_model():
            self._apply_weights()

    @abstractmethod
    def _init_infer(self):
        """Initialize inference modules.

        Subclasses should override this to set up their specific inference modules.
        """
        pass

    def _init_offload_manager(self):
        self.transformer_infer.offload_manager.init_cuda_buffer(self.transformer_weights.offload_block_cuda_buffers, self.transformer_weights.offload_phase_cuda_buffers)
        if self.lazy_load:
            self.transformer_infer.offload_manager.init_cpu_buffer(self.transformer_weights.offload_block_cpu_buffers, self.transformer_weights.offload_phase_cpu_buffers)

    def _should_init_empty_model(self):
        """Determine if model should be initialized empty (for LoRA).

        Returns:
            bool: True if model should be initialized empty
        """
        if self.config.get("lora_configs") and self.config["lora_configs"] and not self.config.get("lora_dynamic_apply", False):
            return True
        return False

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk.

        Returns:
            bool: True if current rank should load weights
        """
        if self.config.get("device_mesh") is None:
            # Single GPU mode
            return True
        elif dist.is_initialized():
            if self.config.get("load_from_rank0", False):
                # Multi-GPU mode, only rank 0 loads
                if dist.get_rank() == 0:
                    logger.info(f"Loading weights from {self.model_path}")
                    return True
            else:
                return True
        return False

    def _apply_weights(self, weight_dict=None):
        """Apply weights to weight containers.

        Args:
            weight_dict: Optional weight dictionary to apply
        """
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()

        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)
        if hasattr(self, "post_weight"):
            self.post_weight.load(self.original_weight_dict)

        # Handle LoRA if needed
        if self.config.get("lora_dynamic_apply", False):
            assert self.config.get("lora_configs", False)
            if hasattr(self, "_register_lora"):
                self._register_lora(self.lora_path, self.lora_strength)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()

    def _load_lora_file(self, file_path):
        if self.device.type != "cpu" and dist.is_initialized():
            device = f"{AI_DEVICE}:{dist.get_rank()}"
        else:
            device = str(self.device)
        if device == "cpu":
            with safe_open(file_path, framework="pt", device=device) as f:
                tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()).pin_memory() for key in f.keys()}
        else:
            with safe_open(file_path, framework="pt", device=device) as f:
                tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()) for key in f.keys()}
        return tensor_dict

    def _register_lora(self, lora_path, strength):
        lora_weight = self._load_lora_file(lora_path)
        self.pre_weight.register_lora(lora_weight, strength)
        self.transformer_weights.register_lora(lora_weight, strength)
        self.pre_weight.register_diff(lora_weight)
        self.transformer_weights.register_diff(lora_weight)

    def _update_lora(self, lora_path, strength):
        lora_weight = self._load_lora_file(lora_path)
        self.pre_weight.update_lora(lora_weight, strength)
        self.transformer_weights.update_lora(lora_weight, strength)
        self.post_weight.update_lora(lora_weight, strength)

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load a safetensors file into a dictionary.

        Args:
            file_path: Path to safetensors file
            unified_dtype: Whether to use unified dtype
            sensitive_layer: Dictionary of sensitive layer patterns

        Returns:
            dict: Dictionary of tensors
        """
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.device.type != "cpu" and dist.is_initialized():
            device = dist.get_rank()
        else:
            device = str(self.device)

        with safe_open(file_path, framework="pt", device=device) as f:
            return {
                key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE()))
                for key in f.keys()
                if not any(remove_key in key for remove_key in remove_keys)
            }

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        """Load checkpoint weights.

        Args:
            unified_dtype: Whether to use unified dtype
            sensitive_layer: Dictionary of sensitive layer patterns

        Returns:
            dict: Dictionary of weights
        """
        if self.config.get("dit_original_ckpt", None):
            safetensors_path = self.config["dit_original_ckpt"]
        else:
            safetensors_path = self.model_path

        if os.path.isdir(safetensors_path):
            if self.lazy_load:
                self.lazy_load_path = safetensors_path
                non_block_file = os.path.join(safetensors_path, "non_block.safetensors")
                if os.path.exists(non_block_file):
                    safetensors_files = [non_block_file]
                else:
                    raise ValueError(f"Non-block file not found in {safetensors_path}. Please check the model path.")
            else:
                safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            if self.lazy_load:
                self.lazy_load_path = safetensors_path
            safetensors_files = [safetensors_path]

        weight_dict = {}
        for file_path in safetensors_files:
            if self.config.get("adapter_model_path", None) is not None:
                if self.config["adapter_model_path"] == file_path:
                    continue
            logger.info(f"Loading weights from {file_path}")
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        """Load quantized checkpoint weights.

        Args:
            unified_dtype: Whether to use unified dtype
            sensitive_layer: Dictionary of sensitive layer patterns

        Returns:
            dict: Dictionary of weights
        """
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.config.get("dit_quantized_ckpt", None):
            safetensors_path = self.config["dit_quantized_ckpt"]
        else:
            safetensors_path = self.model_path

        # Handle GGUF format
        if "gguf" in self.config.get("dit_quant_scheme", ""):
            gguf_path = ""
            if os.path.isdir(safetensors_path):
                gguf_type = self.config.get("dit_quant_scheme").replace("gguf-", "")
                gguf_files = list(filter(lambda x: gguf_type in x, glob.glob(os.path.join(safetensors_path, "*.gguf"))))
                gguf_path = gguf_files[0]
            else:
                gguf_path = safetensors_path
            weight_dict = self._load_gguf_ckpt(gguf_path)
            return weight_dict

        if os.path.isdir(safetensors_path):
            if self.lazy_load:
                self.lazy_load_path = safetensors_path
                non_block_file = os.path.join(safetensors_path, "non_block.safetensors")
                if os.path.exists(non_block_file):
                    safetensors_files = [non_block_file]
                else:
                    raise ValueError(f"Non-block file not found in {safetensors_path}. Please check the model path.")
            else:
                safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            if self.lazy_load:
                self.lazy_load_path = safetensors_path
            safetensors_files = [safetensors_path]
            safetensors_path = os.path.dirname(safetensors_path)

        weight_dict = {}
        for safetensor_path in safetensors_files:
            with safe_open(safetensor_path, framework="pt") as f:
                logger.info(f"Loading weights from {safetensor_path}")
                for k in f.keys():
                    if any(remove_key in k for remove_key in remove_keys):
                        continue
                    if f.get_tensor(k).dtype in [torch.float16, torch.bfloat16, torch.float]:
                        if unified_dtype or all(s not in k for s in sensitive_layer):
                            weight_dict[k] = f.get_tensor(k).to(GET_DTYPE()).to(self.device)
                        else:
                            weight_dict[k] = f.get_tensor(k).to(GET_SENSITIVE_DTYPE()).to(self.device)
                    else:
                        weight_dict[k] = f.get_tensor(k).to(self.device)

        # Load calibration data for nvfp4
        if self.config.get("dit_quant_scheme", "Default") == "nvfp4":
            calib_path = os.path.join(safetensors_path, "calib.pt")
            if os.path.exists(calib_path):
                logger.info(f"[CALIB] Loaded calibration data from: {calib_path}")
                calib_data = torch.load(calib_path, map_location="cpu")
                for k, v in calib_data["absmax"].items():
                    weight_dict[k.replace(".weight", ".input_absmax")] = v.to(self.device)

        return weight_dict

    def _load_gguf_ckpt(self, gguf_path):
        """Load GGUF checkpoint.

        Args:
            gguf_path: Path to GGUF file or directory

        Returns:
            dict: Dictionary of weights
        """
        state_dict = load_gguf_sd_ckpt(gguf_path, to_device=self.device)
        return state_dict

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        """Load and distribute weights from rank 0 to all ranks.

        This is a basic implementation. Subclasses may override for more
        sophisticated distribution (e.g., tensor parallel).

        Args:
            weight_dict: Weight dictionary from rank 0
            is_weight_loader: Whether current rank is the weight loader

        Returns:
            dict: Distributed weight dictionary
        """
        logger.info("Loading distributed weights")
        global_src_rank = 0
        target_device = "cpu" if self.cpu_offload else "cuda"

        if is_weight_loader:
            meta_dict = {}
            for key, tensor in weight_dict.items():
                meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype}

            obj_list = [meta_dict]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]

        distributed_weight_dict = {}
        for key, meta in synced_meta_dict.items():
            distributed_weight_dict[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device)

        if target_device == "cuda":
            dist.barrier(device_ids=[torch.cuda.current_device()])

        for key in sorted(synced_meta_dict.keys()):
            if is_weight_loader:
                distributed_weight_dict[key].copy_(weight_dict[key], non_blocking=True)

            if target_device == "cpu":
                if is_weight_loader:
                    gpu_tensor = distributed_weight_dict[key].cuda()
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()
                else:
                    gpu_tensor = torch.empty_like(distributed_weight_dict[key], device="cuda")
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()

                if distributed_weight_dict[key].is_pinned():
                    distributed_weight_dict[key].copy_(distributed_weight_dict[key], non_blocking=True)
            else:
                dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        if target_device == "cuda":
            torch.cuda.synchronize()
        else:
            for tensor in distributed_weight_dict.values():
                if tensor.is_pinned():
                    tensor.copy_(tensor, non_blocking=False)

        logger.info(f"Weights distributed across {dist.get_world_size()} devices on {target_device}")

        return distributed_weight_dict

    @compiled_method()
    @abstractmethod
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        """Run conditional/unconditional inference.

        Args:
            inputs: Input dictionary
            infer_condition: Whether to infer condition

        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        """Pre-process for sequence parallel mode.

        Args:
            pre_infer_out: Output from pre-inference

        Returns:
            Processed pre_infer_out

        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        """Post-process for sequence parallel mode.

        Args:
            x: Output from transformer inference

        Returns:
            Processed x

        Subclasses must implement this method.
        """
        pass

    def set_scheduler(self, scheduler):
        """Set the scheduler for all inference modules."""
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def to_cpu(self):
        """Move all weights to CPU."""
        self.pre_weight.to_cpu()
        self.transformer_weights.to_cpu()
        if hasattr(self, "post_weight"):
            self.post_weight.to_cpu()

    def to_cuda(self):
        """Move all weights to CUDA."""
        self.pre_weight.to_cuda()
        self.transformer_weights.to_cuda()
        if hasattr(self, "post_weight"):
            self.post_weight.to_cuda()

    @abstractmethod
    @torch.no_grad()
    def infer(self, inputs):
        """Run inference.

        Args:
            inputs: Input dictionary

        Subclasses must implement this method.
        """
        pass
