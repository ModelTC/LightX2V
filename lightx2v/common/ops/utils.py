import os
import re
import threading
import weakref
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

_CUDA_HOST_REGISTER_DEFAULT = 0
_CUDA_HOST_REGISTER_FINALIZER_ATTR = "_lightx2v_cuda_host_register_finalizer"
_READONLY_PINNED_SOURCE_ATTR = "_lightx2v_readonly_pinned_source"
_PINNED_WEIGHT_STATS = {
    "registered_count": 0,
    "registered_bytes": 0,
    "fallback_count": 0,
    "fallback_bytes": 0,
}
_PINNED_WEIGHT_FALLBACK_WARNED = False
_PINNED_WEIGHT_STATS_LOCK = threading.Lock()


def reset_pinned_weight_stats():
    """Reset mmap pinned weight registration counters.

    This helper is used by tests and validation scripts before measuring the
    registration/fallback path of weight loading.
    """
    global _PINNED_WEIGHT_FALLBACK_WARNED
    with _PINNED_WEIGHT_STATS_LOCK:
        for key in _PINNED_WEIGHT_STATS:
            _PINNED_WEIGHT_STATS[key] = 0
        _PINNED_WEIGHT_FALLBACK_WARNED = False


def get_pinned_weight_stats():
    """Get mmap pinned weight registration counters.

    Returns:
        dict: Snapshot of registration and fallback counters.
    """
    with _PINNED_WEIGHT_STATS_LOCK:
        return dict(_PINNED_WEIGHT_STATS)


def _record_pinned_weight_fallback(tensor):
    """Record one fallback from host registration to copy-based pinning.

    Args:
        tensor: Source tensor that could not be registered in place.
    """
    with _PINNED_WEIGHT_STATS_LOCK:
        _PINNED_WEIGHT_STATS["fallback_count"] += 1
    fallback_bytes = tensor.numel() * tensor.element_size()
    try:
        fallback_bytes = tensor.untyped_storage().nbytes()
    except Exception:
        pass
    with _PINNED_WEIGHT_STATS_LOCK:
        _PINNED_WEIGHT_STATS["fallback_bytes"] += fallback_bytes


def _cuda_error_text(error):
    """Convert a CUDA runtime error code to readable text.

    Args:
        error: CUDA runtime error code returned by cudart.

    Returns:
        Error message from CUDA runtime, or the raw error converted to string
        if CUDA error text lookup is unavailable.
    """
    try:
        return torch.cuda.cudart().cudaGetErrorString(error)
    except Exception:
        return str(error)


def cuda_register_host_tensor(tensor):
    """Register an existing CPU tensor storage as CUDA pinned host memory.

    Args:
        tensor: CPU tensor whose underlying storage should be registered with
                ``cudaHostRegister``.

    Returns:
        The input tensor after its storage has been registered as pinned host
        memory. The returned tensor is marked as a read-only pinned source so
        offload code will not write GPU data back into mmap-backed storage.

    Raises:
        RuntimeError: If the tensor is not on CPU, CUDA is unavailable, the
                    current AI device is not CUDA, or CUDA registration fails.
    """
    if tensor.device.type != "cpu":
        raise RuntimeError(f"cudaHostRegister requires a CPU tensor, got {tensor.device}")
    if tensor.numel() == 0:
        return tensor
    if AI_DEVICE != "cuda":
        raise RuntimeError(f"cudaHostRegister is only enabled for cuda AI_DEVICE, got {AI_DEVICE}")
    if not torch.cuda.is_available():
        raise RuntimeError("cudaHostRegister requires CUDA to be available")

    storage = tensor.untyped_storage()
    finalizer = getattr(storage, _CUDA_HOST_REGISTER_FINALIZER_ATTR, None)
    if finalizer is not None and finalizer.alive:
        if not tensor.is_pinned():
            raise RuntimeError("storage has a cudaHostRegister marker but tensor.is_pinned() is false")
        # Keep a Python reference to the registered storage for async H2D users.
        tensor._lightx2v_cuda_host_registered_storage = storage
        setattr(tensor, _READONLY_PINNED_SOURCE_ATTR, True)
        return tensor
    if tensor.is_pinned():
        return tensor

    ptr = storage.data_ptr()
    nbytes = storage.nbytes()
    cudart = torch.cuda.cudart()
    error = cudart.cudaHostRegister(ptr, nbytes, _CUDA_HOST_REGISTER_DEFAULT)
    if error != 0:
        raise RuntimeError(f"cudaHostRegister failed for {nbytes} bytes at 0x{ptr:x}: {_cuda_error_text(error)}")

    def unregister(address):
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaHostUnregister(address)

    # Tie cudaHostUnregister to storage lifetime, not to a temporary tensor view.
    finalizer = weakref.finalize(storage, unregister, ptr)
    setattr(storage, _CUDA_HOST_REGISTER_FINALIZER_ATTR, finalizer)
    # The tensor attribute keeps storage alive after safetensors safe_open closes.
    tensor._lightx2v_cuda_host_registered_storage = storage
    setattr(tensor, _READONLY_PINNED_SOURCE_ATTR, True)
    if not tensor.is_pinned():
        finalizer.detach()
        cudart.cudaHostUnregister(ptr)
        raise RuntimeError("cudaHostRegister succeeded but tensor.is_pinned() is false")

    with _PINNED_WEIGHT_STATS_LOCK:
        _PINNED_WEIGHT_STATS["registered_count"] += 1
        _PINNED_WEIGHT_STATS["registered_bytes"] += nbytes
    return tensor


def is_readonly_pinned_source(tensor):
    """Check whether a tensor is a registered read-only pinned source.

    Args:
        tensor: Tensor to inspect.

    Returns:
        bool: True if the tensor should only be used as a CPU-to-device source.
    """
    return bool(getattr(tensor, _READONLY_PINNED_SOURCE_ATTR, False))


def resolve_block_name(name, block_index, adapter_block_index=None, is_post_adapter=False):
    """Resolve the name according to the block index, replacing the block index in the name with the specified block_index.

    Args:
        name: Original tensor name, e.g. "blocks.0.weight"
        block_index: Target block index
        adapter_block_index: Target adapter block index (optional)
        is_post_adapter: Whether to perform post-adapter block index replacement (optional)

    Returns:
        Resolved name, e.g. "blocks.1.weight" (when block_index=1)

    Example:
        >>> self._resolve_block_name("blocks.0.weight", 1)
        "blocks.1.weight"
    """
    if is_post_adapter:
        return re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", name, count=1)
    else:
        return re.sub(r"\.\d+", lambda m: f".{block_index}", name, count=1)


def get_source_tensor(source_name, weight_dict, lazy_load, lazy_load_file, use_infer_dtype, scale_force_fp32, bias_force_fp32):
    """Get the source tensor from either weight dictionary or lazy loading safetensors file.

    Args:
        source_name: Name of the target tensor to get
        weight_dict: Preloaded weight dictionary
        lazy_load: Whether to enable lazy loading mode
        lazy_load_file: File or directory path for lazy loading
        use_infer_dtype: Whether to convert tensor to inference dtype
        scale_force_fp32: Whether to force weight_scale tensors to float32
        bias_force_fp32: Whether to force bias tensors to float32

    Returns:
        The target tensor retrieved from the source with appropriate dtype conversion applied
    """
    if lazy_load:
        if Path(lazy_load_file).is_file():
            lazy_load_file_path = lazy_load_file
        else:
            lazy_load_file_path = os.path.join(
                lazy_load_file,
                f"block_{source_name.split('.')[1]}.safetensors",
            )
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            if use_infer_dtype:
                return lazy_load_file.get_tensor(source_name).to(GET_DTYPE())
            elif scale_force_fp32 and "weight_scale" in source_name:
                return lazy_load_file.get_tensor(source_name).to(torch.float32)
            elif bias_force_fp32 and "bias" in source_name:
                return lazy_load_file.get_tensor(source_name).to(torch.float32)
            return lazy_load_file.get_tensor(source_name)
    else:
        if use_infer_dtype:
            return weight_dict[source_name].to(GET_DTYPE())
        elif scale_force_fp32 and "weight_scale" in source_name:
            return weight_dict[source_name].to(torch.float32)
        elif bias_force_fp32 and "bias" in source_name:
            return weight_dict[source_name].to(torch.float32)
        return weight_dict[source_name]


def create_pin_tensor(tensor, transpose=False, dtype=None):
    """Create a CPU tensor with pinned memory for faster data transfer to GPU.

    Args:
        tensor: Source tensor to be converted to pinned memory.
        transpose: Whether to transpose the pinned tensor (optional).
        dtype: Target data type of the pinned tensor (optional, defaults to
               source tensor's dtype).

    Returns:
        Pinned memory tensor on CPU with optional transposition applied. When
        the source tensor is a CPU tensor and no dtype conversion is needed, the
        source storage is registered in place with ``cudaHostRegister``. If
        registration fails or dtype conversion is required, a writable pinned
        copy is created instead. The copy path falls back to regular CPU memory
        if pinned allocation fails.
    """
    global _PINNED_WEIGHT_FALLBACK_WARNED

    dtype = dtype or tensor.dtype
    if tensor.device.type == "cpu" and dtype == tensor.dtype:
        try:
            pin_tensor = cuda_register_host_tensor(tensor)
        except Exception as e:
            _record_pinned_weight_fallback(tensor)
            if not _PINNED_WEIGHT_FALLBACK_WARNED:
                logger.warning(f"Failed to register mmap-backed pinned weight; falling back to copy-based pinned memory. First error: {e}")
                _PINNED_WEIGHT_FALLBACK_WARNED = True
        else:
            if transpose:
                base_tensor = pin_tensor
                pin_tensor = base_tensor.t()
                # A transposed view does not own storage; keep the registered
                # base tensor alive for async H2D copies.
                pin_tensor._lightx2v_cuda_host_registered_base = base_tensor
                if is_readonly_pinned_source(base_tensor):
                    setattr(pin_tensor, _READONLY_PINNED_SOURCE_ATTR, True)
            return pin_tensor

    return create_writable_pin_tensor(tensor, transpose=transpose, dtype=dtype)


def create_writable_pin_tensor(tensor, transpose=False, dtype=None):
    """Create a copy-based CPU buffer that may be written by later copy_ calls.

    Args:
        tensor: Source tensor to copy into the new buffer.
        transpose: Whether to transpose the copied tensor (optional).
        dtype: Target data type of the copied tensor (optional, defaults to
               source tensor's dtype).

    Returns:
        Writable pinned CPU tensor. Falls back to regular CPU memory if pinned
        allocation fails.
    """
    dtype = dtype or tensor.dtype
    try:
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=dtype)
    except Exception as e:
        logger.warning(f"Failed to allocate pinned memory (shape={tensor.shape}, dtype={dtype}): {e}. Falling back to regular CPU memory.")
        pin_tensor = torch.empty(tensor.shape, dtype=dtype)
    pin_tensor = pin_tensor.copy_(tensor)
    if transpose:
        pin_tensor = pin_tensor.t()
    del tensor
    return pin_tensor


def move_tensor_back_to_cpu(obj, attr_name, non_blocking=False):
    """Move a device tensor attribute back to its CPU-side attribute.

    Args:
        obj: Object containing ``attr_name`` and optional ``pin_<attr_name>``.
        attr_name: Name of the tensor attribute to move back to CPU.
        non_blocking: Whether to perform non-blocking data transfer (optional).

    Notes:
        Read-only pinned sources are mmap-backed canonical CPU weights. They
        are reused directly on CPU and are not overwritten by GPU results.
        Copy-based pinned buffers remain writable and keep the previous
        offload behavior of copying the device tensor back into ``pin_*``.
    """
    pin_attr_name = f"pin_{attr_name}"
    value = getattr(obj, attr_name, None)
    if hasattr(obj, pin_attr_name) and getattr(obj, pin_attr_name) is not None:
        pin_tensor = getattr(obj, pin_attr_name)
        if is_readonly_pinned_source(pin_tensor):
            setattr(obj, attr_name, pin_tensor)
        elif value is not None:
            setattr(obj, attr_name, pin_tensor.copy_(value, non_blocking=non_blocking).cpu())
        else:
            setattr(obj, attr_name, pin_tensor)
    elif value is not None:
        setattr(obj, attr_name, value.to("cpu", non_blocking=non_blocking))


def get_lazy_load_file_path(lazy_load_file, weight_name_for_block=None):
    """Get the full file path for lazy loading, handling both file and directory inputs.

    Args:
        lazy_load_file: Base file or directory path for lazy loading
        weight_name_for_block: Tensor weight name to generate block-specific file path (optional)

    Returns:
        Resolved full file path for lazy loading
    """
    if weight_name_for_block is None:
        return lazy_load_file
    if Path(lazy_load_file).is_file():
        return lazy_load_file
    else:
        return os.path.join(
            lazy_load_file,
            f"block_{weight_name_for_block.split('.')[1]}.safetensors",
        )


def create_cuda_buffers(base_attrs, weight_dict, lazy_load, lazy_load_file, use_infer_dtype=None, scale_force_fp32=False, bias_force_fp32=False):
    """Create tensor buffers and move them to CUDA device (specified by AI_DEVICE).

    Args:
        base_attrs: [(name, attr_name, transpose), ...] List of tensor loading specifications,
                    where transpose indicates whether transposition is required
        weight_dict: Preloaded weight dictionary
        lazy_load: Whether to use lazy loading mode
        lazy_load_file: File or directory path for lazy loading
        use_infer_dtype: Whether to convert tensors to inference dtype (optional)
        scale_force_fp32: Whether to force weight_scale tensors to float32 (optional)
        bias_force_fp32: Whether to force bias tensors to float32 (optional)

    Returns:
        dict: {attr_name: tensor, ...} Dictionary of tensors located on CUDA device
    """
    result = {}
    for name, attr_name, transpose in base_attrs:
        tensor = get_source_tensor(name, weight_dict, lazy_load, lazy_load_file, use_infer_dtype, scale_force_fp32, bias_force_fp32)
        if transpose:
            tensor = tensor.t()
        result[attr_name] = tensor.to(AI_DEVICE)

    return result


def create_cpu_buffers(base_attrs, lazy_load_file, use_infer_dtype=False, scale_force_fp32=False, bias_force_fp32=False):
    """Create pinned memory tensor buffers on CPU for lazy loading scenario.

    Args:
        base_attrs: [(name, attr_name, transpose), ...] Configuration list,
                    where transpose indicates whether transposition is required
        lazy_load_file: File or directory path for lazy loading
        use_infer_dtype: Whether to convert tensors to inference dtype (optional)
        scale_force_fp32: Whether to force weight_scale tensors to float32 (optional)
        bias_force_fp32: Whether to force bias tensors to float32 (optional)

    Returns:
        dict: {attr_name: tensor, ...} Dictionary of pinned memory tensors on CPU
    """
    result = {}

    for name, attr_name, transpose in base_attrs:
        tensor = get_source_tensor(name, {}, lazy_load=True, lazy_load_file=lazy_load_file, use_infer_dtype=False, scale_force_fp32=False, bias_force_fp32=False)
        dtype = None
        if use_infer_dtype:
            dtype = GET_DTYPE()
        elif scale_force_fp32 and "weight_scale" in name:
            dtype = torch.float32
        elif bias_force_fp32 and "bias" in name:
            dtype = torch.float32
        result[attr_name] = create_pin_tensor(tensor, transpose=transpose, dtype=dtype)

    return result


def create_default_tensors(base_attrs, weight_dict):
    """Create default tensors (device tensors and pinned memory tensors) based on the source weight device.

    Args:
        base_attrs: [(name, attr_name, transpose), ...] Configuration list,
                    where transpose indicates whether transposition is required
        weight_dict: Preloaded weight dictionary

    Returns:
        tuple: (device_tensors_dict, pin_tensors_dict)
        device_tensors_dict: {attr_name: tensor, ...} Tensors located on the original weight device
        pin_tensors_dict: {attr_name: tensor, ...} Tensors with pinned memory on CPU
    """
    device_tensors = {}
    pin_tensors = {}

    if not base_attrs:
        return device_tensors, pin_tensors

    first_tensor_name = base_attrs[0][0]
    device = weight_dict[first_tensor_name].device

    if device.type == "cpu":
        for name, attr_name, transpose in base_attrs:
            if name in weight_dict:
                tensor = weight_dict[name]
                pin_tensors[attr_name] = create_pin_tensor(tensor, transpose=transpose)
                del weight_dict[name]
    else:
        for name, attr_name, transpose in base_attrs:
            if name in weight_dict:
                tensor = weight_dict[name]
                if transpose:
                    tensor = tensor.t()
                device_tensors[attr_name] = tensor

    return device_tensors, pin_tensors


def move_tensor_to_device(obj, attr_name, target_device, non_blocking=False, use_copy=False):
    """Move the specified tensor attribute of an object to the target device,
       with support for pinned memory tensors for faster transfer.

    Args:
        obj: Target object containing the tensor attribute
        attr_name: Name of the tensor attribute to be moved
        target_device: Target device to move the tensor to
        non_blocking: Whether to perform non-blocking data transfer (optional)
        use_copy: Whether to copy the tensor content before moving (optional)
    """
    pin_attr_name = f"pin_{attr_name}"
    if hasattr(obj, pin_attr_name) and getattr(obj, pin_attr_name) is not None:
        pin_tensor = getattr(obj, pin_attr_name)
        if hasattr(obj, attr_name) and getattr(obj, attr_name) is not None and use_copy:
            if target_device == "cpu":
                move_tensor_back_to_cpu(obj, attr_name, non_blocking=non_blocking)
            else:
                setattr(obj, attr_name, pin_tensor.copy_(getattr(obj, attr_name), non_blocking=non_blocking).to(target_device))
        else:
            setattr(obj, attr_name, pin_tensor.to(target_device, non_blocking=non_blocking))
    elif hasattr(obj, attr_name) and getattr(obj, attr_name) is not None:
        setattr(obj, attr_name, getattr(obj, attr_name).to(target_device, non_blocking=non_blocking))


def build_lora_and_diff_names(weight_name, lora_prefix):
    """Build the full names of LoRA (down/up/alpha) and weight difference tensors.

    Args:
        weight_name: Original weight tensor name
        lora_prefix: Prefix string for LoRA tensor names

    Returns:
        tuple: (lora_down_name, lora_up_name, lora_alpha_name, weight_diff_name, bias_diff_name)
        Full names of various LoRA and difference tensors
    """
    base_name = weight_name[:-7]
    parts = base_name.split(".")
    relative_path = ".".join(parts[1:])
    lora_base = f"{lora_prefix}.{relative_path}"
    lora_down_name = f"{lora_base}.lora_down.weight"
    lora_up_name = f"{lora_base}.lora_up.weight"
    lora_alpha_name = f"{lora_base}.alpha"
    weight_diff_name = f"{lora_base}.diff"
    bias_diff_name = f"{lora_base}.diff_b"
    return lora_down_name, lora_up_name, lora_alpha_name, weight_diff_name, bias_diff_name


def move_attr_to_cuda(cls, base_attrs, lora_attrs, non_blocking=False):
    """Move base attributes and LoRA attributes to CUDA device.

    Args:
        cls: Target class instance containing tensor attributes
        base_attrs: [(name, attr_name, transpose), ...] List of base attribute specifications
        lora_attrs: Dictionary mapping LoRA attribute names to their name attributes
        non_blocking: Whether to perform non-blocking data transfer (optional)
    """
    # Base
    for _, base_attr_name, _ in base_attrs:
        move_tensor_to_device(cls, base_attr_name, AI_DEVICE, non_blocking)
    # Lora
    for lora_attr, _ in lora_attrs.items():
        if hasattr(cls, lora_attr) and getattr(cls, lora_attr) is not None:
            setattr(cls, lora_attr, getattr(cls, lora_attr).to(AI_DEVICE, non_blocking=non_blocking))


def move_attr_to_cpu(cls, base_attrs, lora_attrs, non_blocking=False):
    """Move base attributes and LoRA attributes to CPU device.

    Args:
        cls: Target class instance containing tensor attributes
        base_attrs: [(name, attr_name, transpose), ...] List of base attribute specifications
        lora_attrs: Dictionary mapping LoRA attribute names to their name attributes
        non_blocking: Whether to perform non-blocking data transfer (optional)
    """
    # Base
    for _, base_attr_name, _ in base_attrs:
        move_tensor_to_device(cls, base_attr_name, "cpu", non_blocking, use_copy=True)
    # Lora
    for lora_attr, _ in lora_attrs.items():
        if hasattr(cls, lora_attr) and getattr(cls, lora_attr) is not None:
            setattr(cls, lora_attr, getattr(cls, lora_attr).to("cpu", non_blocking=non_blocking))


def state_dict(cls, base_attrs, lora_attrs, destination=None):
    """Generate state dictionary containing base attributes and LoRA attributes.

    Args:
        cls: Target class instance containing tensor attributes
        base_attrs: [(name, attr_name, transpose), ...] List of base attribute specifications
        lora_attrs: Dictionary mapping LoRA attribute names to their name attributes
        destination: Optional destination dictionary to store state dict (if None, creates new dict)

    Returns:
        dict: State dictionary containing all base and LoRA attributes with their corresponding names
    """
    if destination is None:
        destination = {}
    # Base
    for _, base_attr, _ in base_attrs:
        pin_base_attr = getattr(cls, f"pin_{base_attr}", None)
        device_attr = getattr(cls, base_attr, None)
        name_attr = f"{base_attr}_name" if hasattr(cls, f"{base_attr}_name") else None
        if name_attr:
            name = getattr(cls, name_attr)
            destination[name] = pin_base_attr if pin_base_attr is not None else device_attr
    # Lora
    for lora_attr, name_attr in lora_attrs.items():
        if hasattr(cls, lora_attr):
            destination[getattr(cls, name_attr)] = getattr(cls, lora_attr)
    return destination


def load_state_dict(cls, base_attrs, lora_attrs, destination, block_index, adapter_block_index=None):
    """Load state dictionary into class instance, resolving block indices for base and LoRA attributes.

    Args:
        cls: Target class instance to load state dict into
        base_attrs: [(name, attr_name, transpose), ...] List of base attribute specifications
        lora_attrs: Dictionary mapping LoRA attribute names to their name attributes
        destination: Source state dictionary to load from
        block_index: Block index to resolve tensor names
        adapter_block_index: Adapter block index for post-adapter scenarios (optional)
    """
    # Base
    for name, attr_name, _ in base_attrs:
        actual_name = resolve_block_name(name, block_index, adapter_block_index, cls.is_post_adapter)
        cuda_buffer_attr = f"{attr_name}_cuda_buffer"
        if actual_name in destination:
            if hasattr(cls, cuda_buffer_attr):
                setattr(cls, attr_name, getattr(cls, cuda_buffer_attr).copy_(destination[actual_name], non_blocking=True))
        else:
            setattr(cls, attr_name, None)
    # Lora
    for lora_attr, lora_attr_name in lora_attrs.items():
        name = resolve_block_name(getattr(cls, lora_attr_name), block_index)
        if name in destination:
            setattr(cls, lora_attr, getattr(cls, lora_attr).copy_(destination[name], non_blocking=True).to(AI_DEVICE))
