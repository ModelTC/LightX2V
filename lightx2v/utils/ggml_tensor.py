from typing import Optional, Tuple, Union

import gguf
import numpy as np
import torch
from loguru import logger

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)


class GGMLTensor:
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, None] = None,
        orig_shape: Tuple[int, ...] = None,
        dtype: torch.dtype = None,
        gguf_type: gguf.GGMLQuantizationType = None,
        requires_grad: bool = False,
        aligned: bool = True,
        pin_memory: bool = False,
        preallocated: bool = False,
    ):
        super().__init__()

        assert orig_shape is not None
        assert gguf_type is not None

        if isinstance(data, np.ndarray):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                torch_data = torch.from_numpy(data)
        else:
            torch_data = data

            if dtype is not None and torch_data.dtype != dtype:
                torch_data = torch_data.to(dtype)

        self.data = torch_data

        self.gguf_type = gguf_type
        self._orig_shape = orig_shape
        self._aligned = aligned
        self._pinned_memory = pin_memory
        self._requires_grad = requires_grad
        self._preallocated = preallocated

        self._quantized = self._is_quantized_type(gguf_type)
        self._q_type = self._get_quant_type_str(gguf_type)

        if aligned:
            self._make_aligned()
        if pin_memory:
            self._pin_memory()

    def _is_quantized_type(self, gguf_type: gguf.GGMLQuantizationType) -> bool:
        return gguf_type not in TORCH_COMPATIBLE_QTYPES

    def _get_quant_type_str(self, gguf_type: gguf.GGMLQuantizationType) -> str:
        type_mapping = {
            gguf.GGMLQuantizationType.F32: "ggml_f32",
            gguf.GGMLQuantizationType.F16: "ggml_f16",
            gguf.GGMLQuantizationType.Q4_0: "ggml_q4_0",
            gguf.GGMLQuantizationType.Q4_1: "ggml_q4_1",
            gguf.GGMLQuantizationType.Q5_0: "ggml_q5_0",
            gguf.GGMLQuantizationType.Q5_1: "ggml_q5_1",
            gguf.GGMLQuantizationType.Q8_0: "ggml_q8_0",
            gguf.GGMLQuantizationType.Q8_1: "ggml_q8_1",
            gguf.GGMLQuantizationType.Q2_K: "ggml_q2_k",
            gguf.GGMLQuantizationType.Q3_K: "ggml_q3_k",
            gguf.GGMLQuantizationType.Q4_K: "ggml_q4_k",
            gguf.GGMLQuantizationType.Q5_K: "ggml_q5_k",
            gguf.GGMLQuantizationType.Q6_K: "ggml_q6_k",
            gguf.GGMLQuantizationType.Q8_K: "ggml_q8_k",
        }
        return type_mapping.get(gguf_type, "unknown")

    @classmethod
    def empty_pinned(
        cls, shape: Tuple[int, ...], orig_shape: Tuple[int, ...] = None, dtype: torch.dtype = torch.float32, gguf_type: gguf.GGMLQuantizationType = None, aligned: bool = True
    ) -> "GGMLTensor":
        torch_data = torch.empty(shape, pin_memory=True, dtype=dtype)
        return cls(data=torch_data, dtype=dtype, orig_shape=orig_shape, gguf_type=gguf_type, pin_memory=True, aligned=aligned, preallocated=True)

    @classmethod
    def empty_aligned(
        cls, shape: Tuple[int, ...], orig_shape: Tuple[int, ...] = None, dtype: torch.dtype = torch.float32, gguf_type: gguf.GGMLQuantizationType = None, pin_memory: bool = False
    ) -> "GGMLTensor":
        return cls(dtype=dtype, orig_shape=orig_shape, gguf_type=gguf_type, pin_memory=pin_memory, aligned=True, preallocated=True)

    def copy_from(self, source: Union[torch.Tensor, "GGMLTensor"], transpose: bool = False, non_blocking: bool = False) -> "GGMLTensor":
        if not self._preallocated:
            raise RuntimeError("copy_from can only be used with preallocated tensors")

        if transpose:
            source_data = source.data.t().contiguous()
        else:
            source_data = source.data.contiguous()

        if self.shape != source_data.shape:
            raise ValueError(f"Shape mismatch: target {self.shape} vs source {source_data.shape}")

        self.data.copy_(source_data)

        return self

    def copy_(self, target: Union[torch.Tensor, "GGMLTensor"], transpose: bool = False, non_blocking: bool = False) -> "GGMLTensor":
        source_data = self.data
        if transpose:
            source_data = self.t().contiguous()

        if isinstance(target, GGMLTensor):
            target.copy_from(source_data, non_blocking=non_blocking)
        else:
            target.copy_(source_data)

        return self

    def t(self):
        self.data = self.data.t()
        return self

    def _make_aligned(self, alignment: int = 32):
        if not self.data.is_contiguous():
            self.data = self.data.contiguous().data

        ptr = self.data.data_ptr()
        if ptr % alignment == 0:
            return

        if self._pinned_memory:
            aligned_data = torch.empty(self.data.shape, dtype=self.data.dtype, device=self.data.device, pin_memory=True)
        else:
            aligned_data = torch.empty(self.data.shape, dtype=self.data.dtype, device=self.data.device)

        aligned_data.copy_(self.data)
        self.data = aligned_data.data

    def _pin_memory(self) -> "GGMLTensor":
        if self._pinned_memory or self.device.type != "cpu":
            return self

        pinned_data = self.data.pin_memory()
        self.data = pinned_data.data
        self._pinned_memory = True
        return self

    def to_torch(self) -> torch.Tensor:
        return torch.as_tensor(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    @property
    def tensor_type(self) -> gguf.GGMLQuantizationType:
        return self.gguf_type

    @property
    def quant_type(self) -> str:
        return self._q_type

    @property
    def is_quantized(self) -> bool:
        return self._quantized

    @property
    def orig_shape(self) -> Tuple[int, ...]:
        return self._orig_shape

    @property
    def blocksize(self) -> Optional[int]:
        _blocksize, _ = gguf.GGML_QUANT_SIZES[self.qtype]
        return _blocksize

    @property
    def is_pinned(self) -> bool:
        return self._pinned_memory

    def memory_footprint(self) -> int:
        if self._quantized:
            return self.data.numel() * self.element_size()
        else:
            return self.data.numel() * self.element_size()

    def __repr__(self) -> str:
        return f"GGMLTensor(shape={self.data.shape}, orig_shape={self.orig_shape}, dtype={self.data.dtype}, quantized={self.is_quantized}, quant_type='{self.quant_type}', pinned={self.is_pinned})"

    def cuda(self, device: Optional[Union[int, torch.device]] = None, non_blocking: bool = False) -> "GGMLTensor":
        if device is None:
            self.data = self.data.cuda(non_blocking=non_blocking)
        else:
            self.data = self.data.cuda(device=device, non_blocking=non_blocking)
        return self

    def cpu(self, pin_memory: bool = False) -> "GGMLTensor":
        self.data = self.data.cpu()
        return self

    def to(self, *args, **kwargs) -> "GGMLTensor":
        self.data = self.data.to(*args, **kwargs)
        return self


def load_gguf_sd_ckpt(gguf_path, return_arch=False, to_device: Optional[Union[int, torch.device]] = None):
    import warnings

    logger.info(f"Loading gguf-quant dit model from {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        tensor_name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            state_dict[tensor.name] = torch_tensor.to(to_device)
        else:
            state_dict[tensor.name] = GGMLTensor(
                data=torch_tensor,
                gguf_type=tensor.tensor_type,
                orig_shape=shape,
                aligned=True,
                pin_memory=False,
            ).to(to_device)

    if return_arch:
        arch = get_model_architecture(reader)
        return state_dict, arch

    return state_dict


def get_orig_shape(reader, tensor_name: str) -> Optional[Tuple[int, ...]]:
    # TODO 这里正式上线的时候，需要更换
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif isinstance(field_type, str):
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")


def get_model_architecture(reader) -> str:
    arch_str = get_field(reader, "general.architecture", str)
    return arch_str


def dequantize_tensor(tensor, dtype=None):
    qtype = getattr(tensor, "gguf_type", None)
    oshape = getattr(tensor, "orig_shape", tensor.data.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        return dequantize(tensor.to_torch().data, qtype, oshape, dtype=dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {qtype}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)


def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}


if __name__ == "__main__":
    sd = load_gguf_sd_ckpt("/home/SENSETIME/yihuiwen/yihuiwen/workspace/models/city96/Wan2.1-I2V-14B-720P-gguf/wan2.1-i2v-14b-720p-Q4_K_S.gguf", return_arch=False)

    for k, s in sd.items():
        print(k)
        if isinstance(s, GGMLTensor):
            dequantize_tensor(s, torch.float32)
