# Copyright (c) 2024 by SageAttention team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fuse K centering with SageAttention's per-thread INT8 quantization layout."""

import triton
import triton.language as tl


@triton.jit
def center_quant_key(k, mean, output, scale, length: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr):
    block = tl.program_id(0) // 4
    thread = tl.program_id(0) % 4
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rows0 = block * 128 + tl.arange(0, 16) * 8 + thread * 2
    rows1 = rows0 + 1
    columns = tl.arange(0, dim)
    offset = batch * length * heads * dim + head * dim
    x0 = tl.load(k + offset + rows0[:, None] * heads * dim + columns[None, :], rows0[:, None] < length, 0)
    x1 = tl.load(k + offset + rows1[:, None] * heads * dim + columns[None, :], rows1[:, None] < length, 0)
    center = tl.load(mean + (batch * heads + head) * dim + columns)
    # Preserve the BF16/FP16 rounding of the original separate subtraction.
    x0 = (x0.to(tl.float32) - center[None, :].to(tl.float32)).to(k.dtype.element_ty).to(tl.float32)
    x1 = (x1.to(tl.float32) - center[None, :].to(tl.float32)).to(k.dtype.element_ty).to(tl.float32)
    x0 = tl.where(rows0[:, None] < length, x0, 0.0)
    x1 = tl.where(rows1[:, None] < length, x1, 0.0)
    maximum = tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1)))
    factor = maximum / 127.0 + 0.0000001
    q0, q1 = x0 / factor, x1 / factor
    q0 = (q0 + 0.5 * tl.where(q0 >= 0, 1, -1)).to(tl.int8)
    q1 = (q1 + 0.5 * tl.where(q1 >= 0, 1, -1)).to(tl.int8)
    tl.store(output + offset + rows0[:, None] * heads * dim + columns[None, :], q0, rows0[:, None] < length)
    tl.store(output + offset + rows1[:, None] * heads * dim + columns[None, :], q1, rows1[:, None] < length)
    scale_length = triton.cdiv(length, 128) * 4
    tl.store(scale + (batch * heads + head) * scale_length + block * 4 + thread, factor)


@triton.jit
def update_transposed_v_kernel(v, output, length, heads: tl.constexpr, dim: tl.constexpr, padded, start, stride_b, stride_n, stride_h, stride_d):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    batch = tl.program_id(2) // triton.cdiv(dim, 32)
    columns = (tl.program_id(2) % triton.cdiv(dim, 32)) * 32 + tl.arange(0, 32)
    mask = (rows[:, None] < length) & (columns[None, :] < dim)
    values = tl.load(v + batch * stride_b + rows[:, None] * stride_n + head * stride_h + columns[None, :] * stride_d, mask, 0)
    position = start + rows
    # Inverse of Sage's [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] permutation.
    permuted = (position & ~15) | (position & 1) | ((position & 8) >> 2) | ((position & 2) << 1) | ((position & 4) << 1)
    address = ((batch * dim + columns[None, :]) * heads + head) * padded + permuted[:, None]
    tl.store(output + address, values, mask)
