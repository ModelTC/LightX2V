/* Copyright 2025 LightX2V Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <array>
#include <mutex>
#include <utility>

namespace lightx2v_kernel {

constexpr int kMaxCudaDevices = 64;

inline void check_valid_cuda_device_index(int device, char const* op_name) {
  TORCH_CHECK(
      device >= 0 && device < kMaxCudaDevices,
      op_name,
      " requires CUDA device index in [0, ",
      kMaxCudaDevices,
      "), got ",
      device);
}

inline std::pair<int, int> get_cached_device_capability(int device) {
  static std::array<std::once_flag, kMaxCudaDevices> device_once;
  static std::array<int, kMaxCudaDevices> cached_major{};
  static std::array<int, kMaxCudaDevices> cached_minor{};
  std::call_once(device_once[device], [device]() {
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cached_major[device] = prop.major;
    cached_minor[device] = prop.minor;
  });
  return {cached_major[device], cached_minor[device]};
}

inline void check_sm120_or_throw(torch::Tensor const& tensor, char const* op_name) {
  int device = tensor.get_device();
  check_valid_cuda_device_index(device, op_name);
  auto [major, minor] = get_cached_device_capability(device);
  TORCH_CHECK(
      major == 12,
      op_name,
      " is only supported on SM120/SM120a GPUs, got CUDA device ",
      device,
      " with compute capability ",
      major,
      ".",
      minor);
}

inline int getMultiProcessorCount(int device) {
  check_valid_cuda_device_index(device, "getMultiProcessorCount");
  static std::array<std::once_flag, kMaxCudaDevices> device_once;
  static std::array<int, kMaxCudaDevices> cached_mp_count{};
  std::call_once(device_once[device], [device]() {
    int mp_count = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device));
    cached_mp_count[device] = mp_count;
  });
  return cached_mp_count[device];
}

}  // namespace lightx2v_kernel
