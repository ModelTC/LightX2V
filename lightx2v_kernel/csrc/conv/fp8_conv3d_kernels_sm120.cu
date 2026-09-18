/* Copyright 2026 LightX2V Team. All Rights Reserved.

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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/tensor.h"
#include "gemm/sm120_utils.h"

namespace {

using ElementInput = cutlass::float_e4m3_t;
using Layout = cutlass::layout::TensorNDHWC;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

// CUTLASS's legacy Conv3D builder has no SM120 specialization. Sm89 selects
// its public FP8 MMA core; compiling this translation unit for sm_120a lowers
// the instruction shape above to SM120 QMMA.
template <
    typename ElementAccumulator,
    typename ElementOutput_,
    int ThreadblockM,
    int ThreadblockN,
    int ThreadblockK,
    int WarpM,
    int WarpN,
    int WarpK,
    int Stages>
struct ConvDefinition {
  using ElementOutput = ElementOutput_;
  using ThreadblockShape = cutlass::gemm::GemmShape<
      ThreadblockM,
      ThreadblockN,
      ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      float>;
  using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
      ElementInput,
      Layout,
      ElementInput,
      Layout,
      ElementOutput,
      Layout,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm89,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided>::Kernel;
  using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

using F16Config0 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 128, 128, 64, 64, 64, 64, 3>;
using F16Config1 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 128, 128, 64, 64, 64, 64, 4>;
using F16Config2 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 256, 128, 64, 64, 64, 64, 3>;
using F16Config3 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 128, 256, 64, 64, 64, 64, 3>;
using F16Config4 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 64, 128, 64, 64, 64, 64, 4>;
using F16Config5 = ConvDefinition<
    cutlass::half_t, cutlass::half_t, 128, 64, 64, 64, 64, 64, 4>;

// These candidates cover the dominant H3 VAE encoder workloads. Their order
// has no semantic meaning; first-use autotuning selects and caches the winner.
// FP32 accumulation uses smaller warp tiles to avoid spilling accumulator
// fragments to local memory.
using F32Config0 = ConvDefinition<
    float, float, 128, 128, 64, 64, 32, 64, 3>;
using F32Config1 = ConvDefinition<
    float, float, 128, 128, 64, 32, 64, 64, 3>;
using F32Config2 = ConvDefinition<
    float, float, 128, 64, 64, 64, 32, 64, 3>;
using F32Config3 = ConvDefinition<
    float, float, 64, 128, 64, 32, 64, 64, 3>;
using F32Config4 = ConvDefinition<
    float, float, 64, 64, 64, 32, 32, 64, 3>;

struct AutotuneKey {
  int device_index;
  int32_t n;
  int32_t c;
  int32_t d;
  int32_t h;
  int32_t w;
  int32_t k;
  int32_t t;
  int32_t r;
  int32_t s;
  int32_t stride_d;
  int32_t stride_h;
  int32_t stride_w;
  bool use_fp16_accumulator;

  bool operator==(AutotuneKey const& other) const {
    return device_index == other.device_index && n == other.n &&
        c == other.c && d == other.d && h == other.h && w == other.w &&
        k == other.k && t == other.t && r == other.r && s == other.s &&
        stride_d == other.stride_d && stride_h == other.stride_h &&
        stride_w == other.stride_w &&
        use_fp16_accumulator == other.use_fp16_accumulator;
  }
};

struct AutotuneKeyHash {
  size_t operator()(AutotuneKey const& key) const {
    size_t value = std::hash<int>{}(key.device_index);
    auto combine = [&value](auto item) {
      value = value * 31 + std::hash<decltype(item)>{}(item);
    };
    combine(key.n);
    combine(key.c);
    combine(key.d);
    combine(key.h);
    combine(key.w);
    combine(key.k);
    combine(key.t);
    combine(key.r);
    combine(key.s);
    combine(key.stride_d);
    combine(key.stride_h);
    combine(key.stride_w);
    combine(key.use_fp16_accumulator);
    return value;
  }
};

using AutotuneCache =
    std::unordered_map<AutotuneKey, int64_t, AutotuneKeyHash>;

AutotuneCache& autotune_cache() {
  static AutotuneCache cache;
  return cache;
}

std::shared_mutex& autotune_cache_mutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::mutex& autotune_measurement_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::optional<int64_t> cached_config_id(AutotuneKey const& key) {
  std::shared_lock lock(autotune_cache_mutex());
  auto entry = autotune_cache().find(key);
  if (entry == autotune_cache().end()) {
    return std::nullopt;
  }
  return entry->second;
}

void cache_config(AutotuneKey const& key, int64_t config_id) {
  std::unique_lock lock(autotune_cache_mutex());
  autotune_cache()[key] = config_id;
}

void validate(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  TORCH_CHECK(
      input.is_cuda() && weight.is_cuda(),
      "input and weight must be CUDA tensors");
  lightx2v_kernel::check_sm120_or_throw(input, "fp8_conv3d_sm120");
  TORCH_CHECK(
      input.device() == weight.device(),
      "input and weight must be on the same CUDA device");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat8_e4m3fn &&
          weight.scalar_type() == torch::kFloat8_e4m3fn,
      "input and weight must be float8_e4m3fn");
  TORCH_CHECK(
      input.dim() == 5 && weight.dim() == 5,
      "input and weight must be 5D");
  TORCH_CHECK(
      input.is_contiguous(at::MemoryFormat::ChannelsLast3d) &&
          weight.is_contiguous(at::MemoryFormat::ChannelsLast3d),
      "input and weight must be contiguous in channels_last_3d memory format");
  TORCH_CHECK(
      input.size(1) == weight.size(1),
      "input and weight channels must match");
  TORCH_CHECK(
      stride_d > 0 && stride_h > 0 && stride_w > 0,
      "Conv3D strides must be positive");
  TORCH_CHECK(
      input.size(2) >= weight.size(2) &&
          input.size(3) >= weight.size(3) &&
          input.size(4) >= weight.size(4),
      "Conv3D kernel must fit within the input");
  for (int64_t dimension : input.sizes()) {
    TORCH_CHECK(
        dimension > 0 && dimension <= std::numeric_limits<int32_t>::max(),
        "input dimensions must be positive int32 values");
  }
  for (int64_t dimension : weight.sizes()) {
    TORCH_CHECK(
        dimension > 0 && dimension <= std::numeric_limits<int32_t>::max(),
        "weight dimensions must be positive int32 values");
  }
  TORCH_CHECK(
      stride_d <= std::numeric_limits<int32_t>::max() &&
          stride_h <= std::numeric_limits<int32_t>::max() &&
          stride_w <= std::numeric_limits<int32_t>::max(),
      "Conv3D strides must be int32 values");
}

torch::Tensor make_output(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::ScalarType output_dtype,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  int64_t output_frames =
      (input.size(2) - weight.size(2)) / stride_d + 1;
  int64_t output_height =
      (input.size(3) - weight.size(3)) / stride_h + 1;
  int64_t output_width =
      (input.size(4) - weight.size(4)) / stride_w + 1;
  return torch::empty(
      {
          input.size(0),
          weight.size(0),
          output_frames,
          output_height,
          output_width,
      },
      input.options()
          .dtype(output_dtype)
          .memory_format(at::MemoryFormat::ChannelsLast3d));
}

template <typename Definition>
typename Definition::Conv::Arguments make_arguments(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  using Kernel = typename Definition::Kernel;

  int n = input.size(0);
  int c = input.size(1);
  int d = input.size(2);
  int h = input.size(3);
  int w = input.size(4);
  int k = weight.size(0);
  int t = weight.size(2);
  int r = weight.size(3);
  int s = weight.size(4);
  int z = output.size(2);
  int p = output.size(3);
  int q = output.size(4);

  auto input_layout =
      Layout::packed(cutlass::Tensor5DCoord(n, d, h, w, c));
  auto weight_layout =
      Layout::packed(cutlass::Tensor5DCoord(k, t, r, s, c));
  auto output_layout =
      Layout::packed(cutlass::Tensor5DCoord(n, z, p, q, k));
  typename Kernel::TensorRefA input_ref(
      reinterpret_cast<ElementInput*>(input.data_ptr()), input_layout);
  typename Kernel::TensorRefB weight_ref(
      reinterpret_cast<ElementInput*>(weight.data_ptr()), weight_layout);
  typename Kernel::TensorRefC output_ref(
      reinterpret_cast<typename Definition::ElementOutput*>(
          output.data_ptr()),
      output_layout);
  cutlass::conv::Conv3dProblemSize problem(
      cutlass::Tensor5DCoord(n, d, h, w, c),
      cutlass::Tensor5DCoord(k, t, r, s, c),
      cutlass::make_Coord(0, 0, 0),
      cutlass::make_Coord(
          static_cast<int>(stride_d),
          static_cast<int>(stride_h),
          static_cast<int>(stride_w)),
      cutlass::make_Coord(1, 1, 1),
      cutlass::Tensor5DCoord(n, z, p, q, k),
      cutlass::conv::Mode::kCrossCorrelation,
      1);
  return typename Definition::Conv::Arguments(
      problem,
      input_ref,
      weight_ref,
      output_ref,
      output_ref,
      {1.0f, 0.0f});
}

template <typename Definition>
bool can_implement(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  typename Definition::Conv conv;
  auto arguments = make_arguments<Definition>(
      output, input, weight, stride_d, stride_h, stride_w);
  return conv.can_implement(arguments) == cutlass::Status::kSuccess;
}

template <typename Definition>
void launch(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  using Conv = typename Definition::Conv;
  Conv conv;
  auto arguments = make_arguments<Definition>(
      output, input, weight, stride_d, stride_h, stride_w);
  TORCH_CHECK(
      conv.can_implement(arguments) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this FP8 Conv3D shape");

  size_t workspace_size = Conv::get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_pointer = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        input.options().dtype(torch::kUInt8));
    workspace_pointer = workspace.data_ptr();
  }

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  auto status = conv(arguments, workspace_pointer, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS FP8 Conv3D launch failed");
}

using CanImplement = bool (*)(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w);

using Launch = void (*)(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w);

struct ConvCandidate {
  CanImplement can_implement;
  Launch launch;
};

template <typename Definition>
constexpr ConvCandidate make_candidate() {
  return {&can_implement<Definition>, &launch<Definition>};
}

template <bool UseFp16Accumulator>
auto const& kernel_candidates() {
  if constexpr (UseFp16Accumulator) {
    static constexpr std::array candidates{
        make_candidate<F16Config0>(),
        make_candidate<F16Config1>(),
        make_candidate<F16Config2>(),
        make_candidate<F16Config3>(),
        make_candidate<F16Config4>(),
        make_candidate<F16Config5>(),
    };
    return candidates;
  } else {
    static constexpr std::array candidates{
        make_candidate<F32Config0>(),
        make_candidate<F32Config1>(),
        make_candidate<F32Config2>(),
        make_candidate<F32Config3>(),
        make_candidate<F32Config4>(),
    };
    return candidates;
  }
}

AutotuneKey make_key(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    bool use_fp16_accumulator) {
  return {
      input.device().index(),
      static_cast<int32_t>(input.size(0)),
      static_cast<int32_t>(input.size(1)),
      static_cast<int32_t>(input.size(2)),
      static_cast<int32_t>(input.size(3)),
      static_cast<int32_t>(input.size(4)),
      static_cast<int32_t>(weight.size(0)),
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(weight.size(3)),
      static_cast<int32_t>(weight.size(4)),
      static_cast<int32_t>(stride_d),
      static_cast<int32_t>(stride_h),
      static_cast<int32_t>(stride_w),
      use_fp16_accumulator,
  };
}

template <bool UseFp16Accumulator>
int64_t tune_config(
    AutotuneKey const& key,
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  std::lock_guard measurement_lock(autotune_measurement_mutex());
  if (auto cached = cached_config_id(key)) {
    return *cached;
  }

  auto const& candidates = kernel_candidates<UseFp16Accumulator>();
  int count = candidates.size();
  std::vector<bool> viable(count);
  for (int config_id = 0; config_id < count; ++config_id) {
    viable[config_id] = candidates[config_id].can_implement(
        output,
        input,
        weight,
        stride_d,
        stride_h,
        stride_w);
  }
  TORCH_CHECK(
      std::any_of(viable.begin(), viable.end(), [](bool value) {
        return value;
      }),
      "No CUTLASS FP8 Conv3D candidate supports this shape");

  constexpr int kWarmups = 2;
  constexpr int kTrials = 3;
  constexpr int kRepeats = 5;
  std::vector<std::array<float, kTrials>> timings(count);
  for (auto& samples : timings) {
    samples.fill(std::numeric_limits<float>::infinity());
  }

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (int index = 0; index < count; ++index) {
      int config_id = (index + warmup) % count;
      if (viable[config_id]) {
        candidates[config_id].launch(
            output,
            input,
            weight,
            stride_d,
            stride_h,
            stride_w);
      }
    }
  }

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  int initial_offset = static_cast<int>(
      (key.n + key.c + key.d + key.h + key.w + key.k) % count);
  for (int trial = 0; trial < kTrials; ++trial) {
    int offset = (initial_offset + trial) % count;
    for (int index = 0; index < count; ++index) {
      int config_id = (index + offset) % count;
      if (!viable[config_id]) {
        continue;
      }
      c10::cuda::CUDAEvent start(cudaEventDefault);
      c10::cuda::CUDAEvent end(cudaEventDefault);
      start.record(stream);
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        candidates[config_id].launch(
            output,
            input,
            weight,
            stride_d,
            stride_h,
            stride_w);
      }
      end.record(stream);
      end.synchronize();
      timings[config_id][trial] =
          start.elapsed_time(end) / static_cast<float>(kRepeats);
    }
  }

  int64_t best_config_id = 0;
  float best_time = std::numeric_limits<float>::infinity();
  for (int config_id = 0; config_id < count; ++config_id) {
    auto samples = timings[config_id];
    std::sort(samples.begin(), samples.end());
    if (samples[kTrials / 2] < best_time) {
      best_time = samples[kTrials / 2];
      best_config_id = config_id;
    }
  }
  cache_config(key, best_config_id);
  return best_config_id;
}

template <bool UseFp16Accumulator>
torch::Tensor run(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  validate(input, weight, stride_d, stride_h, stride_w);
  c10::cuda::CUDAGuard guard(input.device());
  torch::ScalarType output_dtype =
      UseFp16Accumulator ? torch::kFloat16 : torch::kFloat32;
  auto output = make_output(
      input,
      weight,
      output_dtype,
      stride_d,
      stride_h,
      stride_w);
  AutotuneKey key = make_key(
      input,
      weight,
      stride_d,
      stride_h,
      stride_w,
      UseFp16Accumulator);

  int64_t config_id;
  if (auto cached = cached_config_id(key)) {
    config_id = *cached;
  } else {
    config_id = tune_config<UseFp16Accumulator>(
        key,
        output,
        input,
        weight,
        stride_d,
        stride_h,
        stride_w);
  }
  kernel_candidates<UseFp16Accumulator>()[config_id].launch(
      output,
      input,
      weight,
      stride_d,
      stride_h,
      stride_w);
  return output;
}

}  // namespace

torch::Tensor fp8_conv3d_f16_accum_sm120(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  return run<true>(input, weight, stride_d, stride_h, stride_w);
}

torch::Tensor fp8_conv3d_f32_accum_sm120(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w) {
  return run<false>(input, weight, stride_d, stride_h, stride_w);
}
