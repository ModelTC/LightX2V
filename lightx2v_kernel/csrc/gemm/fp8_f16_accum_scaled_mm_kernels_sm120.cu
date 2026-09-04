// SM120 FP8 GEMM with FP16 accumulation.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

template <
    typename TileShape_,
    typename ElementD_ = cutlass::bfloat16_t,
    bool FuseBias_ = false>
struct GemmDefinition {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = ElementD_;
  using ElementC = void;
  using ElementAccumulator = cutlass::half_t;
  using TileShape = TileShape_;
  static constexpr bool FuseBias = FuseBias_;
  using ClusterShape = Shape<_1, _1, _1>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentAB = 16;
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      float,
      float,
      Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      float,
      float,
      Stride<Int<0>, Int<1>, Int<0>>>;
  using Multiply = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using MultiplyOutput = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementD,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus,
      ElementD,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementD,
      float,
      Stride<Int<0>, Int<1>, Int<0>>,
      AlignmentD>;
  using ScaleBAccum =
      cutlass::epilogue::fusion::Sm90EVT<Multiply, ScaleB, Accum>;
  using ScaledEVT = cutlass::epilogue::fusion::Sm90EVT<
      MultiplyOutput,
      ScaleA,
      ScaleBAccum>;
  using OutputEVT = cutlass::epilogue::fusion::Sm90EVT<
      AddBias,
      ScaledEVT,
      Bias>;
  using EpilogueEVT = std::conditional_t<FuseBias, OutputEVT, ScaledEVT>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          float,
          ElementC,
          LayoutC,
          AlignmentD,
          ElementD,
          LayoutD,
          AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120,
          cutlass::arch::OpClassTensorOp,
          ElementA,
          LayoutA,
          AlignmentAB,
          ElementB,
          LayoutB,
          AlignmentAB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename EpilogueEVT::Arguments prepare_epilogue(
      float* activation_scale,
      float* weight_scale,
      ElementD const* bias) {
    typename ScaleA::Arguments activation_arguments{activation_scale};
    typename ScaleB::Arguments weight_arguments{weight_scale};
    typename ScaleBAccum::Arguments scaled_accumulator{
        weight_arguments,
        {},
        {},
    };
    typename ScaledEVT::Arguments scaled_output{
        activation_arguments,
        scaled_accumulator,
        {},
    };
    if constexpr (FuseBias) {
      typename Bias::Arguments bias_arguments{bias};
      return typename OutputEVT::Arguments{
          scaled_output,
          bias_arguments,
          {},
      };
    } else {
      return scaled_output;
    }
  }
};
using NarrowGemm = GemmDefinition<Shape<_128, _128, _64>>;
using WideGemm = GemmDefinition<Shape<_128, _256, _64>>;
using NarrowGemmFp16 = GemmDefinition<Shape<_128, _128, _64>, cutlass::half_t>;
using WideGemmFp16 = GemmDefinition<Shape<_128, _256, _64>, cutlass::half_t>;

using NarrowGemmWithBias =
    GemmDefinition<Shape<_128, _128, _64>, cutlass::bfloat16_t, true>;
using WideGemmWithBias =
    GemmDefinition<Shape<_128, _256, _64>, cutlass::bfloat16_t, true>;
using NarrowGemmFp16WithBias =
    GemmDefinition<Shape<_128, _128, _64>, cutlass::half_t, true>;
using WideGemmFp16WithBias =
    GemmDefinition<Shape<_128, _256, _64>, cutlass::half_t, true>;

// Bump the ABI whenever a config mapping or candidate implementation changes.
constexpr int64_t kAutotuneCacheAbi = 1;
constexpr int64_t kFallbackConfigId = 0;

struct KernelConfig {
  bool wide_tile;
  int swizzle;
  char const* name;
};

constexpr std::array<KernelConfig, 8> kKernelConfigs = {{
    {false, 1, "tile_128x128x64_swizzle_1"},
    {false, 2, "tile_128x128x64_swizzle_2"},
    {false, 4, "tile_128x128x64_swizzle_4"},
    {false, 8, "tile_128x128x64_swizzle_8"},
    {true, 1, "tile_128x256x64_swizzle_1"},
    {true, 2, "tile_128x256x64_swizzle_2"},
    {true, 4, "tile_128x256x64_swizzle_4"},
    {true, 8, "tile_128x256x64_swizzle_8"},
}};

KernelConfig const& kernel_config(int64_t config_id) {
  TORCH_CHECK(
      config_id >= 0 &&
          config_id < static_cast<int64_t>(kKernelConfigs.size()),
      "FP8-F16 GEMM config_id must be in [0, ",
      kKernelConfigs.size(),
      "), got ",
      config_id);
  return kKernelConfigs[config_id];
}

struct AutotuneKey {
  int device_index;
  int32_t m;
  int32_t n;
  int32_t k;
  torch::ScalarType output_dtype;
  bool has_bias;

  bool operator==(AutotuneKey const& other) const {
    return device_index == other.device_index && m == other.m &&
        n == other.n && k == other.k &&
        output_dtype == other.output_dtype && has_bias == other.has_bias;
  }
};

struct AutotuneKeyHash {
  size_t operator()(AutotuneKey const& key) const {
    size_t value = std::hash<int>{}(key.device_index);
    value = value * 31 + std::hash<int32_t>{}(key.m);
    value = value * 31 + std::hash<int32_t>{}(key.n);
    value = value * 31 + std::hash<int32_t>{}(key.k);
    value = value * 31 + std::hash<int>{}(static_cast<int>(key.output_dtype));
    return value * 31 + std::hash<bool>{}(key.has_bias);
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

std::atomic<bool>& autotune_enabled() {
  static std::atomic<bool> enabled{false};
  return enabled;
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

template <typename Definition>
void launch(
    torch::Tensor output,
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> const& bias,
    int swizzle) {
  using Gemm = typename Definition::Gemm;
  using GemmKernel = typename Definition::GemmKernel;
  using ElementA = typename Definition::ElementA;
  using ElementB = typename Definition::ElementB;
  using ElementD = typename Definition::ElementD;
  using ElementC = typename Definition::ElementC;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  int32_t m = activation.size(0);
  int32_t k = activation.size(1);
  int32_t n = weight.size(1);
  StrideA stride_a = make_stride(
      int64_t(activation.stride(0)), Int<1>{}, int64_t(0));
  StrideB stride_b = make_stride(
      int64_t(weight.stride(1)), Int<1>{}, int64_t(0));
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, make_shape(m, n, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, make_shape(m, n, 1));

  typename GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<ElementA const*>(activation.data_ptr()),
      stride_a,
      reinterpret_cast<ElementB const*>(weight.data_ptr()),
      stride_b,
  };
  typename GemmKernel::EpilogueArguments epilogue{
      Definition::prepare_epilogue(
          activation_scale.data_ptr<float>(),
          weight_scale.data_ptr<float>(),
          bias ? reinterpret_cast<ElementD const*>(bias->data_ptr()) : nullptr),
      nullptr,
      stride_c,
      reinterpret_cast<ElementD*>(output.data_ptr()),
      stride_d,
  };
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop,
      epilogue,
  };
  arguments.scheduler.max_swizzle_size = swizzle;

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this shape");
  TORCH_CHECK(
      Gemm::get_workspace_size(arguments) == 0,
      "Unexpected CUTLASS scheduler workspace");
  auto stream = at::cuda::getCurrentCUDAStream(activation.device().index());
  status = gemm.run(arguments, nullptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS kernel launch failed");
}

void validate(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> const& bias,
    torch::ScalarType output_dtype) {
  TORCH_CHECK(
      activation.is_cuda() && weight.is_cuda() &&
          activation_scale.is_cuda() && weight_scale.is_cuda(),
      "inputs and scales must be CUDA tensors");
  TORCH_CHECK(
      activation.device() == weight.device() &&
          activation.device() == activation_scale.device() &&
          activation.device() == weight_scale.device(),
      "inputs and scales must be on the same CUDA device");
  TORCH_CHECK(
      activation.scalar_type() == torch::kFloat8_e4m3fn &&
          weight.scalar_type() == torch::kFloat8_e4m3fn,
      "activation and weight must be float8_e4m3fn");
  TORCH_CHECK(
      activation.dim() == 2 && weight.dim() == 2,
      "activation and weight must be matrices");
  TORCH_CHECK(
      activation.stride(1) == 1 &&
          activation.stride(0) == activation.size(1),
      "activation must be contiguous");
  TORCH_CHECK(
      weight.stride(0) == 1 && weight.stride(1) == weight.size(0),
      "weight must be a transposed contiguous matrix");
  TORCH_CHECK(
      activation.size(1) == weight.size(0),
      "K dimensions must match");
  TORCH_CHECK(
      activation.size(0) > 0 &&
          activation.size(0) <= std::numeric_limits<int32_t>::max() &&
          weight.size(1) > 0 &&
          weight.size(1) <= std::numeric_limits<int32_t>::max() &&
          activation.size(1) > 0 &&
          activation.size(1) <= std::numeric_limits<int32_t>::max(),
      "M, N and K must be positive int32 values");
  TORCH_CHECK(
      activation_scale.scalar_type() == torch::kFloat32 &&
          weight_scale.scalar_type() == torch::kFloat32,
      "scales must be float32");
  TORCH_CHECK(
      activation_scale.is_contiguous() && weight_scale.is_contiguous(),
      "scales must be contiguous");
  TORCH_CHECK(
      activation_scale.numel() == activation.size(0) &&
          weight_scale.numel() == weight.size(1),
      "scale sizes must match the activation rows and weight columns");
  if (bias) {
    TORCH_CHECK(
        bias->is_cuda() && bias->device() == activation.device(),
        "bias must be on the same CUDA device as the inputs");
    TORCH_CHECK(
        bias->scalar_type() == output_dtype,
        "bias dtype must match the output dtype");
    TORCH_CHECK(
        bias->is_contiguous() && bias->dim() == 1 &&
            bias->size(0) == weight.size(1),
        "bias must be a contiguous vector matching the output columns");
  }
}

template <typename NarrowDefinition, typename WideDefinition>
void launch_config(
    torch::Tensor output,
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> const& bias,
    int64_t config_id) {
  KernelConfig const& config = kernel_config(config_id);
  if (config.wide_tile) {
    launch<WideDefinition>(
        output,
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        config.swizzle);
  } else {
    launch<NarrowDefinition>(
        output,
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        config.swizzle);
  }
}

template <
    typename NarrowDefinition,
    typename NarrowDefinitionWithBias,
    typename WideDefinition,
    typename WideDefinitionWithBias>
int64_t tune_config(
    AutotuneKey const& key,
    torch::Tensor output,
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> const& bias) {
  std::lock_guard measurement_lock(autotune_measurement_mutex());
  if (auto cached = cached_config_id(key)) {
    return *cached;
  }

  auto launch_candidate = [&](int64_t config_id) {
    if (bias) {
      launch_config<NarrowDefinitionWithBias, WideDefinitionWithBias>(
          output,
          activation,
          weight,
          activation_scale,
          weight_scale,
          bias,
          config_id);
    } else {
      launch_config<NarrowDefinition, WideDefinition>(
          output,
          activation,
          weight,
          activation_scale,
          weight_scale,
          bias,
          config_id);
    }
  };

  constexpr int kWarmups = 2;
  constexpr int kTrials = 3;
  constexpr int kRepeats = 5;
  constexpr int kConfigCount = static_cast<int>(kKernelConfigs.size());
  std::array<std::array<float, kTrials>, kConfigCount> timings{};

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    for (int index = 0; index < kConfigCount; ++index) {
      launch_candidate((index + warmup) % kConfigCount);
    }
  }

  auto stream = at::cuda::getCurrentCUDAStream(activation.device().index());
  int initial_offset =
      static_cast<int>((key.m + key.n + key.k) % kConfigCount);
  for (int trial = 0; trial < kTrials; ++trial) {
    int offset = (initial_offset + trial * 3) % kConfigCount;
    for (int index = 0; index < kConfigCount; ++index) {
      int config_id = (index + offset) % kConfigCount;
      c10::cuda::CUDAEvent start(cudaEventDefault);
      c10::cuda::CUDAEvent end(cudaEventDefault);
      start.record(stream);
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        launch_candidate(config_id);
      }
      end.record(stream);
      end.synchronize();
      timings[config_id][trial] =
          start.elapsed_time(end) / static_cast<float>(kRepeats);
    }
  }

  int64_t best_config_id = kFallbackConfigId;
  float best_time = std::numeric_limits<float>::max();
  for (int config_id = 0; config_id < kConfigCount; ++config_id) {
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

template <
    typename NarrowDefinition,
    typename NarrowDefinitionWithBias,
    typename WideDefinition,
    typename WideDefinitionWithBias>
torch::Tensor run(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> const& bias,
    torch::ScalarType output_dtype,
    c10::optional<int64_t> config_id = c10::nullopt) {
  validate(
      activation,
      weight,
      activation_scale,
      weight_scale,
      bias,
      output_dtype);
  c10::cuda::CUDAGuard guard(activation.device());
  int32_t m = activation.size(0);
  int32_t k = activation.size(1);
  int32_t n = weight.size(1);
  auto output = torch::empty(
      {m, n},
      activation.options().dtype(output_dtype));

  if (!config_id) {
    AutotuneKey key{
        activation.device().index(),
        m,
        n,
        k,
        output_dtype,
        bias.has_value(),
    };
    if (auto cached = cached_config_id(key)) {
      config_id = *cached;
    } else if (autotune_enabled().load(std::memory_order_relaxed)) {
      config_id = tune_config<
          NarrowDefinition,
          NarrowDefinitionWithBias,
          WideDefinition,
          WideDefinitionWithBias>(
          key,
          output,
          activation,
          weight,
          activation_scale,
          weight_scale,
          bias);
    } else {
      config_id = kFallbackConfigId;
    }
  }

  if (bias) {
    launch_config<NarrowDefinitionWithBias, WideDefinitionWithBias>(
        output,
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        *config_id);
  } else {
    launch_config<NarrowDefinition, WideDefinition>(
        output,
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        *config_id);
  }
  return output;
}

torch::Tensor run_with_dtype(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::ScalarType out_dtype,
    c10::optional<torch::Tensor> const& bias,
    c10::optional<int64_t> config_id = c10::nullopt) {
  if (out_dtype == torch::kBFloat16) {
    return run<
        NarrowGemm,
        NarrowGemmWithBias,
        WideGemm,
        WideGemmWithBias>(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        out_dtype,
        config_id);
  }
  TORCH_CHECK(
      out_dtype == torch::kFloat16,
      "output dtype must be bfloat16 or float16");
  return run<
      NarrowGemmFp16,
      NarrowGemmFp16WithBias,
      WideGemmFp16,
      WideGemmFp16WithBias>(
      activation,
      weight,
      activation_scale,
      weight_scale,
      bias,
      out_dtype,
      config_id);
}

}  // namespace

torch::Tensor cutlass_scaled_fp8_mm_f16_accum_sm120(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::ScalarType out_dtype,
    c10::optional<torch::Tensor> const& bias) {
  return run_with_dtype(
      activation,
      weight,
      activation_scale,
      weight_scale,
      out_dtype,
      bias);
}

torch::Tensor cutlass_scaled_fp8_mm_f16_accum_with_config_sm120(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::ScalarType out_dtype,
    c10::optional<torch::Tensor> const& bias,
    int64_t config_id) {
  return run_with_dtype(
      activation,
      weight,
      activation_scale,
      weight_scale,
      out_dtype,
      bias,
      config_id);
}

int64_t fp8_f16_accum_autotune_cache_abi_sm120() {
  return kAutotuneCacheAbi;
}

std::vector<std::string> fp8_f16_accum_autotune_configs_sm120() {
  std::vector<std::string> names;
  names.reserve(kKernelConfigs.size());
  for (KernelConfig const& config : kKernelConfigs) {
    names.emplace_back(config.name);
  }
  return names;
}

void set_fp8_f16_accum_autotune_config_sm120(
    int64_t device_index,
    int64_t m,
    int64_t n,
    int64_t k,
    torch::ScalarType out_dtype,
    bool has_bias,
    int64_t config_id) {
  TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
  TORCH_CHECK(
      m > 0 && m <= std::numeric_limits<int32_t>::max() &&
          n > 0 && n <= std::numeric_limits<int32_t>::max() &&
          k > 0 && k <= std::numeric_limits<int32_t>::max(),
      "M, N and K must be positive int32 values");
  TORCH_CHECK(
      out_dtype == torch::kBFloat16 || out_dtype == torch::kFloat16,
      "output dtype must be bfloat16 or float16");
  kernel_config(config_id);

  AutotuneKey key{
      static_cast<int>(device_index),
      static_cast<int32_t>(m),
      static_cast<int32_t>(n),
      static_cast<int32_t>(k),
      out_dtype,
      has_bias,
  };
  cache_config(key, config_id);
}

void set_fp8_f16_accum_autotune_enabled_sm120(bool enabled) {
  autotune_enabled().store(enabled, std::memory_order_relaxed);
}

torch::Tensor get_fp8_f16_accum_autotune_cache_sm120(
    int64_t device_index) {
  TORCH_CHECK(device_index >= 0, "device_index must be non-negative");
  std::vector<std::pair<AutotuneKey, int64_t>> entries;
  {
    std::shared_lock lock(autotune_cache_mutex());
    entries.reserve(autotune_cache().size());
    for (auto const& entry : autotune_cache()) {
      if (entry.first.device_index == device_index) {
        entries.push_back(entry);
      }
    }
  }
  std::sort(
      entries.begin(),
      entries.end(),
      [](auto const& left, auto const& right) {
        auto const& a = left.first;
        auto const& b = right.first;
        return std::tie(
                   a.device_index,
                   a.m,
                   a.n,
                   a.k,
                   a.output_dtype,
                   a.has_bias) <
            std::tie(
                   b.device_index,
                   b.m,
                   b.n,
                   b.k,
                   b.output_dtype,
                   b.has_bias);
      });

  auto result = torch::empty(
      {static_cast<int64_t>(entries.size()), 6},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto rows = result.accessor<int64_t, 2>();
  for (int64_t index = 0; index < static_cast<int64_t>(entries.size()); ++index) {
    auto const& [key, config_id] = entries[index];
    rows[index][0] = key.m;
    rows[index][1] = key.n;
    rows[index][2] = key.k;
    rows[index][3] = key.output_dtype == torch::kBFloat16 ? 0 : 1;
    rows[index][4] = key.has_bias;
    rows[index][5] = config_id;
  }
  return result;
}

void clear_fp8_f16_accum_autotune_cache_sm120(int64_t device_index) {
  std::unique_lock lock(autotune_cache_mutex());
  if (device_index < 0) {
    autotune_cache().clear();
    return;
  }

  for (auto entry = autotune_cache().begin(); entry != autotune_cache().end();) {
    if (entry->first.device_index == device_index) {
      entry = autotune_cache().erase(entry);
    } else {
      ++entry;
    }
  }
}
