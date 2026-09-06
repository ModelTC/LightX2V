#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kMaxWorkspaceBytes = 128ULL * 1024 * 1024;
constexpr int kMaxHeuristicResults = 64;

void check_cublas(cublasStatus_t status, char const* expression) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      expression,
      " failed: ",
      cublasGetStatusString(status));
}

#define CUBLAS_CHECK(expression) check_cublas((expression), #expression)

struct ProblemKey {
  int device;
  int64_t m;
  int64_t n;
  int64_t k;
  uintptr_t stream;

  bool operator==(ProblemKey const& other) const {
    return device == other.device && m == other.m && n == other.n &&
        k == other.k && stream == other.stream;
  }
};

struct ProblemKeyHash {
  size_t operator()(ProblemKey const& key) const {
    size_t result = std::hash<int>{}(key.device);
    auto combine = [&result](size_t value) {
      result ^= value + 0x9e3779b9 + (result << 6) + (result >> 2);
    };
    combine(std::hash<int64_t>{}(key.m));
    combine(std::hash<int64_t>{}(key.n));
    combine(std::hash<int64_t>{}(key.k));
    combine(std::hash<uintptr_t>{}(key.stream));
    return result;
  }
};

struct PlanSet {
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t activation_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  torch::Tensor workspace;
  torch::Tensor beta;
  std::mutex execution_mutex;

  ~PlanSet() {
    if (output_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(output_layout);
    }
    if (activation_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(activation_layout);
    }
    if (weight_layout != nullptr) {
      cublasLtMatrixLayoutDestroy(weight_layout);
    }
    if (operation_desc != nullptr) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  }
};

cublasLtHandle_t cublaslt_handle() {
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t value = nullptr;
    CUBLAS_CHECK(cublasLtCreate(&value));
    return value;
  }();
  return handle;
}

void check_inputs(
    torch::Tensor const& output,
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& alpha,
    torch::Tensor const& bias) {
  TORCH_CHECK(
      output.is_cuda() && activation.is_cuda() && weight.is_cuda(),
      "output, activation, and weight must be CUDA tensors");
  TORCH_CHECK(
      activation_scale.is_cuda() && weight_scale.is_cuda(),
      "scale tensors must be CUDA tensors");
  TORCH_CHECK(alpha.is_cuda() && bias.is_cuda(), "alpha and bias must be CUDA tensors");

  int const device = activation.get_device();
  TORCH_CHECK(
      output.get_device() == device && weight.get_device() == device &&
          activation_scale.get_device() == device && weight_scale.get_device() == device &&
          alpha.get_device() == device && bias.get_device() == device,
      "all tensors must be on the same CUDA device");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      activation.is_contiguous() && weight.is_contiguous(),
      "activation and weight must be contiguous");
  TORCH_CHECK(
      activation_scale.is_contiguous() && weight_scale.is_contiguous(),
      "scale tensors must be contiguous");
  TORCH_CHECK(alpha.is_contiguous() && bias.is_contiguous(), "alpha and bias must be contiguous");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be BF16");
  TORCH_CHECK(activation.scalar_type() == at::ScalarType::Byte, "packed activation must be uint8");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Byte, "packed weight must be uint8");
  TORCH_CHECK(
      activation_scale.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          weight_scale.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "scale tensors must be float8_e4m3fn");
  TORCH_CHECK(alpha.scalar_type() == at::ScalarType::Float, "alpha must be FP32");
  TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be BF16");
  TORCH_CHECK(
      activation.dim() == 2 && weight.dim() == 2 && output.dim() == 2,
      "activation, weight, and output must be matrices");
  TORCH_CHECK(
      activation.sizes()[1] == weight.sizes()[1],
      "activation and weight packed K dimensions must match");

  int64_t const m = activation.sizes()[0];
  int64_t const n = weight.sizes()[0];
  int64_t const k = activation.sizes()[1] * 2;
  TORCH_CHECK(alpha.numel() == 1, "alpha must contain one value");
  TORCH_CHECK(
      output.sizes() == at::IntArrayRef({m, n}),
      "output shape must be (",
      m,
      ", ",
      n,
      ")");
  TORCH_CHECK(bias.numel() == n, "bias must contain ", n, " values");

  auto round_up = [](int64_t value, int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
  };
  int64_t const rounded_m = round_up(m, 128);
  int64_t const rounded_n = round_up(n, 128);
  int64_t const rounded_k_scale = round_up(k / 16, 4);
  TORCH_CHECK(
      activation_scale.sizes() == at::IntArrayRef({rounded_m, rounded_k_scale}),
      "activation scale shape must be (",
      rounded_m,
      ", ",
      rounded_k_scale,
      ")");
  TORCH_CHECK(
      weight_scale.sizes() == at::IntArrayRef({rounded_n, rounded_k_scale}),
      "weight scale shape must be (",
      rounded_n,
      ", ",
      rounded_k_scale,
      ")");
}

void set_dynamic_pointers(
    PlanSet& plan,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& bias) {
  void const* weight_scale_pointer = weight_scale.data_ptr();
  void const* activation_scale_pointer = activation_scale.data_ptr();
  void const* bias_pointer = bias.data_ptr();
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation_desc,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &weight_scale_pointer,
      sizeof(weight_scale_pointer)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation_desc,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &activation_scale_pointer,
      sizeof(activation_scale_pointer)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan.operation_desc,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_pointer,
      sizeof(bias_pointer)));
}

std::shared_ptr<PlanSet> create_plan_set(
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& bias) {
  auto plan = std::make_shared<PlanSet>();
  int64_t const m = activation.sizes()[0];
  int64_t const n = weight.sizes()[0];
  int64_t const k = activation.sizes()[1] * 2;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &plan->operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transpose_weight = CUBLAS_OP_T;
  cublasOperation_t transpose_activation = CUBLAS_OP_N;
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  cublasLtMatmulMatrixScale_t block_scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cudaDataType_t bias_type = CUDA_R_16BF;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transpose_weight,
      sizeof(transpose_weight)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transpose_activation,
      sizeof(transpose_activation)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_POINTER_MODE,
      &pointer_mode,
      sizeof(pointer_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &block_scale_mode,
      sizeof(block_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &block_scale_mode,
      sizeof(block_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation_desc,
      CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
      &bias_type,
      sizeof(bias_type)));

  // Row-major (M,K) storage is column-major (K,M). Compute output^T.
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan->weight_layout, CUDA_R_4F_E2M1, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan->activation_layout, CUDA_R_4F_E2M1, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &plan->output_layout, CUDA_R_16BF, n, m, n));
  set_dynamic_pointers(*plan, activation_scale, weight_scale, bias);

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kMaxWorkspaceBytes,
      sizeof(kMaxWorkspaceBytes)));

  std::array<cublasLtMatmulHeuristicResult_t, kMaxHeuristicResults> results{};
  int returned_results = 0;
  cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      cublaslt_handle(),
      plan->operation_desc,
      plan->weight_layout,
      plan->activation_layout,
      plan->output_layout,
      plan->output_layout,
      preference,
      kMaxHeuristicResults,
      results.data(),
      &returned_results);
  cublasLtMatmulPreferenceDestroy(preference);
  CUBLAS_CHECK(heuristic_status);

  size_t workspace_size = 0;
  for (int index = 0; index < returned_results; ++index) {
    if (results[index].state != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    plan->algorithms.push_back(results[index].algo);
    workspace_size = std::max(workspace_size, results[index].workspaceSize);
  }
  TORCH_CHECK(!plan->algorithms.empty(), "cuBLASLt returned no NVFP4+bias algorithms");

  auto options = torch::TensorOptions().device(activation.device());
  plan->workspace = torch::empty(
      {static_cast<int64_t>(std::max<size_t>(workspace_size, 1))},
      options.dtype(torch::kUInt8));
  plan->beta = torch::zeros({1}, options.dtype(torch::kFloat32));
  return plan;
}

std::shared_ptr<PlanSet> get_plan_set(
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& bias,
    cudaStream_t stream) {
  static std::mutex cache_mutex;
  static std::unordered_map<ProblemKey, std::shared_ptr<PlanSet>, ProblemKeyHash> cache;
  ProblemKey const key{
      activation.get_device(),
      activation.sizes()[0],
      weight.sizes()[0],
      activation.sizes()[1] * 2,
      reinterpret_cast<uintptr_t>(stream)};

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second;
  }
  auto plan = create_plan_set(
      activation, weight, activation_scale, weight_scale, bias);
  cache.emplace(key, plan);
  return plan;
}

}  // namespace

void cublaslt_scaled_nvfp4_mm_bias_sm120(
    torch::Tensor& output,
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& alpha,
    torch::Tensor const& bias,
    int64_t algorithm_index) {
  check_inputs(
      output, activation, weight, activation_scale, weight_scale, alpha, bias);
  c10::cuda::CUDAGuard device_guard(activation.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(activation.get_device());
  auto plan = get_plan_set(
      activation, weight, activation_scale, weight_scale, bias, stream);

  int64_t selected_algorithm = algorithm_index;
  if (selected_algorithm == -1) {
    selected_algorithm = plan->algorithms.size() > 1 ? 1 : 0;
  }
  TORCH_CHECK(
      selected_algorithm >= 0 &&
          selected_algorithm < static_cast<int64_t>(plan->algorithms.size()),
      "cuBLASLt algorithm index ",
      selected_algorithm,
      " is outside [0, ",
      plan->algorithms.size(),
      ")");

  std::lock_guard<std::mutex> lock(plan->execution_mutex);
  set_dynamic_pointers(*plan, activation_scale, weight_scale, bias);
  CUBLAS_CHECK(cublasLtMatmul(
      cublaslt_handle(),
      plan->operation_desc,
      alpha.data_ptr(),
      weight.data_ptr(),
      plan->weight_layout,
      activation.data_ptr(),
      plan->activation_layout,
      plan->beta.data_ptr(),
      output.data_ptr(),
      plan->output_layout,
      output.data_ptr(),
      plan->output_layout,
      &plan->algorithms[selected_algorithm],
      plan->workspace.data_ptr(),
      plan->workspace.numel(),
      stream));
}

int64_t cublaslt_scaled_nvfp4_mm_bias_algo_count_sm120(
    torch::Tensor const& output,
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    torch::Tensor const& alpha,
    torch::Tensor const& bias) {
  check_inputs(
      output, activation, weight, activation_scale, weight_scale, alpha, bias);
  c10::cuda::CUDAGuard device_guard(activation.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(activation.get_device());
  auto plan = get_plan_set(
      activation, weight, activation_scale, weight_scale, bias, stream);
  return static_cast<int64_t>(plan->algorithms.size());
}
