#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <utility>
#include <tuple>
#include <array>
#include <climits>

static void check(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status));
}

// Selected on Thor with cuBLASLt 130100; see tuning/cublaslt_search/report.md.
// Fields: algorithm, tile, split-K, reduction, swizzle, custom, stages, inner, cluster.
// Down-only adaptation; runtime M is part of the descriptor cache key.
// Tile dimensions describe the column-major [N,M] output Y.T.
static constexpr std::array<int, 9> kDownConfig = {66, CUBLASLT_MATMUL_TILE_256x224, 1, CUBLASLT_REDUCTION_SCHEME_NONE, 0, 3, CUBLASLT_MATMUL_STAGES_64xAUTO, CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED, CUBLASLT_CLUSTER_SHAPE_2x2x1};

// Each host thread/device/shape owns its descriptors. No input/weight copies or
// workspace are cached. The fixed algorithm is initialized once; no heuristic search.
struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulHeuristicResult_t result{};
    ~Plan() {
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
    void init(int m, int n, int k, void* bias) {
        check(cublasLtCreate(&handle));
        check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        // Compute Y^T = W X^T as column-major [N,M]. This makes the
        // epilogue's row bias correspond to the original N output channels.
        cublasOperation_t trans = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        auto epilogue = CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
        const auto& cfg = kDownConfig;
        cublasLtMatmulAlgo_t algo;
        check(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, cfg[0], &algo));
        for (int i = 1; i < 9; ++i) {
            // cuBLASLt defines inner/cluster IDs as uint16_t, other fields as 32-bit.
            uint16_t small = cfg[i];
            check(cublasLtMatmulAlgoConfigSetAttribute(&algo,
                static_cast<cublasLtMatmulAlgoConfigAttributes_t>(i),
                i >= 7 ? static_cast<const void*>(&small) : static_cast<const void*>(&cfg[i]),
                i >= 7 ? sizeof(small) : sizeof(int)));
        }
        check(cublasLtMatmulAlgoCheck(handle, op, a, b, c, c, &algo, &result));
        check(result.state);
        TORCH_CHECK(result.workspaceSize <= 32 * 1024 * 1024, "Fixed algorithm exceeds 32 MiB workspace limit");
        result.algo = algo;
    }
};

torch::Tensor run(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
    for (const auto& t : {x, w, bias}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), "Expected contiguous CUDA BF16 tensors");
        TORCH_CHECK(t.device() == x.device(), "Device mismatch");
        TORCH_CHECK(!t.requires_grad(), "Inference only");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "Expected 16-byte alignment");
    }
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && bias.dim() == 1, "Invalid ranks");
    int64_t m = x.size(0), k = x.size(1), n = w.size(0);
    TORCH_CHECK(m > 0 && m <= INT_MAX && k == 13824 && n == 5120 && w.size(1) == k && bias.size(0) == n, "Unsupported shape");
    c10::cuda::CUDAGuard guard(x.device());
    auto prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major == 11 && prop->minor == 0, "This build targets Thor SM110a");
    static thread_local std::map<std::tuple<int, int, int, uintptr_t>, std::unique_ptr<Plan>> plans;
    uintptr_t address = reinterpret_cast<uintptr_t>(bias.data_ptr());
    uintptr_t bias_alignment = address & (~address + 1);
    auto key = std::make_tuple(x.get_device(), int(m), int(k), bias_alignment);
    auto it = plans.find(key);
    if (it == plans.end()) {
        auto plan = std::make_unique<Plan>();
        plan->init(m, n, k, bias.data_ptr());
        it = plans.emplace(key, std::move(plan)).first;
    }
    auto& plan = *it->second;
    void* bias_ptr = bias.data_ptr();
    check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    auto y = torch::empty({m, n}, x.options());
    auto workspace = torch::empty({static_cast<int64_t>(plan.result.workspaceSize)}, x.options().dtype(at::kByte));
    float alpha = 1.f, beta = 0.f;
    check(cublasLtMatmul(plan.handle, plan.op, &alpha, w.data_ptr(), plan.a,
        x.data_ptr(), plan.b, &beta, y.data_ptr(), plan.c, y.data_ptr(), plan.c,
        &plan.result.algo, workspace.data_ptr(), plan.result.workspaceSize,
        at::cuda::getCurrentCUDAStream()));
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
