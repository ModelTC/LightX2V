// CUTE FMHA compile-time tuning policies for validated Intel GPU targets.
#pragma once

namespace cute_fmha_config {

struct BMGTag {};
struct PTLHTag {};

template <
    typename ArchTag,
    int QTile,
    int KvTile,
    int VTile,
    int MmaK,
    int HeadDim,
    int SubgroupLayoutQ,
    int PipelineStages,
    int GrfSize>
struct Config {
  static constexpr int Q_TILE = QTile;
  static constexpr int KV_TILE = KvTile;
  static constexpr int V_TILE = VTile;
  static constexpr int MMA_K = MmaK;
  static constexpr int HEAD_DIM = HeadDim;
  static constexpr int SUBGROUP_LAYOUT_Q = SubgroupLayoutQ;
  static constexpr int PIPELINE_STAGES = PipelineStages;
  static constexpr int GRF_SIZE = GrfSize;
};

// Tuned against the original BMG/B60 workload.
using ConfigBMG = Config<BMGTag, 256, 32, 32, 32, 128, 16, 2, 256>;

// Internal PTL-H representative-workload validation retained this policy.
// Keep a separate type so later PTL-H tuning cannot silently alter BMG.
using ConfigPTLH = Config<PTLHTag, 256, 32, 32, 32, 128, 16, 2, 256>;

#if defined(OMNI_XPU_ARCH_BMG) && defined(OMNI_XPU_ARCH_PTL_H)
#error "Select only one omni_xpu_kernel GPU architecture"
#elif defined(OMNI_XPU_ARCH_BMG)
using ActiveConfig = ConfigBMG;
#elif defined(OMNI_XPU_ARCH_PTL_H)
using ActiveConfig = ConfigPTLH;
#else
#error "Define OMNI_XPU_ARCH_BMG or OMNI_XPU_ARCH_PTL_H"
#endif

}  // namespace cute_fmha_config
