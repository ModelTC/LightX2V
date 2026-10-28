# Codex skills

`skills/` 共 12 个顶层分组、28 个可调用 skill 入口，包括模型接入、性能工具流程与 Intel GPU 内核。Intel XPU 的两个子入口和内核 skills 保持嵌套组织；每个可调用入口的目录名与 frontmatter 的 `name` 一致。详细流程、示例和历史资产保留在各 skill 的 `references/`、`assets/` 中，按任务需要读取。

```text
skills/
├── support-model-warmup/
├── model-enable-intel-xpu/
├── lightx2v-kernel-skills/
├── support-new-model/
├── remove-ai-code-smell/
├── support-contiguous-block-offload/
├── support-cpu-block-offload/
├── support-static-block-residency/
├── benchmark-core-ops/
├── profile-model-module/
├── benchmark-module-parallelism/
└── support-model-compile/
```

上游的 24 个 skill 入口沿用 `policy.allow_implicit_invocation: false`，通过 `$skill-name` 显式调用。`support-static-block-residency` 和三个性能工具 skill 保留默认自动选择，也支持显式调用。

在 LightX2V 仓库根目录或子目录启动 Codex，通过 `/skills` 查看，或用 `$skill-name` 指定入口。Codex 的仓库 skill 搜索从当前工作目录向上至仓库根目录；在 LightX2V 的父目录启动不会因此自动发现子仓库的 skills。发现更新未显示时重启 Codex。规则见 [官方 skill 文档](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。

## 性能工具

| 任务 | 入口 |
|---|---|
| 核心算子后端比较，含独立 SP attention 调优 | [benchmark-core-ops](skills/benchmark-core-ops/SKILL.md) |
| 真实函数/模块性能采集与分析，含多 rank | [profile-model-module](skills/profile-model-module/SKILL.md) |
| 同一完整模块的 TP/SP 执行方案比较 | [benchmark-module-parallelism](skills/benchmark-module-parallelism/SKILL.md) |

任务按比较对象选择入口；模块并行比较按需复用 SP 算子筛选和模块分析。三个入口沿用 [tools 正式导航](../tools/AGENTS.md)及已有报告模板；模型特有输入与状态接线由当前任务补齐。

## 模型与推理

| 任务 | 入口 |
|---|---|
| 新模型原生接入 | [support-new-model](skills/support-new-model/SKILL.md) |
| Intel XPU 接入与排障 | [model-enable-intel-xpu](skills/model-enable-intel-xpu/SKILL.md) |
| XPU 首次运行诊断 | [xpu-exec](skills/model-enable-intel-xpu/xpu-exec/SKILL.md) |
| XPU 接入兼容入口 | [xpu-enable](skills/model-enable-intel-xpu/xpu-enable/SKILL.md) |
| Eager warmup | [support-model-warmup](skills/support-model-warmup/SKILL.md) |
| torch.compile | [support-model-compile](skills/support-model-compile/SKILL.md) |
| CPU block offload 与共享权重 | [support-cpu-block-offload](skills/support-cpu-block-offload/SKILL.md) |
| 静态 block 常驻权重与独立估测 | [support-static-block-residency](skills/support-static-block-residency/SKILL.md) |
| 连续 pinned 权重布局 | [support-contiguous-block-offload](skills/support-contiguous-block-offload/SKILL.md) |
| 去除冗余代码与过度防御 | [remove-ai-code-smell](skills/remove-ai-code-smell/SKILL.md) |

## Intel GPU 内核

- [基础能力索引](skills/lightx2v-kernel-skills/intel-xpu/kernel-basic-skills/README.md)：硬件、ESIMD、内存、DPAS、barrier 与性能分析。
- [具体内核索引](skills/lightx2v-kernel-skills/intel-xpu/kernel-specific-skills/README.md)：GEMM、GEMV、Flash Attention 与量化。
- [Python wheel 集成](skills/lightx2v-kernel-skills/intel-xpu/integration-skills/sycl-esimd-to-python-wheel/SKILL.md)：当前仓库构建入口，以及保留的[旧 Windows 示例工程](skills/lightx2v-kernel-skills/intel-xpu/integration-skills/sycl-esimd-to-python-wheel/assets/lightx2v_kernel_xpu/README.md)。

两个分组的 `README.md` 是普通索引，不是 skill 入口。`SKILL_EN.md` 是英文译文；Codex 读取的入口仍为 `SKILL.md`。`agents/openai.yaml` 维护调用策略及可选 UI 元数据，`agents/openai_en.yaml` 是可选英文模板，不会作为第二个元数据入口自动加载。
