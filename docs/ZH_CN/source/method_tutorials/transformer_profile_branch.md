# Transformer Profile 分支维护契约

## 分支定位

本分支是保留开发工具分支，用于沉淀 DiT 单步和单层 profile。它不要求立即进入 `main`，但必须能够由 AI 编程助手或开发者定期 rebase 到最新 `main`，并以有限、明确的模型适配层继续扩展。

稳定备份为 `yr/bk_torch_profile`；持续维护分支为 `reserved_dev_tool/transformer-profile`。

## 不变量

维护时必须保持以下约束：

1. 未设置 `LIGHTX2V_PROFILE_MODE` 时，不改变模型数值路径、调用次数和 scheduler 状态。
2. profile 只执行一次真实模型调用，不为 profiler schedule 重放 diffusion step 或 block。
3. warmup 不得提前消费目标 step。
4. `full` 只负责完整 step trace；`block` 只包目标 block，并生成 trace、op JSONL 和文本报告。
5. 通用模块不得导入具体模型；模型差异留在 `block_profile.py`、`block_profile_report.py` 和少量自然计算边界 hook 中。
6. 不为了 profile 大规模重写 transformer；优先在既有 self-attention、cross-attention、FFN、MoE、adapter 函数上加 region。
7. 不支持的 compile/offload/layout 必须明确报错，不能静默生成不具代表性的报告。
8. 未知 GPU 不得阻断报告，但不能猜测 peak TFLOPS。
9. 稀疏算子的 FLOPs 必须注明口径；当前动态稀疏 attention 使用 `dense-equivalent`。
10. 不提交 `prof_results/`、模型、媒体结果、个人路径或临时运行配置。

## 文件边界

通用核心：

| 文件 | 职责 |
|---|---|
| `lightx2v/utils/transformer_profile.py` | env 解析、目标校验、一次性 full/block 生命周期、输出目录 |
| `lightx2v/utils/region_profile.py` | 仅在目标 block 激活的 region 与 active model profile |
| `lightx2v/utils/op_shape_trace.py` | 记录逻辑 GEMM / ATTN / MoE shape 和 FLOPs 口径 |
| `lightx2v/utils/torch_trace_profiler.py` | PyTorch profiler trace 导出 |
| `tools/profile/region_event_trace.py` | kernel 分类、region 归属、op 配对、文本报告 |
| `tools/profile/trace_correlation.py` | GPU event 与 CPU runtime launch 关联 |
| `tools/profile/profiler_step_gap.py` | full trace 的 GPU active/gap/sync API 快速统计 |

模型适配层：

| 文件形态 | 职责 |
|---|---|
| `models/networks/<model>/infer/block_profile.py` | 从真实权重和运行时 tensor 绑定逻辑 op shape |
| `models/networks/<model>/infer/block_profile_report.py` | region 顺序、特殊 op 展开、报告配置 |
| `transformer_infer.py` | 在 step 入口调用 `record_transformer()`，在 block loop 调用 `record_block()` |
| attention / MoE / adapter 文件 | 只添加自然 region 或必要的动态 shape hook |

## 支持矩阵

| model class | canonical config / script | full | block | 最新验证状态 |
|---|---|---:|---:|---|
| `hunyuan3d` | `configs/hunyuan3d/hunyuan3d_shape.json` / `scripts/hunyuan3d/run_hunyuan3d.sh` | 是 | 是，含 self/cross attention、dense FFN、MoE | 已适配最新 main 并静态编译；本机缺模型，尚未重跑 E2E |
| `seko_talk` | `configs/seko_talk/seko_talk_01_base.json` / `scripts/seko_talk/run_seko_talk_01_base.sh` | 是 | 是，含 audio adapter | 已适配最新 main，Wan 回归测试通过；本机缺模型，尚未重跑 E2E |
| `wan2.2_moe_distill` | `configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json` / `scripts/wan22/extreme/run_wan22_moe_t2v_extreme.sh` | 是 | 是，含 NVFP4、dynamic sparse attention、text cross-attention | 2026-07-24 在 RTX 5090 单卡完成 full step 1 与 block layer 20 E2E；两次视频、full trace 和三类 block 产物均成功 |

基础 Wan transformer 的 hook 可能自然覆盖更多派生模型，但未进入此表的模型不能视为已支持；必须拿到模型、canonical config 和脚本后再确认。

## 跟进最新 main

每次更新按以下顺序执行：

1. 先把当前稳定 HEAD 备份到新的 `yr/bk_torch_profile[_date]`。
2. 从维护分支执行 `git rebase main`，逐个解决冲突，不用 `ours/theirs` 整体覆盖模型文件。
3. 优先检查这些高冲突位置：
   - transformer 构造函数和 block loop；
   - Wan 的 `torch.compile`、sequence parallel 与 offload 分支；
   - Hunyuan3D MoE 中 expert count 的含义，profile 必须记录 cumsum 前的每 expert token 数；
   - runner warmup 生命周期。
4. 对比 `main..HEAD`，确认通用工具没有重新引入模型语义，模型 hook 没有扩散到无关文件。
5. 执行下文最低验证；没有模型权重的条目必须在支持矩阵中如实标注，不能用静态检查冒充 E2E。

发生 rebase conflict 时：编辑冲突文件保留 main 的新功能和 profile 的必要 hook，`git add <file>` 后执行 `git rebase --continue`；需要放弃本次 rebase 时使用 `git rebase --abort`，稳定备份分支不受影响。

## 适配新模型或配置

1. 确认 canonical model class、config、run script、模型准备方式和预期输出。
2. 找到一次 diffusion step 的 transformer 入口与实际 block loop。
3. 先接 `full`：在入口根据 scheduler 的 0-based step index 调用 `mode_for_step()` 和 `record_transformer()`。
4. 再接 `block`：目标 block 执行前绑定真实权重/输入 shape，用 `should_record_block()` 和 `record_block()` 包住原调用。
5. 按模型自然边界添加 region；不得为追求报告细粒度拆散核心推理结构。
6. 在 `block_profile.py` 记录逻辑 shape。量化权重必须还原逻辑维度，例如 packed FP4 的 K 不能直接使用存储 shape。
7. 用真实 trace 扩展 kernel 分类规则；不要仅凭 kernel 名称猜测 op 顺序。
8. 将 canonical 案例与实测日期加入支持矩阵。

## 最低验证

每次跟进 main：

```bash
python -m py_compile \
  lightx2v/utils/transformer_profile.py \
  lightx2v/utils/region_profile.py \
  lightx2v/models/networks/hunyuan3d/infer/transformer_infer.py \
  lightx2v/models/networks/wan/infer/transformer_infer.py

python -m unittest \
  test_cases.test_transformer_profile \
  test_cases.test_region_event_trace \
  test_cases.test_transformer_block_profile_shapes \
  test_cases.test_wan_mxfp8_fuse_forwarding \
  test_cases.test_wan_feature_cache_cfg_state
```

每个具有本地模型的 support-matrix 条目，还必须执行：

1. 不开 profile 的 canonical E2E，确认媒体/mesh 输出成功。
2. `full` 模式，确认只生成 `*.pt.trace.json`。
3. `block` 模式，确认生成 trace、op JSONL、文本报告。
4. 检查目标 step/layer、主要 token 数、GEMM K/N、region assigned 比例和核心 kernel 配对。
5. 对量化、稀疏、MoE、adapter 等特例检查 FLOPs 口径，不只检查文件存在。

生成物只作为本地验证证据，不进入 Git。提交前运行 `pre-commit run --all-files` 并确认格式化修改被纳入对应功能提交。
