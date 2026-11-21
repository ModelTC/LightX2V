---
name: support-new-model
description: 为 LightX2V 原生接入新的推理模型或任务，覆盖 checkpoint 加载、runner 与请求入口接线、模型和采样实现、CPU 权重 offload、多卡并行推理、torch.compile 编译、内置预热及上游对齐。
---

# 接入 LightX2V 模型

## 确定接入路径

- 从上游可运行的推理入口和目标 checkpoint 出发，核对任务、权重变体、元数据、分片、关键张量的名称、形状与精度。沿条件编码、模型预测和采样的实际调用链读到输出。从上游前向代码找出混合精度计算段，留意 autocast 的作用范围、`.float()` 等显式转换、升精度的归约或累加，以及结果在哪一步转回下游所需的 dtype。
- 按 DiT block、条件输入、token 布局和采样语义选择可复用的组件。允许分别选用不同家族的 runner、权重加载器、encoder、VAE 或 scheduler。按需查看 [Wan、Qwen-Image、MiniMax-H3 实现参考](references/model-components.md)。
- 正式入口执行 LightX2V 的原生核心模型。上游 pipeline 和 DiT 用于对齐，encoder、VAE、solver 等组件可沿用仓库已有的兼容封装及依赖。

## 接通启动与请求

启动沿 `build_startup_config()` → `build_runner()` 装配模型，每次请求经 `prepare_request()` → `run_request()` 进入推理。 [Runner factory](../../../lightx2v/models/runners/runner_factory.py)

1. 在 `RUNNER_MODULES` 添加 `ModelEntry`，模块路径相对于 `lightx2v.models.runners`。runner 用 `@RUNNER_REGISTER` 注册，其同目录 `request_spec.py` 声明 `SPECS`，三处使用相同的 `model_cls`。同一份 Spec 生成 CLI 和 HTTP 请求参数。Spec、loader 及父包的 `__init__.py` 保持轻量，确保 `--help` 只读配置与元数据。 [注册表](../../../lightx2v/models/runners/registry.py) · [CLI](../../../lightx2v/utils/cli.py)
2. 核对共享 loader 实际读到的 checkpoint 元数据。需要选择变体、分区或转换架构字段时，通过 `ModelEntry.config` 接入模型自己的 `load_config()`，用 `merge_config()` 合并部署配置。路径沿用含义匹配的现有配置字段，启动参数优先级以 `build_startup_config()` 为准。 [共享 loader](../../../lightx2v/models/runners/config.py) · [启动合并](../../../lightx2v/utils/set_config.py)
3. 在 Spec 用 `Argument` 声明每个任务的公开字段并复用 `request_args.py`。启动时固定任务用 `RequestSpec`。同一已加载模型支持切换任务时用 `MultiTaskSpec`，执行时按 `input_info.task` 分派，并在请求边界检查任务与已加载权重是否兼容。`BaseRunner` 生成请求字段，`input_info_state` 只声明额外运行状态。 [Spec 定义](../../../lightx2v/utils/request_spec.py) · [公共请求字段](../../../lightx2v/models/runners/request_args.py)
4. 部署参数和固定约束放启动 JSON 顶层，例如固定帧数 attention 的 `num_frames`。CLI 与 HTTP 共同支持且可覆盖的字段，在 `request_defaults` 中设置默认值，例如 `[height, width]` 格式的 `size`。同名字段只配置一次。推理读 `input_info`，初始化和预热按需用 `get_input_value()`。显式值覆盖默认值，`None` 按省略处理，`0`、`False` 和空容器仍是显式值。 [默认值校验](../../../lightx2v/utils/request_defaults.py) · [请求合并](../../../lightx2v/models/runners/base_runner.py)

## 实现推理

- **Runner** 复用 `DefaultRunner` 或最接近的 runner，加载组件并用 `model.set_scheduler()` 绑定 scheduler，处理请求编码、去噪循环、VAE 解码和输出。自定义流程沿用适用的取消与清理钩子，异常时也清理请求状态。用 `close()` 释放 runner 持有的长期资源。保存多个产物或改变输出路径时，用 `record_output()` 登记实际产物。 [DefaultRunner](../../../lightx2v/models/runners/default_runner.py)
- **Model 与权重** 优先复用 `BaseTransformerModel` 的加载逻辑，用 `WeightModule` 绑定已注册的 MM、norm、attention 等算子。兼容的 checkpoint 直接加载，确需重排或转换时把映射留在加载层或转换工具。按实际阶段组织 `pre_infer`、`transformer_infer`、`post_infer`。 [BaseTransformerModel](../../../lightx2v/models/networks/base_model.py) · [WeightModule](../../../lightx2v/common/modules/weight_module.py)
- **Infer** 用已注册算子的 `.apply()` 执行计算，设备和精度沿用该路径的约定。核对 latent、token、Q/K/V、mask、RoPE 和预测的轴顺序与 dtype，保留条件 token 的位置、残差、modulation 和敏感计算的精度。CFG 的分支数量、合成公式和归一化以目标模型为准。只实现与可复用代码有差异的计算。
- **混合精度** 对每个敏感运算，找到 LightX2V 中对应的代码，核对上游函数的输入、计算或累加、输出各自使用的 dtype。可调精度的敏感段用 `GET_SENSITIVE_DTYPE()`，主路径用 `GET_DTYPE()`。启动环境变量 `SENSITIVE_LAYER_DTYPE` 未设置或为 `None` 时，敏感精度跟随 `DTYPE`。按需匹配相关浮点权重和输入的精度，保留算法要求的累加精度，使用融合算子时也要核实这一点。将结果转回下游所需的 dtype。 [上游与 LightX2V 的对应示例](references/model-components.md#混合精度)
- **Scheduler** 管理初始噪声、latent、timestep/sigma 和求解器历史。常规去噪循环走 `step_pre → model.infer → step_post`，由 model 写入该 scheduler 消费的预测。`prepare()` 的签名、起止步和清理行为以具体 scheduler 为准，复用前先核对预测类型、更新方向、时间步和条件区域的更新规则。 [BaseScheduler](../../../lightx2v/models/schedulers/scheduler.py)

每个新模型都必须在正式推理路径中实现 CPU 权重 offload、多卡并行推理、编译和内置启动预热。给已有模型增加任务时，复用或适配这四项能力。量化与缓存按任务要求接入。继承的能力逐项核对实际执行路径，启动时拒绝不兼容的配置。

- **CPU offload** 将 `cpu_offload` 接入核心模型的权重加载和执行路径。按模型结构选择粒度，例如 DiT 可按 block 卸载。明确哪些权重常驻 CPU，以及何时可以复用或释放设备缓冲区。
- **多卡并行** 至少实现一种适合该模型的并行方式，例如序列并行、张量并行或 CFG 并行，让多张卡共同处理同一请求。使用 `parallel` 配置对应的进程组完成各 rank 的计算与通信。 [并行、offload 与缓存示例](references/model-components.md#并行offload-与缓存)
- **编译** 启用 `use_compile` 后，正式请求通过 `torch.compile` 执行模型的核心计算。对 transformer block，复用 `BaseTransformerInfer.init_compile()`，在正式循环中调用 `run_block()`。权重搬运和缓冲区交换放在编译图外。`use_compile=false` 时以 eager 模式执行同一计算。 [编译接入指南](../support-model-compile/SKILL_ZH.md)
- **启动预热** `warmup=true` 时，通过 runner 的初始化钩子在首个正式请求前预热一次。`DefaultRunner` 子类实现或复用 `run_warmup()`，用代表性形状和分支执行该任务的实际推理阶段。隔离预热使用的随机数状态，结束或异常退出时清理临时输入和求解器历史，保留可复用的编译图。`warmup` 和 `use_compile` 默认关闭，可独立启用。 [预热接入指南](../support-model-warmup/SKILL_ZH.md)

## 验证与交付

完整接入新模型时，提供与 checkpoint 和任务匹配的配置，并新建 `scripts/<model>/`。目录根部放 CLI Bash 脚本、英文 `README.md` 和中文 `README_zh.md`。`server/` 放服务启动 Bash 脚本和可直接运行的请求脚本（如 `post_<task>.py`），按当前 Spec 提交最小有效请求。两份 README 都写清权重与配置准备、CLI 用法、服务启动、请求脚本运行方式和结果获取方式。 [英文示例](../../../scripts/minimax_h3_image/README.md) · [中文示例](../../../scripts/minimax_h3_image/README_zh.md) · [请求脚本示例](../../../scripts/minimax_h3_image/server/post_t2i.py)

Bash 脚本把可编辑路径集中在开头，只显式设置运行必需的环境变量。部署默认值沿用项目配置，详细准备步骤写进 README。CLI 和服务脚本分别调用 `python -m lightx2v.infer` 和 `python -m lightx2v.server`。为四项能力提供可运行配置，在中英文 README 写明启动命令、offload 粒度、并行方式和卡数，以及怎样在启动 JSON 中独立设置 `use_compile` 和 `warmup`。多卡推理使用 `torchrun` 启动。

模型使用混合精度时，在启动示例中给出与上游精度选择对应的 `DTYPE` 和 `SENSITIVE_LAYER_DTYPE` 设置。

1. 逐个对交付的 Bash 脚本运行 `bash -n`，核对两份 README 的命令、路径与参数。先跑 `python -m lightx2v.infer --help`，再带 `--model_cls`、`--task`、`--model_path`、`--config_json` 和所需变体参数检查任务级 `--help`。验证默认值、显式覆盖、拒绝未声明字段，以及参数确实参与计算。
2. 用可用权重和最小有效输入跑 CLI 脚本，并通过交付的请求脚本验证服务。检查生成的张量或产物、尺寸与适用的帧数、FPS、采样率。复用 runner 连续执行时检查 seed、状态重置，以及所支持的任务切换。
3. 单卡能容纳模型时，用关闭 offload 和预热的单卡 eager 推理作基线。否则使用相同输入的已验证参考结果，注明比较范围。分别运行 CPU offload 和多卡并行，固定 checkpoint、输入、精度和初始噪声（如适用）。offload 要确认权重实际发生搬运，并记录峰值显存以便对照。多卡要用至少两张卡运行同一请求，核实各 rank 的计算分工。各模式的输出形状应与基线一致。检查预测中有无非预期的 NaN/Inf，结合实际算子、dtype 和任务效果判断数值差异。对声明支持的 offload 与并行组合，以及按任务接入的优化，也做同样的对比。
4. 编译与预热只验收 `use_compile=false, warmup=false`（eager 基线）和 `use_compile=true, warmup=true` 两组。第 3 步的配置、输入和 seed 一致时，直接复用基线结果。两组使用相同的输入和 seed，各自在同一 runner 上连续发两个请求，逐次对比两组输出。全关时预热次数应为零。全开时在启动阶段预热一次，用编译日志或计数器确认首个正式请求复用了预热为相同输入规格生成的编译图。分别记录启动预热、首个请求和后续请求的耗时，编译耗时计入实际发生的阶段。 [Wan 配置和实现](references/model-components.md#编译与启动预热)
5. 用实际条件分支的代表性输入与上游对照，固定 checkpoint、精度设置和适用时的实际初始噪声。更换算子会改变舍入或归约顺序，允许逐元素数值差异，结合模型已有的质量指标或代表性输出判断影响。对可调精度的敏感运算，用小输入抽查实际计算 dtype 以及下游收到的 dtype。出现非预期的 NaN/Inf、无法解释的数值漂移或明显质量退化时，再从相关的条件编码、模型预测、scheduler 单步更新或解码阶段定位原因。
6. 交付代码前，删除本次接入为测试或调试创建的所有临时文件和脚本。用 `git status --short --untracked-files=all` 核对交付内容。
7. 报告已验证的命令和配置、实际算子和 dtype、输出质量，以及观察到的代表性数值差异。缺少权重、上游环境或足够的加速卡时，仍需交付四项能力的实现与可运行配置，把受影响的验证标为“待验证”，写明补验命令和所需条件。
