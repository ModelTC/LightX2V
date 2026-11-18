---
name: support-new-model
description: 为 LightX2V 原生接入新的推理模型或任务，覆盖 checkpoint 加载、runner 与请求入口接线、模型和采样实现及上游对齐。
---

# 接入 LightX2V 模型

## 确定接入路径

- 从上游可运行的推理入口和目标 checkpoint 出发，核对任务、权重变体、元数据、分片、关键张量的名称、形状与精度。沿实际调用链读到输出，确认条件编码、模型预测量和采样过程。
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
- **Scheduler** 管理初始噪声、latent、timestep/sigma 和求解器历史。常规去噪循环走 `step_pre → model.infer → step_post`，由 model 写入该 scheduler 消费的预测。`prepare()` 的签名、起止步和清理行为以具体 scheduler 为准，复用前先核对预测类型、更新方向、时间步和条件区域的更新规则。 [BaseScheduler](../../../lightx2v/models/schedulers/scheduler.py)

先跑通硬件可承载的基线路径，再接入任务要求的 offload、量化、并行或缓存。继承的能力逐项核对实际调用，已知不兼容的配置在入口报错。高级路径按需参考 [并行、offload 与缓存](references/model-components.md#并行offload-与缓存)。

## 验证与交付

完整接入新模型时，提供与 checkpoint 和任务匹配的配置，并新建 `scripts/<model>/`。目录根部放 CLI Bash 脚本、英文 `README.md` 和中文 `README_zh.md`。`server/` 放服务启动 Bash 脚本和可直接运行的请求脚本（如 `post_<task>.py`），按当前 Spec 提交最小有效请求。两份 README 都写清权重与配置准备、CLI 用法、服务启动、请求脚本运行方式和结果获取方式。 [英文示例](../../../scripts/minimax_h3_image/README.md) · [中文示例](../../../scripts/minimax_h3_image/README_zh.md) · [请求脚本示例](../../../scripts/minimax_h3_image/server/post_t2i.py)

Bash 脚本把可编辑路径集中在开头，只显式设置运行必需的环境变量。部署默认值沿用项目配置，详细准备步骤写进 README。CLI 和服务脚本分别调用 `python -m lightx2v.infer` 和 `python -m lightx2v.server`，多卡示例沿用仓库的 `torchrun` 用法。

1. 逐个对交付的 Bash 脚本运行 `bash -n`，核对两份 README 的命令、路径与参数。先跑 `python -m lightx2v.infer --help`，再带 `--model_cls`、`--task`、`--model_path`、`--config_json` 和所需变体参数检查任务级 `--help`。验证默认值、显式覆盖、拒绝未声明字段，以及参数确实参与计算。
2. 用可用权重和最小有效输入跑 CLI 脚本，并通过交付的请求脚本验证服务。检查生成的张量或产物、尺寸与适用的帧数、FPS、采样率。复用 runner 连续执行时检查 seed、状态重置，以及所支持的任务切换。对范围内的高级模式，用相同输入和基线比对。
3. 上游对齐时固定 checkpoint、输入、精度和实际初始噪声。按条件编码 → scheduler 初始状态 → pre-infer → 实际执行的代表性 block → 模型预测和单步更新 → VAE 解码定位首个差异。比较 shape 和数值误差，覆盖实际使用的条件分支。
4. 交付代码前，删除本次接入为测试或调试创建的所有临时文件和脚本。用 `git status --short --untracked-files=all` 核对交付内容。
5. 报告已验证的命令、配置、结果和误差，以及权重、上游环境或硬件缺失所限制的验证范围。按实际运行证据说明支持范围。
