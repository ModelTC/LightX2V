# OpenPI pi0.5-LIBERO 接入报告

## 接入结论

LightX2V 使用 OpenPI 官方转换器生成的 PyTorch 权重，并在本地直接执行
`pi05_libero`：

```text
Orbax/JAX checkpoint
  -> OpenPI convert_jax_model_to_pytorch.py
  -> model.safetensors
  -> LightX2V OpenPIModel
  -> OpenPIRunner
  -> local LIBERO rollout/evaluation
```

公共入口仍是 `python -m lightx2v.infer`。OpenPIRunner 启动并同步等待本地
子进程，使 OpenPI 的 Transformers 4.53.2 补丁不影响 LightX2V 主进程。

## 权重

- OpenPI 源码基线：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- 转换配置：`pi05_libero`
- 精度：`bfloat16`
- tensor keys：812
- 参数量：3,616,757,520
- 文件大小：7,233,650,408 bytes
- key/shape/dtype manifest SHA-256：
  `ee81d609ff73d395731f9f3df3b0caefcbd17f83d2ff153d26166e1bd024e20d`
- `model.safetensors` SHA-256：
  `812e78eb87ddcf6acd87acaf1561a0adcda7ed035b5733b4371064c2f1a16a77`

转换目录同时保存 tokenizer、LIBERO normalization statistics 和
`SHA256SUMS`。加载使用 SafeTensors strict 模式，不重写参数名。

## 模型结构

| 部分 | 配置 |
| --- | --- |
| Vision prefix | SigLIP，hidden 1152，27 层，16 heads |
| Language prefix | Gemma 2B，width 2048，18 层，8 heads |
| Action expert | Gemma 300M，width 1024，18 层，8 heads |
| Action horizon | 10 |
| Internal action dimension | 32 |
| LIBERO action dimension | 7 |
| Sampler | 10-step Euler flow matching |

输入处理：

```text
agentview RGB + wrist RGB + 8-D robot state + task text
  -> resize-with-pad 224 x 224
  -> add masked right-wrist placeholder
  -> q01/q99 state normalization and pad to 32
  -> PaliGemma tokenization and pad to 200
  -> pi0.5 sample_actions: (1, 10, 32)
  -> q01/q99 action denormalization
  -> LIBERO actions: (10, 7)
```

## 代码边界

```text
configs/openpi/
  pi05_libero.json
  pi05_libero_eval.json

lightx2v/models/networks/openpi/
  config.py
  model.py
  observation.py
  infer/
  weights/
  pi0.py
  gemma.py
  preprocessing.py
  transformers_replace/

lightx2v/models/runners/openpi/
  openpi_runner.py
  libero_rollout.py
  libero_evaluate.py

lightx2v_ros/src/inference/inference/openpi_node/
  main.py
```

`pi0.py`、`gemma.py`、`preprocessing.py` 和
`transformers_replace/` 保持 OpenPI 官方 PyTorch 参数树和计算路径。
LightX2V 自己的代码只负责配置、权重加载、前后处理、runner 调度和 ROS bridge。

模型没有继承视频生成专用的 `BaseTransformerModel`，避免改变官方
SafeTensors 参数层级。`OpenPIModel.forward()` 保留官方训练 loss 入口。

## 运行时隔离

OpenPI PyTorch 代码要求 Transformers 4.53.2 及五个 replacement 文件；其他
LightX2V 模型使用主环境中的 Transformers。两套依赖不会进入同一个进程：

```text
LightX2V process
  -> registry selects OpenPIRunner
  -> child PYTHONPATH prepends private OpenPI runtime
  -> child imports OpenPIModel and patched Transformers
```

私有 runtime 包含：

- `transformers==4.53.2`
- `huggingface-hub==0.32.3`
- `tokenizers==0.21.1`
- OpenPI Gemma、PaliGemma 和 SigLIP replacement files

`USE_FLAX=0` 只设置在 OpenPI 子进程。主环境的包不被安装、卸载或覆盖。

## LIBERO 协议

本地 rollout 和定量评测都直接调用 LIBERO MuJoCo 环境。成功条件是 policy
阶段的 `env.step(action)` 返回 `done=True`。图像采用 LIBERO simulator
输出的官方 180 度方向修正，network 和 ROS node 不重复翻转。

定量评测保留以下边界：

- 一个 suite 内连续使用 policy RNG，suite 边界重置
- 每个任务使用真实 init-state 数量检查
- episode 异常计入失败，CUDA OOM 终止
- episode 记录原子落盘
- 只在完整 suite 边界恢复
- 模型、任务输入、源码和协议 fingerprint 不一致时拒绝混用结果
- 部分运行不生成可误认为官方结果的完整成功率

## 验证记录

- 官方转换器完成 JAX 到 PyTorch 转换
- tensor manifest 和文件 checksum 通过
- SafeTensors strict load 通过
- OpenPI 与 LightX2V 预处理逐元素一致
- 固定输入和固定 flow noise 的最大动作误差：
  `2.613627914094252e-08`
- parity tolerance：`atol=1e-6`，通过
- `libero_spatial/task0/init0` 闭环在 68 个 policy steps 后成功
- 输出动作 shape `(68, 7)`，dtype `float32`，全部 finite
- Shell、JSON、Python compile、Ruff 和 pre-commit 检查通过

ROS 代码复用 LightX2V 现有 simulator contract。需要可 source 的 ROS2
workspace 才能进行运行时闭环验证。

## 训练边界

当前 LightX2V 接入提供模型结构、严格权重加载和训练 `forward()`，但没有新增
一套未经验证的训练调度器。训练仍以 OpenPI 的
`scripts/train_pytorch.py pi05_libero` 为基线。

后续在 LightX2V 内增加训练时，需要补齐 LIBERO dataset/collator、quantile
normalization、32 维 action padding、optimizer/DDP 和 checkpoint 保存约定。
保存权重时应使用 `core_model.state_dict()`，避免增加外层
`core_model.` 前缀。
