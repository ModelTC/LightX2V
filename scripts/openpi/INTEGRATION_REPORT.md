# OpenPI pi0.5-LIBERO PyTorch 转换与 LightX2V 接入报告

## 结论

本次接入采用以下单一路径：

```text
官方 pi05_libero Orbax/JAX checkpoint
  -> 官方 convert_jax_model_to_pytorch.py
  -> model.safetensors
  -> LightX2V 原生 PyTorch network
  -> OpenPI runner
  -> 离线 i2va 或 ROS LIBERO 闭环
```

LightX2V 运行时不 import `/data/liuhongda/openpi`、JAX、Flax 或 Orbax，也不启动
OpenPI server、网页或 viewer。

## 权重转换结果

- 本地 OpenPI 源码版本：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- 转换入口：OpenPI 官方 `examples/convert_jax_model_to_pytorch.py`
- 转换配置：`pi05_libero`，输出精度 `bfloat16`
- JAX 输入：
  `/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero`
- PyTorch 输出：
  `/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero_pytorch`
- `model.safetensors`：7,233,650,408 bytes，812 个 tensor keys
- 逻辑 state dict 为 813 keys；其中 1 个是 SafeTensors metadata 记录的合法 tied-weight alias
- key/shape/dtype manifest SHA256：
  `ee81d609ff73d395731f9f3df3b0caefcbd17f83d2ff153d26166e1bd024e20d`
- SHA256：
  `812e78eb87ddcf6acd87acaf1561a0adcda7ed035b5733b4371064c2f1a16a77`
- 参数总量：3,616,757,520
- 严格加载结果：`missing=[]`、`unexpected=[]`

完整权重目录：

```text
pi05_libero_pytorch/
├── model.safetensors
├── config.json
├── SHA256SUMS
└── assets/
    ├── paligemma_tokenizer.model
    └── physical-intelligence/libero/norm_stats.json
```

`assets` 是显式补齐的：官方转换脚本查找 `checkpoint_dir.parent/assets`，而本地发布
权重把它放在 `checkpoint_dir/assets`。

## 模型结构

- vision prefix：SigLIP，hidden 1152，27 层，16 heads
- language prefix：Gemma 2B，width 2048，18 层，8 heads，1 KV head
- action expert：Gemma 300M，width 1024，18 层，8 heads，1 KV head
- π0.5 AdaRMS：由 timestep MLP 条件控制 action expert
- action：内部 padding 到 32 维，horizon 为 10
- sampler：10 步 Euler flow matching
- LIBERO 输出：反归一化后截取前 7 维，得到 `(10, 7)`

关键参数树保持官方名字不变，例如：

```text
paligemma_with_expert.paligemma.*
paligemma_with_expert.gemma_expert.*
action_in_proj.*
action_out_proj.*
time_mlp_in.*
time_mlp_out.*
```

## LightX2V 文件组织

```text
configs/openpi/pi05_libero.json
lightx2v/models/networks/openpi/
├── config.py
├── gemma.py
├── image_tools.py
├── model.py
├── observation.py
├── pi0.py
├── preprocessing.py
├── infer/{pre_infer.py,transformer_infer.py,post_infer.py}
├── weights/loader.py
└── transformers_replace/...
lightx2v/models/runners/openpi/openpi_runner.py
lightx2v/models/runners/openpi/libero_rollout.py
lightx2v_ros/src/inference/inference/openpi_node/main.py
scripts/openpi/
├── convert_pi05_libero_to_pytorch.sh
├── setup_pytorch_runtime.sh
├── prepare_libero_sample.py
├── validate_pytorch_parity.py
├── run_libero_i2va.sh
└── run_libero_ros_i2va.sh
```

网络按 LightX2V 的 `networks / infer / weights / runners` 边界组织。没有强行继承
视频生成用的 `BaseTransformerModel`，因为那会改变官方 SafeTensors 参数树。

## 输入到输出

```text
agentview RGB + wrist RGB + state(8) + task description
  -> 两张有效图 + 一张全零、mask=false 的右腕占位图
  -> resize-with-pad 224x224
  -> state q01/q99 quantile normalize + pad 到 32
  -> PaliGemma SentencePiece tokenize + pad 到 200
  -> pi0.5 sample_actions，输出 (1,10,32)
  -> actions q01/q99 unnormalize
  -> slice 前 7 维，输出 float32 (10,7)
```

与官方 OpenPI 的 image、mask、state、token 和 token mask 预处理逐元素完全一致。

## 环境隔离

base 未被修改：

```text
/opt/conda/bin/python
torch 2.8.0+cu128
transformers 5.14.1
```

OpenPI 启动脚本只在该进程的 `PYTHONPATH` 前置：

```text
/data/liuhongda/openpi_data/python_deps/openpi_pytorch_runtime
```

其中包含 `transformers 4.53.2`、`huggingface-hub 0.32.3`、
`tokenizers 0.21.1` 和官方五个 Transformers replacement 文件。通用
OpenPI 不修改通用 `lightx2v.infer` 入口。静态单观测推理直接调用 OpenPI 自己的
`lightx2v.models.runners.openpi.single_observation`，避免隔离的 Transformers
4.53.2 eager import 需要更新 Transformers API 的无关模型。

两个启动脚本还设置 `USE_FLAX=0`，防止 Transformers 因 base 中存在 Flax 而
自动加载 JAX。实测导入 patched Transformers/SigLIP 后 `jax_loaded=False`、
`flax_loaded=False`。

## 启动方式

转换和私有运行时：

```bash
cd /data/liuhongda/LightX2V
bash scripts/openpi/convert_pi05_libero_to_pytorch.sh
bash scripts/openpi/setup_pytorch_runtime.sh
```

本地 LIBERO 闭环推理并录制机器人执行视频：

```bash
bash scripts/openpi/run_libero_i2va.sh
```

默认输出 MP4、实际执行动作轨迹和成功指标。切换任务：

```bash
LIBERO_BENCHMARK=libero_goal \
LIBERO_TASK_ID=3 \
LIBERO_INIT_STATE_ID=0 \
OPENPI_SAVE_VIDEO_PATH=/absolute/output/rollout.mp4 \
OPENPI_SAVE_ACTION_PATH=/absolute/output/rollout.actions.npy \
bash scripts/openpi/run_libero_i2va.sh
```

单帧静态 smoke inference 仍可用：

```bash
OPENPI_RUN_MODE=single_observation \
OPENPI_IMAGE_PATH=/absolute/input_dir \
OPENPI_STATE_PATH=/absolute/input_dir/state.npy \
OPENPI_TASK_DESCRIPTION="pick up the black bowl and place it on the plate" \
OPENPI_SAVE_ACTION_PATH=/absolute/output/actions.npy \
bash scripts/openpi/run_libero_i2va.sh
```

ROS 闭环：

```bash
LIBERO_BENCHMARK=libero_10 \
LIBERO_TASK_ID=5 \
LIBERO_INIT_STATE_ID=0 \
bash scripts/openpi/run_libero_ros_i2va.sh
```

ROS 复用 LightX2V 的共享 LIBERO simulator contract。simulator 已执行官方要求的
180 度图像旋转，所以 OpenPI node、runner 和 network 均不再次 flip。

## 已完成验证

- 官方转换命令成功完成
- SafeTensors 六组关键参数前缀存在
- SafeTensors 的 key/shape/dtype manifest、参数量和 checksum 全部通过
- 官方模型严格加载 36.17 亿参数，无 missing/unexpected keys
- LightX2V 本地模型严格加载成功
- 官方预处理与 LightX2V 预处理逐元素一致
- H200 上使用 base Python 完成本地真实前向
- H200 上完成 `libero_spatial/task0/init0` 本地闭环：68 个 policy steps 后
  BDDL success
- 视频：
  `/data/liuhongda/LightX2V/save_results/output_openpi_pi05_libero.mp4`
- 视频属性：H.264 MP4、256x256、10 FPS、69 帧、6.9 秒
- 实际执行动作：float32、shape `(68, 7)`、全部 finite
- 指标：
  `/data/liuhongda/LightX2V/save_results/output_openpi_pi05_libero.metrics.json`
- 官方 OpenPI PyTorch 与 LightX2V 固定输入/固定 noise：
  max abs error `2.613627914094252e-08`，`atol=1e-6` 通过
- 对齐报告：
  `/data/liuhongda/openpi_data/results/pi05_libero_pytorch_parity.json`
- Python AST/compile、JSON、shell、`git diff --check` 均通过

非 ROS 本地 MuJoCo 闭环已经实际跑通。当前机器没有可 source 的 ROS2/colcon 环境，
因此 ROS 节点完成了静态契约检查，但没有执行 ROS 版本的闭环 rollout。

## 训练边界

`PI0Pytorch.forward()`、训练 loss 和 gradient-checkpointing 接口已随官方 PyTorch
架构保留在 `networks/openpi`，可作为后续 LightX2V kernel/offload/量化优化入口。
本次两个目标 runner 是离线推理和 ROS 推理；数据加载、optimizer、DDP 与 checkpoint
保存仍以 OpenPI README 中的 `scripts/train_pytorch.py pi05_libero ...` 为已验证基线，
没有在本次改动中伪造一套未验证的 LightX2V 训练调度器。

如果后续在 LightX2V 内补训练，需要同时补 LIBERO dataset/collator、action 的
quantile normalize + 32 维 padding、optimizer/DDP 和 checkpoint 保存约定；应保存
`core_model.state_dict()`，避免外层 `OpenPIModel` 自动增加 `core_model.` 前缀。
