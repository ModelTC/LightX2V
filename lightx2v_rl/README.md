# LightX2V RL

LightX2V 的少步图像与视频强化学习模块。算法、模型适配、数据与 reward 接口位于本模块，固定版本的 [verl-omni](../third_party/verl-omni) 提供分布式调度、worker、FSDP 训练引擎和权重同步。

| 能力 | 当前范围 |
| --- | --- |
| 算法 | CPS、Flow-GRPO、Geo-FSRL（Geo，对应 FewStepRL 中的 IMMB） |
| 图像模型 | Qwen-Image-2.1-Turbo；携带固定采样日程的 Qwen-Image-2.1 四步模型 |
| 图像任务 | T2I、单参考图 I2I、多参考图 I2I |
| 视频模型 | Wan2.1 单 transformer T2V，使用完整 Diffusers 格式权重 |
| Reward | UnifiedReward-Edit 9B / 27B、HPSv3、VideoAlign |
| 训练方式 | 同步在线 RL、全参数训练；Actor 默认 FP32 主参数、BF16 计算 |

Qwen rollout 使用官方 Diffusers pipeline，Wan rollout 复用 vLLM-Omni。LightX2V 原生加速 rollout、LoRA 训练、Wan2.2 双专家和 H3 尚未接入。

本文命令均使用 Bash，在 **LightX2V 仓库根目录**执行。配置中的模型、数据清单和输出路径相对仓库根目录解析；清单内部的图片路径相对清单所在目录解析。多机运行时，各节点须使用相同的共享挂载路径。

## 环境准备

### 系统依赖

推荐 Linux、Python 3.12、兼容所安装 PyTorch / vLLM CUDA runtime 的 NVIDIA 驱动。完整训练栈要求 Python ≥ 3.11；UnifiedReward 的固定 GPU wheel 要求 glibc ≥ 2.35。环境安装应在 GPU 节点或同规格镜像上完成。

Ubuntu 24.04 示例：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  python3.12 python3.12-dev python3-pip \
  git curl build-essential ca-certificates
nvidia-smi
```

Python 开发头文件必须与运行解释器一致。共享虚拟环境不能代替每个节点的系统依赖；建议将这些依赖预装到公共镜像。

### 资源目录与缓存

```text
lightx2v_rl/
├── .venv-rl/             # Actor、rollout、Ray、reward 客户端
├── .venv-reward/         # 可选：HPSv3 / VideoAlign 服务
├── .venv-unifiedreward/  # 可选：UnifiedReward 服务
├── models/              # 生成模型、rewards/ 与下载缓存，可使用相对软链接
├── data/                # 原始数据入口、适配后的 JSONL / Parquet 清单
└── outputs/             # 日志、checkpoint 与验证结果
```

以下示例将模型实体放在仓库同级的 `models/`，已存在的目录或链接会保留：

```bash
set -euo pipefail
mkdir -p ../models lightx2v_rl/data lightx2v_rl/outputs/setup
if [[ ! -e lightx2v_rl/models && ! -L lightx2v_rl/models ]]; then
  ln -s ../../models lightx2v_rl/models
fi
RL_MODELS_DIR="$(cd lightx2v_rl/models && pwd)"
export UV_CACHE_DIR="$RL_MODELS_DIR/.uv-cache"
export PIP_CACHE_DIR="$RL_MODELS_DIR/.pip-cache"
export HF_HOME="$RL_MODELS_DIR/.hf-runtime"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"
```

继续在同一终端安装；新终端需要重新设置缓存变量。下载器直接写入指定模型目录，不使用 Hugging Face 默认权重缓存。模型、数据、虚拟环境和运行输出不纳入版本管理。`weights/` 仅作为旧 reward 路径的兼容入口，新配置统一使用 `models/rewards/`。

### 安装训练环境

```bash
if ! command -v uv >/dev/null 2>&1; then
  python3.12 -m pip install --target lightx2v_rl/.bootstrap-tools uv
  export PATH="$PWD/lightx2v_rl/.bootstrap-tools/bin:$PATH"
fi
uv --version

git submodule update --init third_party/verl-omni
bash lightx2v_rl/scripts/setup/setup_qwen21_env.sh
```

安装脚本创建 `.venv-rl`，安装固定的 verl-omni / vLLM-Omni、模型依赖覆盖和本模块，并检查 CUDA 与 QwenImage21 导入。Wan 适配复用此训练环境。依赖以 [qwen_image21.txt](requirements/qwen_image21.txt) 和安装脚本为准；仅 `pip install -e lightx2v_rl` 不包含完整 GPU 训练栈。

安装时会应用 `patches/` 中的通用扩展接口补丁，不改变 submodule 的 Git 版本。可单独检查或在补丁更新后重新应用：

```bash
# 只检查，不修改文件。
lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/setup/prepare_verl_omni.py --check

# 首次准备或补丁更新时应用。
lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/setup/prepare_verl_omni.py
```

这些补丁是运行依赖。使用父仓库记录的 submodule 版本，保留应用补丁后的工作区差异。

### 安装 reward 环境

评分模型与训练模型依赖不同版本的 Transformers / vLLM，通过独立进程和 HTTP 通信。只安装实际启用的服务环境：

| 环境 | 用途 | 固定依赖 |
| --- | --- | --- |
| `.venv-reward` | HPSv3 / VideoAlign | [reward.txt](requirements/reward.txt)：Torch 2.8.0、Transformers 4.51.3、PEFT 0.17.0 |
| `.venv-unifiedreward` | UnifiedReward-Edit | [unifiedreward.txt](requirements/unifiedreward.txt)：vLLM 0.20.2、Transformers 5.5.4 |

```bash
# HPSv3 / VideoAlign
REWARD_PYTHON_VERSION="$(command -v python3.12)" \
REWARD_BOOTSTRAP_PYTHON="$(command -v python3.12)" \
  bash lightx2v_rl/scripts/setup/setup_reward_env.sh

# UnifiedReward-Edit
LIGHTX2V_SETUP_PYTHON="$(command -v python3.12)" \
  bash lightx2v_rl/scripts/setup/setup_unifiedreward_env.sh
```

脚本优先使用 uv，无需激活环境。多机使用兼容镜像时，可在一台 GPU 节点安装共享环境，其余节点复用；不要同时从多个节点安装，也不要修改运行中作业使用的环境。依赖安装不包含模型权重。

### 检查环境

在 GPU 节点检查训练环境；评分环境可将命令中的 `.venv-rl` 替换成对应目录：

```bash
lightx2v_rl/.venv-rl/bin/python - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
PY

lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.train \
  --config qwen_image21_turbo_geo --check-config
```

`--check-config` 只检查配方约束，不加载模型或启动服务。

## 模型与数据

### 模型资源

| 资源 | 默认入口，相对仓库根目录 |
| --- | --- |
| 生成模型 | `lightx2v_rl/models/<模型名>/` |
| HPSv3 | `lightx2v_rl/models/rewards/hpsv3/` |
| VideoAlign | `lightx2v_rl/models/rewards/videoalign/` |
| UnifiedReward 默认别名 | `lightx2v_rl/models/rewards/unifiedreward/` |

Qwen-Image-2.1-Turbo 与 UnifiedReward 使用固定版本下载器，支持续传与完成校验：

```bash
python3 lightx2v_rl/scripts/assets/download_qwen21_assets.py \
  --models qwen unifiedreward --connections 8

# 可选：27B，保存到独立目录。
python3 lightx2v_rl/scripts/assets/download_qwen21_assets.py \
  --models unifiedreward27b --connections 8
```

分别保存到 `models/Qwen-Image-2.1-Turbo/`、`models/rewards/UnifiedReward-Edit-qwen35-9b/` 和 `models/rewards/UnifiedReward-Edit-qwen35-27b/`，这里的 `models/` 指模块资源目录。准备 9B 后建立服务默认别名与旧 `weights/` 兼容入口：

```bash
python3 lightx2v_rl/scripts/assets/link_reward_weights.py --models unifiedreward
```

保留下载完成的 `download-manifest.json`，启动时会校验。四步 Qwen 模型需另行准备**完整 Diffusers pipeline**，其 `model_index.json` 必须携带对应的固定 `sample_sigmas`；本地蒸馏导出还须保留 `export_manifest.json`，供预检验证模型来源与结构。不能只把 Turbo 的步数改成 4。Wan 同样需要包含 transformer、文本编码器、VAE 和配置的完整 Diffusers 目录。

HPSv3 / VideoAlign 使用包含 checkpoint、基座、tokenizer 和 `reward-assets.json` 的自包含 bundle。已有 bundle 可直接复制或软链接；从已有 FewStepRL 资源整理时执行：

```bash
python3 lightx2v_rl/scripts/assets/prepare_reward_assets.py \
  --source-root ../FewStepRL --hf-cache ../models/.hf-runtime/hub --dry-run
python3 lightx2v_rl/scripts/assets/prepare_reward_assets.py \
  --source-root ../FewStepRL --hf-cache ../models/.hf-runtime/hub
```

此工具不下载权重。源目录需含 VideoAlign checkpoint，本地 HF 缓存需含 HPSv3 checkpoint、Qwen2-VL-7B 和 Qwen2-VL-2B 基座。默认生成 `lightx2v_rl/models/rewards/fewstep/`；按实际资源位置修改参数。当前 bundle 包含两套资产，服务可只加载其中一种。首次建立默认入口：

```bash
python3 lightx2v_rl/scripts/assets/link_reward_weights.py --models hpsv3 videoalign
python3 lightx2v_rl/scripts/assets/prepare_reward_assets.py --check
```

链接工具的 `--models` 仅选择已经准备的 reward，不要求其他 reward 存在；可用 `--storage-dir` 指定共享权重位置。也可用 `LIGHTX2V_RL_REWARD_ASSETS` 指向完整 bundle。运行时无需 FewStepRL 源仓库。

### 数据清单

Qwen 使用 JSONL，每行包含唯一 `id`、`task_type`、`prompt` 与有序 `source_images`：

```jsonl
{"id":"text-001","task_type":"t2i","prompt":"A cat swimming in clear water.","source_images":[]}
{"id":"edit-001","task_type":"i2i","prompt":"Replace the background with a garden.","source_images":["images/input.png"]}
{"id":"multi-001","task_type":"i2i","prompt":"Add the object from <image2> to <image1>.","source_images":["images/canvas.png","images/reference.png"]}
```

图片路径相对清单目录，允许该目录中的图片入口使用软链接。RL 不要求标准输出图。多参考图按“第一张为待编辑画布，其余为素材参考”传入；生成结果在评分请求中放在最后。多 ref 的 UR rubric 是本模块的模板扩展，其评分准确性仍需结合任务样例评估。

准备 OpenGPT-4o-Image 的示例：

```bash
if [[ ! -e lightx2v_rl/data/OpenGPT-4o-Image && \
      ! -L lightx2v_rl/data/OpenGPT-4o-Image ]]; then
  ln -s ../../../OpenGPT-4o-Image lightx2v_rl/data/OpenGPT-4o-Image
fi
lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/assets/prepare_opengpt4o.py \
  --validation-size 256 --seed 42
```

默认输出 `data/opengpt4o/train.jsonl`、`val_t2i.jsonl`、`val_i2i.jsonl`，按关联 prompt / 图像分组划分训练与验证。需要多参考图时添加 `--include-multi-reference`。已有清单可直接通过 `data.train_files`、`data.val_files` 指定。已有共享模型和原始数据目录时，也可用 `scripts/assets/link_resources.py --models-dir ../models --dataset-dir ../OpenGPT-4o-Image` 一次建立两个资源入口。

默认按完整 iter 交替 T2I / I2I。可用 `data.task_cycle='[t2i,i2i_single,t2i,i2i_multi]'` 指定三类任务的轮换；清单必须包含所选任务。参考图数量增加时，需同步配置 `actor_rollout_ref.rollout.pipeline.max_reference_images`、`max_reference_tokens` 和服务端 `LIGHTX2V_UNIFIEDREWARD_LIMIT_MM_PER_PROMPT`；服务端图数上限包含生成结果。

## Reward 服务

### 独立部署

在单独终端启动评分服务，选择未用于训练的 GPU。以下为本机示例：

```bash
# GPU 4：HPSv3；增加 VideoAlign 时可用 --models hpsv3,videoalign 分别指定设备。
CUDA_VISIBLE_DEVICES=4 bash lightx2v_rl/scripts/rewards/start_reward_server.sh \
  --models hpsv3 --hps-device cuda:0 --host 0.0.0.0 --port 8791
```

```bash
# GPU 5：UnifiedReward-Edit 9B。
source lightx2v_rl/scripts/rewards/profiles/unifiedreward_edit9b_h100.sh
CUDA_VISIBLE_DEVICES=5 LIGHTX2V_UNIFIEDREWARD_HOST=0.0.0.0 \
  bash lightx2v_rl/scripts/rewards/start_unifiedreward_server.sh
```

UR 默认端口 8792，模型服务名为 `UnifiedReward`。选择 27B 时，将 profile 换成 `unifiedreward_edit27b_h100.sh`（单实例并发 1）或 `unifiedreward_edit27b_h100_concurrency2.sh`（并发 2）。这些是每实例一张 H100 80G 的部署配方；实际容量取决于输入尺寸、图数和上下文。profile 在启动时生效，不会修改已运行的服务。

服务默认关闭 prefix caching，避免固定 vLLM 版本多图请求的 Mamba 分块对齐停滞。客户端并发由 `reward.num_workers` 与 `reward.custom_reward_function.reward_kwargs.max_concurrent_samples` 控制，服务端并发由 `LIGHTX2V_UNIFIEDREWARD_MAX_NUM_SEQS` 控制。

```bash
# 检查服务身份与连通性。
lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.rewards.client \
  --endpoint http://127.0.0.1:8791

# 实际发送 T2I 和两参考图 I2I 请求；不加载生成模型。
lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.rewards.preflight \
  --endpoint http://127.0.0.1:8792 --i2i-source-count 2
```

Qwen 启动预检会对每个 UR 实例执行真实评分请求，图数跟随数据。内部健康检查与评分请求不使用环境中的 HTTP 代理。远端部署需把训练配置中的 endpoint 改为服务的实际地址；`0.0.0.0` 仅用于监听。平台启动入口会自动分配 reward GPU 和地址，无需手工启动以上服务。

### 评分与 advantage

| Reward | 使用方式 |
| --- | --- |
| UnifiedReward T2I | alignment / coherence / style 各 1–5 分，标量为 `(均值 - 1) / 4` |
| UnifiedReward I2I | edit success / preservation 各 0–25 分，标量为 `均值 / 25` |
| HPSv3 | 图像偏好评分；视频按帧评分后聚合 |
| VideoAlign | 视频质量、运动质量与文本对齐，保留 checkpoint 的标准化方式 |

Qwen 默认 UR 权重为 1，T2I 的 HPSv3 权重为 0.5，I2I 为 0；多节点变体可使用不同权重，以实际配方为准。修改以下参数会同步改变评分请求和 advantage 权重，权重为零的项不发送请求：

```text
reward.custom_reward_function.reward_kwargs.hps_weight=0.5
reward.custom_reward_function.reward_kwargs.hps_weight_i2i=0.2
```

默认 `algorithm.weight_advantages=true`：各 reward 先独立计算组内相对 advantage，再按任务权重合并；`max_group_std` 使用该 reward 在本轮同任务各组中的最大标准差作分母。也支持 `group` / `global` 标准差。权重不是固定的梯度贡献比例。设为 `false` 可使用先合成 scalar reward、再计算 GRPO advantage 的路径。

UR 输出格式无效时，整项 UR 使用量表中间分（T2I 三项为 3，I2I 两项为 12.5，归一化均为 0.5）继续，并记录 `ur_format_error_fraction`；其他 reward 仍使用真实分数。中间分仍参与组内 advantage。连接失败、HTTP 错误、超时或候选缺失则停止当前作业，避免不完整组参与更新。默认评分 HTTP 超时为 600 秒，不包含本地并发队列等待。

## 训练

### 配方与公共参数

| 模型 | Geo | CPS | Flow-GRPO |
| --- | --- | --- | --- |
| Qwen-Image-2.1-Turbo | `qwen_image21_turbo_geo` | `qwen_image21_turbo_cps` | `qwen_image21_turbo_flow_grpo` |
| Wan2.1 T2V | `wan_geo` | `wan_cps` | `wan_flow_grpo` |

`--config` 接受配方名或 YAML 路径，后接普通 `key=value` 覆盖，不使用 Hydra 的 `+` / `~` 前缀。可先用 `--print-config` 查看展开配置。

Qwen 默认：8 个组内候选、共享初始噪声和所选训练位置、每轨迹随机选择一个非末尾转移进行 bridge / replay；其余转移沿用确定性路径。不同候选的 bridge 噪声独立。Actor 使用 FP32 主参数和 BF16 计算，学习率 `2e-6`，每 iter 两个 mini-batch、各更新一次。四步配方读取模型自己的采样日程，其他训练规则一致。

训练默认 5000 iter；Qwen 在训练前验证，之后每 20 iter 验证，每 100 iter 保存 checkpoint，最多保留最近 3 份 Actor checkpoint。对应配置为 `trainer.total_training_steps`、`trainer.test_freq`、`trainer.save_freq` 和 `trainer.max_actor_ckpt_to_keep`。

### 单机 Qwen

准备好模型、数据和评分服务后，在新终端仅暴露训练卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash lightx2v_rl/scripts/train/train_qwen21.sh \
  trainer.experiment_name=qwen_geo \
  trainer.default_local_dir=lightx2v_rl/outputs/qwen_geo \
  trainer.validation_data_dir=lightx2v_rl/outputs/qwen_geo/validation
```

默认选择 `qwen_image21_turbo_geo`。通过 `LIGHTX2V_RL_CONFIG` 切换配方，通过 `LIGHTX2V_RL_PYTHON` 指定训练解释器。启动脚本使用离线模型模式，所有权重须提前准备。

### 多节点平台

平台须在每个节点执行同一启动命令，并提供共同的 `MASTER_ADDR`、`MASTER_PORT`；所有节点各 8 张 GPU。脚本内部启动每节点一个协调进程，按 rank 建立 Ray 与评分服务，只有 rank 0 启动训练 driver。不要再套一层八进程 `torchrun`。

| 配方 | 总节点 | 训练卡 | 独立 reward 服务 |
| --- | ---:| ---:| --- |
| `qwen_image21_turbo_geo_40gpu` | 5 | 32 | 6 UR + 2 HPSv3 |
| `qwen_image21_4step_geo_6nodes` | 6 | 32 | 12 UR + 4 HPSv3 |
| `qwen_image21_4step_geo_7nodes_hps02` | 7 | 32 | 20 UR + 4 HPSv3 |
| `qwen_image21_4step_geo_7nodes_ur_only` | 7 | 32 | 24 UR |

Turbo 五节点也提供 CPS / Flow-GRPO 对应配方。五节点启动示例：

```bash
export LIGHTX2V_RL_CONFIG=qwen_image21_turbo_geo_40gpu
export LIGHTX2V_RDZV_ID=qwen-geo-run1
bash lightx2v_rl/scripts/train/train_qwen21_platform.sh
```

七节点四步示例，模型路径需指向已准备的完整四步 pipeline。`LIGHTX2V_RL_DATA_DIR` 指定包含 `train.jsonl`、`val.jsonl` 的数据目录；没有固定预览清单时将 `trainer.validation_preview_file` 设为 `null`：

```bash
export LIGHTX2V_RL_CONFIG=qwen_image21_4step_geo_7nodes_hps02
export LIGHTX2V_QWEN_4STEP_MODEL=lightx2v_rl/models/qwen-image21-4step
export LIGHTX2V_RL_DATA_DIR=lightx2v_rl/data/opengpt4o_multiref
export LIGHTX2V_UNIFIEDREWARD_PROFILE=lightx2v_rl/scripts/rewards/profiles/unifiedreward_edit27b_h100_concurrency2.sh
export LIGHTX2V_RDZV_ID=qwen-4step-run1
bash lightx2v_rl/scripts/train/train_qwen21_4step_7nodes.sh \
  trainer.validation_preview_file=null \
  trainer.experiment_name=qwen_4step_geo \
  trainer.default_local_dir=lightx2v_rl/outputs/qwen_4step_geo
```

四步平台配方按 `T2I → 单 ref → T2I → 多 ref` 轮换，使用 1024×1024 画布、每 iter 64 个 prompt × 8 候选。七节点混合 reward 配方使用 UR 27B，T2I / I2I 的 HPS 权重分别为 0.5 / 0.2；只使用 UR 时改用 `train_qwen21_4step_7nodes_ur_only.sh`。每个作业使用不同 rendezvous ID。`MASTER_PORT` 须与 Ray 的 6379 不同。

非自动分发平台可使用 `scripts/train/qwen21_cluster.sh head|worker|train` 和 `scripts/rewards/start_qwen21_reward_pool.sh` 手动部署；必需环境变量与节点布局见各脚本的 `--help`。

### Wan 视频训练

准备完整 Diffusers 模型、Parquet 数据和 HPSv3 / VideoAlign 服务后执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash lightx2v_rl/scripts/train/train_wan_reward.sh \
  --config wan_geo --reward-config hpsv3_videoalign \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.model.path=lightx2v_rl/models/wan-fewstep-diffusers \
  'data.train_files=[lightx2v_rl/data/wan/train.parquet]' \
  'data.val_files=[lightx2v_rl/data/wan/val.parquet]'
```

`--reward-config` 也可选 `hpsv3` 或 `videoalign`。数据中的 `reward_model.ground_truth` 应为生成视频的 prompt；远端服务通过 `reward.custom_reward_function.reward_kwargs.endpoint` 指定。

## 日志、验证与 checkpoint

默认输出 console；使用 W&B 官方云端时先登录，再给训练命令添加 `trainer.logger='[console,wandb]'`。也可由平台提供 `WANDB_API_KEY`：

```bash
unset WANDB_BASE_URL
lightx2v_rl/.venv-rl/bin/wandb login
```

默认 project 为 `lightx2v_rl`，可用 `trainer.project_name` 修改。精简视图记录各 reward、policy / KL loss、学习率、梯度、格式错误比例与耗时，不显示加权总 reward。Qwen 的训练与验证 reward 按 `t2i`、`i2i_single`、`i2i_multi` 分开；`time/rollout_and_reward_s` 同时包含生成和等待评分。

验证始终评分完整验证集。默认可视化 T2I / I2I 各 8 条，通过 `trainer.validation_samples_per_task` 调整；指定 `trainer.validation_preview_file` 时，展示该 JSONL 清单列出的全部 ID，并拆成三类任务表。预览 ID 必须属于完整验证集。I2I 展示所有原始参考图，图像、prompt 和评分明细同时保存到本地。

| 内容 | 路径，相对模块目录 |
| --- | --- |
| 平台协调与训练 stdout | `outputs/platform/<run>/rank<rank>-<pid>/` |
| 独立评分实例日志 | `outputs/reward_pool/<run-id>/` |
| 完整原始指标 | `outputs/metrics/*.jsonl` |
| checkpoint | `trainer.default_local_dir/global_step_<iter>/` |
| 验证图像与 `samples.jsonl` | `trainer.validation_data_dir/global_step_<iter>/` |

Ray Dashboard 默认监听 head 的 `127.0.0.1:8265`，可经平台端口转发或 SSH 隧道访问。平台需要按节点 IP 转发时，可设置 `LIGHTX2V_RAY_DASHBOARD_HOST=0.0.0.0` 并使用平台访问控制。独立 reward 进程日志不在 Ray worker 日志中。

新实验应设置独立的 `trainer.default_local_dir` 和 `trainer.experiment_name`，并用 `trainer.resume_mode=disable` 从初始模型开始；恢复同一实验用 `auto`。恢复会加载优化器、学习率调度器和数据采样器状态，改变学习率初始化值不会覆盖已恢复的优化器状态；数据、batch 或任务周期变化后不应复用旧采样器状态。

Qwen FSDP2 checkpoint 可导出为单个完整 transformer safetensors，显式指定来源、基座和目标：

```bash
lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/assets/export_qwen21_rl.py \
  --checkpoint lightx2v_rl/outputs/qwen_4step_geo/global_step_100/actor \
  --base-model lightx2v_rl/models/qwen-image21-4step \
  --output lightx2v_rl/models/qwen-image21-4step-rl \
  --dtype bfloat16
```

可先加 `--dry-run` 检查路径与架构。导出不需要 GPU，需足够 CPU 内存和磁盘空间；输出目录必须不存在。文件包含训练后的完整 transformer，不包含 VAE、文本编码器和完整 pipeline，推理时继续使用对应基座与采样日程。

## 开发结构

```text
lightx2v_rl/
├── lightx2v_rl/
│   ├── algorithms/fewstep/  # bridge、log-prob、loss 和选步规则
│   ├── rollout/             # 模型无关的轨迹契约
│   ├── models/              # 模型注册、qwen_image/、wan/ 适配
│   ├── backends/verl_omni/   # 后端配置、advantage 与运行时接入
│   ├── rewards/             # HTTP 客户端、评分服务和模板
│   ├── data/                # 数据适配与任务采样器
│   ├── training/            # 配置、指标、验证展示与候选组检查
│   ├── deployment/          # 集群启动与健康检查
│   └── configs/             # 算法、模型、reward 和部署配方
├── scripts/{setup,assets,rewards,train}/
├── requirements/
└── patches/                 # 固定 verl-omni 的通用扩展补丁
```

顶层 `models/`、`data/` 是资源目录，Python 包内同名目录存放代码。新增模型需实现 learner 重算、rollout、条件输入和配置校验，在 `models/registry.py` 注册 `ModelSpec`，由 `actor_rollout_ref.model.rl_adapter` 选择。模型特有的 timestep / sigma / 参考图逻辑留在模型目录；共享算法和 reward 不依赖具体模型。submodule 仅维护必要的通用接口补丁。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `Python.h` 缺失 | 在运行节点安装与解释器匹配的 `python3.x-dev` |
| `ensurepip` 缺失 | 安装 uv 后重新运行环境脚本，或安装匹配的 `python3.x-venv` |
| 无法导入 `QwenImage21Pipeline` | 运行 `scripts/setup/repair_qwen21_diffusers.sh` 恢复固定官方 Diffusers |
| submodule 补丁检查失败 | 保持父仓库固定版本，运行 `prepare_verl_omni.py` 应用补丁 |
| startup 等待评分服务 | 检查对应 `reward.log` 与实例日志；健康接口正常后仍需通过实际评分预检 |
| rendezvous 断连 | 先定位最早的训练或服务异常，断连通常是作业退出的后续现象 |
| 资源或 manifest 缺失 | 检查相对链接、下载完成状态，保留资产清单 |

参考：[verl-omni 安装说明](../third_party/verl-omni/docs/start/install.md)、[diffusion v1 trainer](../third_party/verl-omni/docs/start/diffusion_v1.md)。
