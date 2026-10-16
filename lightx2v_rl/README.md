# LightX2V RL

LightX2V 的少步图像与视频强化学习模块，与 `lightx2v_train` 并列。提供 CPS、Flow-GRPO、Geo-FSRL（简称 Geo）算法、模型适配、数据处理及 reward 接口，复用固定版本的 verl-omni 完成分布式调度、worker 管理、训练引擎和权重同步。

| 能力 | 支持范围 |
| --- | --- |
| 算法 | CPS、Flow-GRPO、Geo-FSRL；Geo 对应 FewStepRL 中的 IMMB |
| 图像模型 | Qwen-Image-2.1-Turbo：T2I、单参考图 I2I |
| 视频模型 | Wan2.1 单 transformer T2V，使用完整 Diffusers 格式权重 |
| Reward | HPSv3、VideoAlign、UnifiedReward-Edit-qwen35-9b |
| 训练后端 | verl-omni / FSDP2；支持独立 HTTP reward 服务 |

Qwen 使用官方 Diffusers pipeline，Wan rollout 复用 vLLM-Omni。LightX2V 原生加速推理后端、Wan2.2 双专家和 H3 尚需适配。

本文所有命令均在 **LightX2V 仓库根目录**执行，即包含 `lightx2v_rl/` 与 `third_party/` 的目录；仓库目录可以任意命名。配置中的本地资源路径相对仓库根目录解析。安装与启动命令使用 Bash。

[环境准备](#环境准备) · [模型与数据](#模型与数据) · [Reward 服务](#reward-服务) · [训练](#训练) · [日志与监控](#日志与监控) · [开发](#开发) · [常见问题](#常见问题)

## 环境准备

### 系统要求

- Linux、可用的 NVIDIA GPU 和与所安装 PyTorch / vLLM CUDA runtime 兼容的驱动。
- 推荐 Python 3.12；完整 verl-omni 训练运行时要求 Python ≥ 3.11。
- Git、curl、C/C++ 编译工具及与解释器匹配的 Python 开发头文件。
- UnifiedReward 的固定 GPU wheel 要求 glibc ≥ 2.35。

GPU 数量、显存需求由模型、分辨率、batch 和评分服务决定。后文的单机与五节点配置是部署示例，不是安装环境的前提。驱动、CUDA 与容器的版本要求参见所固定版本的 [verl-omni 安装说明](../third_party/verl-omni/docs/start/install.md)；不同虚拟环境的 CUDA runtime 可能不同，需分别检查。

以 Ubuntu 24.04、系统 Python 3.12 为例，在容器内安装系统依赖：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  python3.12 python3.12-dev python3-pip \
  git curl build-essential ca-certificates

python3.12 --version
nvidia-smi
```

其他发行版安装对应的软件包。使用其他 Python 版本时，开发头文件必须与解释器一致。系统依赖应预装到公共镜像，或在每个运行服务的节点安装；共享虚拟环境不包含这些系统依赖。

### 目录与缓存

模型、reward、数据和运行输出统一通过本模块下的目录访问：

```text
仓库根目录/
├── third_party/verl-omni/     # 固定版本的 submodule
└── lightx2v_rl/
    ├── .venv-rl/             # 训练与 rollout 环境
    ├── .venv-reward/         # 可选：HPSv3 / VideoAlign 环境
    ├── .venv-unifiedreward/  # 可选：UnifiedReward 环境
    ├── models/              # 生成模型、rewards/ 与下载缓存；可为软链接
    ├── data/                # 原始数据入口与适配后的清单
    └── outputs/             # 安装日志、训练日志、验证结果与 checkpoint
```

推荐将大文件放在共享存储，通过**相对软链接**接入。以下示例让模型实体保存在仓库同级的 `models/`；已有有效目录或链接时直接复用：

```bash
set -euo pipefail
mkdir -p ../models lightx2v_rl/data lightx2v_rl/outputs/setup
if [[ ! -e lightx2v_rl/models && ! -L lightx2v_rl/models ]]; then
  ln -s ../../models lightx2v_rl/models
fi
test -d lightx2v_rl/models

RL_MODELS_DIR="$(cd lightx2v_rl/models && pwd)"
export UV_CACHE_DIR="$RL_MODELS_DIR/.uv-cache"
export PIP_CACHE_DIR="$RL_MODELS_DIR/.pip-cache"
export HF_HOME="$RL_MODELS_DIR/.hf-runtime"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"
```

在同一终端继续执行安装命令。缓存路径在运行时展开，避免安装脚本切换目录后写入其他位置，不包含固定挂载路径。新终端需要重新设置这些变量。模型下载脚本直接写入 `models/`，不使用 Hugging Face 默认权重缓存。

数据也可以按数据集分别建立相对软链接。迁移工作目录时保持链接与目标的相对关系。`weights/` 是旧 reward 路径的兼容入口，新配置统一使用 `models/rewards/`。

### Python 环境划分

训练和评分模型依赖不同版本的 Transformers / vLLM，采用独立环境，通过 HTTP 通信。只安装任务需要的 reward 环境。

| 环境 | 用途 | 依赖定义 |
| --- | --- | --- |
| `.venv-rl` | Ray、Actor、reference、rollout、评分客户端 | 固定 verl-omni + [模型依赖覆盖](requirements/qwen_image21.txt) |
| `.venv-reward` | HPSv3 / VideoAlign 服务 | [reward.txt](requirements/reward.txt)：Torch 2.8.0、Transformers 4.51.3、PEFT 0.17.0 |
| `.venv-unifiedreward` | UnifiedReward-Edit vLLM 服务 | [unifiedreward.txt](requirements/unifiedreward.txt)：vLLM 0.20.2、Transformers 5.5.4 |

脚本显式选择目标解释器，无需激活环境。多机共享环境时，在一台兼容的 GPU 节点安装一次，其余节点复用；节点须使用兼容镜像、相同的共享挂载路径及可用的基础 Python。不要向正在被训练使用的环境安装包，也不要让多个节点同时修改同一环境。

### 安装训练环境

先检查 uv；镜像未提供时安装到本模块的工具目录：

```bash
if ! command -v uv >/dev/null 2>&1; then
  python3.12 -m pip install --target lightx2v_rl/.bootstrap-tools uv
  export PATH="$PWD/lightx2v_rl/.bootstrap-tools/bin:$PATH"
fi
uv --version

git submodule update --init third_party/verl-omni
```

使用父仓库记录的 submodule 版本。当前完整 GPU 安装入口为 `setup_qwen21_env.sh`：它安装公共 verl-omni / vLLM-Omni 训练栈、本模块，以及 QwenImage21 所需的依赖覆盖；Wan 适配也复用这套训练环境。

```bash
(
  set -euo pipefail
  if [[ ! -e lightx2v_rl/.venv-rl && ! -L lightx2v_rl/.venv-rl ]]; then
    uv venv --python "$(command -v python3.12)" --seed lightx2v_rl/.venv-rl
  fi
  bash lightx2v_rl/scripts/setup/setup_qwen21_env.sh \
    2>&1 | tee lightx2v_rl/outputs/setup/install_training.log
)
```

在 GPU 节点执行；脚本会检查 CUDA 和官方 QwenImage21 导入。依赖版本以安装脚本和 requirements 为准，避免单独升级 Torch、Diffusers、Transformers 或 vLLM，破坏组合兼容性。仅 `pip install -e lightx2v_rl` 不会安装完整训练栈。

安装脚本还会运行 `prepare_verl_omni.py`，应用本模块维护的通用接口补丁。准备完成后可只读检查：

```bash
lightx2v_rl/.venv-rl/bin/python \
  lightx2v_rl/scripts/setup/prepare_verl_omni.py --check
```

训练依赖这些补丁；`--check` 只验证、不修改文件，无需每个 iter 执行。首次准备或补丁更新后，可去掉 `--check` 应用补丁。submodule 中由此产生的工作区差异应保留。

### 安装可选 reward 环境

需要 HPSv3 或 VideoAlign 时执行：

```bash
(
  set -euo pipefail
  REWARD_PYTHON_VERSION="$(command -v python3.12)" \
  REWARD_BOOTSTRAP_PYTHON="$(command -v python3.12)" \
    bash lightx2v_rl/scripts/setup/setup_reward_env.sh \
    2>&1 | tee lightx2v_rl/outputs/setup/install_reward.log
)
```

需要 UnifiedReward-Edit 时执行：

```bash
(
  set -euo pipefail
  LIGHTX2V_SETUP_PYTHON="$(command -v python3.12)" \
    bash lightx2v_rl/scripts/setup/setup_unifiedreward_env.sh \
    2>&1 | tee lightx2v_rl/outputs/setup/install_unifiedreward.log
)
```

两者使用系统 Python 创建独立环境，优先通过 uv 安装，不依赖系统 `ensurepip`。仅部署 reward 服务时，准备好系统依赖、目录、缓存和 uv 后即可安装，无需先安装训练栈。安装完成只表示 Python 依赖就绪；模型权重需另行准备，服务也需单独启动。Qwen 默认配方同时使用 UnifiedReward 和 HPSv3，因此需要两套 reward 环境。

### 验证环境

在 GPU 节点执行。下面检查所有已创建的环境，未安装的可选环境会跳过：

```bash
set -euo pipefail
for RL_ENV in .venv-rl .venv-reward .venv-unifiedreward; do
  [[ -d "lightx2v_rl/$RL_ENV" ]] || continue
  "lightx2v_rl/$RL_ENV/bin/python" - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA unavailable"
PY
done

lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.train --config wan_geo --check-config
lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.train \
  --config qwen_image21_turbo_geo --check-config
```

`--check-config` 检查配置约束，不加载真实模型；开始训练前仍需准备权重、数据和评分服务。

## 模型与数据

### 资源约定

| 资源 | 默认入口，相对仓库根目录 |
| --- | --- |
| 生成模型 | `lightx2v_rl/models/<模型名>/` |
| HPSv3 | `lightx2v_rl/models/rewards/hpsv3/` |
| VideoAlign | `lightx2v_rl/models/rewards/videoalign/` |
| UnifiedReward-Edit | `lightx2v_rl/models/rewards/unifiedreward/` |
| 原始数据 / 适配后数据 | `lightx2v_rl/data/<数据集名>/` |

入口可以指向共享目录中的完整资源。下载与迁移工具生成的 `download-manifest.json` / `reward-assets.json` 用于启动前校验，应与权重一起保留。虚拟环境、模型、数据与运行输出均不提交到 Git。

### 生成模型与 UnifiedReward

提供 Qwen-Image-2.1-Turbo 和 UnifiedReward-Edit 的固定版本下载器，可续传并校验已有文件：

```bash
python3 lightx2v_rl/scripts/assets/download_qwen21_assets.py \
  --models qwen unifiedreward --connections 8
```

按需将 `--models` 改成 `qwen` 或 `unifiedreward`。默认分别保存到 `models/Qwen-Image-2.1-Turbo/`、`models/rewards/UnifiedReward-Edit-qwen35-9b/`，这里的 `models/` 指模块资源目录。下载 UnifiedReward 后建立服务使用的别名：

```bash
if [[ ! -e lightx2v_rl/models/rewards/unifiedreward && \
      ! -L lightx2v_rl/models/rewards/unifiedreward ]]; then
  ln -s UnifiedReward-Edit-qwen35-9b lightx2v_rl/models/rewards/unifiedreward
fi
```

也可设置 `LIGHTX2V_UNIFIEDREWARD_MODEL` 指向另一个已校验的本地快照。Wan 权重需另行准备完整 Diffusers 目录，包含模型配置、文本编码器和 VAE；训练时用 `actor_rollout_ref.model.path` 指定。

### HPSv3 / VideoAlign 资产

服务读取包含 checkpoint、基座、tokenizer 和 `reward-assets.json` 的自包含 bundle。已有 bundle 可以直接复制或通过目录软链接接入，运行时不依赖 FewStepRL 仓库。

从已有 FewStepRL 资源整理 bundle 时，使用本地迁移工具：

```bash
python3 lightx2v_rl/scripts/assets/prepare_reward_assets.py \
  --source-root ../FewStepRL \
  --hf-cache ../models/.hf-runtime/hub
```

示例参数需替换成实际的相对目录。该工具**不下载权重**：源仓库需包含 VideoAlign checkpoint，本地 HF 缓存需包含 HPSv3 checkpoint、Qwen2-VL-7B 和 Qwen2-VL-2B 基座；可先加 `--dry-run` 查看缺失资源。默认产物位于 `lightx2v_rl/models/rewards/fewstep/`，当前 bundle 校验要求包含 HPSv3 与 VideoAlign 两套资产，服务可只加载其中一种。

准备好 bundle 后，首次建立默认入口：

```bash
for RL_REWARD_ENTRY in hpsv3 videoalign reward-assets.json; do
  if [[ ! -e "lightx2v_rl/models/rewards/$RL_REWARD_ENTRY" && \
        ! -L "lightx2v_rl/models/rewards/$RL_REWARD_ENTRY" ]]; then
    ln -s "fewstep/$RL_REWARD_ENTRY" "lightx2v_rl/models/rewards/$RL_REWARD_ENTRY"
  fi
done
python3 lightx2v_rl/scripts/assets/prepare_reward_assets.py --check
```

自定义 bundle 位置时，设置 `LIGHTX2V_RL_REWARD_ASSETS` 指向包含 `reward-assets.json`、`hpsv3/` 和 `videoalign/` 的目录。相对路径按仓库根目录解释。`--check --verify-hashes` 可进一步校验全部文件内容。

若使用仓库同级 `models/` 的共享布局，且两套评分模型的 bundle 与 UnifiedReward 均已准备，可用 `scripts/assets/link_reward_weights.py` 统一建立上述别名及旧 `weights/` 兼容链接，随后用 `--check` 验证。该工具要求全部目标已存在，不作为仅安装一种 reward 的前置步骤。

### 数据准备

数据清单通过 `data.train_files`、`data.val_files` 配置，适配器通过 `data.custom_cls` 选择。自定义数据集应明确 prompt、任务类型、参考图/视频和 reward 所需元数据。

Qwen 配方提供 OpenGPT-4o-Image 适配器。以下示例假设原始数据保存在仓库同级目录：

```bash
test -d ../OpenGPT-4o-Image
if [[ ! -e lightx2v_rl/data/OpenGPT-4o-Image && \
      ! -L lightx2v_rl/data/OpenGPT-4o-Image ]]; then
  ln -s ../../../OpenGPT-4o-Image lightx2v_rl/data/OpenGPT-4o-Image
fi
lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/assets/prepare_opengpt4o.py \
  --validation-size 256 --seed 42
```

输出 `data/opengpt4o/train.jsonl`、`val_t2i.jsonl`、`val_i2i.jsonl` 及数据审计信息。划分按关联 prompt 和图像分组，减少训练/验证泄漏；默认联合训练只使用 T2I 与单参考图 I2I，多参考图样本单独保存。已有清单可直接复用，恢复训练时应保持清单和 sampler 配置一致。

对于上述共享布局，也可用 `scripts/assets/link_resources.py` 同时建立模型和原始数据链接；该工具要求两个源目录均已存在。它只建立入口，不下载或重新划分数据。

## Reward 服务

| Reward | 输入 | 可用指标 / 能力 |
| --- | --- | --- |
| HPSv3 | 图像或视频帧 + 文本 | `hpsv3_standard`、`hpsv3_general`、`hpsv3_percentile` |
| VideoAlign | 视频 + 文本 | `videoalign_mq` 运动质量、`videoalign_ta` 文本对齐、`videoalign_vq` 视觉质量 |
| UnifiedReward-Edit-qwen35-9b | prompt + 生成图；编辑时附一张参考图 | T2I 对齐/一致性/风格；I2I 编辑成功/内容保留 |

HPSv3 基于单帧评分，不直接衡量跨帧运动。UnifiedReward 当前适配不包含视频或多参考图编辑。安装相应环境并准备好权重后，在独立终端启动需要的服务。

HPSv3 / VideoAlign 服务示例：

```bash
# 一个服务、两张 GPU，分别加载 HPSv3 和 VideoAlign。
CUDA_VISIBLE_DEVICES=0,1 bash lightx2v_rl/scripts/rewards/start_reward_server.sh \
  --models hpsv3,videoalign --hps-device cuda:0 --videoalign-device cuda:1 \
  --host 0.0.0.0 --port 8791
```

仅需要一种时，将 `--models` 改成 `hpsv3` 或 `videoalign`，并选择对应设备。默认使用 SDPA，无需额外安装 FlashAttention。健康检查：

```bash
lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.rewards.client \
  --endpoint http://127.0.0.1:8791
```

UnifiedReward-Edit 服务示例：

```bash
CUDA_VISIBLE_DEVICES=2 LIGHTX2V_UNIFIEDREWARD_HOST=0.0.0.0 \
  bash lightx2v_rl/scripts/rewards/start_unifiedreward_server.sh
```

默认端口 8792、TP=1、服务模型名 `UnifiedReward`，通过 `LIGHTX2V_UNIFIEDREWARD_PORT`、`LIGHTX2V_UNIFIEDREWARD_TP` 等变量调整。可用 `curl --fail http://127.0.0.1:8792/health` 检查服务。

手工部署时，训练与 reward 选择不同 GPU；脚本不会自动为 Ray 预留评分卡。远端客户端使用服务节点的实际地址，`0.0.0.0` 仅用于监听。一个 HPSv3 / VideoAlign 实例串行处理评分，可增加独立实例扩容。

评分组合由配方控制：

- Qwen 默认 `UnifiedReward_normalized + 0.5 × HPSv3_general_raw`，同一图像的两项评分并发请求。0.5 是乘数，不代表固定贡献比例。T2I 的三个 1–5 分维度归一化为 `(均值 - 1) / 4`，I2I 的两个 0–25 分维度归一化为 `均值 / 25`。
- Wan 可选 `--reward-config hpsv3`、`videoalign`、`hpsv3_videoalign`。默认聚合为 `sum(weight_i × reward_i) / 指标数量`，VideoAlign 保留 checkpoint 的 z-score 标准化。

两条路径返回 scalar reward，再计算 GRPO advantage。评分失败或非有限结果会报错；UnifiedReward 有限的越界分数会先按量表裁剪，并保留原始值和裁剪指标。具体权重、指标与 endpoint 见 [Qwen 配置](lightx2v_rl/configs/qwen_image21_turbo/base.yaml) 和 [视频 reward 配置](lightx2v_rl/configs/rewards/)。

## 训练

### 配置入口

```bash
lightx2v_rl/.venv-rl/bin/python -m lightx2v_rl.train \
  --config qwen_image21_turbo_geo --print-config
```

`--config` 接受内置配方名或 YAML 路径，后接普通 `key=value` 覆盖，不使用 Hydra 的 `+` / `~` 前缀。切换算法时选择完整配方，保持 Actor、rollout 与 loss 参数一致。

| 模型 | Geo | CPS | Flow-GRPO |
| --- | --- | --- | --- |
| Qwen-Image-2.1-Turbo | `qwen_image21_turbo_geo` | `qwen_image21_turbo_cps` | `qwen_image21_turbo_flow_grpo` |
| Wan2.1 T2V | `wan_geo` | `wan_cps` | `wan_flow_grpo` |

默认每 20 iter 验证一次，可用 `trainer.test_freq` 调整。Qwen 默认训练前也验证，T2I / I2I 按 batch 1:1 交替；一个 iter 只处理一种任务。训练使用所选随机 bridge，验证使用确定性生成路径。

### Qwen：单机示例

此示例使用 4 张训练 GPU、1 张 HPSv3 GPU 和 1 张 UnifiedReward GPU。先完成模型与 OpenGPT-4o-Image 数据准备，在三个独立终端依次执行：

```bash
# 终端 1：HPSv3。
CUDA_VISIBLE_DEVICES=4 bash lightx2v_rl/scripts/rewards/start_reward_server.sh \
  --models hpsv3 --hps-device cuda:0 --port 8791

# 终端 2：UnifiedReward-Edit。
CUDA_VISIBLE_DEVICES=5 bash lightx2v_rl/scripts/rewards/start_unifiedreward_server.sh

# 终端 3：预检并训练。
CUDA_VISIBLE_DEVICES=0,1,2,3 lightx2v_rl/.venv-rl/bin/python \
  -m lightx2v_rl.models.qwen_image.preflight --config qwen_image21_turbo_geo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash lightx2v_rl/scripts/train/train_qwen21.sh
```

通过 `LIGHTX2V_RL_CONFIG` 选择其他 Qwen 配方。默认每个输入生成 4 个候选；调整训练卡数时，同时检查 batch、mini-batch 和模型显存需求。

### Qwen：五节点示例

提供 5 台 × 8 卡的配方，按 H100 80G 规划：4 个训练节点共 32 卡，1 个评分节点运行 6 个 UnifiedReward 与 2 个 HPSv3 实例。这是固定部署配方，其他拓扑需要同步适配配置和启动器。

平台须在**每个节点执行同一条命令一次**，提供各节点一致的 `MASTER_ADDR`、`MASTER_PORT`，工作目录设为仓库根目录：

```bash
set -euo pipefail
export LIGHTX2V_RL_CONFIG=qwen_image21_turbo_geo_40gpu
export LIGHTX2V_RDZV_ID=qwen-rl-run-001

lightx2v_rl/.venv-rl/bin/python lightx2v_rl/scripts/setup/prepare_verl_omni.py --check
bash lightx2v_rl/scripts/train/train_qwen21_platform.sh
```

`LIGHTX2V_RDZV_ID` 每个作业取唯一值、所有节点一致；`MASTER_PORT` 不使用 Ray 的 6379。启动器内部每节点只运行一个协调进程，无需再套多进程 torchrun。协调 rank 0 启动 Ray head 和唯一训练 driver，rank 1–3 启动 Ray worker，rank 4 启动评分池。评分节点不加入训练 Ray 资源池。

`40gpu` 指申请总卡数，配置中的 `trainer.nnodes=4` 仅统计训练节点。其他算法对应 `qwen_image21_turbo_cps_40gpu`、`qwen_image21_turbo_flow_grpo_40gpu`。默认每 iter 为 32 个同任务输入、每输入 4 个候选、1 次 optimizer update。

该入口依赖平台向所有节点分发命令；在普通 SSH 会话只运行一次不会自动启动其他机器。手工部署可使用 `scripts/train/qwen21_cluster.sh` 和 `scripts/rewards/start_qwen21_reward_pool.sh`，参数见各脚本的 `--help`。

### Wan：视频训练示例

先启动所需的 HPSv3 / VideoAlign 服务，再指定本地模型与数据清单。以下路径是示例，需要准备对应文件：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash lightx2v_rl/scripts/train/train_wan_reward.sh \
  --config wan_geo --reward-config hpsv3_videoalign \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.model.path=lightx2v_rl/models/wan-fewstep-diffusers \
  'data.train_files=[lightx2v_rl/data/wan/train.parquet]' \
  'data.val_files=[lightx2v_rl/data/wan/val.parquet]'
```

数据中的 `reward_model.ground_truth` 使用生成视频的原始 prompt。远端评分服务通过 `reward.custom_reward_function.reward_kwargs.endpoint` 指定。

## 日志与监控

默认输出到 console，W&B 为可选项。使用官方云端时，先登录或由平台提供 `WANDB_API_KEY`：

```bash
unset WANDB_BASE_URL
lightx2v_rl/.venv-rl/bin/wandb login
```

训练命令追加以下参数即可启用；默认 project 为 `lightx2v_rl`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash lightx2v_rl/scripts/train/train_qwen21.sh \
  'trainer.logger=[console,wandb]' \
  trainer.project_name=lightx2v_rl \
  trainer.experiment_name=qwen_geo \
  trainer.log_val_generations=8
```

多机时将同样参数传给平台启动脚本。运行进程应能读取凭据；离线记录可设置 `WANDB_MODE=offline`。仅使用自建 W&B 时才需设置对应的 `WANDB_BASE_URL`。

默认采用精简指标视图，展示 reward、policy / KL loss、clip 比例、学习率、梯度及耗时；Qwen 按 T2I / I2I 区分训练和验证曲线。`time/rollout_and_reward_s` 包含生成与等待评分，不能视为纯推理耗时。完整原始指标另存为 `outputs/metrics/*.jsonl`。

Qwen 验证分项显示在 `eval_t2i/`、`eval_i2i/` 下：`reward` 为组合总分，`unifiedreward` 为未乘权重的归一化分数，`hpsv3_raw` 为未乘权重的 HPS 分数，`ur_clipped_fraction` 为评分越界裁剪比例。任务独有的原始维度保存在完整指标和验证明细中，不适用的维度不按零分参与均值。更新运行时补丁后需重新启动训练进程，新产生的 eval 才会记录这些明细。

| 内容 | 模块内位置 |
| --- | --- |
| 安装日志 | `outputs/setup/` |
| 平台协调及训练 stdout | `outputs/platform/<run>/rank<rank>-<pid>/` |
| 独立评分实例日志 | `outputs/reward_pool/<run-id>/` |
| 原始训练指标 | `outputs/metrics/` |
| checkpoint、验证结果 | 配方的 `trainer.default_local_dir`、`trainer.validation_data_dir` |

Ray Dashboard 默认监听 head 节点的 `127.0.0.1:8265`，通过平台端口转发或 SSH 隧道访问。平台需要按节点 IP 转发时，可在启动前设置 `LIGHTX2V_RAY_DASHBOARD_HOST=0.0.0.0`，配合平台访问控制。Ray 展示训练集群任务、Actor 和进程日志；独立 reward 节点需结合其服务日志与平台 GPU 监控查看。

## 开发

```text
lightx2v_rl/
├── lightx2v_rl/              # Python 源码
│   ├── algorithms/fewstep/  # CPS、Flow-GRPO、Geo 的共享数学实现
│   ├── rollout/             # 模型无关的轨迹字段与构建
│   ├── models/              # registry、qwen_image/、wan/ 等模型适配
│   ├── backends/verl_omni/   # 配置、loss 与通用运行时接入
│   ├── rewards/             # 评分模型、HTTP 协议、解析与聚合
│   ├── data/                # 数据集适配和采样器
│   ├── training/            # 配方组装、指标记录
│   ├── deployment/          # 集群启动与进程管理
│   └── configs/             # 公共、模型、算法、reward 与部署配方
├── scripts/{setup,assets,rewards,train}/
├── requirements/            # 环境依赖约束
└── patches/                 # 固定 verl-omni 的通用扩展补丁
```

顶层 `models/`、`data/` 是运行资源入口；Python 包内同名目录存放代码。

新增模型时，在 `lightx2v_rl/models/<模型族>/` 实现 Actor 重算、rollout 与条件输入适配，在 `models/registry.py` 注册 `ModelSpec`，通过配方的 `actor_rollout_ref.model.rl_adapter` 选择。模型特有的 timestep、sigma、参考输入和 checkpoint 契约归模型目录；共享 bridge、logprob 和 loss 归算法目录。

新增 reward 放在 `rewards/`，新增数据集放在 `data/`。算法和模型专属逻辑留在本模块，submodule 只维护必要的通用接口补丁。升级 verl-omni 时需重新验证补丁、配置和 Actor / rollout 数据契约。

当前 advantage 路径为 scalar reward → GRPO，尚未迁移 FewStepRL 的逐 reward advantage 与 `max_group_std` estimator。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `uv` 存在，但模型依赖导入失败 | uv 是安装工具；运行对应环境的安装脚本与验证命令 |
| `ensurepip is not available` | 安装 uv 后重跑；使用标准库 venv 时需安装匹配的 `python3.x-venv` |
| `Python.h: No such file or directory` | 在运行服务的节点安装与解释器匹配的 `python3.x-dev` |
| 无法导入 `QwenImage21Pipeline` | 执行 `bash lightx2v_rl/scripts/setup/repair_qwen21_diffusers.sh` 恢复固定官方 Diffusers |
| submodule 补丁检查失败 | 运行 `prepare_verl_omni.py` 应用补丁；保持父仓库记录的 submodule 版本 |
| 资源或 manifest 缺失 | 检查相对链接目标、下载/迁移完成状态，保留资产清单 |
| 多节点出现 rendezvous 断连 | 优先查看最早的训练或服务异常，以及对应节点日志；断连可能是作业退出的后续现象 |
| reward 连接失败 | 检查服务健康、endpoint、端口与节点网络；避免手工服务和平台入口重复占用同一端口 |

参考：[verl-omni 安装](../third_party/verl-omni/docs/start/install.md)、[diffusion v1 trainer](../third_party/verl-omni/docs/start/diffusion_v1.md)、[LICENSE](LICENSE)、[第三方授权说明](THIRD_PARTY_NOTICES.md)。
