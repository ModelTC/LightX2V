# VGGT GSO 评测（A100）

## 环境准备

准备 NVIDIA A100、NVIDIA 驱动、Python、Docker 和 NVIDIA Container Toolkit。容器镜像需要包含 CUDA 版 PyTorch、torchvision 及 LightX2V 基础运行依赖。

准备 NumPy、SciPy、Pillow、Transformers 等图像处理和评测依赖。

将 `EVAL_IMAGE` 替换为所使用的镜像名称，并检查容器能否访问 GPU：

```bash
export EVAL_IMAGE="your-registry/lightx2v-eval"
docker run --rm --gpus device=0 --entrypoint python "$EVAL_IMAGE" \
  -c "import torch, torchvision; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

评测项目的附加 Python 依赖由 `runtime-requirements.txt` 声明，在启动单对象验证时通过 `--install-deps` 安装。

## 数据与权重

准备 GSO 的同步 RGB、深度、相机位姿和内参渲染数据，以及模型权重。数据清单包含 1023 个对象，输入文件的相对路径由 `configs/common/suite.json` 指定。

```text
data/
  fixtures/    # 按 suite.json 放置输入和真值
  models/      # 放置模型权重
  deps/        # 放置依赖模型权重
```

权重目录由 `configs/nvidia/paths.json` 配置，文件清单和哈希由 `configs/common/weight-lock.json` 指定。将文件按配置中的相对路径放在数据根目录下。

| 权重 | 配置项 |
|---|---|
| VGGT 权重（含 `model.pt`） | `weights.vggt` |

## 配置本机运行环境

在同一终端中设置项目、数据和输出目录，选择一张空闲 GPU：

```bash
PROJECT="/path/to/LightX2V/tools/evaluation/vggt"
DATA_ROOT="/path/to/data"
RESULT_ROOT="/path/to/results/vggt"
CARD=0
cd "$PROJECT"
mkdir -p runtime cache "$RESULT_ROOT"
nvidia-smi -i "$CARD"
```

配置 NVIDIA 容器入口、依赖搜索路径和设备映射：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path("configs/nvidia/runtime.json")
config = json.loads(path.read_text())
config["container_image"] = os.environ["EVAL_IMAGE"]
config["docker_device_args"] = ["--gpus", "device={card}"]
config["docker_common_mounts"] = []
config["visibility_environment"] = {
    "CUDA_VISIBLE_DEVICES": "0",
    "EVAL_PHYSICAL_CARD": "{card}",
}
config["environment"]["PYTHONPATH"] = "/eval/runtime"
config["environment"].pop("TRITON_LIBCUDA_PATH", None)
config["idle_memory_mib_max"] = 1
path.write_text(json.dumps(config, indent=2) + "\n")
PY
```

`--card` 指定宿主机的物理 GPU 编号，容器内使用单张逻辑 GPU。模型后端由 `configs/nvidia/model.json` 配置。

## 启动评测

准备模型源码并校验输入、真值和权重：

```bash
bash scripts/platform/nvidia/run_eval.sh --prepare-sources
bash scripts/platform/nvidia/run_eval.sh \
  --check-inputs --rehash-weights \
  --data-root "$DATA_ROOT" \
  --output "$RESULT_ROOT/input-check"
```

安装附加依赖并进行单对象验证：

```bash
bash scripts/platform/nvidia/run_eval.sh \
  --install-deps --smoke --card "$CARD" --repeats 1 \
  --data-root "$DATA_ROOT" \
  --output "$RESULT_ROOT/smoke"
```

运行完整 GSO 清单，每个对象一次正式请求：

```bash
bash scripts/platform/nvidia/run_eval.sh \
  --card "$CARD" --repeats 1 \
  --data-root "$DATA_ROOT" \
  --output "$RESULT_ROOT/full"
```

每次运行使用新的输出目录。结果目录中的 `report.md`、`summary.json`、`metrics.csv` 和 `requests.csv` 分别保存报告、汇总、逐对象精度和逐请求记录。失败对象保留记录，指标按成功输出统计。

E2E 包含读图、预处理、模型推理、后处理和 CPU 输出转换；不包含模型加载、预热、结果写盘和离线评分。

## 测试结果

| 指标 | 结果（成功输出） |
|---|---:|
| CD-L1 均值 ↓ | 0.146713018 |
| CD-L1 中位数 ↓ | 0.140144518 |
| E2E 均值 | 0.437304 秒 |
| E2E 中位数 | 0.431934 秒 |
