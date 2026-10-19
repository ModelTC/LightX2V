# WorldMirror 示例

本目录存放 HY-WorldMirror-2.0 的使用示例。它是一个单步多头 3D 重建模型（同时输出 depth / normal / camera / points / Gaussian splats），通过 LightX2V runner 运行。

## 推理效果

<div align="center">
  <img src="../../assets/figs/hyworld/pic_acceleration.png" alt="HY-WorldMirror-2.0" width="60%">
</div>

## 模型下载

在使用示例脚本之前，需要先下载对应的模型权重，可通过以下地址获取：

3D 重建模型
- [HY-WorldMirror-2.0](https://huggingface.co/tencent/HY-World-2.0)

下载后的目录应包含 `HY-WorldMirror-2.0/` 子文件夹，其中有 `model.safetensors` 和 `config.json`。

## 使用方式一：使用 Bash 脚本（强烈推荐）

环境搭建推荐使用我们提供的 Docker 镜像，详见 [快速上手](../../docs/ZH_CN/source/getting_started/quickstart.md)。

```
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V/scripts/worldmirror

# 运行下列脚本前，修改脚本中的 lightx2v_path、model_path、--input_path 和 --save_result_path。
```

3D 重建模型
```
# fp32 精度推理（默认精度，与上游参考输出的 MAE 在 1e-3 以内）
bash run_worldmirror_recon.sh

# fp8-pertensor 量化推理，显存峰值下降约 0.6 GB，速度略快
# （与参考输出的 MAE 在 1e-2 以内）。所需 input-scale 标定文件
# configs/worldmirror/worldmirror_input_scales.safetensors 已在 config 中引用。
bash run_worldmirror_recon_fp8.sh
```

两个脚本默认 `RENDER_VIDEO=1`，会额外渲染一段 Gaussian-splat 相机插值漫游视频，
落盘到 `<SAVE_RESULT_PATH>/<case>/<timestamp>/rendered/rendered_rgb.mp4`。
设 `RENDER_VIDEO=0` 可关闭渲染；设 `RENDER_DEPTH=1` 可额外输出一段 depth 漫游。

## Python API

安装后，从仓库根目录运行下列代码。使用 FP8 时将配置文件改成
`worldmirror_recon_fp8.json`；使用 BF16 时，在构建 runner 前设置
`config["enable_bf16"] = True`。

```python
import json

from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config

with open("configs/worldmirror/worldmirror_recon.json") as f:
    config = json.load(f)
config["model_path"] = "/path/to/HY-World-2.0"
runner = build_runner(build_startup_config(config))
request = runner.prepare_request({
    "input_path": "/path/to/scene_dir",
    "save_result_path": "/path/to/output",
    "return_result_tensor": True,
})
result = runner.run_request(request)
print(result["output_dir"])
print(result["timings"])
```

WorldMirror 通过文件交付重建结果。省略 `save_result_path` 或传入 `None` 时，
默认写入 `./inference_output/<场景>/<时间戳>/`；`strict_output_path` 优先，
直接写入指定目录。返回值始终包含 `output_dir`。这里的 `return_result_tensor=True`
额外返回 `timings`（rank 0 为计时字典，其他 rank 为 `None`），不返回重建张量。
这保留了程序化读取结果和计时的用法。

## 交互式和进阶 CLI

[run_worldmirror.py](run_worldmirror.py) 支持逐 head 禁用、mask 控制、相机和深度先验、
训练 YAML 与 checkpoint 加载、渲染，以及 `>>>` 交互循环。
省略 `--no_interactive` 可复用已加载的模型继续处理其他场景。

```bash
python scripts/worldmirror/run_worldmirror.py \
    --input_path /path/to/scene_dir \
    --pretrained_model_name_or_path /path/to/HY-World-2.0 \
    --lightx2v_config configs/worldmirror/worldmirror_recon_fp8.json \
    --save_rendered --no_interactive
```
