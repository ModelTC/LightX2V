# 生成第一个视频

本教程使用 Wan2.1-T2V-1.3B。先完成[环境安装](quickstart.md)，再从 LightX2V 仓库根目录执行下面的命令。

## 下载模型

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir /path/to/models/Wan2.1-T2V-1.3B
```

也可以通过 ModelScope 下载 `Wan-AI/Wan2.1-T2V-1.3B`，将下载后的本地目录作为 `model_path`。

## 配置并运行

打开 [run_wan_t2v.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/run_wan_t2v.sh)，设置以下内容：

| 设置 | 内容 |
| --- | --- |
| `lightx2v_path` | 当前仓库的绝对路径 |
| `model_path` | 下载后模型目录的绝对路径 |
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU，例如 `0` |
| `--prompt` / `--negative_prompt` | 希望生成的场景与需要避免的内容 |
| `--size` | 输出尺寸，按 `高 宽` 填写；脚本默认 `480 832` |
| `--num_frames` | 输出帧数；脚本默认 `81` |
| `--save_result_path` | 输出视频路径 |

脚本选择 [wan_t2v.json](https://github.com/ModelTC/LightX2V/blob/main/configs/wan/wan_t2v.json)。采样步数、引导强度等启动设置在 JSON 中修改，请求参数在启动脚本中修改。

脚本显式传入的 `--size`、`--num_frames` 会覆盖 JSON 中的同名默认值，因此仅修改 JSON 不会改变当前脚本的输出尺寸和帧数。请修改脚本中的这两个参数；若希望由 JSON 统一控制，先删除脚本中的这两个参数，再设置 JSON 的 `size`（`[高, 宽]`）和 `num_frames`。

该 JSON 使用 `flash_attn3`，请根据显卡和安装环境选择支持的算子，见[注意力机制](../method_tutorials/attention.md)。需要降低显存占用时，参考[参数卸载](../deploy_guides/for_low_resource.md)。

```bash
bash scripts/wan/run_wan_t2v.sh
```

推理完成后，打开 `--save_result_path` 对应的视频。脚本默认写入仓库下的 `save_results/output_lightx2v_wan_t2v.mp4`。

## 继续使用

- [Wan2.1 与 VACE](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/README_zh.md)：其他模型规格、蒸馏权重、服务和多卡运行。
- [Wan2.2](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/README_zh.md)、[Qwen Image](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image/README_zh.md)：对应模型的启动入口和配置。
- [Python API](python_api.md)：在应用中调用生成、复用同一个 Pipeline。
- [服务化部署](../deploy_guides/deploy_service.md)：通过接口提交生成请求。Wan 专用的服务和客户端入口见 Wan README。

比较 CLI、服务与 Python 的结果时，应使用同一模型、JSON 和请求参数，包括 seed、尺寸和帧数。手动配置的 Python 示例可能采用不同的采样参数。
