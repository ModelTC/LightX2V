# HY-Motion-1.0 → LightX2V (A100)

目标是 `tencent/HY-Motion-1.0` 完整版，权重 revision
`620dd559f8d964aac2f82f1204fe6a35ad8ad14d`，不是 Lite 版本。
接入代码移植自 Tencent HY-Motion 源码 `4e426f5a1021cbcf7f375458c37b840ee7225229`；
模型目录保留原始许可证。运行时不需要原仓库、FBX SDK、Gradio 或 transformers 模型类。

先在独立 Python 3.10 环境中执行 `bash scripts/hymotion/install_a100.sh`。
准备本地权重，目录如下；已经下载的组件可用符号链接复用：

```text
weights/hymotion/
  HY-Motion-1.0/config.yml
  HY-Motion-1.0/latest.ckpt
  Qwen3-8B/                    # Qwen/Qwen3-8B 完整 safetensors + tokenizer/config
  clip-vit-large-patch14/      # openai/clip-vit-large-patch14
```

动作反归一化参数和固定人体几何资产随适配代码提供。神经网络权重全部检查；官方检查点缺少的
5 个 `body_model.*` 几何缓冲区从固定资产加载，其他缺失或多余参数均报错。
Qwen 和 CLIP 目录也可分别通过 `QWEN_MODEL_PATH`、`CLIP_MODEL_PATH` 指定，
或使用部署 JSON 的 `qwen_model_path`、`clip_model_path` 字段。

```bash
MODEL_PATH=/path/to/HY-Motion-1.0 \
QWEN_MODEL_PATH=/path/to/Qwen3-8B \
CLIP_MODEL_PATH=/path/to/clip-vit-large-patch14 \
PROMPT='A person walks forward at a steady pace.' \
SAVE_RESULT_PATH=save_results/hymotion/walk.npz \
bash scripts/platforms/nvidia/run_hymotion.sh --num_frames 120
```

默认 50 步 Euler、CFG=5、seed=42、30 FPS。`num_frames` 接受 1–4999；与原生实现一样，
实际帧数截断到 20–360，不做视频 VAE 的帧数对齐。输出 NPZ 包含原生 `poses`、`trans`、`betas`、`Rh`、
`rot6d`、`keypoints3d` 和 `latent_denorm`，可用于动作可视化和后续动画处理。

目前支持 NVIDIA A100 单卡、不量化、无 offload/cache。MLU、MetaX、Ascend 的 JSON
与同名脚本是显式占位，执行会说明未实现并退出。

CLIP 文本清理默认为 `clip_text_cleanup: basic`，复用 Transformers 的 BasicTokenizer，与原项目仅安装其 requirements.txt、没有额外 ftfy 时的行为一致。这样，其他模型安装 ftfy 不会悄然改变弯引号等 Unicode 输入的输出。若参考原项目环境显式安装了 ftfy，可在模型配置中设置 `clip_text_cleanup: ftfy` 对齐该清理方式。
