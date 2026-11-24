# SDXL base 1.0 → LightX2V (A100)

目标为文生图模型 `stabilityai/stable-diffusion-xl-base-1.0`，revision
`462165984030d82259a11f4367a4eed129e94a7b`，不包含 refiner。
先在独立 Python 3.10 环境中执行 `bash scripts/three_models/install_a100.sh`。

保持官方 Diffusers 权重目录布局：`unet/`、`vae/`、`text_encoder/`、`text_encoder_2/`、
`tokenizer/`、`tokenizer_2/`、`scheduler/`。优先读取 `.fp16.safetensors`，只访问本地文件。
Diffusers 是文件布局；运行时不实例化 Diffusers 的模型、pipeline 或 scheduler。
transformers 仅提供 tokenizer。

```bash
MODEL_PATH=/path/to/stable-diffusion-xl-base-1.0 \
PROMPT='A red apple on a wooden table, studio photograph.' \
SAVE_RESULT_PATH=save_results/sdxl/apple.png \
bash scripts/platforms/nvidia/run_sdxl.sh
```

默认 1024×1024、30 步 Euler、CFG=5、seed=42。支持 `--negative_prompt`、
`--size HEIGHT WIDTH`，尺寸至少 128 且为 64 的倍数。
PNG 旁的 `.png.json` 给出从分词至 PNG 写出、GPU 同步完成的时延，排除权重加载。

`weights/` 只加载/保存权重、配置和算子绑定；`infer/` 实现 UNet 与 VAE；
共享 `three_models/clip.py` 实现双 CLIP 文本推理；`models/schedulers/sdxl/` 实现 Euler。
MM、Conv2d、LayerNorm、Attention 复用公共注册表；GroupNorm、残差、GEGLU 和条件拼接
保留 SDXL 语义。UNet/CLIP 用 FP16，VAE 用 FP32，解码前先上转换再除 scaling factor。

Euler 的 sigma 缓存保留在 CPU，与原生标量提升规则一致；线性绑定保留非连续张量的
矩阵乘法折叠和 bias 舍入顺序。随机 GenEval、Parti、COCO 各一条提示词的 30 步潜变量、
解码张量及最终 PNG 已与原生实现逐元素对照，PNG 的 SHA256 一致。

A100 单卡已实现；MLU、MetaX、Ascend 脚本与 JSON 为明确占位。暂不支持
refiner、LoRA、量化、并行、offload 或 feature cache。
