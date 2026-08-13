# Wan-Animate-2

LightX2V 原生支持 Wan-Animate-2 Base 和 Distilled。它直接读取身份参考图和驱动视频，不需要旧版 Wan2.2-Animate 的 pose、face 预处理结果。

源码对齐模式要求 CUDA、BF16、PyTorch 2.7+（需提供 `torch.nn.attention.flex_attention`）以及 FlashAttention 2。官方 720p 配置建议使用 8 卡 SP8；单卡配置主要用于 block-offload 调试，720×1280 下仅静态 reference K/V 就约占 62 GiB。

## 模型目录

从 `Wan-AI/Wan2.2-Animate-2-14B` 下载后的目录应保持官方布局：

```text
Wan2.2-Animate-2-14B/
├── videomodel/Wan-AI/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   ├── umt5-xxl/
│   ├── xlm-roberta-large/
│   └── vae.pth
└── wan_animate_2/
    ├── wan_animate_2_bf16.safetensors
    └── wan_animate_2_bf16_distillation.safetensors
```

配置中的 checkpoint 路径相对 `--model_path` 解析，无需写本机绝对路径。

## 运行

四个脚本都采用 Wan 示例的扁平写法。模型、参考图、驱动视频和 GPU
直接写在脚本顶部；默认使用本地模型与上游 `examples/demo1` 输入。
需要换输入时直接修改对应脚本顶部的 `model_path`、`image_path` 和
`video_path`。

Base 单卡配置使用 block CPU offload：

```bash
bash scripts/wan22/run_wan22_animate2.sh
```

10 步 Distilled、无 CFG：

```bash
bash scripts/wan22/run_wan22_animate2_distill.sh
```

上游的 8×A800 运行形态使用 SP8，VAE 在各 rank 复制：

```bash
bash scripts/wan22/run_wan22_animate2_8gpu.sh

bash scripts/wan22/run_wan22_animate2_distill_8gpu.sh
```

`image_path` 是身份参考图；`video_path` 是原始驱动视频。主 `prompt` 应描述
参考角色的外观与背景，不要描述动作；`prompt_ref` 默认是
`人物动作的参考视频`。脚本中的 seed 为 `-1`，与上游一样在每次请求时
随机选择；对拍时请在脚本里改成相同的固定 seed。

## Parity profiles

| Profile | DiT checkpoint | Steps | CFG | `log_scale` | Parallel |
| --- | --- | ---: | ---: | ---: | --- |
| Base | `wan_animate_2_bf16.safetensors` | 40 | 3.0 | 0.0 | 单卡 + block offload / SP8 |
| Distilled | `wan_animate_2_bf16_distillation.safetensors` | 10 | disabled | -1.3 | 单卡 + block offload / SP8 |

两种 profile 都使用上游原生 pipeline 实际调用的 shifted-flow、二阶 midpoint DPM-Solver++。上游 YAML 和 Diffusers 示例中的 `flow_solver: euler` 并不是原生 pipeline 的实际执行路径。

默认 clip 长度为 81 帧、重叠为 1 帧、输出为 24 FPS。长驱动视频按 clip 自回归生成；每个新 clip 会把上一 clip 的最后一个像素帧重新送入 VAE，而不是复用生成分支的滚动 KV cache。Base 的 CFG 无条件分支还会跳过第 10 个 Transformer block，这些行为均由 Wan-Animate-2 runner 负责。

单卡配置的 CPU offload 只改变参数驻留位置。做严格的上游 8 卡执行形态对比时，使用 `_8gpu` 配置和固定相同的参考图、驱动视频、两个 prompt、negative prompt、seed 与 BF16 环境。

本实现已在 H100 上与该源码仓库完成分层和端到端对拍：两份权重结构、FPS/resize/padding/clip 计划、初始噪声、2/10/40 步 DPM 更新、VAE、T5、CLIP，以及完整 Transformer block/head 均逐 tensor 对齐。相同 FA2/SP2 拓扑下，Distilled 10-step 的每一步 `noise_pred` 和 latent（包括最终 latent）均 bitwise exact；Base 40-step/CFG3 的编码前 24 帧 RGB 逐元素 bitwise exact。82 帧输入还覆盖了 `81 + 29` 的跨 clip 像素回灌与 overlap 裁剪。

严格对拍必须固定相同 GPU/SP 拓扑、FlashAttention 版本、两个 prompt、negative prompt 与 seed，并比较编码前 RGB tensor；MP4 会额外叠加视频编码差异。官方安装命令提供的是 FlashAttention 2，本配置因此显式使用其 varlen kernel；如果环境另外提供 `flash_attn_interface`，上游会优先选择 FA3，比较时两边必须使用相同 backend。

上游和本实现的 generation self-attention 都使用 `torch.compile(..., mode="max-autotune")` 的 FlexAttention。其独立冷启动可能选到数值略有差异的合法内核；同一实现用相同输入独立启动也不保证 bitwise 可复现。逐元素严格对拍需同时固定 PyTorch/Triton、GPU 拓扑与编译缓存/内核选择；普通验收应比较编码前 tensor，并为独立进程保留合理浮点容差。

## 权重结构检查

下面的检查只读取 safetensors JSON header，不导入 Torch，也不会加载约 32 GB 的 tensor payload：

```bash
python tools/convert/check_wan_animate2_checkpoint.py \
    /path/to/Wan2.2-Animate-2-14B
```

它会验证 Base/Distilled 都是 40 层、1303 个 BF16 tensors、`blocks.N.block.*` key 布局，确认旧版 pose/face 模块不存在，并要求两份 checkpoint 的 key、dtype 和 shape 完全一致。
