# Wan CPU Offload 常驻权重

[English](README.md) | 简体中文

单卡运行 Wan2.1 图生视频，对比两种权重使用方式：`baseline` 将所有 block 的权重放在 CPU，计算时按需搬到显存；常驻策略让部分 block 的权重在去噪期间留在显存，减少重复搬运。

先运行 `estimate` 测量一个 block 的计算和搬运耗时，生成常驻数量，再运行 `run` 推理。常驻 block 会均匀分布在模型各层中。`run_baseline` 用于对比，也可以单独运行。

安装对应平台的项目依赖后，在仓库根目录执行以下命令。

## NVIDIA GPU

使用 `Wan2.1-I2V-14B-720P-Lightx2v` 的 FP8 权重。先设置三个脚本开头的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`，三处保持一致。

在 `configs/wan/residency/baseline.json` 中设置以下权重路径，建议使用绝对路径。这些配置不会随脚本中的 `model_path` 自动改变。

- `dit_quantized_ckpt`：FP8 DiT 权重目录。
- `t5_quantized_ckpt`：量化 T5 文件，示例为 `models_t5_umt5-xxl-enc-fp8.pth`。
- `clip_quantized_ckpt`：量化 CLIP 文件，示例为 `clip-fp8.pth`。

```bash
bash scripts/wan/inference/offload_residency/estimate.sh
bash scripts/wan/inference/offload_residency/run.sh
bash scripts/wan/inference/offload_residency/run_baseline.sh
```

## Ascend NPU

使用原始 `Wan2.1-I2V-14B-720P` BF16 权重，包含文本编码器、CLIP 和 VAE。设置三个 NPU 脚本开头的 `lightx2v_path`、`model_path` 和 `ASCEND_RT_VISIBLE_DEVICES`，基础配置为 `configs/wan/residency/baseline_npu.json`。

```bash
bash scripts/wan/inference/offload_residency/estimate_npu.sh
bash scripts/wan/inference/offload_residency/run_npu.sh
bash scripts/wan/inference/offload_residency/run_baseline_npu.sh
```

## 修改输入和配置

输入图片在估测脚本的 `--image-path` 和两个推理脚本的 `--image_path` 中设置，三处应使用同一张图片。两个推理脚本的正负提示词和 `seed` 也应保持一致。

默认生成 81 帧，推理 40 步。需要调整帧数、尺寸或步数时，先修改对应平台基础配置中的 `num_frames`、`size`、`infer_steps`，再运行估测脚本，生成配套的常驻配置。

`cpu_offload_layout` 默认是 `per_tensor`。如果要使用连续权重搬运，将基础配置中的该参数改为 `contiguous`，再重新估测；baseline 和常驻策略应使用相同布局。

## 查看结果与对比

文件保存在 `save_results/wan_residency/`，重复运行会覆盖同名结果。

| 内容 | NVIDIA GPU | Ascend NPU |
| --- | --- | --- |
| 生成的常驻配置 | `resident.json` | `resident_npu.json` |
| 测量报告 | `resident.profile.json` | `resident_npu.profile.json` |
| 常驻策略视频 | `resident.mp4` | `resident_npu.mp4` |
| baseline 视频 | `baseline.mp4` | `baseline_npu.mp4` |

生成配置中的 `offload_resident_blocks` 就是常驻 block 数量。测量报告中的 `resident_blocks`、`streamed_blocks` 分别是常驻和按需搬运的 block 数量，`compute_ms`、`transfer_ms` 分别是单个 block 的计算和搬运时间，单位为毫秒。测出零常驻是正常结果，表示工具估计当前输入无需额外常驻权重。

需要对比耗时时，将两个推理脚本中的 `PROFILING_DEBUG_LEVEL` 都设为 `1`，在同一设备上依次运行，查看日志中每步的 `Run Dit every step` 耗时。首步包含初始化开销，比较稳定速度时应单独看后续步骤。

常驻数量越多，占用的显存越多。工具不会根据剩余显存自动调低数量；需要节省显存时，可调低生成配置中的 `offload_resident_blocks`。重新估测会覆盖手动修改。

更换设备、模型权重、量化方式、attention 实现、输入图片、帧数、尺寸或搬运布局后，应重新估测。估测结果用于选择常驻数量，实际加速效果以推理对比为准。
