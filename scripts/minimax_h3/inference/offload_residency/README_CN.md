# MiniMax-H3 常驻权重

[English](README.md) | 简体中文

单卡运行 MiniMax-H3 文生音视频，对比零常驻 baseline 和部分 block 常驻策略。先用独立工具测量一个 block 的计算时间与搬运时间，生成常驻数量；推理直接读取结果，将常驻层均匀分布在全部 DiT blocks 中。

## 运行前准备

按[模型说明](../../README_zh.md)准备完整 BF16 Diffusers 组件。在本目录脚本顶部设置 `lightx2v_path`、`model_path` 和设备编号。默认任务为 `t2av`，使用 `fl2av` 模型权重；输入为 124 帧、768×1344、29 步。提示词在脚本的 `--prompt` 中修改，估测、baseline 和常驻推理应保持一致。

先按[模型的 AdaLN 缓存说明](../../README_zh.md#adaln-缓存)生成匹配缓存。现有缓存生成入口为 `tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh`：设置模型路径、对应平台，以及下面的 baseline 配置，选择 `--model-variant fl2av`。同一平台上，baseline、估测和常驻推理共用缓存。旧缓存校验不通过时，先保留旧目录，再重新生成。

## NVIDIA GPU

配置为 `configs/minimax_h3/residency/baseline.json`。设置三个脚本中的 `CUDA_VISIBLE_DEVICES`，从项目根目录运行：

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate.sh
bash scripts/minimax_h3/inference/offload_residency/run.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline.sh
```

## Ascend NPU

配置为 `configs/minimax_h3/residency/baseline_npu.json`。设置三个 NPU 脚本中的 `ASCEND_RT_VISIBLE_DEVICES`：

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline_npu.sh
```

baseline 可单独运行；常驻推理需要先完成对应平台的估测。

## 结果在哪里

所有结果保存在 `save_results/minimax_h3_residency/`，重复运行覆盖同名文件。

| 文件 | 内容 |
| --- | --- |
| `resident.json` / `resident_npu.json` | 生成的推理配置，包含 `offload_resident_blocks` |
| `resident.profile.json` / `resident_npu.profile.json` | block 计算、搬运耗时，常驻数量与层编号 |
| `resident.mp4` / `resident_npu.mp4` | 常驻策略生成的音视频 |
| `baseline.mp4` / `baseline_npu.mp4` | 零常驻 baseline 生成的音视频 |

估测仅加载一个真实 block，构造与输入规模对应的文本、音频和视频激活，并加载正式 AdaLN 缓存。计算与搬运分别预热 3 次、采样 10 次，取中位数。它不运行完整去噪，不包含文本编码和 VAE，因此完成得比推理快。

常驻数量按 `r = max(0, N - floor(N*c/d))` 估算，其中 `N` 是 block 总数，`c`、`d` 分别是计算和搬运时间。得到 `r=0` 是正常结果。该总量估计不保证每次搬运都完全隐藏，也不按剩余显存自动降低 `r`。

baseline 设置 `r=0`：所有 DiT block 主权重在 CPU，需要时搬入两个设备缓冲。常驻策略也保留全部 CPU 主权重，但选中的 `r` 层在去噪期间另有设备副本，跨步骤复用，并在 VAE 解码前释放。因此 `r=0` 不代表推理不占显存。

## 修改配置与比较

修改 baseline 配置中的帧数、尺寸、步数或算子后，重新运行估测；更换硬件、权重、精度或提示词长度也应重新估测。调整步数、采样偏移或权重时，还需要重新生成匹配的 AdaLN 缓存。生成配置中的 `r` 可以手动调低，但再次估测会覆盖它。

默认 `cpu_offload_layout="per_tensor"`。连续布局是独立选项：需要时将 baseline 配置改为 `"contiguous"`，再重新估测。两组对比使用相同布局。

本入口针对 BF16、单设备 eager、`NoCaching`。不支持量化、共享 CPU 权重、并行、LoRA、compile、磁盘流式加载或 fused QKV。估测输入目前只覆盖 `t2av`；其他任务不能直接沿用这个数量作为测量结论。

脚本默认关闭详细 profiling。若比较去噪耗时，可将两份推理脚本的 `PROFILING_DEBUG_LEVEL` 都设为 `1`，比较相同输入与 seed 下的 `Run Dit every step`。该日志包括调度与模型计算；首步还包含常驻权重准备，不能将估测报告里的单 block 时间当作整步时间。
