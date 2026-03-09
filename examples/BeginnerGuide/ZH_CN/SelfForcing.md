
# Self-Forcing (Wan2.1-T2V-1.3B)

本文档介绍如何在 LightX2V 中运行 `Wan2.1-T2V-1.3B` 的 Self-Forcing 加速。

## 准备环境

请参考[01.PrepareEnv](01.PrepareEnv.md)

## 下载

### 1. BF16 原始模型 (Wan2.1-T2V-1.3B)

从 Self-Forcing 的 HuggingFace 仓库下载 BF16 原始模型：

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

### 2. Self-Forcing checkpoint

下载 Self-Forcing 的 checkpoint（`self_forcing_dmd.pt`）：

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 3. 量化模型 (FP8 / NVFP4)

从 LightX2V 的 HuggingFace Collection 下载量化模型：

https://huggingface.co/collections/lightx2v/ar-lightx2v

## 开始运行

我们提供三个脚本：

1. `scripts/self_forcing/run_wan_t2v_sf.sh`：BF16 原始模型 + Self-Forcing checkpoint
2. `scripts/self_forcing/run_wan_t2v_sf_fp8.sh`：FP8 量化模型
3. `scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh`：NVFP4 量化模型

运行下面任意脚本之前，你需要先把脚本中的路径变量填好。

### 1) BF16 + Self-Forcing checkpoint

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf.sh
```

`run_wan_t2v_sf.sh` 中关键字段：

- `lightx2v_path`：你的 LightX2V 仓库路径
- `model_path`：原始 BF16 `Wan2.1-T2V-1.3B` 模型目录路径
- `sf_model_path`：`self_forcing_dmd.pt` 的路径（从 `gdhe17/Self-Forcing` 下载）

该脚本实际会执行：

- `python -m lightx2v.infer`
- `--model_cls wan2.1_sf`：使用 Wan2.1 的 Self-Forcing pipeline
- `--task t2v`：文生视频
- `--sf_model_path`：Self-Forcing checkpoint
- `--config_json configs/self_forcing/wan_t2v_sf.json`：Self-Forcing 的运行配置

注意：`configs/self_forcing/wan_t2v_sf.json` 中默认 `enable_cfg=false`。

### 2) FP8 量化模型

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

关于 FP8 的额外说明：

- 该脚本使用 `configs/self_forcing/wan_t2v_sf_fp8.json`。
- 你需要把该配置文件中的 `dit_quantized_ckpt` 修改为从 HuggingFace collection 下载的 FP8 权重文件的**绝对路径**。
- `dit_quant_scheme` 为 `fp8-vllm`。

### 3) NVFP4 量化模型

修改并运行：

```
bash scripts/self_forcing/run_wan_t2v_sf_nvfp4.sh
```

关于 NVFP4 的额外说明：

- 该脚本使用 `configs/self_forcing/wan_t2v_sf_nvfp4.json`。
- 你需要把该配置文件中的 `dit_quantized_ckpt` 修改为从 HuggingFace collection 下载的 NVFP4 权重文件的**绝对路径**。
- `dit_quant_scheme` 为 `nvfp4`。
