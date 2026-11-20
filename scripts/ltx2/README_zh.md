# LTX 音视频生成

[English](README.md) | [简体中文](README_zh.md)

## 准备

完成[环境安装](../../docs/ZH_CN/source/getting_started/quickstart.md)，按模型版本和任务选择入口：

| 版本或任务 | 入口 |
| --- | --- |
| LTX-2 文生音视频 | [run_ltx2_t2av.sh](run_ltx2_t2av.sh) |
| LTX-2 图生音视频 | [run_ltx2_i2av.sh](run_ltx2_i2av.sh) |
| LTX-2.3 文生 / 图生音视频 | [run_ltx2_3_t2av.sh](ltx2_3/run_ltx2_3_t2av.sh) / [run_ltx2_3_i2av.sh](ltx2_3/run_ltx2_3_i2av.sh) |
| LTX-2.3 音频、关键帧及动作控制 | [ltx2_3](ltx2_3) |
| LTX-2.5 蒸馏模型，八卡 | [run_ltx2_5_t2av_distilled.sh](ltx2_5/run_ltx2_5_t2av_distilled.sh) / [run_ltx2_5_i2av_distilled.sh](ltx2_5/run_ltx2_5_i2av_distilled.sh) |

下载对应的 [LTX-2](https://huggingface.co/Lightricks/LTX-2)、[LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) 或 [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) 权重。在所选 Shell 脚本中设置 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`，并修改提示词和输入路径。

- **LTX-2 / LTX-2.3**：在所选 JSON 中将 `dit_original_ckpt` 指向对应权重文件，将 `gemma_original_ckpt` 指向完整的 [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) HF 目录，包含权重、tokenizer 和 processor。未配置该字段时，加载器会在 `model_path` 下查找 Gemma 资产。
- **LTX-2.5**：在所选 JSON 中设置各组件权重路径，包括 Transformer、文本编码器、视频 / 音频 VAE、时长预测器和上采样器。启动脚本使用 [ltx2_5_distilled_8gpu.json](../../configs/ltx2/ltx2_5_distilled_8gpu.json)。

启动前先准备模型架构配置：LTX-2 可使用官方的 `transformer/config.json`；没有独立配置的一体式 LTX-2.3 权重，应从对应 checkpoint metadata 一次性导出 Transformer 及音视频启动参数，保存为 `model_path` 下的 `config.json`。模型目录须与 checkpoint 版本对应，不要将 LTX-2 配置直接用于 LTX-2.3。现有目录加载器读取配置，显式部署参数最后生效。Gemma 仍是独立资产，通过 `gemma_original_ckpt` 指定。

LTX-2 / LTX-2.3 的 Gemma 3 可设置 `gemma_cpu_offload=true`、`gemma_offload_granularity="block"`，词表保留在 CPU，只搬运选中的 token embedding，并分阶段搬运 decoder 层、final norm、projection 和音视频 connectors。该开关独立于 DiT 的 `offload_granularity`。同步搬运保留原 CPU 权重存储，不 pin 或预取权重；默认 `"model"` 保持整模块 offload。同一 encoder 只支持串行调用，直接 API 多线程共享 encoder 的并发调用尚不支持。LTX-2.5 设置 Gemma block offload 时会在加载前报错。

## 运行

使用 LTX-2 基础示例时，先修改 [ltx2.json](../../configs/ltx2/ltx2.json) 中的 `dit_original_ckpt`，然后在仓库根目录运行：

```bash
bash scripts/ltx2/run_ltx2_t2av.sh
```

基础文生 / 图生脚本将结果保存到 `save_results/output_lightx2v_ltx2_t2av.mp4` 或 `save_results/output_lightx2v_ltx2_i2av.mp4`，可通过 `--save_result_path` 修改。分辨率、帧数和采样参数在所选 JSON 中，配置目录为 [configs/ltx2](../../configs/ltx2)。

LTX-2 多卡推理可选择以 `_cfg_parallel.sh`、`_ulysses.sh` 或 `_tp.sh` 结尾的脚本，显卡数量需与 `--nproc_per_node` 和 JSON 中的并行设置一致。

## Python API

| LTX-2 权重 | 文生音视频 | 图生音视频 |
| --- | --- | --- |
| 基础模型 | [ltx2_t2av.py](inference/ltx2_t2av.py) | [ltx2_i2av.py](inference/ltx2_i2av.py) |
| FP8 蒸馏模型 | [ltx2_t2av_distilled_fp8.py](inference/ltx2_t2av_distilled_fp8.py) | [ltx2_i2av_distilled_fp8.py](inference/ltx2_i2av_distilled_fp8.py) |

修改所选示例中的模型目录、权重文件、输入和输出路径，然后运行：

```bash
python scripts/ltx2/inference/ltx2_t2av.py
```

通用用法见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。
