# MiniMax-H3

[English](README.md) | [简体中文](README_zh.md)

运行前，设置所选脚本中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。

## 模型权重与任务

从官方 Hugging Face 仓库 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 下载 Diffusers 格式权重，`model_path` 指向本地模型目录。

| 任务 | 请求输入 | Transformer |
| --- | --- | --- |
| `t2av` | `prompt` | `transformer` |
| `i2av` | `prompt`、`image_path`（首帧） | `transformer` |
| `l2av` | `prompt`、`last_frame_path` | `transformer` |
| `fl2av` | `prompt`、`image_path`、`last_frame_path` | `transformer` |
| `ref2av` | `prompt`、参考图片和/或视频，以及可选的参考音频 | `transformer_ref` |

启动时须指定 `--model-variant`：`fl2av` 支持表中前四种任务，`ref2av` 支持参考任务。请求通过 `task` 选择任务，基础分支切换任务无需重新加载。两个分支可共用推理 JSON，LoRA 和量化权重须与分支匹配。

## AdaLN 缓存

AdaLN 缓存会预先保存推理中可复用的计算结果，减少重复计算。**示例配置已启用缓存，首次推理或启动服务前需先生成；已有匹配缓存可跳过。**

1. **确认缓存配置。** 推理 JSON 中保留以下字段（示例配置已包含）：

   ```json
   {
     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"
   }
   ```

2. **编辑[缓存生成脚本](../../tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh)。** 按下表设置参数：

   | 参数 | 如何设置 |
   | --- | --- |
   | `lightx2v_path`、`model_path` | 与推理脚本使用相同的仓库路径和模型路径。 |
   | `CUDA_VISIBLE_DEVICES` | 选择用于生成缓存的 GPU，例如 `0`；生成缓存只需单卡。 |
   | `--config_json` | 指向推理使用的 JSON。 |
   | `--model-variant` | `t2av`、`i2av`、`l2av`、`fl2av` 均填 `fl2av`；参考任务填 `ref2av`。 |

3. **生成缓存。** 在仓库根目录运行：

   ```bash
   bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
   ```

   日志出现 `AdaLN cache saved to` 即表示完成，随后可运行下方的离线推理或启动服务。缓存保存在 `adaln_cache_dir` 下的子目录中，推理会自动读取，无需手动指定缓存文件。

> **缓存复用**：更换基座分支／权重、时间步序列，或更改影响 AdaLN／末层的 LoRA（含 strength、alpha、模式、`LORA_DTYPE`）时，需重新生成缓存。导出与推理使用相同设置；不修改调制层的 LoRA 可共用基座 cache。

## 离线推理

五个任务脚本共用 `configs/minimax_h3/minimax_h3.json`：单卡 BF16，默认输出 124 帧、`[高度, 宽度] = [544, 960]`。

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
bash scripts/minimax_h3/run_minimax_h3_i2av.sh
bash scripts/minimax_h3/run_minimax_h3_l2av.sh
bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

基础配置使用 SageAttention2 和 SGL；注意力、量化和并行配置须与设备及已安装算子匹配。

通过 `--config_json` 选择配置，下表列出 `configs/minimax_h3/` 中的代表配置。单卡使用对应任务脚本；SP、TP 和混合并行参考 `run_minimax_h3_t2av_parallel.sh`，其他任务同步设置分支、任务及输入。

| `configs/minimax_h3/` 下的配置 | 运行方式 |
| --- | --- |
| `minimax_h3.json` | 单 GPU BF16 |
| `minimax_h3_compile.json` | 编译并在启动时预热 |
| `minimax_h3_block_offload.json` | 单 GPU BF16，分块卸载 |
| `minimax_h3_sp.json` | SP4 |
| `minimax_h3_tp.json` | TP2 |
| `minimax_h3_tp_sp.json` | TP2 × SP2 |
| `fp8/minimax_h3.json` | 单 GPU DiT FP8 |
| `distill/4steps/minimax_h3_bf16_4step.json` | 单 GPU BF16，四步 LoRA |
| `distill/4steps/minimax_h3_fp8_4step.json` | SP4，FP8 + 四步 LoRA |
| `distill/4steps/minimax_h3_ref2av_4step.json` | SP8，参考任务四步 LoRA |

多卡脚本默认 SP4，切换配置时同步设置 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`，与配置中的并行规模保持一致。

编译使用 `minimax_h3_compile.json`，启用 `use_compile`、`warmup`，并设置 `rope_type: "torch_real_rope"`；服务也可选用。

在推理 JSON 中添加 `video_codec_options` 设置 MP4 编码选项，例如：

```json
{
  "video_codec_options": {
    "preset": "ultrafast",
    "crf": "18"
  }
}
```

## Python API

[四步 DMD T2AV 示例](inference/distill/minimax_h3_t2av_dmd.py) 使用
`configs/minimax_h3/distill/4steps/minimax_h3_bf16_4step.json`。安装后，修改脚本中的
`MODEL_PATH`、`OUTPUT_PATH` 和该 JSON 中的本地 LoRA 路径，再从仓库根目录运行：

```bash
python scripts/minimax_h3/inference/distill/minimax_h3_t2av_dmd.py
```

生成器创建与配置优先级见 [Python API 指南](../../docs/ZH_CN/source/getting_started/python_api.md)。

## 少步草片与 RGB 视频重生成

低成本抽卡的草片筛选、高清重生成与启动脚本，参见 [MiniMax-H3 Roll 文档](../minimax_h3_roll/README_zh.md)。

## LoRA

模式写在推理 JSON 的 `lora_mode` 中，不设置时默认 `merge`。`distill/` 下所有配置均使用 `consumer`：

```json
{
  "lora_mode": "consumer"
}
```

精度仍使用环境变量，**导出 cache 和推理时保持一致**：

```bash
export LORA_DTYPE=BF16  # BF16、FP16、FP32；不设置时跟随 DTYPE
```

| DiT 权重 | `lora_mode` | 行为 |
| --- | --- | --- |
| 原始 BF16 | `merge`（默认）、`separate`、`consumer` | 分别表示加载时合并、独立分支计算、将分支相加融合进 H3 的激活/残差算子。 |
| 量化 FP8 或 INT8，包括 ConvRot | `separate`、`consumer` | 保留现有量化 GEMM 和浮点 LoRA 分支，不支持将 LoRA 合并进量化 DiT 权重。 |

`consumer` 目前只支持 CUDA 上的 `minimax_h3`，支持 eager 和 `use_compile: true`，不支持 TP
归约或 GEMM 校准；可以使用序列并行和 model/block offload。不支持的组合直接
报错，不切换模式。融合算子保留中间 dtype 舍入，也支持 BF16 投影搭配 FP32
gate/residual。具体后端的数值一致性和速度需要实测，不承诺统一加速幅度。
编译可能改变 block 其他计算的舍入方式，因此不保证 eager/compile 逐位一致。
量化编译还要求所选 GEMM/attention 后端支持 tracing；consumer 不替换这些后端。

`separate` 和 `consumer` 都保留请求间切换 LoRA 的能力，不修改基座权重。
运行时切换能力与计算模式是两个概念。

启动前设置 `export LORA_DTYPE=FP32`（也可选 `BF16`／`FP16`）；未设置时跟随
`DTYPE`，不跟随 `SENSITIVE_LAYER_DTYPE`。它统一控制三种模式的 LoRA 权重和计算精度。
merge 最终转回基座权重 dtype；separate／consumer 在两路**相加后**转回基座投影输出
的 dtype，不改变基座精度和量化 GEMM。FP32 merge 仍有最终的权重舍入，不等价于独立分支。
旧 JSON 字段 `lora_merge_force_fp32` 已移除，不需要在 JSON 中增加 `lora_dtype`。

基础任务四步 v1.0 LoRA 示例：在 `minimax_h3.json` 中添加或修改以下字段，`path` 填本地 LoRA 路径：

```json
{
  "lora_mode": "consumer",
  "infer_steps": 4,
  "video_flow_shift": 6.0,
  "audio_flow_shift": 3.0,
  "scheduler_type": "euler",
  "lora_configs": [
    {
      "path": "/path/to/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
      "strength": 1.0,
      "alpha": 128
    }
  ]
}
```

支持 H3 Turbo 的原生（非 ComfyUI）LoRA。alpha 优先级：逐层 alpha tensor →
文件 metadata 的 alpha → `lora_configs[].alpha`；内嵌 alpha 存在时忽略配置值。
日志打印来源、数值和层数。rank 从权重形状读取，AdaLN 和 `norm_out.linear` 不用另配。
`configs/minimax_h3/distill/8steps/` 下的 BF16、FP8 和 INT8 ConvRot 配置均使用
`minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors`，8 步 Euler、shift 12/3、strength=1。
将 LoRA 路径改为本地文件；alpha 直接读取权重中的逐层值，不在 JSON 中统一覆盖。
这些配置已设为 `lora_mode: "consumer"`，也可改成 `separate`；量化基座不支持 merge。
LoRA 本身仍是 BF16 权重，FP8／INT8 只对应基座。首次运行前按下文导出匹配的 AdaLN cache，
导出和推理使用相同的配置（含 `lora_mode`）及 `LORA_DTYPE` 环境变量。
最早的 FL2V v0.1 没有内嵌 alpha，需要配置指定。仓库中的完整 FP8 transformer 不是 LoRA。

### 调度器与带 LoRA 的 AdaLN cache

统一字段为 `scheduler_type`：支持 `euler`、`lcm`，基座默认保留 `reference_blend`。
不兼容旧字段名。LCM 对齐 Pika H3 的“预测干净样本再加噪”更新，不是少跑几步的 Euler。
匹配的六步 LCM 权重使用：

```json
{
  "scheduler_type": "lcm",
  "infer_steps": 6,
  "distilled_sigma_values": [1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0],
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0
}
```

这里是**未做 shift** 的基础 sigma，包含首尾端点，必须有 `infer_steps + 1` 个值，
从 1 严格递减到 0；视频和音频各自做一次 shift。不要随意将 Euler 权重改用 LCM。

使用**同一份推理 JSON（含 `lora_mode`）和相同的 `LORA_DTYPE` 环境变量**（含 LoRA、strength 和调度器配置）导出 cache：

```bash
PYTHONPATH=. python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path /path/to/MiniMax-H3 --model-variant fl2av \
  --config_json configs/minimax_h3/distill/8steps/minimax_h3_bf16_8step.json
```

Ref 基座改成 `--model-variant ref2av`。缓存标识包含基座、时间步、相关 LoRA、strength
及合并精度；`separate`/`consumer` 的调制计算相同，可共用 cache，`merge` 单独导出。
原有 H3 cache 需重新生成一次。已写进 cache 的 AdaLN／末层 LoRA 不会在线重复应用。
运行时切换或移除 LoRA 前，先导出对应 cache；缺少 cache 时在更改当前 adapter 前报错。
没有调制层更新的旧版 LoRA 可以共用基座 cache。

### Pika-H3-Flex 六步 LoRA

[官方实现](https://github.com/Pika-Labs/Pika-H3-Flex)发布了独立的 FL2VA 和 Ref2VA adapter，
两者均为 rank/alpha=128/128、strength=1，使用上面的六步 LCM 时间表及 shift 12/3。
发布权重没有内嵌 alpha，因此以下配置显式设置 `alpha: 128`，无需转换格式：

- [FL2AV 配置](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json)：使用 `transformer/`，支持 T2AV、I2AV、L2AV、FL2AV。
- [Ref2AV 配置](../../configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json)：使用 `transformer_ref/`，不要混用 adapter。

下载后将配置中的 `lora_configs[0].path` 改为实际文件路径。基座仍使用 MiniMax-H3：

```bash
hf download Pika-Labs/Pika-H3-Flex fl2va/adapter.safetensors ref2va/adapter.safetensors --local-dir /path/to/pika-h3-flex

export DTYPE=BF16
export LORA_DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export PYTHONPATH=.
model_path=/path/to/MiniMax-H3
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_fl2av_6step.json

python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant fl2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant fl2av --task t2av \
  --model_path "$model_path" --config_json "$config_path" \
  --prompt 'integrated_multimodal_description: A fox walks through a snowy forest. overall_soundscape: Wind and crunching snow. non_diegetic_music: N/A' \
  --seed 42 --save_result_path output_pika_t2av.mp4
```

I2AV 增加 `--image_path`；L2AV 增加 `--last_frame_path`；FL2AV 同时传两张图，并对应修改 `--task`。
Ref2AV 使用独立配置和 cache：

```bash
config_path=configs/minimax_h3/distill/6steps/minimax_h3_pika_ref2av_6step.json
python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
  --model_path "$model_path" --model-variant ref2av --config_json "$config_path"
python -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant ref2av --task ref2av \
  --model_path "$model_path" --config_json "$config_path" \
  --image_path /path/to/reference.png \
  --prompt 'Animate the subject in <Picture 1>, preserving its appearance and setting. Natural motion and ambient sound, no music.' \
  --seed 42 --save_result_path output_pika_ref2av.mp4
```

示例默认 768×1344、124 帧、单卡 block offload，未量化；官方的 LoRA 分支使用 BF16，
因此显式设置 `LORA_DTYPE=BF16`。也可设置 `lora_mode: "separate"`；merge 的权重舍入不等价于独立分支。
Ref 示例统一使用 `reference_image_resize_mode: "match"`，按输出画布面积约束参考图、保留宽高比且不放大。
这是本项目选用的预处理，与 Pika 官方 768p 档位的参考图短边 2048 缩放不等价。
LightX2V 与官方使用的 attention/kernel 仍可能产生数值差异。

## 服务与 POST 请求

基础服务设置 `--model-variant fl2av`，启动时不传 `--task`。按卡数选择入口，模型加载后在另一终端提交请求。

单卡：

```bash
bash scripts/minimax_h3/server/start_server.sh
```

四卡 SP4 使用 `minimax_h3_sp.json`：

```bash
bash scripts/minimax_h3/server/start_server_parallel.sh
```

单卡和多卡服务共用以下 POST 客户端：

```bash
python scripts/minimax_h3/server/post_t2av.py
python scripts/minimax_h3/server/post_i2av.py
python scripts/minimax_h3/server/post_l2av.py
python scripts/minimax_h3/server/post_fl2av.py
```

基础请求须指定 `task`，输入见任务表。图片示例上传客户端本地文件的 Base64，也可传服务端可访问的绝对路径。

参考任务服务使用 `--model-variant ref2av`：

```bash
bash scripts/minimax_h3/server/start_server_ref2av.sh
```

设置客户端的参考图片路径，在另一终端发送请求：

```bash
python scripts/minimax_h3/server/post_ref2av.py
```

## 请求参数与启动配置

- **启动与请求**：启动时设置模型路径、分支、权重／LoRA、算子、卸载、并行、编译和预热；通过 Python、CLI 或 POST 传入 `prompt`、媒体路径、`seed` 和输出路径。
- **输出规格**：JSON 提供 `num_frames`、`size` 的默认值，请求可覆盖。`size` 为 `[高度, 宽度]`，宽高须为 32 的倍数；帧数向上对齐到 `17*n+5`，对齐后须在 124–362 之间，如 125 → 141。输出为 MP4，包含 24 FPS 视频和 32 kHz 立体声音频。
- **随机种子**：`seed` 省略或传 `null` 时使用 42；可指定包含 0 在内的非负整数。
- **结果保存**：CLI 不指定输出路径时跳过保存。服务的 `save_result_path` 建议使用 `./minimax_h3_t2av.mp4` 等相对路径，按服务输出目录解析；省略或传 `null` 时跳过保存，已保存的结果可下载。
