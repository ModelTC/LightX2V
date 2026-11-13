# LoRA 模型部署与相关工具

`lora_mode` 只有三个取值：`merge`（默认）、`separate`、`consumer`。`separate` 独立计算基座和适配器分支，支持的模型可在运行时切换适配器。`consumer` 将两路相加融合到 MiniMax-H3 的激活和残差算子；其他模型使用该模式会报错。H3 consumer 支持 CUDA eager 和 `use_compile: true`，可以使用序列并行，不支持张量并行和 GEMM 校准。不会自动切换为其他模式。

LoRA (Low-Rank Adaptation) 是一种高效的模型微调技术，通过低秩矩阵分解显著减少可训练参数数量。LightX2V 全面支持 LoRA 技术，包括 LoRA 推理、LoRA 提取和 LoRA 合并等功能。

## 🎯 LoRA 技术特性

- **灵活部署**：支持动态加载和移除 LoRA 权重
- **多种格式**：支持多种 LoRA 权重格式和命名约定
- **工具完善**：提供完整的 LoRA 提取、合并工具链

## 📜 LoRA 推理部署

### 配置文件方式

在配置文件中指定 LoRA 路径：

```json
{
  "lora_mode": "merge",
  "lora_configs": [
    {
      "path": "/path/to/your/lora.safetensors",
      "strength": 1.0
    }
  ]
}
```

**配置参数说明：**

- `lora_configs[].path`: 每个 LoRA 条目的权重文件路径；静态合并支持多个条目
- `lora_configs[].strength`: LoRA 强度系数，控制 LoRA 对原模型的影响程度

执行模式通过 **JSON 中的 `lora_mode`** 设置：`merge`（默认）、`separate` 或 `consumer`。它是推理、服务和 cache 导出共用的启动配置；精度仍由环境变量 `LORA_DTYPE` 控制。

### LoRA 精度

使用环境变量 `LORA_DTYPE=BF16|FP16|FP32`，三种模式共用。不设置时跟随 `DTYPE`（通常是 BF16），不跟随 `SENSITIVE_LAYER_DTYPE`。

```bash
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
export LORA_DTYPE=FP32
# 然后正常启动推理或服务；torchrun 各 rank 继承这些设置。
```

| 模式 | `LORA_DTYPE` 控制什么 | 哪些 dtype 不变 |
|---|---|---|
| `merge` | A/B 权重、低秩增量计算及与基座相加的精度；相加不降低基座精度 | 合并后保存的基座权重 |
| `separate` | A/B 权重及两次 LoRA GEMM；与基座输出相加后再转回 | 基座 GEMM、量化方案和该层输出 |
| `consumer` | 与 `separate` 相同的分支精度和输出舍入边界，只把相加融合进后续算子 | 基座 GEMM 和后续激活／残差精度 |

FP32 LoRA **不会把整个模型转成 FP32**。FP32 merge 最后仍会转回基座 dtype，因此不等价于保留独立分支。FP32 分支增加显存与计算开销，不保证更快或画质更好。merge 多份 LoRA 时，按配置顺序逐份相加并转回基座 dtype。

旧 JSON 字段 `lora_merge_force_fp32` 已移除；原来设为 `true` 的场景改用 `export LORA_DTYPE=FP32`，JSON 中不新增 `lora_dtype`。环境变量需在启动前设置，修改后重启进程。独立权重转换工具仍使用各自的 dtype 参数。

### 命令行方式

先将上面的 `lora_configs` 字段合并到完整的 Wan 配置中，再通过 `--config_json` 启动。当前 CLI 不提供单独的 LoRA 路径或强度选项：

```bash
python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task t2v \
  --model_path /path/to/model \
  --config_json /path/to/config.json \
  --prompt "Your prompt here"
```

### 多LoRA配置

要使用多个具有不同强度的LoRA，请在配置JSON文件中指定：

```json
{
  "lora_mode": "merge",
  "lora_configs": [
    {
      "path": "/path/to/first_lora.safetensors",
      "strength": 0.8
    },
    {
      "path": "/path/to/second_lora.safetensors",
      "strength": 0.5
    }
  ]
}
```

### Wan 的组合限制与 Wan2.2 命名

本页 JSON 是配置片段，需合并到对应模型的完整配置中。Wan 默认采用静态合并，不支持同时启用 `dit_quantized` 或 `lazy_load`。启用 `lora_mode: "separate"` 时，每个子模型仅使用匹配列表中的第一个 LoRA 条目；不要把多条目示例当成动态叠加。

Wan2.2 MoE 需要通过 `name` 将 LoRA 分配到高噪声和低噪声模型：

```json
{
    "lora_configs": [
        {
            "name": "high_noise_model",
            "path": "/path/to/high_noise_lora.safetensors",
            "strength": 1.0
        },
        {
            "name": "low_noise_model",
            "path": "/path/to/low_noise_lora.safetensors",
            "strength": 1.0
        }
    ]
}
```

### MiniMax-H3 原生 LoRA

H3 Turbo 发布的非 ComfyUI 权重使用 `*.lora_A.default.weight` / `*.lora_B.default.weight`。alpha 优先级为 **逐层 alpha tensor → 文件 metadata alpha → `lora_configs[].alpha`**；有内嵌 alpha 的层忽略配置值。日志打印来源、数值和层数。rank 从权重形状读取，AdaLN 和 `norm_out.linear` 不用额外设置；没有内嵌 alpha 的文件才需要配置 alpha。

H3 的 `separate`／`consumer` 支持单个激活 LoRA，可搭配 FP8／INT8 基座 GEMM，不向量化权重合并。接入 consumer 不代表每一种量化后端都已实测；编译兼容性还取决于所选 GEMM／attention 后端。

使用持久化 AdaLN cache 时，导出和推理必须使用相同的 LoRA、strength、alpha、调度器、**`lora_mode`** 及 **`LORA_DTYPE`**。separate 和 consumer 的调制计算相同、共用 cache；merge 单独区分。缓存标识包含解析后的 alpha 和 LoRA dtype，缓存中的调制层更新不会在线重复应用。切换／移除 LoRA 前先生成对应的新／基座 cache。命令见 [H3 推理与缓存说明](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README_zh.md)。

### 支持的 LoRA 格式

LightX2V 支持多种 LoRA 权重命名约定：

| 格式类型 | 权重命名 | 说明 |
|----------|----------|------|
| **标准 LoRA** | `lora_A.weight`, `lora_B.weight` | 标准的 LoRA 矩阵分解格式 |
| **Down/Up 格式** | `lora_down.weight`, `lora_up.weight` | 另一种常见的命名约定 |
| **差值格式** | `diff` | `weight` 权重差值 |
| **偏置差值** | `diff_b` | `bias` 权重差值 |
| **调制差值** | `diff_m` | `modulation` 权重差值 |

### 推理脚本示例

**步数蒸馏 LoRA 推理：**

```bash
# T2V LoRA 推理
bash scripts/wan/inference/distill/run_wan_t2v_distill_lora.sh

# I2V LoRA 推理
bash scripts/wan/inference/distill/run_wan_i2v_distill_lora.sh
```

### API 服务中使用 LoRA

在 API 服务中通过 [config 文件](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json) 指定，对 [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh) 中的启动命令进行修改：

```bash
python -m lightx2v.server \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json \
  --port 8000
```

## 🔧 LoRA 提取工具

使用 `tools/extract/lora_extractor.py`，按 JSON/YAML 配置对选定 Linear 的权重差做 FP32 SVD。运行以下命令前，先修改配置中的模型与输出路径。相对路径以运行命令的工作目录为准，支持 `~` 和环境变量。

需要基于校准数据优化 Qwen-Image 2.1 LoRA 时，使用独立入口 `tools/extract/lora_reconstructor.py`。支持 Q-Diffusion 风格的逐块重建，以及 TFMQ 风格的时间通路联合重建后再逐块重建；支持单卡和 `torchrun` DDP。示例配置使用 rank=64、alpha=8，并单独保存 SVD 初始化 LoRA。校准数据采集、算法细节和运行命令见[重建说明](https://github.com/ModelTC/LightX2V/blob/main/tools/extract/RECONSTRUCTION_zh.md)。

### 基本用法

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/wan.json

python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json
```

### 参数说明

| 配置字段 | 默认值 | 说明 |
|----------|--------|------|
| `source_model` | 必填 | 基础模型本地路径 |
| `target_model` | 必填 | 微调模型、导出学生权重或原始学生 checkpoint 路径 |
| `output` | 必填 | 输出 `.safetensors` 路径 |
| `layers` | 必填 | 不含 `.weight` 的模块名，支持 `*` 等 glob 通配符 |
| `rank` | `32` | LoRA rank，超过矩阵维度时逐层截断 |
| `alpha` | 与 rank 相同 | LoRA alpha；因子补偿 `alpha / 实际rank`，不会改变重建增量 |
| `dtype` | `bf16` | 因子 dtype：`bf16` / `fp16` / `fp32` 或完整名称；alpha 标量固定 FP32 |
| `device` | `auto` | 优先 CUDA，否则 CPU；也可指定 `cuda:0` 等 |
| `key_prefix` | `auto` | `transformer` / `diffusion_model`；自动根据模型键名选择 |

路径自动识别完整模型目录中的 `transformer/`、独立 Transformer 目录、safetensors 文件/分片、PyTorch state dict，以及下文的 train/RL checkpoint。无需 `--source-type` 或 `--target-type`。输入为本地文件，不自动下载模型。

Wan 配置覆盖 T2V 的 Attention、FFN、文本/时间嵌入及输出 Linear，兼容 native Wan 和 Diffusers Wan 命名；I2V 特有层可自行加入 `layers`。Qwen 配置覆盖 32 个 block 的 224 个 Linear 和 8 个全局 Linear，不输出归一化参数差分。Qwen 合并的 `gate_up` 会按 `gate_layer` / `proj` 分开提取，使用 `transformer.` 前缀兼容 ComfyUI。

所有选中的矩阵均做低秩分解，包括小矩阵；不再提供 `--diff-only`，也不自动输出 bias、norm 或 modulation 参数的直接差分。完全相同的矩阵会跳过。层名无匹配、目标缺层或形状不一致时直接报错，不保存不完整 LoRA。

### 高级用法示例

**提取高秩 LoRA：**

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json \
  --target-model /path/to/exported/student \
  --output /path/to/high_rank_lora.safetensors \
  --rank 64 \
  --alpha 64 \
  --output-dtype fp16
```

命令行参数覆盖文件中的同名字段；只覆盖 rank 时，文件中显式设置的 alpha 保持不变。

### 导出 train / RL 学生模型

`export_student.py` 在 CPU 上导出全参数学生模型，输出独立 Transformer 目录及分片索引，不复制文本编码器、VAE 或采样配置。导出结果可直接作为提取器的 `target_model`。

```bash
python tools/extract/export_student.py \
  --checkpoint /path/to/train/checkpoint-000001000 \
  --base-model /path/to/base/model \
  --output /path/to/exported/student

python tools/extract/export_student.py \
  --checkpoint /path/to/rl/global_step_100 \
  --base-model /path/to/base/model \
  --output /path/to/exported/rl_student \
  --dtype bf16
```

支持 train 的 `model_state.pt`、`student_consolidated/`、`dist_state/student` 中的 `model`，以及 `dist_state` 中的 `student_model` 或 `model`；仅读取学生参数，不加载 optimizer/fake。RL 支持 `actor/` 下的一维 FSDP2 DTensor 分片和完整副本；运行时使用与训练 checkpoint 兼容的现代 PyTorch。LoRA 训练 checkpoint 需要先合并成全参数权重。

默认保留保存精度，`--max-shard-size-gb` 默认 2；单个张量不拆分，可能超过此上限。`--base-model` 用于校验参数名称/形状并提供 `config.json`，源目录已有配置时可以省略。DCP 若有刻意不保存的冻结参数，可显式加 `--fill-missing-from-base` 补齐；默认缺层报错。导出目录必须不存在，失败不会留下半成品。safetensors/DCP 逐层读取，现代 PyTorch 下的 RL 分片通过 CPU mmap 读取，无需启动分布式训练进程。

## 🔀 LoRA 合并工具

使用 `tools/extract/lora_merger.py` 将 LoRA 权重合并到基础模型中，以进行后续量化等操作。

合并工具支持 `diffusion_model.` 和 `transformer.` 前缀，并应用文件内每层的 `alpha / rank`。命令行 `--alpha` 仍表示整体合并强度，默认 1；旧 LoRA 未保存 alpha 时保持单位缩放。

### 基本用法

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged/model.safetensors \
  --alpha 1.0
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--source-model` | str | ✅ | 无 | 基础模型路径 |
| `--lora-model` | str | ✅ | 无 | LoRA 权重路径 |
| `--output` | str | ✅ | 无 | 输出合并模型路径 |
| `--source-type` | str | ❌ | `safetensors` | 基础模型格式 |
| `--lora-type` | str | ❌ | `safetensors` | LoRA 权重格式 |
| `--output-format` | str | ❌ | `safetensors` | 输出格式 |
| `--alpha` | float | ❌ | `1.0` | LoRA 合并强度 |
| `--output-dtype` | str | ❌ | `bf16` | 输出数据类型 |

### 高级用法示例

**部分强度合并：**

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model \
  --lora-model /path/to/lora.safetensors \
  --output /path/to/merged_model.safetensors \
  --alpha 0.7 \
  --output-dtype fp32
```

**多格式支持：**

```bash
python tools/extract/lora_merger.py \
  --source-model /path/to/base/model.pt \
  --source-type pytorch \
  --lora-model /path/to/lora.safetensors \
  --lora-type safetensors \
  --output /path/to/merged_model.safetensors \
  --output-format safetensors \
  --alpha 1.0
```
