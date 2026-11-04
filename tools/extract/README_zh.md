# LoRA 提取工具

[English](README.md)

从 base 与完整 target 模型之间提取 LoRA，使 **base + LoRA 尽量逼近 target**。以下命令均在仓库根目录运行。

## 导出学生模型权重

`export_student.py` 在 CPU 上导出 `lightx2v_train` / `lightx2v_rl` checkpoint 中的完整 student/actor 权重，支持 DCP、RL FSDP、PyTorch state dict 和 safetensors。

```bash
python tools/extract/export_student.py \
  --checkpoint /path/to/training_checkpoint \
  --base-model lightx2v_train/models/Qwen-Image-2.1 \
  --output outputs/student_transformer
```

- `--checkpoint`：要导出的训练 checkpoint 或完整模型权重。
- `--base-model`：对应的 base 模型，用于获取 transformer 配置及校验权重名称、形状。
- `--output`：新建的 transformer 输出目录，不能已存在。包含 safetensors 权重、索引、`config.json` 和导出记录。
- 可选参数：`--dtype bf16` 转换导出精度，默认保留原始精度；`--max-shard-size-gb` 设置分片大小，默认 2 GB。仅当 checkpoint 有意省略冻结参数时，使用 `--fill-missing-from-base` 从 base 补齐。

产物是完整 transformer 权重，不是 LoRA，也不包含其他推理组件。已有所需完整权重时可跳过导出。

## 使用 SVD 提取 LoRA

`lora_extractor.py` 对 target 与 base 的权重差进行 FP32 SVD 分解，无需校准数据或训练。

```bash
python tools/extract/lora_extractor.py \
  --config tools/extract/configs/qwen_image_21.json
```

运行前修改配置中的关键字段：

- `source_model`、`target_model`、`output`：分别为原始 base、希望逼近的完整 target 权重、输出 LoRA 的 `.safetensors` 路径。
- `rank`、`alpha`、`dtype`：LoRA 的秩、缩放参数和导出精度。Qwen-Image 2.1 示例使用 rank 64、alpha 8。
- `layers`：需要提取的 Linear 层名称或通配符。

使用上一步导出的权重时，在命令后追加 `--target-model outputs/student_transformer`。Wan 示例见 [configs/wan.json](configs/wan.json)；`lora_merger.py` 的合并用法见[部署指南](../../docs/ZH_CN/source/deploy_guides/lora_deploy.md)。

## 使用 TFMQ 重建 LoRA

`lora_reconstructor.py` 的用法、校准数据和配置说明见 [RECONSTRUCTION_zh.md](RECONSTRUCTION_zh.md) / [English](RECONSTRUCTION.md)。

## Qwen-Image 2.1 RL 对比结果

使用相同的 base 和 target，rank 64、alpha 8；独立验证集包含 512 条 T2I、256 条单 ref、256 条多 ref 请求，每条请求采集 teacher 的四个时间步。MSE 比较模型最终的目标图像 velocity 输出，对每个请求、每个时间步等权平均。

| 方法 | 总体 MSE |
|---|---:|
| SVD | 0.0097887147 |
| **TFMQ** | **0.0018547516** |
| 在线联合微调 | 0.0038252176 |

本次实验中，TFMQ 的 MSE 相比 SVD 降低 **81.1%**，相比在线联合微调降低 **51.5%**。这些数值衡量输出拟合误差，不是图像质量评分。
