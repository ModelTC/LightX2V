# 注意力机制

## LightX2V支持的注意力机制

| 名称               | 类型名称         | GitHub 链接 |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sol-Attn           | `sol_attn`       | [Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends) |
| Sparge Attention   | `sparge_attn`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

---

## 配置示例

注意力机制的config文件在[这里](https://github.com/ModelTC/lightx2v/tree/main/configs/attentions)

通过指定--config_json到具体的config文件，即可以测试不同的注意力机制

比如对于radial_attn，配置如下：

```json
{
  "self_attn_1_type": "radial_attn",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
}
```

如需更换为其他类型，只需将对应值替换为上述表格中的类型名称即可。

Wan 的 `cross_attn_2_type` 用于图像交叉注意力。Wan2.1 的 I2V/FLF2V 和 Wan2.2 的 Animate/Animate2 配置需要保留；T2V/VACE，以及关闭图像编码器的 Wan2.2 MoE、TI2V 和 S2V 配置无需此字段。

tips: radial_attn因为稀疏算法原理的限制只能用在self attention

如需进一步定制注意力机制的行为，请参考各注意力库的官方文档或实现代码。

### 在 Wan2.1 上使用 Sol-Attn

Sol-Attn 是仅支持前向计算的非因果 self-attention 后端，要求连续的 BF16 张量和 128 维 attention head。RTX 5090（SM120）可使用仓库提供的固定版本环境：

```bash
docker build -f dockerfiles/Dockerfile_5090_cu130_sol_attn -t lightx2v-sol-attn:sm120 .
```

按[快速入门](../getting_started/quickstart.md)启动容器，将镜像替换为 `lightx2v-sol-attn:sm120`。H100/H200（SM90）按 [Sol-Attn 上游说明](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends)安装后端。

运行前，修改 `scripts/wan/inference/run_wan_i2v_sol_attn.sh` 中的 `lightx2v_path`、`model_path` 和 `CUDA_VISIBLE_DEVICES`。再编辑 `configs/attentions/wan_i2v_sol_attn.json`，按 GPU 选择后端：

| GPU | `sol_attn_setting.dense_backend` | `cross_attn_1_type` / `cross_attn_2_type` |
| --- | --- | --- |
| H100/H200 | `flash_attn3`（默认） | `flash_attn3`（默认） |
| RTX 5090 | 改为 `sage_attn2` 或 `torch_sdpa` | 改为 `flash_attn2` 等已安装且支持的实现 |

```bash
bash scripts/wan/inference/run_wan_i2v_sol_attn.sh
```

首次调用会编译对应 shape 的 kernel，性能计时应排除该次调用。

#### 调整与实现说明

- 默认 40 步配置中，前 8 步和第 0 个 Transformer 层使用 dense 后端，其余使用 Sol-Attn。通过 `dense_steps` 和 `dense_layers` 调整；后者支持 `[0, 1]` 或 `"0-1"`。
- 示例使用 `tau=0.8`；减小该值可采用更保守的稀疏率，增大则可能降低画质。`sample_shift=3` 与标准 Wan2.1 I2V 配置一致。
- Morton3D 重排在整个 Transformer block stack 外执行一次，并同步 RoPE；strict 模式下，安装无效会直接报错。
- SM120 使用 `kv_splits=1`，`"auto"` 也会解析为 1。

### 在 MiniMax-H3 上使用 Sol-Attn

MiniMax-H3 的示例沿用 15 秒、768p、block CPU offload 配置：

运行前，将 `scripts/minimax_h3/run_minimax_h3_t2av.sh` 中的 `lightx2v_path` 和 `model_path` 改为本地目录，并将其 `--config_json` 参数改为 `"${lightx2v_path}/configs/minimax_h3/minimax_h3_sol_block_offload.json"`。

```bash
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
```

对应配置为 `configs/minimax_h3/minimax_h3_sol_block_offload.json`。Sol-Attn 仅用于 50 层主 Transformer，短文本 refiner 使用 dense Torch SDPA；前 6 个去噪步骤和第 0 层通过 `dense_backend=sage_attn2` 使用 SageAttention2。H3 的 attention 序列按 `[text | audio | video]` 混合打包，不是单一的三维视频网格，因此该配置使用 `reorder=none`，不能直接启用 Wan 的 Morton3D 重排。
