# 注意力机制

## LightX2V支持的注意力机制

| 名称               | 类型名称         | GitHub 链接 |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sol-Attn           | `sol_attn`       | [Sol-Attn](https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends) |
| Sparge Attention   | `sparge_ckpt`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

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

tips: radial_attn因为稀疏算法原理的限制只能用在self attention

如需进一步定制注意力机制的行为，请参考各注意力库的官方文档或实现代码。

### 在 Wan2.1 上使用 Sol-Attn

Sol-Attn 是仅前向、非因果的自注意力后端。官方 kernel 要求连续的 BF16 张量且 head dimension 为 128，与 Wan2.1 自注意力匹配。在 H200 上先用 `scripts/install_sol_attn.sh` 安装固定且已验证的 Sol-Attn 和 CUTLASS DSL 版本，再运行：

```bash
MODEL_PATH=/path/to/Wan2.1-I2V-14B-480P \
    bash scripts/wan/run_wan_i2v_sol_attn.sh
```

示例配置为 `configs/attentions/wan_i2v_sol_attn.json`。它启用了 Morton3D token 重排和 strict 模式，安装无效时会直接报错而不会静默回退到 dense SDPA。Wan 会在完整 Transformer block stack 外只执行一次 Morton 重排并同步 RoPE，不再在每层 attention 中搬运 Q/K/V/输出。按照论文的质量保护策略，40 步 I2V 配置的前 8 个去噪步骤和第 0 个 Transformer 层使用 FlashAttention 3，其余调用使用 Sol-Attn。考虑到论文未直接验证 I2V 纵向分辨率，示例使用较保守的 `tau=0.5`；增大 `tau` 会提高稀疏率和速度，但也可能降低画质。`sample_shift=3` 与标准 Wan2.1 I2V 配置保持一致，避免将采样噪声调度造成的颜色或曝光变化误判为注意力误差。`dense_steps` 和 `dense_layers` 可在 `sol_attn_setting` 中调整；`dense_layers` 同时支持 `[0, 1]` 和 `"0-1"` 两种形式。首次 Sol-Attn 调用会编译对应 shape 的 kernel，计时时应排除首次调用。
