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

示例配置为 `configs/attentions/wan_i2v_sol_attn.json`。它启用了 Morton3D token 重排和 strict 模式，安装无效时会直接报错而不会静默回退到 dense SDPA。首次调用会编译对应 shape 的 kernel，计时时应排除首次调用。
