# 注意力机制

## LightX2V支持的注意力机制

| 名称               | 类型名称         | GitHub 链接 |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
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
