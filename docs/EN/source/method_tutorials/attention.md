# Attention Mechanisms

## Attention Mechanisms Supported by LightX2V

| Name               | Type Name        | GitHub Link |
|--------------------|------------------|-------------|
| Flash Attention 2  | `flash_attn2`    | [flash-attention v2](https://github.com/Dao-AILab/flash-attention) |
| Flash Attention 3  | `flash_attn3`    | [flash-attention v3](https://github.com/Dao-AILab/flash-attention) |
| Sage Attention 2   | `sage_attn2`     | [SageAttention](https://github.com/thu-ml/SageAttention) |
| Radial Attention   | `radial_attn`    | [Radial Attention](https://github.com/mit-han-lab/radial-attention) |
| Sparge Attention   | `sparge_ckpt`     | [Sparge Attention](https://github.com/thu-ml/SpargeAttn) |

---

## Configuration Examples

The configuration files for attention mechanisms are located [here](https://github.com/ModelTC/lightx2v/tree/main/configs/attentions)

By specifying --config_json to a specific config file, you can test different attention mechanisms.

For example, for radial_attn, the configuration is as follows:

```json
{
  "self_attn_1_type": "radial_attn",
  "cross_attn_1_type": "flash_attn3",
  "cross_attn_2_type": "flash_attn3"
}
```

To switch to other types, simply replace the corresponding values with the type names from the table above.

Tips: radial_attn can only be used in self attention due to the limitations of its sparse algorithm principle.

For further customization of attention mechanism behavior, please refer to the official documentation or implementation code of each attention library.
