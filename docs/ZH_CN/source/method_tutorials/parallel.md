# 并行推理

LightX2V 支持分布式并行推理，能够利用多个 GPU 进行推理。DiT部分支持两种并行注意力机制：**Ulysses** 和 **Ring**，同时还支持 **VAE 并行推理**。并行推理，显著降低推理耗时和减轻每个GPU的显存开销。

## DiT 并行配置

DiT 并行是通过 `parallel_attn_type` 参数控制的,支持两种并行注意力机制：

### 1. Ulysses 并行

**配置方式：**
```json
{
    "parallel_attn_type": "ulysses"
}
```

### 2. Ring 并行


**配置方式：**
```json
{
    "parallel_attn_type": "ring"
}
```


## VAE 并行配置

VAE 并行是通过 `parallel_vae` 参数控制：

```json
{
    "parallel_vae": true
}
```

**配置说明：**
- `parallel_vae: true`：启用 VAE 并行推理（推荐设置）
- `parallel_vae: false`：禁用 VAE 并行，使用单 GPU 处理

**使用建议：**
- 在多 GPU 环境下，建议始终启用 VAE 并行
- VAE 并行可与任何注意力并行方式（Ulysses/Ring）组合使用
- 对于内存受限的场景，VAE 并行可显著减少内存使用


## 使用方式

并行推理的config文件在[这里](https://github.com/ModelTC/lightx2v/tree/main/configs/dist_infer)

通过指定--config_json到具体的config文件，即可以测试并行推理

[这里](https://github.com/ModelTC/lightx2v/tree/main/scripts/dist_infer)有一些运行脚本供使用。
