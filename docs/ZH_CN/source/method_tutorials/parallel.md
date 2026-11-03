# 并行推理

以下 JSON 为配置片段，需合并到完整模型配置中。

使用 `torchrun` 启动时，进程数须与并行配置匹配。仅使用序列并行和 CFG 并行时，`N = seq_p_size × cfg_p_size`（未设置的值按 1 计）；例如 4 路序列并行加 2 路 CFG 并行需要 8 个进程。

CFG 并行需要 `enable_cfg=true`，不能用于关闭 CFG 的蒸馏配置。

LightX2V 支持分布式并行推理，能够利用多个 GPU 进行推理。DiT部分支持两种并行注意力机制：**Ulysses** 和 **Ring**，同时还支持 **Cfg 并行推理**。并行推理，显著降低推理耗时和减轻每个GPU的显存开销。

## DiT 并行配置

### 1. Ulysses 并行

**配置方式：**
```json
{
    "parallel": {
        "seq_p_size": 4,
        "seq_p_attn_type": "ulysses"
    }
}
```

### 2. Ring 并行


**配置方式：**
```json
{
    "parallel": {
        "seq_p_size": 4,
        "seq_p_attn_type": "ring"
    }
}
```

## Cfg 并行配置

**配置方式：**
```json
{
    "enable_cfg": true,
    "parallel": {
        "cfg_p_size": 2
    }
}
```

## 混合并行配置

**配置方式：**
```json
{
    "enable_cfg": true,
    "parallel": {
        "seq_p_size": 4,
        "seq_p_attn_type": "ulysses",
        "cfg_p_size": 2
    }
}
```


## 使用方式

并行推理的config文件在[这里](https://github.com/ModelTC/lightx2v/tree/main/configs/dist_infer)

通过指定--config_json到具体的config文件，即可以测试并行推理

[这里](https://github.com/ModelTC/lightx2v/tree/main/scripts/dist_infer)有一些运行脚本供使用。
