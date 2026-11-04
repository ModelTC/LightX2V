# Parallel Inference

Merge the JSON fragments below into a complete model config.

Launch with `torchrun` and a process count matching the parallel configuration. With only sequence and CFG parallelism, `N = seq_p_size × cfg_p_size` (unset values count as 1); for example, four-way sequence parallelism with two-way CFG parallelism needs 8 processes.

CFG parallelism requires `enable_cfg=true` and cannot be used with CFG-disabled distilled configs.

LightX2V supports distributed parallel inference, enabling the utilization of multiple GPUs for inference. The DiT component supports two parallel attention mechanisms: **Ulysses** and **Ring**, while also supporting **Cfg parallel inference**. Parallel inference significantly reduces inference time and alleviates memory overhead on each GPU.

## DiT Parallel Configuration

### 1. Ulysses Parallel

**Configuration method:**
```json
{
    "parallel": {
        "seq_p_size": 4,
        "seq_p_attn_type": "ulysses"
    }
}
```

### 2. Ring Parallel

**Configuration method:**
```json
{
    "parallel": {
        "seq_p_size": 4,
        "seq_p_attn_type": "ring"
    }
}
```

## Cfg Parallel Configuration

**Configuration method:**
```json
{
    "enable_cfg": true,
    "parallel": {
        "cfg_p_size": 2
    }
}
```

## Hybrid Parallel Configuration

**Configuration method:**
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

## Usage

Parallel inference configuration files are available [here](https://github.com/ModelTC/lightx2v/tree/main/configs/dist_infer)

By specifying --config_json to a specific config file, you can test parallel inference.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/dist_infer) are some run scripts for your use.
