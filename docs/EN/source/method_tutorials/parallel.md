# Parallel Inference

LightX2V supports distributed parallel inference, enabling the use of multiple GPUs for inference. The DiT part supports two parallel attention mechanisms: **Ulysses** and **Ring**, and also supports **VAE parallel inference**. Parallel inference significantly reduces inference time and alleviates the memory overhead of each GPU.

## DiT Parallel Configuration

DiT parallel is controlled by the `parallel_attn_type` parameter and supports two parallel attention mechanisms:

### 1. Ulysses Parallel

**Configuration:**
```json
{
    "parallel_attn_type": "ulysses"
}
```

### 2. Ring Parallel

**Configuration:**
```json
{
    "parallel_attn_type": "ring"
}
```

## VAE Parallel Configuration

VAE parallel is controlled by the `parallel_vae` parameter:

```json
{
    "parallel_vae": true
}
```

**Configuration Description:**
- `parallel_vae: true`: Enable VAE parallel inference (recommended setting)
- `parallel_vae: false`: Disable VAE parallel, use single GPU processing

**Usage Recommendations:**
- In multi-GPU environments, it is recommended to always enable VAE parallel
- VAE parallel can be combined with any attention parallel method (Ulysses/Ring)
- For memory-constrained scenarios, VAE parallel can significantly reduce memory usage

## Usage

The config files for parallel inference are available [here](https://github.com/ModelTC/lightx2v/tree/main/configs/dist_infer)

By specifying --config_json to the specific config file, you can test parallel inference.

Some running scripts are available [here](https://github.com/ModelTC/lightx2v/tree/main/scripts/dist_infer) for use.
