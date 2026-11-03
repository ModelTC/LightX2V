# CPU and Disk Offload

Offload reduces GPU weight residency by moving weights between GPU, CPU memory, and optionally disk. It trades memory use for transfer latency; it does not remove activation memory requirements. The settings below describe Wan configurations. Other models may support a different subset; start from their checked-in examples.

## Choose a granularity

| Setting | Meaning |
|---------|---------|
| `cpu_offload` | Enable DiT CPU offload |
| `offload_granularity: "block"` | Transfer transformer blocks as needed |
| `offload_granularity: "phase"` | Offload and prefetch weights for the self-attention, cross-attention, and FFN phases within each transformer block |
| `t5_cpu_offload` | Offload the Wan T5 encoder |
| `clip_cpu_offload` | Offload the CLIP encoder for variants that use it |
| `lazy_load` | Enable disk-backed loading in supported configurations |
| `num_disk_workers` | Disk-loading worker count when lazy loading is enabled |

Merge these fields into a complete Wan config to enable block offload:

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "t5_cpu_offload": true
}
```

Use a complete [CPU block configuration](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/block) or [disk-loading configuration](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/disk) as the starting point. Configure the model and repository paths in [run_wan_i2v_lazy_load.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/run_wan_i2v_lazy_load.sh) before running it.

## Memory and compatibility

CPU offload needs enough host memory for the offloaded weights and temporary buffers. Disk loading also depends on storage capacity and throughput. Resolution, frame count, checkpoint precision, attention backend, and VAE settings affect peak memory; an offload switch alone does not establish a minimum GPU size. See [Low-resource deployment](../deploy_guides/for_low_resource.md) for the related Wan settings.

Wan static LoRA merging cannot be combined with `lazy_load` or a quantized DiT. Check [LoRA deployment](../deploy_guides/lora_deploy.md) before combining these options.

## NUMA-local CPU weights

On Linux CUDA hosts with known GPU NUMA topology, static block offload can optionally place each rank's pinned CPU weights in a local NUMA arena:

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "cpu_offload_numa": true
}
```

This option defaults to `false`. It rejects lazy loading, tensor parallelism, LoRA, shared weights, contiguous offload, and specialized NVFP4/GGUF/Marlin/`fp8-pertensor` storage paths. Ordinary backend hardware and checkpoint requirements still apply.

Release resources with `runner.close()` after use, or `pipeline.runner.close()` in a `finally` block. See the [server's NUMA reference](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/server/README.md#numa-local-cpu-offload-weights) for the implementation constraints.

For the original design discussion, see the [Offload article](https://github.com/ModelTC/LightX2V-BLOG/blob/main/_articles/Offload.md).
