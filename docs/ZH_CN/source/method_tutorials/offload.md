# CPU 与磁盘卸载

Offload 在 GPU、CPU 内存以及可选的磁盘之间搬运权重，降低 GPU 上驻留的权重规模，代价是传输延迟；它不能消除激活值的显存需求。下文以 Wan 配置为例，其他模型支持的选项可能不同，应从对应模型的仓库示例开始配置。

## 选择卸载粒度

| 配置项 | 含义 |
|--------|------|
| `cpu_offload` | 开启 DiT 的 CPU 卸载 |
| `offload_granularity: "block"` | 按需搬运 Transformer block |
| `offload_granularity: "phase"` | 按每个 Transformer block 内的 self-attention、cross-attention、FFN 阶段卸载和预取权重 |
| `t5_cpu_offload` | 卸载 Wan 的 T5 编码器 |
| `clip_cpu_offload` | 卸载使用 CLIP 的模型变体的编码器 |
| `lazy_load` | 在支持的配置中启用从磁盘按需加载 |
| `num_disk_workers` | lazy loading 的磁盘加载线程数 |

将以下字段合并到完整的 Wan 配置，可启用 block offload：

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "t5_cpu_offload": true
}
```

建议从完整的 [CPU block 配置](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/block) 或[磁盘加载配置](https://github.com/ModelTC/LightX2V/tree/main/configs/offload/disk) 开始。使用 [run_wan_i2v_lazy_load.sh](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/run_wan_i2v_lazy_load.sh) 前，先修改其中的模型和仓库路径。

## 内存与组合限制

CPU 卸载仍需足够的主机内存容纳权重和临时缓冲区；磁盘加载还依赖存储容量和吞吐量。分辨率、帧数、权重精度、注意力后端和 VAE 配置都会影响显存峰值，不能只凭卸载开关判断最低显存要求。相关 Wan 设置见[低资源部署](../deploy_guides/for_low_resource.md)。

Wan 静态 LoRA 合并不能与 `lazy_load` 或量化 DiT 同时使用。组合这些选项前，请查看 [LoRA 部署](../deploy_guides/lora_deploy.md)。

## NUMA 本地 CPU 权重

在已知 GPU NUMA 拓扑的 Linux CUDA 主机上，静态 block offload 可选择将各 rank 的 pinned CPU 权重放到本地 NUMA arena：

```json
{
    "cpu_offload": true,
    "offload_granularity": "block",
    "cpu_offload_numa": true
}
```

设为 `true` 时必须成功绑定 NUMA 节点。Qwen 蒸馏预设使用 `"auto"`。若 GPU NUMA 拓扑、节点绑定或 System V IPC 不可用，或 IPC 权限、段大小及配额限制导致共享内存段创建失败，则回退到普通 CPU block offload。内存不足、段挂接、CUDA 注册或资源清理失败时仍会报错。默认值为 `false`。普通（非分层）Qwen Image T2I/I2I 使用非量化 DiT 时，可在启动时将 LoRA 在线合并到 CPU 权重，再将合并结果放入 NUMA 本地内存。同一进程的后续请求复用这组权重。其他 LoRA 模式、lazy loading、张量并行、共享权重、contiguous offload 及 NVFP4/GGUF/Marlin/`fp8-pertensor` 特殊存储路径暂不支持。仍需满足对应后端的硬件和权重要求。

使用结束后调用 `runner.close()`；Python pipeline 可在 `finally` 中调用 `pipeline.runner.close()`。具体约束见[服务端 NUMA 说明](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/server/README.md#numa-local-cpu-offload-weights)。

原理介绍可参考 [Offload 技术文章](https://github.com/ModelTC/LightX2V-BLOG/blob/main/_articles/Offload.md)。
