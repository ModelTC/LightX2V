# Qwen-Image-2512 block offload 配置

| 配置 | 并行 | CPU block 共享 |
| --- | --- | --- |
| `qwen_image_t2i_2512_block.json` | 单卡 | 关闭 |
| `qwen_image_t2i_2512_block_sp8.json` | Ulysses SP8 | 关闭 |
| `qwen_image_t2i_2512_block_shared_host_sp8.json` | Ulysses SP8 | 每 host / IPC namespace 一份 |
| `qwen_image_t2i_2512_block_shared_numa_sp8.json` | Ulysses SP8 | 每个参与推理的 GPU NUMA 节点一份 |

全部配置使用 `cpu_offload=true`、`offload_granularity=block`，保留 2512 基线的 50 步、16:9、CFG 4 和 flash_attn3。

共享配置字段：

- `shared_cpu_weights`：启用 CPU block 权重共享。
- `shared_cpu_weight_backend=sysv`：使用 SysV 共享内存。
- `shared_cpu_weight_scope=host|numa|auto`：副本分组策略；`auto` 在 GPU NUMA 节点均已知时按 NUMA 分组，否则按 host 分组。
- `shared_cpu_weight_strict_numa=true`：NUMA 内存绑定失败时报错。
- `shared_cpu_weight_register_chunk_mb=128`：CUDA host registration 的目标分块大小，实际边界避开 tensor 内部。

`numa` 模式按实际 GPU 拓扑选择节点、自动选出各组 leader，由 leader 将全套 block 权重填入该节点的共享 arena，其他 rank 直接映射。无需手工设置 NUMA 节点数量；副本数等于实际参与的 GPU NUMA 节点数。

启动方法及适配范围见 [脚本说明](../../../scripts/qwen_image/offload/README.md)。
