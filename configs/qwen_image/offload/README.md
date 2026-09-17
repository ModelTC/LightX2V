# Qwen-Image-2512 block offload 配置

| 配置 | 默认并行 | CPU block 共享 |
| --- | --- | --- |
| `qwen_image_2512_block_shared.json` | Ulysses SP8 | 默认 host；启动时可用 `SHARED_CPU_WEIGHT_SCOPE=numa` 切换 |

共享入口只有 `qwen_image_2512_block_shared_offload.sh`。它默认使用 8 卡，根据 `CUDA_VISIBLE_DEVICES` 在临时副本中调整 SP；显式传入 `CONFIG_JSON` 时保留指定拓扑并核对卡数。

配置保留 2512 基线的 50 步、16:9、CFG 4 和 FlashAttention 3。共享使用 SysV、CUDA pinned memory，`shared_cpu_weight_strict_numa=true`，`shared_cpu_weight_register_chunk_mb=128`。host 在同机同一 IPC 域共享一份，NUMA 按参与 GPU 的 NUMA 域建立副本。

完整命令、参数优先级和适配范围见 [脚本说明](../../../scripts/qwen_image/offload/README.md)。
