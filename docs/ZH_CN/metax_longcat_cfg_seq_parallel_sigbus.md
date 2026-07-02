# MetaX LongCat CFG + Ulysses Seq Parallel SIGBUS 调试记录

## 背景

在 MetaX 平台运行 LongCat Image T2I 多卡脚本时，原 8 卡并行配置会在第一步 DiT 前向阶段触发 SIGBUS：

```json
"parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 2
}
```

同样的 LongCat 配置在 NVIDIA 平台可以跑通，因此重点排查 MetaX 平台后端、PyTorch distributed collective、Ulysses seq parallel attention 的组合兼容性。

## 现象

MetaX 8 卡 LongCat T2I 在所有 rank 完成 encoder 和 `step_pre` 后，于第一步 `infer_main` 期间退出：

```text
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
traceback : Signal 7 (SIGBUS) received
```

失败位置不是配置路径、模型加载、文本编码、VAE 或 scheduler，而是 DiT transformer 前向。

## 复现矩阵

已验证结果如下：

| 配置 | 卡数 | 结果 | 说明 |
| --- | --- | --- | --- |
| `cfg_p_size=1`, `seq_p_size=1` | 1 | 通过 | 单卡 LongCat 1 step 通过 |
| `cfg_p_size=2`, `seq_p_size=1` | 2 | 通过 | 仅 CFG 并行通过 |
| `cfg_p_size=1`, `seq_p_size=2` | 2 | 通过 | 仅 Ulysses seq 并行通过 |
| `cfg_p_size=1`, `seq_p_size=4` | 4 | 通过 | 4 卡 seq 并行通过 |
| `cfg_p_size=2`, `seq_p_size=2` | 4 | 失败 | 最小稳定复现 SIGBUS |
| `cfg_p_size=2`, `seq_p_size=4` | 8 | 失败 | 原 8 卡配置 SIGBUS |
| `cfg_p_size=1`, `seq_p_size=8` | 8 | 失败 | MetaX LongCat 8 路 seq 也不稳定 |

当前可用的稳定规避配置：

```json
"parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 1
}
```

该配置已完成 50 step 实际运行验证，并保存结果图。

## 定位过程

### 1. 缩小到 DiT 前向

日志显示所有 rank 都完成：

- 模型加载
- 文本 encoder
- target shape 设置
- `step_pre`

之后在 `LongCatImageRunner.run()` 的 `self.model.infer(self.inputs)` 内部失败。

### 2. 缩小到 LongCat CFG + Seq 组合

LongCat CFG 并行逻辑位于：

```text
lightx2v/models/networks/longcat_image/model.py
```

CFG 并行时：

- `cfg_p_rank == 0` 跑 cond prompt
- `cfg_p_rank == 1` 跑 uncond prompt
- 每个 CFG 分支内部继续执行 seq parallel
- 最后通过 CFG group `all_gather` 汇总 cond/uncond noise prediction

最小复现证明：单独 CFG、单独 seq 都能跑，二者组合才触发 SIGBUS。

### 3. 缩小到第一个 double-stream block

在 LongCat transformer 内部加临时探针后发现：

- uncond 分支可以跑完 10 个 double blocks + 20 个 single blocks
- cond 分支进入 `double_block[0]` 后失败
- cond 分支已经完成：
  - image/text norm
  - QKV projection
  - RoPE
- 失败发生在 `calculate_parallel.apply(...)` 内部

对应代码位置：

```text
lightx2v/models/networks/longcat_image/infer/transformer_infer.py
```

### 4. 缩小到 Ulysses 第一条 all-to-all

继续在 Ulysses attention 内部加临时探针后，最后成功日志停在：

```text
[UlyssesDebug] rank=1 group_rank=1 stage=before_all_to_all_q shape=(2, 2016, 12, 128)
[UlyssesDebug] rank=0 group_rank=0 stage=before_all_to_all_q shape=(2, 2016, 12, 128)
```

下一条 `after_all_to_all_q` 没有打印，随后 SIGBUS。

对应代码位置：

```text
lightx2v/common/ops/attn/ulysses_attn.py
```

触发语句：

```python
dist.all_to_all_single(output_q, img_q, group=seq_p_group)
```

触发张量形状：

```text
(2, 2016, 12, 128)
```

该形状来自 LongCat 16:9 输出：

- packed image tokens: `4032`
- `seq_p_size=2` 后每 rank image shard: `2016`
- attention heads: `24`
- 每个 seq rank heads: `12`
- head dim: `128`

## 排除项

### 不是路径或配置文件引用错误

路径已验证，模型和配置均能加载，单卡和部分多卡组合能跑通。

### 不是普通 2D mesh subgroup all-to-all 完全不可用

额外写了纯 PyTorch distributed 复现脚本：

```text
/tmp/metax_mesh_alltoall_repro.py
```

测试内容：

- 初始化同样的 2x2 device mesh
- 取 `seq_p` group
- 对同形状 `(2, 2016, 12, 128)` bf16 tensor 执行 `dist.all_to_all_single`

结果：通过。

### 不是连续 q/k/v all-to-all 本身必挂

额外写了更接近 Ulysses 的纯 collective 复现脚本：

```text
/tmp/metax_ulysses_alltoall_repro.py
```

测试内容：

- 构造 `(2528, 24, 128)` q/k/v
- 切出 image tokens
- reshape/permute 成 `(2, 2016, 12, 128)`
- 连续执行 q/k/v 三次 `all_to_all_single`
- 循环 30 轮

结果：通过。

因此问题不是 MetaX 上所有 `all_to_all_single` 都不可用，而是更窄的组合条件触发。

### `seq_p_head_parallel=true` 不能规避

尝试打开：

```json
"seq_p_head_parallel": true
```

仍然 SIGBUS，说明不是单纯“大块 all-to-all 太大”导致。

## 当前结论

该问题更准确的描述是：

> MetaX 平台在 LongCat Image 真实 DiT 前向中，当 CFG 并行和 Ulysses seq parallel 同时启用时，Ulysses parallel attention 的第一条 `dist.all_to_all_single` 会触发 SIGBUS。NVIDIA 平台同配置可跑通，因此更可能是 MetaX PyTorch/distributed 后端或相关通信实现对该组合场景兼容性不足。

触发组合：

- LongCat Image
- `enable_cfg=true`
- `cfg_p_size=2`
- `seq_p_size>=2`
- `seq_p_attn_type=ulysses`
- MetaX 平台
- 第一步 DiT forward
- 第一个 double-stream block
- Ulysses `all_to_all_single(output_q, img_q, group=seq_p_group)`

## 建议规避方案

在 MetaX LongCat T2I 脚本中使用 4 卡 seq-only 并行：

```json
"parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 1
}
```

这样会关闭 CFG parallel，让 cond/uncond 在每个 seq group 内顺序执行。实际验证可跑完整 50 step。

对应输出验证：

```text
/data/LightX2V/save_results/longcat_image_t2i_metax_dist4.png
```

对应日志示例：

```text
/data/LightX2V/logs/platforms/metax/dist/longcat_image_t2i_20260628_001927.log
```

## 给 MetaX 平台侧的排查建议

建议平台侧优先检查：

1. `torch.distributed.all_to_all_single` 在真实模型前向中处理 bf16、非叶子张量、连续张量、多个 process group 并发时是否存在 SIGBUS 风险。
2. 2D device mesh 中两个 seq groups 同时进行 all-to-all，且另一个 cfg 分支存在不同执行进度时，后端通信资源管理是否可靠。
3. MetaX 对 PyTorch NCCL-like backend 的 async/sync 行为、stream 使用、内存注册、页对齐或 DMA 访问是否对真实模型 tensor 有额外限制。
4. LongCat cond/uncond prompt embedding 内容不同但 shape 相同，为什么 cond 分支更容易在第一个 double block 触发，而 uncond 分支可跑完。
5. `flash_attn2` 在当前环境日志中出现 fallback 信息，虽然本次定位停在 all-to-all 之前，但仍建议确认 MetaX flash attention/fallback attention 与 distributed stream 的交互。

## 相关临时诊断日志

本次调试过程中生成过以下临时日志，可用于回溯：

```text
/tmp/longcat_cfg2_seq2_ulysses_rank01.log
/tmp/longcat_cfg2_seq2_double_debug.log
/tmp/metax_mesh_alltoall_repro.log
/tmp/metax_ulysses_alltoall_repro.log
```

这些日志位于 `/tmp`，不属于仓库文件，后续环境清理后可能消失。
