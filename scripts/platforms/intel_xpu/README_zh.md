# Intel XPU 多卡使用注意事项

LightX2V 在 Intel XPU 上使用 `torchrun` 启动多卡推理，并通过 PyTorch XCCL/oneCCL
完成卡间通信。建议优先从本目录 `dist_infer/` 中与模型、并行方式相匹配的脚本开始修改。

## 1. 启动前检查

- 确认每张卡都能被 PyTorch 识别：

  ```bash
  python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())'
  ```

- PyTorch、Intel GPU 驱动、Level Zero 和 oneCCL 必须来自相互兼容的版本。若使用
  Intel 提供的环境初始化脚本，应在启动 LightX2V 前先加载该环境。
- 多个进程必须使用同一套 Python 环境、模型文件和 LightX2V 代码。启动前确认
  `PLATFORM=intel_xpu`。

## 2. 卡数必须与并行配置一致

配置文件中的并行度必须满足：

```text
torchrun 的进程数 = tensor_p_size × seq_p_size × cfg_p_size
```

未配置的并行度按 `1` 计算。例如：

```json
"parallel": {
  "tensor_p_size": 2,
  "seq_p_size": 2,
  "cfg_p_size": 1
}
```

应使用 4 个进程启动：

```bash
torchrun --standalone --nproc_per_node=4 -m lightx2v.infer ...
```

并行度与进程数不一致会直接报错；每个进程通常对应一张可见 XPU。

## 3. 正确设置可见设备

Intel XPU 使用 `ZE_AFFINITY_MASK` 选择设备，不要用 `CUDA_VISIBLE_DEVICES` 代替：

```bash
export ZE_AFFINITY_MASK=0,1,2,3
```

`--nproc_per_node` 不应超过可见 XPU 数量。设备包含 tile 时，Level Zero 可能使用
`卡号.tile号`（如 `0.0`）表示子设备；应先依据当前机器的 `xpu-smi`/`sycl-ls`
输出确认编号，避免把 tile 数误当成物理卡数。同一机器同时运行多个任务时，应为各任务
分配互不重叠的设备，并使用不同的 rendezvous 端口。

## 4. oneCCL 通信设置

使用 Ulysses 序列并行时会执行 all-to-all。请保留仓库示例脚本中的以下 oneCCL
环境变量，除非已经针对当前驱动和 oneCCL 版本完成验证：

```bash
export CCL_SYCL_ALLTOALL_ARC_LL=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1
export CCL_SYCL_CCL_BARRIER=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
```

这些变量在 `dist_infer/` 的 SP/SP+TP 示例中已设置。纯 TP 脚本不需要照搬
all-to-all 专用设置。不要将 NVIDIA NCCL 的调优变量直接套用到 XPU；XPU 后端为
`xccl`。

## 5. 并行方式选择

- **TP（张量并行）**：适合单卡放不下模型或希望拆分线性层计算。相关维度（例如
  attention heads）必须能被 `tensor_p_size` 整除。
- **SP（序列并行）**：适合较长视频或较大分辨率。使用 Ulysses 时，attention heads
  以及序列切分相关维度必须满足并行切分要求。
- **CFG 并行**：用两组设备分别计算有条件（positive prompt）和无条件/负条件
  （negative prompt）分支，再聚合两组预测结果。它适合推理时确实执行传统
  Classifier-Free Guidance、且两个分支计算量接近的模型，通常设置
  `enable_cfg=true`、`cfg_p_size=2`。相较串行 CFG，它能降低一次去噪 step 中两次
  DiT 前向的延迟，但至少需要 2 张卡，且不会把单个分支的模型显存继续拆小；如果目的是
  解决单分支 OOM，应优先选择 TP/SP。
- **SP + TP**：总卡数为两者乘积，通信量也更大。先分别验证 TP 和 SP，再组合使用，
  更容易定位配置或通信问题。

多卡并不一定线性加速。分辨率、帧数或模型较小时，通信开销可能超过计算收益。

## 6. 内存与输出

- 多卡并行仍可能在每个 rank 上复制 text encoder、VAE 或缓存，不能按卡数简单地将
  单卡显存除法估算。显存不足时，结合配置中的 CPU offload、量化和较小输入逐步验证。
- 保持各 rank 的模型与输入配置完全一致，不要在不同进程中设置不同随机种子或 prompt。
- 结果只应由主进程写入最终路径。请为并发任务设置不同的 `OUTPUT_PATH`，避免互相覆盖。

## 7. 常见问题排查

- **初始化卡住**：检查进程数、可见设备数和并行度乘积；确认无残留进程占用 rendezvous
  端口，并确认 oneCCL/驱动版本匹配。
- **all-to-all 或 all-reduce 卡住**：先缩减为 2 卡，确认示例脚本中的 oneCCL 环境变量
  未被覆盖；再分别测试纯 TP 和纯 SP。
- **某个 rank 提前退出**：查看所有 rank 的日志，最早出现的异常通常才是根因。可设置
  `PYTHONFAULTHANDLER=1` 获取更完整的错误信息。
- **结果错误或不稳定**：先使用 BF16 和仓库默认算子配置建立基线，再逐项启用 FP8、INT8、
  自定义 kernel 或 compile，避免同时引入多个变量。

可直接参考：

- `dist_infer/run_minimax_h3_t2av_tp.sh`：4 卡 TP
- `dist_infer/run_minimax_h3_t2av_sp_tp.sh`：8 卡 SP + TP
- `dist_infer/run_wan22_ti2v_t2v_sp.sh`：2 卡 SP
