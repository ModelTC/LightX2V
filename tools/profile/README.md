# 模型范围采集

在同进程真实同步函数/实例方法上采集TensorBoard时间线与可选核心事实。入口实际调用目标；工具不重放，也不提供正式无开销请求计时。

```bash
python -m tools.profile.capture --target my_model:Block.forward \
  --output /tmp/profile-clean --warmup 3 my_entry.py
python -m tools.profile.capture --target my_model:Block.forward \
  --output /tmp/profile-facts --warmup 3 --workloads my_entry.py
tensorboard --logdir /tmp/profile-clean
```

同一进程至少调用skip+warmup+count次。scheduler warmup默认1，active默认1；调用不足为capture_incomplete并CLI非零退出。模型warmup与采集器warmup用途不同，不保证跨进程缓存。进入active前完成当前CUDA设备预热队列；--synchronize-cuda仅在整个目标前后同步，不逐kernel同步。

API：`with Capture(output, enabled=True, warmup=3) as capture:`，由入口循环调用`capture.call(model.forward, input)`。默认enabled=False。普通同步Python函数/实例方法可临时包装；异步/生成器/静态或类方法描述符、缓存别名不接管。退出/异常恢复原方法和argv/path。目标owner与实际__main__入口同源时提前拒绝，改用独立driver或Capture API。

record_shapes默认关闭，--record-shapes只供ATen调试，不据它计算FLOPs。--workloads复用共用算子统计并给实际apply唯一标记；--moe-distribution额外记录生产专家分布，实际稀疏区域随统计自动取得，--stats-max-calls默认每类4096，超限有coverage gap。core_ops写盘在apply标记外，backend统计和workloads写盘仍可能在apply内；CPU归约/搬运可能同步，判断生产空隙须另采干净时间线。

--observe-only只记录host边界，不启动profiler；--stats-only只采计算量、不启动profiler，无scheduler warmup。异步GPU的host返回不等于完成。多设备队列边界由调用方明确。

产物为session.json、逐次calls.jsonl、一份.pt.trace.json.gz及可选call_*.stats。trace为PyTorch/Kineto格式，面向TensorBoard PyTorch Profiler插件；runtime分类兼容转换保留原分类。session保存状态、参数和实际加载Python身份；失败/不完整产物保留。统计与逐次runtime边界只覆盖实际core/backend统计入口，编译图/graph内部缺口不猜。

--library-log-dir用于尚未导入torch的新CLI进程，在库初始化前打开cuBLAS/cuBLASLt日志；退出恢复环境并保存实际库/设置/日志。日志可能在进程退出最后flush，用library_precision.read_library_logs读取最终文件。它包含预热，是进程级API证据，不按时间或shape猜逐次归属；设置许可/API computeType不证明物理部分归约。

源码顺序：capture.py真实调用/scheduler→workload_stats.py显式标记→library_precision.py设置及新进程日志。原TorchTraceProfileContext和analyze_torch_trace_gap.py不变。

## SP 算子的真实流水

已有 SP benchmark 选项可以直接组合采集 API，无需复制 Ulysses/Ring 流水。由 `diagnostic_configuration_from_report` 核对 suite、实测候选和基础 seed，再从当前候选目录取得完整配置。它不代表产品采用决定。

```python
from tools.benchmarks.sp_bench import _prepare_operation
from tools.profile.capture import Capture

actual_seed = base_seed + repeat * 1000
fn, dense_equivalent_flops = _prepare_operation(case, candidate, device, actual_seed, rank, inputs=inputs)
with Capture(rank_output, enabled=True, warmup=warmup, count=1,
             workloads=joint_statistics, metadata=metadata) as capture:
    for _ in range(warmup + 1):
        result = capture.call(fn)
```

`case/candidate/inputs/device/rank` 均沿用生产准备，保留实际 `head_parallel/head_parallel_group_size`，包括非整除尾组；不要改用只支持 bulk 的组件模拟诊断。每个 rank 采集自己的目录；metadata 的 identity.target 包含完整候选配置，另记录共同批次、suite/case、输入文件与源码身份、基础及实际 seed、repeat 和真实组。完整 SP 入口是组合边界，计算量来自实际本地叶子调用，不能把返回的稠密等价量分给每个 kernel。时间线采集先关闭统计，必要时另跑同配置的联合统计。


MoE可分离的实际专家调用复用trace_ranges记录compute_trace_marker；只在已采样联合诊断时标记，不新增导出/预热/计算量通道。
