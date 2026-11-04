# GPU 资源观测

宿主Linux/NVIDIA整卡、非MPS工具；只记录和核对资源，不启动、终止或重试工作负载。

```bash
python tools/runtime/gpu_watch.py capture --devices GPU-完整UUID --output /new/watch --seconds 120
python tools/runtime/gpu_watch.py check --samples /new/watch/samples.jsonl --task /run/task.json
```

先后台启动capture，确认有成功的空闲采样，再启动任务。任务退出后等待一个采样，向capture发SIGTERM会再采一次后退出；到期也会最后采一次。查询逐次有界；到期尾采样可能因剩余预算不足而失败，须提前取得有效后侧样本。确认capture退出、文件写完后再运行check。采集时长不是工作负载的超时保证。
`task.json`由实际运行记录提供：`pid`为宿主任务根PID，`start_ticks`为其`/proc/PID/stat`启动tick；`start/end`为宿主同一单调时钟的任务起止；另含`boot_id`和`devices`（沿用capture规范化后的完整UUID）。任务区间包含模型加载、预热和正式测量；容器内PID不能直接填写。
check要求起止两侧采样、指定设备完整、起跑无GPU进程/利用率且显存≤256MiB、相邻采样完成间隔≤3s。外来GPU进程、身份/查询未知或覆盖缺失均不通过；两侧边界保守纳入。阈值可显式调整，不靠慢样本判竞争。
MPS等代理可能合并多个调用者的进程归属，本版不支持；发现`M`/`M+C`类型会拒绝，但这不能替代部署核实。利用率来自历史采样窗，进程清单描述当前上下文，两者不保证同刻；利用率只用于起跑空闲检查，任务中的空清单加非零利用率不能证明是外来任务，也不证明归属。
输出`resources_clear`只表示该区间未观察到竞争，退出码0才通过；不证明命令成功、清理完成、充分预热或质量。离散采样会漏短瞬时竞争，CPU/网络干扰不在覆盖内。
