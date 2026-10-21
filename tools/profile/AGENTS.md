# 采集工具的Agent用法

代码负责确定性采集；API/指标查README。先核真实目标/输入/源码及授权，选择最小同步函数，入口在同进程真实预热与调用，不暗中重放或跨进程预热。

正常时间线关闭record_shapes/统计，需要shape/实际区域再联合诊断。统计可能改变空隙，不能作无开销请求性能。缺计算量在owning core/backend补最小合同；未知精度查实际native API，必要时新进程库日志，不根据输入dtype猜累加。

只用获准空闲设备，动态监测干扰；保存命令、PID启动身份、日志/产物/退出，恢复先核实再重试。按源码/输入/机制身份复用证据，不为小改名重复GPU。机器路径、权重和一次性材料留本地；不自动改默认、批准质量、提交或推送。

SP实测配置和输入沿用生产_prepare_operation返回的(fn, dense_equivalent_flops)，按真实seed/repeat/rank和完整head流水参数组织采集；不复制流水，不用bulk模拟替代head。资源观测复用tools/runtime/gpu_watch，launcher提供期限和本人任务收尾。
