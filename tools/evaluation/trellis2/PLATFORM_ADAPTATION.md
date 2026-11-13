# 新平台适配步骤

1. 阅读README、DATA和`configs/README.md`，确认新机器上的GSO文件与权重哈希。
2. 新建`configs/<platform>/`，按现有平台提供runtime、model、paths、source-lock和evaluation-origin。source-lock只固定LightX2V源代码；推理统一走LightX2V runner。
3. 新建`scripts/platform/<platform>/run_eval.sh`及必要监控适配。监控需报告设备身份、利用率、显存和进程归属；容器只映射分配给本次评测的一张卡。
4. 核对设备API对应的torch namespace、同步、计时事件、显存统计、autocast和随机种子。平台相关差异放配置或设备适配层。
5. 用单对象smoke检查输入输出、请求数和精度，再运行完整1023对象、每对象1次正式请求。所有失败保留在分母内。
6. 保存镜像/源码/权重/配置哈希、设备与依赖信息、汇总和逐对象指标、失败记录及遥测。只将精简可审查结果提交PR；原始NPZ、缓存和大文件放独立产物存储。

计时范围固定为读图、预处理、LightX2V推理和输出CPU化；模型加载、预热、写盘、离线评测不计入端到端时延。
