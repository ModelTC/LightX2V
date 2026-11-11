# 平台适配速查

评测协议和指标在 `configs/common/` 与 `evaluate.py` 中共用；设备、容器、模型后端及来源锁由 `configs/<platform>/` 选择。当前有 MLU 配置，以及供 NVIDIA 环境填写镜像后使用的配置模板；NVIDIA 尚未在目标卡上验证。MetaX 等新平台需先按下列接口增加配置和监控脚本。

1. 复制 `configs/mlu/` 或 `configs/nvidia/` 到 `configs/<platform>/`。填写 `runtime.json` 的 PyTorch 设备 namespace/type、注册模块、镜像/Python、容器设备参数、可见卡变量、dtype、包版本和监控命令。`monitor_parser: "json"` 的监控脚本必须输出 `card`、`util_percent`、`memory_mib`、`capacity_mib`、`pids`，可选输出温度和功耗。
2. 从该平台的模型配置选择 attention、稀疏卷积、deform conv 等后端写入 `model.json`；不要沿用未在目标平台验证的组合。`paths.json` 指向本机数据和权重。`source-lock.json` 与兼容补丁必须固定该平台实际使用的参考源码。
3. 添加 `scripts/platform/<platform>/run_eval.sh`，设置 `EVAL_PLATFORM=<platform>` 后执行 `evaluate.py --platform <platform>`；提供满足上述 JSON 协议的宿主机监控脚本，并在 `runtime.json` 指向它。
4. 先用目标卡运行 `--smoke`，确认空闲卡检查、设备计时/显存、两实现输出和报告完整后，再运行正式评测。只有目标硬件链路通过后，才把该平台标为已验证。

示例：`bash scripts/platform/nvidia/run_eval.sh --smoke --card 0 --data-root /path/to/data`。NVIDIA profile 的镜像为空，运行前需设置 `EVAL_IMAGE`。新增平台时保留公共 suite 不变，并将新配置快照和平台身份记录在结果中。
