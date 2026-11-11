# VGGT 端到端评测

比较原项目与LightX2V的精度、输出差值、时间和显存。`evaluate.py`包含平台共用流程，平台由脚本及`configs/<platform>/`选择。**MLU已验证；NVIDIA配置尚未在目标卡上验证。**

## 准备

- Linux、Git、Python 3.10+、Docker权限和一张空闲目标设备。
- 与所选`--platform`匹配的 PyTorch 镜像。MLU默认镜像及容器Python路径在`configs/mlu/runtime.json`；NVIDIA运行时需设置`EVAL_IMAGE`。镜像和模型依赖必须与平台配置匹配。
- 按 [DATA.md](DATA.md) 放置GSO同步视图包和权重。模型/数据不随PR提供；原始GSO网格不等于所需的RGB/depth/pose包。

在LightX2V仓库根目录执行；若只复制了评测目录，直接进入该目录并跳过下面的`cd`。

```bash
cd tools/evaluation/vggt

# 首次准备：从公开仓库固定提交重建两实现；不下载权重或运行模型
bash scripts/platform/mlu/run_eval.sh --prepare-sources

# 检查数据、源码和权重；/path/to/data是DATA.md所列根目录
bash scripts/platform/mlu/run_eval.sh --check-inputs --rehash-weights --data-root /path/to/data

# 首对象完整链路，首次运行补装冻结的额外依赖
bash scripts/platform/mlu/run_eval.sh --install-deps --smoke --card 0 --data-root /path/to/data

# 完整1023对象GSO评测；两实现各预热1次、每对象重复2次
bash scripts/platform/mlu/run_eval.sh --card 0 --data-root /path/to/data
```

NVIDIA运行使用`bash scripts/platform/nvidia/run_eval.sh ...`并设置`EVAL_IMAGE`。平台适配步骤见[PLATFORM_ADAPTATION.md](PLATFORM_ADAPTATION.md)。

源码准备固定使用模型PR [#1675](https://github.com/ModelTC/LightX2V/pull/1675)的LightX2V提交，以及锁定的参考源码和兼容修改；准确版本见所选平台的`source-lock.json`。不要求先合并模型PR，也不依赖旧机器的源码目录。

## 结果

终端最后打印报告路径。每次自动创建新的`artifacts/evaluation-时间戳/`，包含：

- `report.md`：质量、性能、差值与失败说明。
- `summary.json`、`comparison.json`：汇总及完整逐对象数据。
- `requests.csv`、`metrics.csv`、`summary.csv`：逐请求时间/显存和指标表。
- `reference/`、`lightx2v/`：原始NPZ、环境、请求、预热、设备遥测；另有阶段日志、失败清单、配置快照与哈希。
- `depth-example.png`：首个可用对象的输入、深度和误差图。

完整评测覆盖1023个GSO对象，两实现各重复2次，共4092次正式请求，另各预热1次。`--limit N`用于小样本诊断；smoke结果不代表完整GSO成绩。CPU指标在两种模型进程结束后自动执行；失败/缺失不从分母删除，未完成返回非零退出码。`--output`指定新目录，已有目录不会覆盖。

每个正式请求记录端到端墙钟时间，并分别记录评测适配器调用、输出转为CPU NumPy数组、评测框架开销。`comparison.json`和`report.md`还汇总适配器可分辨的内部阶段；模型事件时间是端到端时间的子项，不与上述部分重复相加。模型加载、预热、NPZ写盘和离线精度计算不计入端到端时间。

已有预测可直接重算，时间/显存沿用原记录：

```bash
bash run_eval.sh --reuse-runs /path/to/run/reference /path/to/run/lightx2v --output artifacts/recalculated
bash run_eval.sh --help
```

数据与协议：[DATA.md](DATA.md)。配置目录：[configs/README.md](configs/README.md)。新增A100等平台的工程步骤及结果交付：[PLATFORMS.md](PLATFORMS.md)。

Ascend 910B 配置、依赖镜像与参考兼容补丁已提供；当前仅完成设备 API/配置检查，模型 smoke 和正式评测待执行。详见 [Ascend 配置](configs/ascend_npu/README.md)。
