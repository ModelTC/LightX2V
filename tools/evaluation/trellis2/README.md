# TRELLIS.2 GSO 端到端评测

本项目只通过 LightX2V runner 执行 TRELLIS.2 推理，并用固定的 GSO 真值计算精度、端到端时延和显存。不会下载或运行模型官方原版。`evaluate.py`包含共用流程，平台通过启动脚本与`configs/<platform>/`选择。MLU已验证；NVIDIA配置尚未在目标卡验证。

## 准备与运行

需要 Linux、Git、Python 3.10+、Docker、匹配所选平台的 PyTorch 镜像、一张空闲设备，以及 DATA.md 中列出的 GSO 输入包和权重。数据与模型权重不随PR提供。

在LightX2V仓库根目录执行（若只复制评测目录，则进入该目录）：

```bash
cd tools/evaluation/trellis2
bash scripts/platform/mlu/run_eval.sh --prepare-sources
bash scripts/platform/mlu/run_eval.sh --check-inputs --rehash-weights --data-root /path/to/data
bash scripts/platform/mlu/run_eval.sh --install-deps --smoke --card 0 --data-root /path/to/data
bash scripts/platform/mlu/run_eval.sh --card 0 --data-root /path/to/data
```

完整运行处理1023个对象，每个对象测量2次，共2046次正式请求，另有1次预热。`--smoke`只运行1个对象；`--limit N`用于小样本诊断。失败和缺失保留在分母中，未完成返回非零退出码。NVIDIA使用`scripts/platform/nvidia/run_eval.sh`并设置`EVAL_IMAGE`。新增平台流程见[PLATFORMS.md](PLATFORMS.md)和[PLATFORM_ADAPTATION.md](PLATFORM_ADAPTATION.md)。

源码准备仅从source-lock中固定的LightX2V提交重建；不下载模型官方原版。准确提交见所选平台的`configs/<platform>/source-lock.json`。

## 结果与时延边界

每次在`artifacts/evaluation-时间戳/`创建新目录：

- `report.md`、`summary.json`、`comparison.json`：完成状态、LightX2V对GSO真值的质量/性能指标及重复差异。
- `requests.csv`、`metrics.csv`、`summary.csv`：逐请求时延、显存与逐对象质量指标。
- `lightx2v/`：预测NPZ、环境、请求、预热和设备遥测。
- 阶段日志、失败清单、配置快照和哈希。

端到端墙钟时延覆盖读图、预处理、推理、模型后处理及输出转到CPU；不含模型加载、预热、NPZ写盘或离线精度计算。适配器调用、CPU输出转换和评测框架开销分别记录；模型事件时间属于端到端的子项。

重算已有结果（沿用原请求的时间和显存记录）：

```bash
bash run_eval.sh --reuse-runs /path/to/run/lightx2v --output artifacts/recalculated
bash run_eval.sh --help
```

数据协议见[DATA.md](DATA.md)，平台配置见[configs/README.md](configs/README.md)。

Ascend 910B 配置已加入，尚未完成模型 smoke 或全量评测；详见 [Ascend 配置](configs/ascend_npu/README.md)。
