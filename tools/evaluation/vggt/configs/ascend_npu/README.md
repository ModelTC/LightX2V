# Ascend 运行配置

目标环境为 Linux x86_64、Ascend 910B、BF16。MLU 和 NVIDIA 配置、公共 suite、权重锁和计时边界保持原样。当前仅完成配置、依赖、源码重建与小张量设备 API 检查；尚未运行模型 smoke 或完整评测。正式评测要求另行确定后再启动。

请使用新的项目目录准备此平台，避免复用其他平台的 `sources/`、`runtime/` 或缓存。

在 `tools/evaluation/vggt/` 中构建完整依赖镜像：

```bash
docker build -f scripts/platform/ascend_npu/Dockerfile -t lightx2v-eval-vggt:ascend-910b .
bash scripts/platform/ascend_npu/run_eval.sh --prepare-sources
bash scripts/platform/ascend_npu/run_eval.sh --check-inputs --rehash-weights \
  --data-root /your/evaluation-data --output /your/results/vggt-input-audit
```

镜像从公开的 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.10` 构建，安装 PyTorch 2.12.0 CPU wheel、torch-npu 2.12.0 及固定评测依赖。CPU wheel 配合 torch-npu 提供 NPU 运算；不安装 CUDA 包。镜像不含数据、权重或模型源码，运行源码由 `--prepare-sources` 从固定公开提交下载并校验。无需先构建模型 PR 的镜像，也无需 `--install-deps`。若使用该选项，会根据 `runtime.json:requirements_file` 安装 Ascend 的 Python 3.10 兼容依赖表；其中 kornia/imageio 固定为 0.8.1/2.37.0（通用表的 0.8.3/2.38.1 要求 Python ≥3.11）。宿主机须安装兼容的 Ascend 驱动、`npu-smi` 和 Docker。

`runtime.json` 中的 `docker_device_args` 映射选定物理卡、管理设备和只读驱动目录，`ASCEND_RT_VISIBLE_DEVICES={card}` 将该物理卡映射为容器内唯一逻辑 `npu:0`。`--card` 或 `ASCEND_CARD` 选择物理卡，默认 0；`--image` 或 `EVAL_IMAGE` 可覆盖镜像名。驱动路径或设备路径不同的机器需相应修改这些参数。该驱动环境需要 `--cap-add ALL`；仅模型容器获得设备和该权限，指标/报告容器保持无设备访问。`eval-python` 在覆盖容器入口后重新加载 CANN 环境。

监控通过宿主机 `npu-smi info` 获取 HBM、AICore 利用率、PCI 标识与进程。空闲设备仍有驱动保留 HBM，因此 `idle_memory_mib_max` 默认 4096 MiB；它是启动前阈值，不会从记录的显存中扣除。启动还必须满足利用率为 0 且没有进程。此阈值可按目标驱动的空闲基线调整。监控解析失败时会阻止启动。

数据和权重使用 `DATA.md` 的冻结目录与哈希；`paths.json` 仅记录相对路径，`--data-root`/`EVAL_DATA_ROOT` 由用户指定位置。所有输出请用 `--output` 指向本项目目录之外，权重、源码缓存、运行产物均不提交。

源码准备只下载source-lock中固定的LightX2V提交；本评测不下载或运行官方原版。

VGGT 保持 5 视图、518 像素、BF16，LightX2V 使用 `torch_sdpa`/`DefaultBF16`。LightX2V使用`torch_sdpa`/`DefaultBF16`配置。

后续按确定的评测要求先运行一次 `--smoke`，验证LightX2V推理、重复请求、输出与报告后，再决定是否运行全量。当前 `evaluation-origin.json` 保持空记录，`validation_status` 明确标记模型 smoke 待验证。
