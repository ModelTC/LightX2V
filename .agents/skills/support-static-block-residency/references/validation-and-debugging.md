# 验证与故障定位

简体中文 | [English](validation-and-debugging_en.md)

## 按改动选择验证

先确认当前测试与目标运行环境；只执行覆盖变更及未解决问题的检查。仅改文档时检查链接和内容，改脚本时补充语法、路径和参数检查，不因此重跑所有模型实验。

| 改动 | 重点验证 |
|---|---|
| 数量计算、分布 | `d<=c` 得到零；中间值与接近全常驻；最小整数预算；索引数量、范围、唯一性、均匀性 |
| 模型估测 case | 正式权重与 dtype、输入 shape/文本长度、RoPE、结果有限值、计算重置不进入计时区间 |
| 配置输出 | 原配置其他设置保留，数量与报告一致，写入失败不留下半写入 JSON |
| 推理回调、slot 调度 | block 顺序、状态顺序、实际搬运层、ready/free 依赖、两种布局的数值一致性 |
| 生命周期 | 连续执行、release 后重建、取消/异常、warmup 与 VAE 前释放、CPU 源保留 |
| 脚本或配置 | 检查 shell 语法；从仓库外启动，核对配置/输入/输出路径与工作目录未改变；估测输出被 run 使用；baseline 独立运行；输入与环境一致；profiling 沿用 `base.sh` |

## 当前验证入口

按目标模型选择已有启动目录：

- [Wan](../../../../scripts/wan/inference/offload_residency/README_CN.md)
- [Qwen Image](../../../../scripts/qwen_image/inference/offload_residency/README_CN.md)
- [Qwen-Image-2.1-Turbo](../../../../scripts/qwen_image_21/inference/offload_residency/README_CN.md)
- [MiniMax-H3](../../../../scripts/minimax_h3/inference/offload_residency/README_CN.md)

各目录提供 `estimate.sh`、`run_baseline.sh`、`run.sh` 及对应 NPU 入口。需要验证完整调用链时，先生成常驻配置，再运行 baseline 和常驻推理，核对配置读取、输出及日志。

`test_cases/` 已与上游对齐，原有 `*_static_residency.py` 检查文件已移除。公式、均匀分布、事件执行及释放重建的验证，应根据当前实现和本次改动选择检查方式；不要继续引用不存在的 pytest 或 `python -m test_cases.*` 命令。

记录检查结果及未覆盖范围。设备缺失或跳过检查不能算通过；小 shape 的 block 检查不等同于默认分辨率、完整 pipeline 或目标显存峰值通过。涉及两种布局时分别检查 `per_tensor` 和 `contiguous`。

## 验证机制确实生效

读取生成配置与报告，核对 `offload_resident_blocks == resident_blocks`、`resident_blocks + streamed_blocks == num_blocks`，以及运行日志中的实际索引。实际准备应遵守均匀分布，而非只修改日志或配置字段。

必要时在验证过程中统计 `copy_block_to_buffer()`：一次完整 block 序列内，应只搬运非常驻层，按逻辑顺序共 `N-r` 次。常驻准备的首次复制单独计，不混入这个数字。Qwen 当前两路 CFG 每步执行两个序列时，对应 `2*(N-r)` 次；其他模型或条件以实际调用次数为准。

对照 `r=0`、部分常驻、`r=N` 等有代表性的数量，确认所有层仍计算，常驻层跳过重复搬运。在同一实例上连续执行并在 release 后重新运行，检查 CPU 源地址/状态没有被替换为丢失的临时副本。可控异常后确认资源清理且能再次执行。

该类机制测试不进入 `estimate.sh`，也不用于自动搜索最优 `r`。旧路径回归需移除 `offload_resident_blocks`；生产 baseline 脚本保留事件路径与零常驻，以便比较常驻本身的影响。

## 正确性与真实入口

同一平台 baseline/常驻对比保持 checkpoint、task、输入、seed、shape、steps、dtype、计算算子、权重布局和其他组件 offload 相同。只改变常驻数量及由估测产生的必要配置，不通过降低输入规模或更换算子冒充目标配置通过。

优先比较最终 latent、单 block/中间状态，以及图像像素或解码后的视频帧。同计算路径且可重复时使用 bitwise 比较；存在非确定性时使用有依据的容差并记录误差。只凭视觉接近不能证明正确，视频容器字节差异也不必然代表模型输出差异。

输出不一致时，依次检查实际权重、dtype/scale、单 block、首次去噪，再到最终输出，避免多步扩散放大误差掩盖来源。Qwen 包含文本状态的对照，不能只检查图像分支。

完整接入还应实际运行目标平台的估测、常驻和 baseline 入口，检查生成配置能被推理读取及输出成功。baseline 可独立运行；不要在已有有效测量且条件未变时为了形式反复估测。NPU 缺硬件时完成可做的检查并明确未实测；不能引用 CUDA 通过作为替代。

## 计时与内存

仅在任务要求性能结论时做代表性比较。使用同一设备和运行条件，依次运行两条路径，记录冷/热状态及重复测量结果，不用单次最快值作普遍结论。脚本沿用 `base.sh` 中的 `PROFILING_DEBUG_LEVEL`，当前为 `2`，可输出相应计时。需要调整时在 `base.sh` 中修改，两条路径保持相同级别。

| 模型 | 日志标签 | 当前覆盖范围 |
|---|---|---|
| Wan | `Run Dit every step` | scheduler 的 step_pre、模型推理与 step_post；见 [default_runner.py](../../../../lightx2v/models/runners/default_runner.py) |
| Qwen Image | `infer_main` | 模型推理，包括开启时的正负 CFG 分支与权重搬运；不含 scheduler 前后处理、文本编码和 VAE；见 [qwen_image_runner.py](../../../../lightx2v/models/runners/qwen_image/qwen_image_runner.py) |

分别记录工具估测时间、加载/编码、首步、后续稳定步骤、VAE、端到端；不要把不同范围的标签或单 block 的 `c` 直接比较。首次常驻准备和 kernel 初始化可能进入首步，需单列。自定义异步计时必须等待设备完成；Python enqueue 耗时不能代表实际设备耗时。

报告实际 `N`、`r`、搬运数和常驻索引，以及 denoise 期间 allocated/reserved 峰值。区分额外常驻权重、两个 slots、其他权重、激活和 workspace；`r*block_bytes` 不等于完整显存用量。释放后 reserved 不下降可能是缓存分配器行为，先检查 allocated 与仍存活的引用。

常驻增加显存并减少重复搬运，速度收益依赖输入和硬件。若需证明隐藏了多少传输，使用设备时间线；公式或总速度提升本身不证明每次复制都已完全隐藏。

## 故障定位

| 现象 | 优先检查 |
|---|---|
| 估测很快结束 | 工具只测一个真实 block；每项 3 次预热和 10 次采样；检查输出报告而非要求完整去噪 |
| `r=0` | 报告是否 `d<=c`；这是正常结果；再核对输入规模与 dtype 是否匹配 |
| 常驻后仍慢 | 运行是否用了生成配置、实际索引/搬运次数、首步成本及非 DiT 阶段；总预算不保证局部完全隐藏 |
| 显存不足 | `r`、slot 和激活等内存；当前不自动裁减。需要节省显存时明确调整生成数量，不静默改变任务输入 |
| 修改配置却未生效 | run 读取的是生成配置；修改基础配置后是否重新估测；CLI、环境和独立 checkpoint 路径是否一致 |
| 第二次执行失败或输出随机错误 | slot ready/free、caller 输入/输出存活、释放前同步、CPU 源恢复 |
| 连续布局结果漂移 | 正式布局绑定、dtype/scale 转换和估测/推理布局一致性 |
| NPU 入口报错 | `PLATFORM=ascend_npu`、有效 `AI_DEVICE`、平台依赖、NPU 算子、模型路径；不要自动切 CUDA 冒充验收 |

最终报告记录当前代码版本、实际配置、平台与资产、测试结果和未覆盖范围。历史测得的 `r`、加速比和测试数量只作背景，不写成新机器的保证。
