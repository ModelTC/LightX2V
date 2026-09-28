# 验证与结果说明

简体中文 | [English](validation_en.md)

根据本次改动选择检查，不为文档或简单脚本修改无条件启动完整模型。更改存储、复制或调度时，先验证数据与生命周期，再做生成对照。下述命令从仓库根目录执行，使用匹配目标设备的 Python 环境。

## 入口与有效配置

核对脚本实际传入的模型、配置、任务、路径和环境变量。`scripts/base/base.sh` 可能覆盖调用前设置的环境，尤其是调试计时；读取其当前内容，必要时在 source 后明确设置。当前 Qwen layout 脚本在 source 后设置 BF16、敏感层跟随主精度和 `PROFILING_DEBUG_LEVEL=0`。

脚本语法、JSON 解析以及用参数捕获替代 Python 推理进程，可以验证命令拼接、含空格路径、工作目录独立性和环境覆盖。它们不代表模型执行成功，不为这些低影响检查新增长期测试框架。

同平台的两个配置应仅在布局开关上有必要差异。若模型权重、精度、输入或算子不同，先恢复可比条件，或明确此实验不能归因于连续布局。

## 存储与传输

复用现有测试，并补充本次改动涉及的行为：

| 文件 | 已有关注点 |
|---|---|
| [test_block_buffer.py](../../../../test_cases/test_block_buffer.py) | 外部 storage 边界、地址对齐、转置 views、pin、owner 存活、异步双缓冲与辅助状态 |
| [test_block_offload_groups.py](../../../../test_cases/test_block_offload_groups.py) | Wan/Qwen 加载与 baseline 对照、普通调度、多 group、checkpoint 原始 dtype、未描述状态、清理及平台导入隔离 |
| [test_platform_norm_storage.py](../../../../test_cases/test_platform_norm_storage.py) | 浮点 Norm 模板继承、无参数状态、精度及 slot 复用 |
| [test_platform_quant_storage.py](../../../../test_cases/test_platform_quant_storage.py) | 逐通道量化模板、scale/bias 精度、转置方向、checkpoint 消费、双缓冲与平台原生 MM 对照 |
| [test_platform_weight_copy.py](../../../../test_cases/test_platform_weight_copy.py) | 平台复制接口、转置 host view、多轮往返及相邻内存保护 |

CUDA 上的现有测试入口：

```bash
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 \
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
python -m pytest -q \
  test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py \
  test_cases/test_platform_norm_storage.py test_cases/test_platform_quant_storage.py \
  test_cases/test_platform_weight_copy.py
```

NPU 上先核对测试 fixture 的平台算子和 skip 条件，再用 `PLATFORM=ascend_npu`、`ASCEND_RT_VISIBLE_DEVICES=0` 运行适用用例。算子 storage contract 在 CUDA 上的验证要标为替代设备测试；只测 projection 或复制的用例不代表执行过完整 NPU block。

MLU 上将命令首行替换为 `PLATFORM=cambricon_mlu MLU_VISIBLE_DEVICES=0 \`；MUSA 上替换为 `PLATFORM=musa MUSA_VISIBLE_DEVICES=0 \`，各自在已配置好厂商运行时的环境中执行。测试按平台注册项选择设备；MUSA 沿用现有 torchada 兼容路径。量化用例只在算子所属平台执行其原生内核；CUDA 可替代验证这些算子的存储和复制，其余平台跳过非本平台算子的设备用例，避免要求无关量化精度支持。不能把替代设备的通过结果当成 MLU/MUSA 原生计算已经通过。

关键不变量：

- 每个计划中的 CPU tensor 属于目标 block 的 storage，地址与 offset、dtype、shape、stride 一致且已 pin；越界或未消费项明确失败。
- baseline tensor 可能因分配器碰巧相邻。判断连续布局看共同 storage 与完整布局，不能只看相邻指针，也不能把单个 tensor 的 `is_contiguous()` 当成整个 block 连续。
- 两个设备 slots 重复 H2D 后值正确、地址稳定，辅助状态跟随对应 block；主 CPU 源在推理期间不改变。
- 覆盖首块、末块、回绕、下一 step、CFG 分支，以及适用的连续请求和异构 group 切换。
- 错误 checkpoint 精度、shape、前缀遗漏/越界、重复所有权及不支持算子在消费权重前失败；计划完成后的绑定不能逃离预分配 storage。
- 所有 group 的 transfer 均正确关闭，不能只清理当前选中的 group。

涉及 NPU CPU 目标拷贝时单独检查连续和转置/非连续 host 目标，多轮往返后与原始权重比较。数据移动预期保持逐位一致；不要用放宽数值容差掩盖复制损坏。

## 真实推理对照

在同一设备和环境中，固定 checkpoint、dtype、seed、输入、分辨率/帧数、采样步数、CFG、算子和组件 offload。分别使用正式 baseline 与 contiguous 路径，不用替代 kernel 或自制推理循环作为最终验收。

1. 先确认 baseline 可正常生成。两者都有异常时，排查共同的加载、计算和 VAE 路径。
2. 用适量真实 blocks/steps 做 smoke test，发现数据与调度问题后再运行目标规模。
3. 比较适用的中间/最终 latents 和解码输出。确定性路径优先精确对照；确有非确定性时，先用同一 baseline 重复运行界定波动，再解释容差与质量判断。
4. 修改公共或平台分派代码后保留 CUDA 默认行为的回归证据。不能只证明新的 NPU 配置可以实例化。

每个平台单独报告“未验证、存储测试通过、部分真实推理通过、完整生成通过”等状态及具体范围。示例记录应包含代码版本（有未提交改动时附 diff 或校验信息）、设备、软件版本、配置、checkpoint、输入、执行范围和结果位置。不要固定测试通过数量；整套用例被 skip 不等于验证通过。

历史通过记录只证明当时的版本。没有 NPU 就明确说明没有 910B 实机结果；给出目标机器可执行命令和待检查输出，不伪造通过状态，也不把当前限制改写为永久不支持。

## 性能与内存，仅按需测量

正确性通过后再分析收益。使用匹配条件的 warmup 和多次测量，分别报告端到端、denoise、H2D 提交时间与设备传输区间；标明 profiler 是否启用及其扰动。

- 连续方案没有推理时 pack，不能为比较方便人为加回 staging 整合。
- 设备 event 区间可能包含 host 提交空隙或等待，不等同于纯 DMA 时间；嵌套、重叠的计算、传输和 host wait 不能直接相加。
- 字节量按实际复制计数，区分 block payload、padding、辅助状态和 CFG 多次遍历；`GB/s` 注明十进制或二进制单位。
- 分开记录 CPU checkpoint 加载峰值、常驻 pinned 存储、设备 slots 和其他组件内存。减少独立分配不等于减少主体权重字节量。
- CPU pack 后缓存状态只是特定实验的待验证因素，不能从 H2D 变慢直接断言 DRAM 未更新、缓存一致性错误或固定硬件瓶颈。

计时代码不自动进入正式推理路径，不恢复用户已删除的 benchmark 基础设施。根据用户要求保存必要证据，并区分测量事实和原因推测。
