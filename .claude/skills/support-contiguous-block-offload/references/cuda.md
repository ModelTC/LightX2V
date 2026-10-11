# NVIDIA GPU

简体中文 | [English](cuda_en.md)

## 默认行为与平台隔离

CUDA 路径由 [offload/__init__.py](../../../../lightx2v_platform/ops/offload/__init__.py) 的 `_DEFAULT_BLOCK_OFFLOAD_BACKENDS` 按平台注册名 `cuda` 选择 `TorchBlockOffload`，无需修改 [CudaDevice](../../../../lightx2v_platform/base/nvidia.py)。平台类显式声明的 `block_offload_backend` 优先于默认映射。该 backend 在 CPU 分配 pinned storage，在设备分配字节 buffer，并在当前 stream 提交复制。沿用公共布局和 manager，不另建 CUDA 模型 loader。

修改前记录目标模型未开启连续布局时的配置和关键输出。保留 CUDA 默认算子、checkpoint 转换、精度、shape 和调度；改动公共代码后验证 per-tensor baseline 与 contiguous 两条路径。

- NVIDIA 运行不能依赖 `torch_npu`、CANN 或 Ascend 算子注册成功。平台模块按既有平台注册入口加载。
- 仅所选算子需要的依赖放入对应加载分支。参考 [QwenImageTransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/transformer_infer.py)：`modulate_type=triton` 才导入 Qwen Triton 调制内核，默认选择保持原样。
- 不因 NPU 适配而把 CUDA 默认 attention、RoPE 或 norm 全部换成通用 PyTorch 实现。
- 不因为其他芯片报告 `device.type == "cuda"` 或继承 `CudaDevice` 就启用该能力；默认映射仅匹配平台注册名 `cuda`，其他平台仍需声明支持。保持 `nvidia.py` 原有设备定义，不向其中添加 offload 后端属性。

## 配置与权重

入口使用目标 NVIDIA 设备，必要时显式设置 `PLATFORM=cuda`，避免从 NPU shell 继承错误平台。CUDA 示例使用 `CUDA_VISIBLE_DEVICES`；不要同时把 NPU 环境选择写入同一个入口。

对比时固定主 dtype、敏感层 dtype 和量化方案。当前 Wan CUDA 示例是 FP8-vLLM 权重配 BF16 计算，Qwen 示例是原始 BF16。两者仅是 [模型案例](model-examples.md)，不能据此将所有 CUDA 权重限制为 FP8 或 BF16。

新增量化方案前检查算子的 `describe_storage()` 与加载行为是否一致，尤其是 transpose、scale、bias 精度和设备辅助状态。已有量化计算内核不等于已有连续布局存储契约。

baseline 与 contiguous 使用同一平台算子。记录实际传输的是哪些权重及其字节量，不把全模型参数量当成每步 H2D 字节量。

## 验证重点

按 [验证流程](validation.md) 选择相关检查，特别关注：

1. CUDA import isolation：禁止 Ascend 算子模块导入，仍能初始化公共加载和模型权重类。
2. 两个 slot 的地址稳定、CPU 源保持 pinned；异步复制后按必要同步检查值和 stride。
3. 各模型的 manager 独立使用自己的 group，拒绝重复注册或错误 blocks，清理时关闭各自 transfer。
4. 同配置的 baseline/contiguous 生成对照，以及改动前后 CUDA 默认计算的回归。

仅用小矩阵检查权重复制，不能证明完整 attention、RoPE、文本编码器和 VAE 正确；缩短步数或分辨率的 smoke test 要注明范围。NPU 改动若触及共用函数，CUDA 的上述检查仍需按影响范围覆盖。

修改、替换 stream 或 event 协议时验证依赖和重叠，不通过删除同步或增加全设备同步来掩盖错误。性能测量只在用户需要时执行。
