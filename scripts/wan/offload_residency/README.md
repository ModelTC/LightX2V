# Wan 静态常驻 block

先运行独立工具，再运行推理。工具只读取一个真实 block，用目标形状的模拟输入测量，不加载 T5、CLIP、VAE 或完整 DiT。

```bash
bash scripts/wan/offload_residency/estimate.sh
bash scripts/wan/offload_residency/run.sh
bash scripts/wan/offload_residency/run_baseline.sh
```

先修改脚本中的项目和模型路径，以及 `configs/wan/residency/baseline.json` 中的 checkpoint 路径。NPU 使用对应的 `_npu.sh` 和 `baseline_npu.json`。

工具分别对计算、H2D 做 3 次预热、10 次有效采样，取中位数 `c`、`d`，直接计算：

```text
r = max(0, ceil(N * (1 - c / d)))
resident_indices = [k * N // r for k in range(r)]
```

`r=0` 表示所有 block 从 CPU 按需搬运，GPU 仍保留两个执行 buffer。`r=N` 表示全部常驻，当前加载器持有的两个 buffer 仍计入显存。常驻层保留 CPU 原始权重，在一次去噪任务内复用 GPU 权重。正常推理和预热都会在去噪结束、进入 VAE 解码前释放 GPU 常驻副本，异常退出时同样释放。

默认输出：

- `save_results/wan_residency/resident.json`：完整推理配置，含 `offload_resident_blocks`。
- `save_results/wan_residency/resident.profile.json`：测量数据、形状、常驻索引及权重显存估计，推理不读取此文件。

工具不搜索候选、不验证重叠、不缓存测量。公式是搬运总量估计，不保证双 buffer 在每个位置都没有等待。推理严格使用配置中的整数 `r`，不自动调整。

换设备、量化、attention 实现、输入帧数或尺寸后重新运行工具。I2V 工具与推理应使用同一输入图片，以保证实际宽高比相同。`cpu_offload_layout` 可以选择 `per_tensor` 或 `contiguous`；工具按同一布局测搬运，公式与布局无关。

也可直接运行工具：

```bash
python -m tools.estimate_offload_residency \
  --task i2v \
  --model-path models/Wan2.1-I2V-14B-720P-Lightx2v \
  --config configs/wan/residency/baseline.json \
  --image-path assets/inputs/imgs/img_0.jpg \
  --output-config save_results/wan_residency/resident.json
```

省略 `--output-config` 时，写到输入配置旁的 `*.resident.json`。若明确指定与输入相同的路径，则原子替换该配置。T2V 使用 `--task t2v` 和对应 checkpoint、推理配置，无需 `--image-path`。

当前接入范围：Wan2.1 标准 I2V/T2V、单设备、固定权重、eager block offload。未接入 Wan2.2 双模型切换、adapter、动态 LoRA、分布式和特征缓存路径；遇到这些组合会报错，不会忽略 `r`。Flux 保持现有实现。

开发验证：

```bash
python -m pytest -q test_cases/test_wan_static_residency.py
python -m test_cases.check_wan_static_residency \
  --model-path models/Wan2.1-I2V-14B-720P-Lightx2v \
  --config configs/wan/residency/baseline.json \
  --image-path assets/inputs/imgs/img_0.jpg \
  --layout per_tensor \
  --output save_results/wan_residency/check.json
```

第二条命令加载完整 DiT，用原 offload 输出校验多个常驻数量；它只用于开发检查，不参与估算工具或正式推理。
