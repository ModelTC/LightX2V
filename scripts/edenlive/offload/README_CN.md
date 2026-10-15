# EdenLive block offload、量化和预热

编辑 `run_edenlive_block_offload.sh` 的代码、模型和输入路径，然后从仓库根目录运行。默认入口是 Wan2.1 14B AR I2V；输入为首帧图片、录屏视频和文字。`model_path` 指向完整的 EdenLive-x2v 目录，T5、CLIP、DINO、VAE 继续使用该目录的默认组件。

`configs/edenlive/offload/` 为四种模型提供 BF16 配置，`configs/edenlive/quantized/` 为四种模型分别提供 FP8 配置。切换模型时一起修改 `--model_cls` 和 config；`few_step` 对应 `_ar`，`bidir` 对应整段双向推理。

| 配置 | 权重与计算 |
| --- | --- |
| BF16 block | CPU 保存 DiT blocks，两个 GPU staging blocks 交替预取与计算 |
| `fp8-sgl` | 单文件 FP8 权重，使用现有 SGL FP8 MM |
| `int8-convrot` | 单文件 INT8 权重、scale 和旋转组大小，使用 comfy-kitchen ConvRot MM |

这些示例使用私有 CPU 权重，不启用 `shared_cpu_weights`。编码器和 VAE 的 offload 沿用 Wan 默认实现。INT8 ConvRot 转换和推理需要 `comfy-kitchen`，转换器还使用仓库已有的 `qtorch`。

## 转换

先完成原始 EdenLive 到 native EdenLive-x2v 的转换。然后量化所选 native DiT；量化模式始终输出一个 safetensors 文件：

```bash
python tools/convert/convert_edenlive.py \
  --source /data/nvme3/gushiqiao/models/EdenLive-x2v/dit-14b \
  --output /data/nvme2/gushiqiao/models/dit-quant \
  --quantize fp8 --output_name dit-14b-fp8 --device cuda:0

python tools/convert/convert_edenlive.py \
  --source /data/nvme3/gushiqiao/models/EdenLive-x2v/dit-14b \
  --output /data/nvme2/gushiqiao/models/dit-quant \
  --quantize int8-convrot --output_name dit-14b-int8-convrot --device cuda:0
```

`dit-14b` 和 `dit-5b` 是 AR 权重；双向模型使用各自的 `dit-14b-bidir` 和 `dit-5b-bidir`。转换其他变体时同时修改 source 和输出名称。已有输出默认保留，显式传 `--overwrite` 才覆盖。推理 config 的 `dit_quantized_ckpt` 是生成的单文件绝对路径，架构参数仍从 `model_path` 的 DiT config 读取。

## 开关与生命周期

- `cpu_offload=true`、`offload_granularity="block"`：复用 Wan 私有 CPU block offload；也可单独使用现有 model offload。
- `ar_config.async_vae_decode=true`：AR 每个 chunk 提交缓存式 VAE 解码，VAE 驻留 GPU，结束后恢复原 offload 状态。同步模式保持整段解码。
- `use_compile=true`：复用公共 block dispatcher，H2D、prefetch 和 swap 留在图外。双向 block offload 按 staging 对象缓存两个 callable；AR 按逻辑层缓存，KV 管理及自回归注意力留在 eager 边界。
- `warmup=true`：首个请求前执行一次真实编码、Step 0、AR clean-cache commit 和解码。5B 固定覆盖 512×512、704×1280；14B 固定覆盖 480×480、480×832。帧数取 config；未指定时 AR 覆盖 attention window、一个额外 chunk 和尾帧。

warmup 清空 generator、solver、KV、prompt 和临时输入，保留编译 callable 与 kernel cache。正式请求使用自己的 seed。新的分辨率、chunk 长度或动态 prompt 布局可能产生新的 compile signature；预热不能消除模型上下卡或 block 权重复制成本。

phase offload 未接入。公共 warmup gate 仍拒绝 disagg、unload_modules 和 feature caching。默认配置启用 compile、warmup，AR 还启用异步解码；可分别关闭这些开关做对照。`scripts/base/base.sh` 默认 BF16，外部设置的环境变量仍有优先级。

## 验证范围

| 路径 | 功能验证 |
| --- | --- |
| BF16 AR block + compile + warmup + async VAE | 5B 704×1280、14B 480×832，121 帧，覆盖 KV 滚动窗口和尾 chunk |
| BF16 AR normal / model offload + compile + warmup | 5B 704×1280，25 帧 |
| BF16 双向 block + compile + warmup | 5B 704×1280、14B 480×832，各 25 帧 |
| 同步 / 异步 VAE 输出 | 5B、14B 各 25 帧，解码视频逐像素一致 |
| eager block 预热前后输出 | 5B 704×1280，25 帧，解码视频逐像素一致 |
| 单文件 FP8 + normal / block + eager / compile | 小规模 native DiT 转换和真实算子推理测试通过；完整量化模型未转换 |
| INT8 ConvRot | 已提供配置和转换入口；当前环境未安装依赖，尚未实测 |
| lazy block / SP / TP / LoRA 组合 | 本轮未验证 |

`test_cases/test_edenlive_warmup.py` 和 `test_cases/test_edenlive_quantized.py` 共 13 项通过，4 项 ConvRot 测试因缺少依赖跳过。视频测试使用仓库 `cryo-crossing` 输入；compile 和不同组件 offload 路径允许 BF16 数值差异。5B 25 帧同模式 eager/compile 对比的解码 RGB 平均绝对差约 4/255，PSNR 约 30.94 dB。以上是功能验证结果。
