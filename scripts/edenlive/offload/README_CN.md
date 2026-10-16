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

## Wan2.2 LightVAE TLR decoder

`configs/edenlive/quantized/wan22_5b_few_step_fp8_lightvae.json` 为 5B AR 提供 FP8 + compile + warmup + LightVAE 配置，`num_frames=null`。运行时通过安装的 `lightvae` 包调用 `lightvae.wan22_lightvae_decoder.Decoder`，保留包内 TLR 低秩计算及自动选择的 Triton kernel。模型结构直接读取权重中的 metadata。

LightVAE 是可选依赖。包索引提供发行版时可用 `python -m pip install lightvae`。当前验证环境的索引尚未提供该包，使用本地 LightVAE 工程进行普通 pip 安装：

```bash
python -m pip install /path/to/LightVAE
```

无需把 LightVAE 源码加入 `PYTHONPATH`。配置中指定：

```json
"use_lightvae": true,
"lightvae_path": "/data/nvme6/gushiqiao/models/LightVAE/lightvae-lite-wan22-decoder.safetensors"
```

原版 Wan2.2 VAE 继续编码首帧；LightVAE decoder 使用同一个编码器的 mean/std，将 DiT latent 反归一化后解码。组件加载、lazy load 和 VAE CPU offload 沿用现有流程。推理入口使用 `--model_cls wan22_edenlive_ar`，将 `--config_json` 切换为上述配置即可。

默认仍在 AR 去噪完成后整段解码。逐块解码使用 `configs/edenlive/streaming/` 下的 LightTAE / LightVAE 同步和异步配置，均为 16 FPS、FP8 + compile + warmup、`num_frames=null`。`streaming_vae=true` 开启同步出块，`ar_config.async_vae_decode=true` 将解码放到独立 CUDA stream，与下一块 DiT 重叠。

LightVAE 流式模式需要安装带 `Decoder.new_cache()` 和 `Decoder.forward(..., cache=...)` 的新版包；更新本地工程后重新执行上面的 pip 安装命令即可。跨块保留 TLR 和时间上采样缓存，原版 VAE 继续负责首帧编码。LightTAE 同样保留 MemBlock 历史，只对首块裁掉启动帧。

通过 Python API 接收每个解码完成的块：

```python
def on_chunk(pixels, index, is_last):
    # pixels: GPU tensor, [1, 3, T, H, W]，范围 [-1, 1]；index 从 0 开始。
    # 3 个 latent 的首块为 9 帧，后续完整块为 12 帧。
    print(index, pixels.shape, is_last)

pipe.runner.set_chunk_callback(on_chunk)
pipe.generate(...)  # 正常传入 prompt、image_path、video_path 等参数
```

同步模式在推理线程回调；异步模式由独立工作线程提交 VAE 计算，并在该块 CUDA event 完成后按序回调，主线程可继续提交下一块 DiT。回调应尽快返回，慢速传输可交给自己的队列。推理结束前会等待全部回调，回调异常会向调用方抛出。现有完整视频返回和保存流程仍保留，因此当前还会累计像素块供最终拼接。

`scripts/edenlive/wan22_5b_few_step.sh` 默认选择 LightTAE、16 FPS、同步流式配置。可选设置 `ar_config.overlap_kv_vae=true`，让同一块的 KV commit 和 VAE 解码在独立 CUDA stream 并行；两者完成后在推理线程回调出块，再开始下一块 DINO / DiT。该模式保持 `ar_config.async_vae_decode=false`，不跨块重叠解码。当前 H200 1280×704 单轮实测没有提速，因此默认关闭。
