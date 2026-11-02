# Qwen Image VAE TensorRT 加速

TensorRT 可加速 Qwen Image 的 VAE Encoder 和 Decoder。固定分辨率的 T2I 使用静态引擎；需要适配多种输入尺寸的 I2I 使用多 profile 引擎。

## 准备与启动

1. 完成 [LightX2V 环境安装](../getting_started/quickstart.md)，准备对应的 Qwen Image 模型。
2. 安装与当前 CUDA 环境及引擎构建版本匹配的 TensorRT，并准备预先构建的 `.trt` 文件。设置配置路径不会自动构建引擎。
3. 选择下表中的完整配置，修改 `trt_vae_config.trt_engine_path`，并确认 `attn_type` 与 GPU 和已安装的后端匹配。引擎文件的目录结构见下文。

| 任务 | 完整配置 | 引擎模式 |
| --- | --- | --- |
| Qwen-Image-2512 T2I | [T2I TensorRT 配置](https://github.com/ModelTC/LightX2V/blob/main/configs/qwen_image/qwen_image_t2i_2512_trt.json) | 静态，`multi_profile=false` |
| Qwen-Image-Edit-2511 I2I | [I2I TensorRT 配置](https://github.com/ModelTC/LightX2V/blob/main/configs/qwen_image/qwen_image_i2i_2511_trt.json) | 多 profile，`multi_profile=true` |

在推理环境中确认 TensorRT 可导入：

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

配置完成后，从仓库根目录启动 T2I：

```bash
python -m lightx2v.infer \
  --model_cls qwen_image \
  --task t2i \
  --model_path /path/to/Qwen-Image-2512 \
  --config_json configs/qwen_image/qwen_image_t2i_2512_trt.json \
  --prompt "A cat sitting beside a window." \
  --seed 42 \
  --save_result_path save_results/qwen_trt.png
```

I2I 使用 `--task i2i`、对应的模型和 I2I 配置，并添加 `--image_path /path/to/input.png`。任务类型须通过启动参数指定。查看启动和推理日志，确认加载了 TensorRT 引擎，未回退到 PyTorch。

## 静态引擎目录与配置

目录树按 **宽×高** 标注。静态模式仅匹配预置尺寸，未匹配时回退到 PyTorch VAE。引擎按所用分辨率加载，切换分辨率时释放旧引擎。

*   **引擎目录示例**：`path/to/vae_trt_t2i_static`
*   **目录结构要求**：
    按以下子目录名称存放对应分辨率的引擎文件：

    ```text
    vae_trt_t2i_static/
    ├── 16_9/        # 对应尺寸 1664x928 (宽x高)
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 9_16/        # 对应尺寸 928x1664
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 1_1/         # 对应尺寸 1328x1328
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    ├── 4_3/         # 对应尺寸 1472x1104
    │   ├── vae_encoder.trt
    │   └── vae_decoder.trt
    └── 3_4/         # 对应尺寸 1104x1472
        ├── vae_encoder.trt
        └── vae_decoder.trt
    ```

### 配置片段
将以下字段合并到完整 JSON 配置中，设置 `vae_type` 为 `tensorrt`，并配置对应存放各特定分辨率目录的根目录，且关闭 `multi_profile` 模式：

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "trt_engine_path": "path/to/vae_trt_t2i_static",
        "multi_profile": false
    }
}
```

## 多 profile 引擎目录与配置

以下 profile 按 **高×宽** 标注，引擎的 profile 顺序须与列表一致。
*   **引擎目录示例**：`path/to/vae_trt_extended_mp`
*   **内置包含的 Optimization Profile 档位与尺寸**：
    1.  `1_1_512` (512x512)
    2.  `1_1_1024` (1024x1024)
    3.  `16_9_480p` (480x848)
    4.  `16_9_720p` (720x1280)
    5.  `16_9_1080p` (1088x1920)
    6.  `9_16_720p` (1280x720)
    7.  `9_16_1080p` (1920x1088)
    8.  `4_3_768p` (768x1024)
    9.  `3_2_1080p` (1088x1620)

*(最大支持图片的高/宽上限为 1920 像素)*

### 配置片段
`trt_engine_path` 必须指向包含 `vae_encoder_multi_profile.trt` 和 `vae_decoder_multi_profile.trt` 的目录，不能指向单个引擎文件；同时将 `multi_profile` 设为 `true`：

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "trt_engine_path": "path/to/vae_trt_extended_mp",
        "multi_profile": true
    }
}
```

## 历史性能数据

以下数据未重新测量。静态表中的 4:3 `1472x1140` 和 3:4 `768x1024` 与当前引擎映射不同，不能用这些耗时代表当前对应尺寸的性能。

### 静态引擎（T2I）

#### VAE 独立测试性能 (H100, 纯 VAE 推理)

| 分辨率 | 尺寸 (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc 加速 | PT Dec (ms) | TRT Dec (ms) | Dec 加速 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16:9** | 1664x928 | 66.53 | **32.70** | **2.03x** | 103.65 | **49.66** | **2.09x** |
| **9:16** | 928x1664 | 65.72 | **32.22** | **2.04x** | 103.02 | **50.71** | **2.03x** |
| **1:1** | 1328x1328 | 78.16 | **41.95** | **1.86x** | 121.91 | **61.52** | **1.98x** |
| **4:3** | 1472x1140 | 73.99 | **37.23** | **1.99x** | 114.45 | **54.75** | **2.09x** |
| **3:4** | 768x1024 | 31.74 | **17.33** | **1.83x** | 50.77 | **26.86** | **1.89x** |

> 综合平均加速比：Encoder ~1.95x, Decoder ~2.02x

#### 端到端服务模式性能 (H100, Qwen-Image-2512, 5 step, VAE Decoder)

> 注意：T2I 任务不涉及 VAE Encoder（无输入图像），故仅统计 VAE Decoder。

| 分辨率 | 尺寸 (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec 加速 | 首次加载 (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16:9** | 1664x928 | 189.3 | **88.4** | **2.14x** | 343.9 |
| **9:16** | 928x1664 | 179.6 | **85.6** | **2.10x** | 226.4 |
| **1:1** | 1328x1328 | 157.6 | **106.2** | **1.48x** | 304.1 |
| **4:3** | 1472x1140 | 148.7 | **94.7** | **1.57x** | 238.0 |
| **3:4** | 768x1024 | 70.4 | **46.1** | **1.53x** | 178.2 |

> 综合平均 Decoder 加速比：~1.8x。「首次加载」列为 Lazy Load 切换分辨率时的一次性开销（含引擎反序列化），后续同分辨率请求不再产生。

#### 两组数据差异说明

端到端绝对耗时高于独立测试，主要原因：
1. **后处理开销**：端到端 `decode()` 包含 `image_processor.postprocess()`（tensor → PIL 图片转换），独立测试返回 tensor 无此开销。
2. **GPU 资源争抢**：端到端场景 Transformer (~30GB) + Text Encoder (~15GB) 常驻显存，L2 Cache 被大模型权重覆盖，VAE 推理时需重新从 HBM 读取，增加了显存访问延迟。
3. **显存碎片化**：PyTorch 动态分配器与 TRT 固定 context 交替使用，PyTorch 侧受碎片影响更大（故其绝对差距更显著）。

### 多 profile 引擎（I2I）

#### VAE 独立测试性能 (H100, 纯 VAE 推理, 10 轮平均)

**Encoder**:

| 分辨率 | 尺寸 (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc 加速 |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 11.00 | **8.53** | **1.29x** |
| **1024x1024** | 1024x1024 | 42.85 | **27.56** | **1.55x** |
| **480p 16:9** | 848x480 | 17.25 | **12.00** | **1.44x** |
| **720p 16:9** | 1280x720 | 38.00 | **25.35** | **1.50x** |
| **768p 4:3** | 1024x768 | 31.98 | **21.76** | **1.47x** |

> Encoder 平均加速比：~1.45x

**Decoder**:

| 分辨率 | 尺寸 (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec 加速 |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 17.60 | **12.78** | **1.38x** |
| **1024x1024** | 1024x1024 | 68.16 | **44.93** | **1.52x** |
| **480p 16:9** | 848x480 | 27.67 | **18.85** | **1.47x** |
| **720p 16:9** | 1280x720 | 60.24 | **40.80** | **1.48x** |
| **768p 4:3** | 1024x768 | 51.14 | **34.92** | **1.46x** |

> Decoder 平均加速比：~1.46x。综合 Encoder + Decoder 平均加速比：~1.45x

#### 端到端服务模式性能 (H100, qwen-image-edit-251130, 4 step)

**VAE Encoder**:

| 分辨率 | 尺寸 (WxH) | PT Enc (ms) | TRT Enc (ms) | Enc 加速 |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 48.5 | **28.8** | **1.68x** |
| **1024x1024** | 1024x1024 | 48.2 | **28.4** | **1.70x** |
| **480p 16:9** | 848x480 | 48.7 | **29.6** | **1.64x** |
| **720p 16:9** | 1280x720 | 48.6 | **30.1** | **1.62x** |
| **768p 4:3** | 1024x768 | 49.2 | **29.8** | **1.65x** |

> Encoder 平均加速比：~1.66x

**VAE Decoder**:

| 分辨率 | 尺寸 (WxH) | PT Dec (ms) | TRT Dec (ms) | Dec 加速 |
| :---: | :---: | :---: | :---: | :---: |
| **512x512** | 512x512 | 138.4 | **134.0** | **1.03x** |
| **1024x1024** | 1024x1024 | 152.7 | **133.3** | **1.15x** |
| **480p 16:9** | 848x480 | 140.4 | **134.4** | **1.04x** |
| **720p 16:9** | 1280x720 | 139.0 | **134.2** | **1.04x** |
| **768p 4:3** | 1024x768 | 152.8 | **134.8** | **1.13x** |

> Decoder 平均加速比：~1.08x

#### 两组数据差异说明

**Encoder 端到端加速更好（1.66x > 独立 1.45x）**：
- Qwen I2I 分别用 `CONDITION_IMAGE_SIZE` 和 `VAE_IMAGE_SIZE` 计算视觉语言编码器与 VAE 的尺寸，保持原图比例并对齐到 32 的倍数。分析加速比时，应对照实际 VAE 输入尺寸和 TensorRT profile。
- GPU 资源争抢对 PyTorch 动态内存分配影响更大，进一步拉开差距。

**Decoder 端到端加速大打折扣（1.08x < 独立 1.46x）**：
- `decode()` 中的 `image_processor.postprocess(output_type="pil")` 会执行 tensor → PIL 图片转换（denormalize → clamp → permute → to(uint8) → ToPILImage），这是 **CPU 密集操作**，耗时约 **80-90ms**，TRT 无法加速。
- 独立测试使用 `return_result_tensor=True` 跳过了此开销，端到端服务模式默认 `return_result_tensor=False`。
- 加一个恒定常数必然稀释加速比：以 1024x1024 为例，独立 PT=68ms / TRT=45ms → 1.52x；加 ~85ms 后 PT=153ms / TRT=130ms → 1.18x，与实测 1.15x 吻合。

> **结论**：端到端 Decoder 加速比低不是 TRT 引擎变慢，而是端到端的 vae decoder 计时范围内包含了 TRT 无法加速的 CPU 后处理操作。若需精确衡量 TRT 内核加速效果，应参考独立测试数据。
