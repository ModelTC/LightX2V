# Diffusion Model 分离部署指南

对于大规模生成模型（如 Wan、Qwen Image 等），Text Encoder、Image Encoder 以及 VAE 编解码器往往常驻显存，极大挤压核心 DiT 模型的可用空间，在高分辨率、长时生成场景下容易导致 OOM。

LightX2V 提供了原生的 **Disaggregation Mode（分离部署模式）**，通过高性能 **Mooncake 传输引擎**，将推理流水线拆分为 **Encoder、Transformer、Decoder** 三段，分别部署在不同显卡或节点上，支持 RDMA / TCP 通信。**Wan** 与 **Qwen Image** 均已支持完整三段式部署（含 VAE Decoder 独立节点）。分离部署模式支持各段并发处理不同请求，从而提升多请求吞吐。

---

## 方案对比


| 特性       | **Baseline（常规单机部署）** | **Disagg Mode（分离微服务部署）**                                                 |
| -------- | -------------------- | ------------------------------------------------------------------------ |
| **部署架构** | 所有模型聚合在同一进程中         | 三段式：**Encoder 节点** → **Transformer 节点** → **Decoder 节点**（VAE 解码独立）       |
| **显存占用** | 极高                   | **按需分配**（各节点只加载自身所需部分，Decoder 仅 VAE Decoder）                             |
| **通信底层** | 进程内原生 Tensor 共享      | **Mooncake 引擎**（Phase1: Encoder→Transformer；Phase2: Transformer→Decoder） |
| **适用场景** | 显存充裕的单机环境、快速验证       | **显存受限**、长帧视频、多机分布式高并发生产环境                                               |


---

## Quick Start

```bash
# 安装mooncake engine
pip install mooncake-transfer-engine
```

#### Wan2.1-T2V-14B（50 Steps, 480×832, 81 帧）

```bash
# 指定各阶段部署的卡
# GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0

# 启动三段式分离部署服务
bash scripts/server/disagg/wan/start_wan_t2v_disagg.sh

# 成功启动后测试
python scripts/server/disagg/wan/post_wan_t2v.py
```

#### Wan2.1-I2V-14B-480P（40 Steps, 480×832, 81 帧）

```bash
# 指定各阶段部署的卡
# GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0

# 启动三段式分离部署服务
bash scripts/server/disagg/wan/start_wan_i2v_disagg.sh

# 成功启动后测试
python scripts/server/disagg/wan/post_wan_i2v.py
```

### qwen-image-edit-release-251130（t2i）

```bash
# 指定各阶段部署的卡
# GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0

# 启动三段式分离部署服务
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg.sh

# 成功启动后测试
python scripts/server/disagg/qwen/post_qwen_t2i.py
```

### qwen-image-edit-release-251130（i2i）

```bash
# 指定各阶段部署的卡
# GPU_ENCODER=0 GPU_TRANSFORMER=1 GPU_DECODER=0

# 启动三段式分离部署服务
bash scripts/server/disagg/qwen/start_qwen_i2i_disagg.sh

# 成功启动后测试
python scripts/server/disagg/qwen/post_qwen_i2i.py
```

## 性能实测 Benchmark

测试环境：`sage_attn2`，`BF16`，`PROFILING_DEBUG_LEVEL=2`，Mooncake RDMA 协议。
Baseline 使用单张 GPU 加载全部模型；Disagg Mode 可将 Encoder、Transformer、Decoder 分别部署在不同 GPU 上。

### RTX 4090 24GB（显存/PCIe 受限场景）

该组数据用于说明 **显存受限显卡**（例如 4090 24GB）上，Baseline 往往必须依赖 `cpu_offload=block` 才能跑通；而 Disagg 将 Encoder/DiT/VAE Decoder 拆分到不同进程/显卡后，可以显著降低单卡常驻显存压力，从而降低 offload 总体latency并改善延迟与吞吐。

#### Wan2.1-I2V-14B（BF16，40 steps）


| 指标            | Baseline（1×4090，block offload）      | Disagg（2×4090，block offload）        | 备注                               |
| ------------- | ----------------------------------- | ----------------------------------- | -------------------------------- |
| 峰值显存          | 480P：9GB；720P：14GB                  | —                                   | Baseline 依赖 offload 才能跑通         |
| DiT 单步        | 480P：24.24 s/step；720P：90.71 s/step | 480P：19.02 s/step；720P：62.80 s/step | 4090 上 DiT 更容易被显存/PCIe 限制        |
| Image Encoder | 480P：0.57s；720P：0.52s               | 480P：0.28s；720P：0.20s               | Disagg 资源解耦后 Encoder 更稳定         |
| VAE Encoder   | 480P：3.02s；720P：6.83s               | 480P：3.22s；720P：7.36s               | 受分辨率影响显著                         |
| Text Encoder  | 480P：2.14s；720P：1.90s               | 480P：0.20s；720P：0.20s               | 4090 上 Baseline 受 offload/调度影响更大 |
| Encoder 总计    | 480P：6.09s；720P：9.78s               | 480P：4.08s；720P：8.32s               |                                  |


#### Wan2.1-I2V-14B（INT8）


| 指标            | Baseline（block offload）   | Disagg（no offload） | Disagg（block offload） |
| ------------- | ------------------------- | ------------------ | --------------------- |
| 说明            | `use_offload=false` 会 OOM | —                  | —                     |
| DiT 单步        | 12.94 s/step              | 12.55 s/step       | 12.91 s/step          |
| Image Encoder | 0.79s                     | 0.19s              | 0.19s                 |
| VAE Encoder   | 3.11s                     | 2.93s              | 2.93s                 |
| Text Encoder  | 1.67s                     | 0.20s              | 0.20s                 |
| Encoder 总计    | 5.90s                     | 3.69s              | 3.70s                 |


#### 4090 并发压测（BF16 + block offload，Wan2.1 480P 4 steps Distill端到端压测）

该组并发数据用于观察在**多请求排队/资源争用**下，Disagg 是否更容易提升 QPS 与尾延迟。


| N(concurrency) | mode     | ok/total | QPS    | P50(s) | P95(s) | P99(s) | note              |
| -------------- | -------- | -------- | ------ | ------ | ------ | ------ | ----------------- |
| 1              | baseline | 1        | 0.0091 | 109.45 | 109.45 | 109.45 | wan2.1 480P 4step |
| 2              | baseline | 2        | 0.0092 | 162.52 | 211.01 | 215.32 | wan2.1 480P 4step |
| 4              | baseline | 4        | 0.0093 | 269.85 | 414.94 | 427.83 | wan2.1 480P 4step |
| 8              | baseline | 8        | 0.0093 | 485.82 | 824.24 | 854.32 | wan2.1 480P 4step |
| 1              | disagg   | 1        | 0.0117 | 85.38  | 85.38  | 85.38  | wan2.1 480P 4step |
| 2              | disagg   | 2        | 0.0122 | 125.83 | 160.18 | 163.23 | wan2.1 480P 4step |
| 4              | disagg   | 4        | 0.0126 | 201.85 | 305.73 | 314.94 | wan2.1 480P 4step |
| 8              | disagg   | 8        | 0.0129 | 358.20 | 595.12 | 616.48 | wan2.1 480P 4step |


#### 4090 并发压测（Qwen Image 2512，T2I，5 steps Distill端到端压测）


| N(concurrency) | mode     | ok/total | QPS    | P50(s) | P95(s) | P99(s) | note                  |
| -------------- | -------- | -------- | ------ | ------ | ------ | ------ | --------------------- |
| 1              | baseline | 1        | 0.0207 | 48.30  | 48.30  | 48.30  | qwen-image-2512 5step |
| 2              | baseline | 2        | 0.0207 | 72.57  | 94.31  | 94.24  | qwen-image-2512 5step |
| 4              | baseline | 4        | 0.0212 | 118.77 | 181.44 | 187.33 | qwen-image-2512 5step |
| 8              | baseline | 8        | 0.0216 | 208.64 | 354.94 | 367.90 | qwen-image-2512 5step |
| 1              | disagg   | 1        | 0.0452 | 22.11  | 22.11  | 22.11  | qwen-image-2512 5step |
| 2              | disagg   | 2        | 0.0510 | 29.68  | 38.25  | 39.02  | qwen-image-2512 5step |
| 4              | disagg   | 4        | 0.0528 | 48.52  | 73.00  | 75.17  | qwen-image-2512 5step |
| 8              | disagg   | 8        | 0.0534 | 85.78  | 143.37 | 148.62 | qwen-image-2512 5step |


### NVIDIA H100 SXM5 80 GB

### Wan2.1-T2V-1.3B（小模型参考）

测试条件：Baseline 混合部署在 `cuda:0`；Disagg 模式将 Transformer 部署于 `cuda:0`，Encoder 挂载于 `cuda:1`。


| 测试指标（T2V 50 Steps）  | **Baseline（单卡）**     | **Disagg Mode（双卡）**            | 备注                       |
| ------------------- | -------------------- | ------------------------------ | ------------------------ |
| **峰值显存**            | ~16.4 GB (16823 MiB) | Enc: ~11.3 GB / Trans: ~9.2 GB | Transformer 显存压缩至 9.2 GB |
| **Text Encoder 耗时** | ~0.282 s             | ~0.294 s                       | 通过 RDMA 几乎无损             |
| **DiT 单步延迟**        | 0.945 s/step         | 0.882 s/step                   | 独占 VRAM 带宽，加速 ~6.6%      |
| **VAE Decoder 耗时**  | 2.360 s              | 2.434 s                        | 基本不受网络影响                 |
| **端到端总耗时**          | **52.46 s**          | **49.27 s**                    | Disagg 缩减约 3 s           |


### Wan2.1-14B（H100 单卡 Baseline vs 双卡 Disagg 对比）

#### Wan2.1-T2V-14B（50 Steps, 480×832, 81 帧）


| 阶段                       | Baseline（单卡）          | Disagg Encoder | Disagg Transformer    |
| ------------------------ | --------------------- | -------------- | --------------------- |
| **Encoder Pipeline 总耗时** | **0.84 s**            | **0.89 s**     | —                     |
| DiT 推理（50 steps）         | 252.7 s（~4.95 s/step） | —              | 252.8 s（~4.95 s/step） |
| VAE Decoder              | 2.25 s                | —              | 2.38 s                |
| **端到端 Pipeline 总耗时**     | **253.2 s**           | —              | **253.5 s**           |


**分析**：两种模式的端到端延迟几乎持平。Disagg 能支持 Encoder 与 Transformer 并发处理不同请求，提升多请求吞吐。

#### Wan2.1-I2V-14B-480P（40 Steps, 480×832, 81 帧）


| 阶段                           | Baseline（单卡）          | Disagg Encoder   | Disagg Transformer    |
| ---------------------------- | --------------------- | ---------------- | --------------------- |
| Image Encoder（CLIP ViT-H/14） | 0.68 s                | **0.29 s** ↓ 57% | —                     |
| VAE Encoder                  | 1.90 s                | **1.35 s** ↓ 29% | —                     |
| Text Encoder（T5）             | 0.17 s                | 0.18 s           | —                     |
| **Encoder Pipeline 总耗时**     | 4.75 s                | **3.17 s** ↓ 33% | —                     |
| DiT 推理（40 steps）             | 208.2 s（~5.06 s/step） | —                | 207.7 s（~5.06 s/step） |
| VAE Decoder                  | 2.03 s                | —                | 2.22 s                |
| **端到端 Pipeline 总耗时**         | **211.6 s**           | —                | **210.9 s**           |


**分析**：I2V Disagg 模式下 端到端延迟两者持平。

#### 显存占用对比


| 模型                  | 部署模式     | 所需 GPU 数 | 各卡显存峰值                                |
| ------------------- | -------- | -------- | ------------------------------------- |
| Wan2.1-T2V-14B      | Baseline | 1×       | ~39 GB                                |
| Wan2.1-T2V-14B      | Disagg   | 2×       | Encoder: ~11 GB / Transformer: ~28 GB |
| Wan2.1-I2V-14B-480P | Baseline | 1×       | ~48 GB                                |
| Wan2.1-I2V-14B-480P | Disagg   | 2×       | Encoder: ~13 GB / Transformer: ~32 GB |


### qwen-image-edit-release-251130

Qwen Image 在 H100 上对 T2I/I2I 任务的测试，Disagg 模式 Text Encoder 使用 `lightllm_kernel` 加速。


| 任务      | 部署模式            | Text Encoder (s) | VAE Encoder (s) | DiT Total (s)    | 端到端 (s)   |
| ------- | --------------- | ---------------- | --------------- | ---------------- | --------- |
| **T2I** | Baseline        | 0.430            | N/A             | 23.24 (50 steps) | 25.50     |
| **T2I** | Disagg (Kernel) | **0.300**        | N/A             | 22.04            | **25.05** |
| **I2I** | Baseline        | 0.924            | 0.844           | 33.81 (40 steps) | 38.83     |
| **I2I** | Disagg (Kernel) | **0.746**        | **0.137**       | 31.39            | **33.90** |


**核心发现：**

1. **Text Encoder 加速**：`lightllm_kernel` 使 T2I Text Encoder 耗时从 0.43s 降至 0.30s（加速 ~30%），I2I 从 0.92s 降至 0.75s（加速 ~19%）。
2. **VAE Encoder 差异**：Disagg 模式下 VAE Encoder 表现出显著加速，源于 GPU 资源独占带来的调度优化。
3. **计算资源解耦**：Text Encoder/VAE 与 DiT 分离至不同 GPU，端到端延迟显著降低（I2I 降低约 5 秒）。

---

## 1. Disagg 分离架构解析

通过配置参数 `disagg_mode`，推理 Pipeline 被物理拆分为 **三段式** 独立服务，数据流经 **Phase1（Encoder → Transformer）** 与 **Phase2（Transformer → Decoder）** 两次 Mooncake 传输：

- **Encoder 角色（`disagg_mode="encoder"`）**：
  - 仅加载 Text Encoder、Image Encoder（I2V / I2I 时）以及 VAE Encoder，**跳过 DiT 与 VAE Decoder**。
  - 执行特征提取，将 `context`、`clip_encoder_out`、`vae_encoder_out`、`latent_shape` 等通过 Mooncake **Phase1** 投递给 Transformer 节点。
- **Transformer 角色（`disagg_mode="transformer"`）**：
  - 仅加载 DiT 模型，**跳过 Encoder 与 VAE Decoder**（三段式下由 Decoder 节点承担解码）。
  - 启动后等待 Phase1 数据，收到后执行哈希校验、拼装输入并完成去噪；若配置了 `decoder_engine_rank`，将去噪后的潜空间通过 Mooncake **Phase2** 发送给 Decoder 节点，**不本地做 VAE 解码**。
- **Decoder 角色（`disagg_mode="decode"`）**：
  - 仅加载 **VAE Decoder**，**跳过 Text/Image Encoder 与 DiT**。
  - 启动后进入 Phase2 接收等待状态，收到 Transformer 发来的潜空间后执行 VAE 解码并保存输出视频/图像，**任务完成状态与结果文件均落在 Decoder 节点**。

---

## 2. 配置方法

所有分离部署参数统一在 config json 的 `disagg_config` 字段中配置。

### 2.1 T2V 配置示例

**Encoder 端（`configs/disagg/wan/wan_t2v_disagg_encoder.json`）**：

```json
{
    "infer_steps": 50,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 480,
    "target_width": 832,
    "self_attn_1_type": "sage_attn2",
    "cross_attn_1_type": "sage_attn2",
    "cross_attn_2_type": "sage_attn2",
    "sample_guide_scale": 5,
    "sample_shift": 5,
    "enable_cfg": true,
    "cpu_offload": false,
    "fps": 16,
    "disagg_mode": "encoder",
    "disagg_config": {
        "bootstrap_addr": "127.0.0.1",
        "bootstrap_room": 0,
        "sender_engine_rank": 0,
        "receiver_engine_rank": 1,
        "protocol": "rdma",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE"
    }
}
```

**Transformer 端（`configs/disagg/wan/wan_t2v_disagg_transformer.json`）**：将 `disagg_mode` 改为 `"transformer"`，并增加 Phase2 相关配置，用于向 Decoder 发送潜空间：

```json
{
    "disagg_mode": "transformer",
    "disagg_config": {
        "bootstrap_addr": "127.0.0.1",
        "bootstrap_room": 0,
        "sender_engine_rank": 0,
        "receiver_engine_rank": 1,
        "protocol": "rdma",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE",
        "decoder_engine_rank": 2,
        "decoder_bootstrap_room": 1
    }
}
```

**Decoder 端（`configs/disagg/wan/wan_t2v_disagg_decode.json`）**：仅加载 VAE Decoder，Phase2 接收端；`bootstrap_room` 需与 Transformer 的 `decoder_bootstrap_room` 一致：

```json
{
    "disagg_mode": "decode",
    "disagg_config": {
        "bootstrap_addr": "127.0.0.1",
        "bootstrap_room": 1,
        "sender_engine_rank": 1,
        "receiver_engine_rank": 2,
        "protocol": "rdma",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE"
    }
}
```

### 2.2 I2V 配置示例

I2V 使用 **ViT-H/14** CLIP 图像编码器，其输出为完整序列特征（257 个 token × 1280 维），因此**必须额外指定 `clip_embed_dim`** 以保证 RDMA buffer 正确分配：

**Encoder 端（`configs/disagg/wan/wan_i2v_disagg_encoder.json`）**：

```json
{
    "infer_steps": 40,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 480,
    "target_width": 832,
    "self_attn_1_type": "sage_attn2",
    "sample_guide_scale": 5,
    "sample_shift": 3,
    "enable_cfg": true,
    "cpu_offload": false,
    "fps": 16,
    "clip_embed_dim": 329216,
    "disagg_mode": "encoder",
    "disagg_config": {
        "bootstrap_addr": "127.0.0.1",
        "bootstrap_room": 2,
        "sender_engine_rank": 0,
        "receiver_engine_rank": 1,
        "protocol": "rdma",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE"
    }
}
```

> `clip_embed_dim = 257 × 1280 = 329216`。T2V 无 CLIP 编码器，无需此字段；I2V 若缺少此字段会在 Mooncake buffer copy 时报错。
>
> **Transformer 端**同样需要 `clip_embed_dim: 329216`，与 Encoder 端保持一致。

### 2.3 Decoder 配置示例（三段式）

Wan T2V Decoder 见上文；Qwen Image I2I Decoder 示例（`configs/disagg/qwen/qwen_image_i2i_disagg_decode.json`）：

```json
{
    "task": "i2i",
    "disagg_mode": "decode",
    "infer_steps": 40,
    "vae_z_dim": 16,
    "vae_stride": [1, 8, 8],
    "target_video_length": 1,
    "target_height": 1664,
    "target_width": 1664,
    "disagg_config": {
        "bootstrap_addr": "127.0.0.1",
        "bootstrap_room": 2,
        "sender_engine_rank": 1,
        "receiver_engine_rank": 2,
        "protocol": "rdma",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE"
    }
}
```

Decoder 的 `bootstrap_room` 必须与 Transformer 配置中的 `**decoder_bootstrap_room**` 相同；`sender_engine_rank` / `receiver_engine_rank` 对应 Phase2 的 Transformer（发送方）与 Decoder（接收方）引擎 rank。

### 2.4 关键参数说明


| 参数                       | 说明                                                               |
| ------------------------ | ---------------------------------------------------------------- |
| `disagg_mode`            | 分离部署服务角色：`"encoder"`、`"transformer"` 或 `"decode"`                |
| `bootstrap_addr`         | 对端节点 IP（Encoder 连 Transformer；Decoder 连 Transformer）             |
| `bootstrap_room`         | Phase1/Phase2 房间号，**收发端必须一致**；多组服务需不同 room                       |
| `sender_engine_rank`     | Phase1 中 Encoder 的 rank；Phase2 中 Transformer 的 rank              |
| `receiver_engine_rank`   | Phase1 中 Transformer 的 rank；Phase2 中 Decoder 的 rank              |
| `decoder_engine_rank`    | **仅 Transformer 配置**：Phase2 中 Decoder 的 rank，用于建立 Phase2 发送      |
| `decoder_bootstrap_room` | **仅 Transformer 配置**：Phase2 房间号，需与 Decoder 的 `bootstrap_room` 一致 |
| `protocol`               | Mooncake 传输协议：`"rdma"`（推荐）或 `"tcp"`                              |
| `local_hostname`         | 本节点主机名/IP，用于 Mooncake P2P 握手                                     |
| `metadata_server`        | Mooncake 元数据服务，单节点使用 `"P2PHANDSHAKE"` 即可                         |
| `clip_embed_dim`         | CLIP 输出的展平元素总数（**仅 I2V 需要**，ViT-H/14 为 329216）                   |


---

## 3. 启动服务与请求流程

LightX2V 使用 `lightx2v.server` 启动 HTTP API 服务。**三段式**分离部署的通用原则如下：

**启动顺序（建议）：**

1. **先启动 Decoder 服务**（进入 Phase2 接收等待）。
2. **再启动 Transformer 服务**（等待 Phase1 数据，并准备向 Decoder 发送 Phase2）。
3. **最后启动 Encoder 服务**。

**请求与轮询顺序：**

1. **先向 Decoder 发请求**，拿到 `decoder_task_id`（Decoder 开始等待 Phase2 数据）。
2. **再向 Transformer 发相同 payload 的请求**（Transformer 等待 Phase1，收到后跑 DiT，再通过 Phase2 发给 Decoder）。
3. **再向 Encoder 发相同 payload 的请求**（Encoder 编码并通过 Phase1 发给 Transformer）。
4. **向 Decoder 轮询任务状态**：结果文件与 `completed` 状态均在 Decoder 节点，使用 Decoder 的 `task_id` 与 Decoder 的 URL 轮询。

> 原因：Transformer 收到请求后才开始等待 Phase1；Encoder 发送后触发 Transformer 的 DiT；Transformer 完成后通过 Phase2 把潜空间发给已等待的 Decoder；Decoder 完成 VAE 解码并落盘，故任务完成与结果路径以 Decoder 为准。

### 3.1 Wan2.1 T2V 启动示例（三段式）

参考脚本：`scripts/server/disagg/wan/`。配置使用带 `decoder_engine_rank` 的 transformer 与 decode 配置（见 2.1、2.3）。

**启动顺序：** Decoder → Transformer → Encoder（三端可绑定不同 `CUDA_VISIBLE_DEVICES` 与端口）。

先启动 Decoder（例如 port 8004）：

```bash
python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/disagg/wan/wan_t2v_disagg_decode.json \
    --host 0.0.0.0 \
    --port 8004
```

再启动 Transformer（例如 port 8003）、最后启动 Encoder（例如 port 8002）。

发起请求时，Wan 的 T2V / I2V 走视频任务接口 `/v1/tasks/video/`。**三段式请求顺序**：先 POST Decoder → 再 POST Transformer → 再 POST Encoder → 轮询 Decoder 状态：

```python
import requests
import time

DECODER_URL = "http://localhost:8004"
TRANSFORMER_URL = "http://localhost:8003"
ENCODER_URL = "http://localhost:8002"
PAYLOAD = {
    "prompt": "Two anthropomorphic cats in boxing gear fight on a spotlighted stage.",
    "negative_prompt": "色调艳丽，过曝，静态",
    "seed": 42,
    "save_result_path": "/path/to/output.mp4",
}

# 1. Decoder 先收请求，进入 Phase2 等待
resp_d = requests.post(f"{DECODER_URL}/v1/tasks/video/", json=PAYLOAD)
decoder_task_id = resp_d.json()["task_id"]

# 2. Transformer 收请求，等待 Phase1 并跑 DiT，再发 Phase2
requests.post(f"{TRANSFORMER_URL}/v1/tasks/video/", json=PAYLOAD)

# 3. Encoder 编码并发送 Phase1
requests.post(f"{ENCODER_URL}/v1/tasks/video/", json=PAYLOAD)

# 4. 轮询 Decoder 获取完成状态与结果路径
while True:
    status = requests.get(f"{DECODER_URL}/v1/tasks/{decoder_task_id}/status").json()
    if status["status"] == "completed":
        print(f"Done: {status['save_result_path']}")
        break
    if status["status"] == "failed":
        raise RuntimeError(status.get("error"))
    time.sleep(5)
```

### 3.2 Qwen Image T2I / I2I 启动示例（三段式）

参考脚本：`scripts/server/disagg/qwen`。Qwen Image 的 T2I / I2I 均支持 Encoder + Transformer + Decoder 三段式，走图片任务接口 `/v1/tasks/image/`（与 Wan 的 `/v1/tasks/video/` 不同）。

**启动顺序：** 先 Decoder → 再 Transformer → 再 Encoder。端口示例：Encoder 8012、Transformer 8013、Decoder 8014。

#### Qwen T2I 三段式

依次启动 Decoder、Transformer、Encoder（使用 `qwen_image_t2i_disagg_decode.json`、`qwen_image_t2i_disagg_transformer.json`、`qwen_image_t2i_disagg_encoder.json`）。请求顺序：先 POST Decoder 拿 `task_id` → 再 POST Transformer → 再 POST Encoder → 轮询 **Decoder** 的 `task_id` 状态，结果图在 Decoder 节点保存。

```bash
# 启动三端后再执行（脚本内已配置 ENCODER_URL / TRANSFORMER_URL / DECODER_URL 与 ENDPOINT）
python scripts/server/disagg/qwen/post_qwen_t2i.py
```

#### Qwen I2I 三段式

同样先启动 Decoder、Transformer、Encoder（使用 `qwen_image_i2i_disagg_decode.json`、`qwen_image_i2i_disagg_transformer.json`、`qwen_image_i2i_disagg_encoder.json`）。可直接使用提供的 3-way 请求脚本：

```bash
# 启动三端后再执行（脚本内已配置 ENCODER_URL / TRANSFORMER_URL / DECODER_URL 与 ENDPOINT）
python scripts/server/disagg/qwen/post_qwen_i2i.py
```

脚本逻辑：先向 Decoder 发请求 → 再向 Transformer 发请求 → 再向 Encoder 发请求 → 轮询 Decoder 完成；结果图默认保存为 `save_results/qwen_i2i_disagg_3way.png`（在 Decoder 节点）。执行前请在脚本中确认 `IMAGE_PATH` 指向有效输入图。

### 3.3 接口约定总结


| 模型         | 任务        | 请求接口               | 结果与状态轮询        |
| ---------- | --------- | ------------------ | -------------- |
| Wan2.1     | T2V / I2V | `/v1/tasks/video/` | **Decoder** 节点 |
| Qwen Image | T2I / I2I | `/v1/tasks/image/` | **Decoder** 节点 |


启动成功后日志一般会出现：

```text
INFO:     Application startup complete.
```

---

## 4. RDMA vs TCP 协议选择


| 特性         | **RDMA**                       | **TCP**          |
| ---------- | ------------------------------ | ---------------- |
| **传输机制**   | 网卡直接内存访问（Zero-Copy），完全绕过 OS 内核 | 走 Kernel 网络协议栈   |
| **CPU 开销** | 极低                             | 较高               |
| **延迟**     | 微秒级                            | 毫秒级              |
| **硬件要求**   | 需 InfiniBand / RoCE 网卡         | 任意网络             |
| **推荐场景**   | 生产多机部署                         | 单机验证、无 RDMA 硬件环境 |


> 如无 RDMA 硬件，将 `disagg_config` 中 `protocol` 设为 `"tcp"` 即可正常工作。
