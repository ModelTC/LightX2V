# Diffusion Model 分离部署指南

对于大规模生成模型（如 Wan、Qwen Image 等），Text Encoder、Image Encoder 以及 VAE 编解码器往往常驻显存，极大挤压核心 DiT 模型的可用空间，在高分辨率、长时生成场景下容易导致 OOM。

LightX2V 提供了原生的 **Disaggregation Mode（分离部署模式）**，通过高性能 **Mooncake 传输引擎**，将 Encoder 与 Transformer 部署在不同显卡或节点上，支持 RDMA / TCP 通信。分离部署模式还支持 Encoder 与 Transformer 并发处理不同请求，从而提升多请求吞吐。

---

## 方案对比


| 特性       | **Baseline（常规单机部署）** | **Disagg Mode（分离微服务部署）**                   |
| -------- | -------------------- | ------------------------------------------ |
| **部署架构** | 所有模型聚合在同一进程中         | 拆分为 **Encoder 节点** 与 **Transformer 节点**    |
| **显存占用** | 极高                   | **按需分配**（各节点只加载自身所需部分）                     |
| **通信底层** | 进程内原生 Tensor 共享      | **Mooncake 引擎**（支持 Zero-Copy RDMA 与标准 TCP） |
| **适用场景** | 显存充裕的单机环境、快速验证       | **显存受限**、长帧视频、多机分布式高并发生产环境                 |


---

## 性能实测 Benchmark

测试环境：NVIDIA H100 SXM5 80 GB，`sage_attn2`，`BF16`，`PROFILING_DEBUG_LEVEL=2`，Mooncake RDMA 协议。
Baseline 使用单张 GPU 加载全部模型；Disagg Mode 将 Encoder 与 Transformer 部署在不同 GPU 上。

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

通过配置参数 `disagg_mode`，推理 Pipeline 被物理拆分为两个独立服务：

- **Encoder 角色（`disagg_mode="encoder"`）**：
  - 仅加载 Text Encoder、Image Encoder（I2V 时）以及 VAE Encoder，**跳过 DiT 加载**。
  - 执行特征提取，将 `context`、`clip_encoder_out`、`vae_encoder_out`、`latent_shape` 等结果通过 Mooncake 投递给 Transformer 节点。
- **Transformer 角色（`disagg_mode="transformer"`）**：
  - 仅加载 DiT 模型与 VAE Decoder，**跳过 Encoder 加载**。
  - 启动后进入 Mooncake 接收等待状态，收到数据后执行哈希校验、拼装输入并完成去噪与解码，最终保存输出视频。

---

## 2. 配置方法

所有分离部署参数统一在 config json 的 `disagg_config` 字段中配置。

### 2.1 T2V 配置示例

**Encoder 端（`configs/wan/wan_t2v_disagg_encoder.json`）**：

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

**Transformer 端（`configs/wan/wan_t2v_disagg_transformer.json`）**：将上述 `disagg_mode` 改为 `"transformer"` 即可，其余参数保持一致。

### 2.2 I2V 配置示例

I2V 使用 **ViT-H/14** CLIP 图像编码器，其输出为完整序列特征（257 个 token × 1280 维），因此**必须额外指定 `clip_embed_dim`** 以保证 RDMA buffer 正确分配：

**Encoder 端（`configs/wan/wan_i2v_disagg_encoder.json`）**：

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

### 2.3 关键参数说明


| 参数                     | 说明                                               |
| ---------------------- | ------------------------------------------------ |
| `disagg_mode`          | 分离部署服务角色：`"encoder"` 或 `"transformer"`           |
| `bootstrap_addr`       | Transformer 节点的 IP 地址（Encoder 据此建立 Mooncake 连接）  |
| `bootstrap_room`       | 通信房间号，**收发端必须一致**；多组 disagg 服务并存时需各自使用不同 room    |
| `sender_engine_rank`   | Encoder 侧 Mooncake 引擎 rank                       |
| `receiver_engine_rank` | Transformer 侧 Mooncake 引擎 rank                   |
| `protocol`             | Mooncake 传输协议：`"rdma"`（推荐，需 IB/RoCE 网卡）或 `"tcp"` |
| `local_hostname`       | 本节点主机名/IP，用于 Mooncake P2P 握手                     |
| `metadata_server`      | Mooncake 元数据服务，单节点使用 `"P2PHANDSHAKE"` 即可         |
| `clip_embed_dim`       | CLIP 输出的展平元素总数（**仅 I2V 需要**，ViT-H/14 为 329216）   |


---

## 3. 启动服务与请求流程

LightX2V 使用 `lightx2v.server` 启动 HTTP API 服务。分离部署时的通用原则如下：

1. **先启动 Transformer 服务，再启动 Encoder 服务。**
2. **先向 Transformer 发请求，再向 Encoder 发同样的请求。**
3. **最终结果由 Transformer 保存，任务状态也应从 Transformer 轮询。**

> 原因：Transformer 的 `run_pipeline` 会在收到 HTTP 请求后进入 `receive_encoder_outputs()`，阻塞等待 Encoder 通过 Mooncake 发送编码结果。如果不先请求 Transformer，它不会开始等待，Encoder 侧即使完成编码和发送，也不会触发后续 DiT 推理。

### 3.1 Wan2.1 T2V 启动示例

参考脚本：

- `scripts/server/disagg/wan/`

先启动 Transformer：

```bash
#!/bin/bash
lightx2v_path=
model_path=/data/nvme0/models/Wan-AI/Wan2.1-T2V-14B

export CUDA_VISIBLE_DEVICES=

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/wan/wan_t2v_disagg_transformer.json \
    --host 0.0.0.0 \
    --port 8003
```

再启动 Encoder：

```bash
#!/bin/bash
lightx2v_path=
model_path=/data/nvme0/models/Wan-AI/Wan2.1-T2V-14B

export CUDA_VISIBLE_DEVICES=

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/wan/wan_t2v_disagg_encoder.json \
    --host 0.0.0.0 \
    --port 8002
```

发起请求时，Wan 的 T2V / I2V 都走视频任务接口 `/v1/tasks/video/`：

```python
import requests
import time

TRANSFORMER_URL = "http://localhost:8003"
ENCODER_URL = "http://localhost:8002"
PAYLOAD = {
    "prompt": "Two anthropomorphic cats in boxing gear fight on a spotlighted stage.",
    "negative_prompt": "色调艳丽，过曝，静态",
    "seed": 42,
    "save_result_path": "/path/to/output.mp4",
}

resp_t = requests.post(f"{TRANSFORMER_URL}/v1/tasks/video/", json=PAYLOAD)
task_id = resp_t.json()["task_id"]

requests.post(f"{ENCODER_URL}/v1/tasks/video/", json=PAYLOAD)

while True:
    status = requests.get(f"{TRANSFORMER_URL}/v1/tasks/{task_id}/status").json()
    if status["status"] == "completed":
        print(f"Done: {status['save_result_path']}")
        break
    if status["status"] == "failed":
        raise RuntimeError(status.get("error"))
    time.sleep(5)
```

### 3.2 Qwen Image T2I / I2I 启动示例

参考脚本：

- `scripts/server/disagg/qwen`

#### Qwen T2I

先启动 Transformer：

```bash
bash LightX2V/scripts/server/disagg/qwen/start_qwen_t2i_disagg_transformer.sh
```

再启动 Encoder：

```bash
bash LightX2V/scripts/server/disagg/qwen/start_qwen_t2i_disagg_encoder.sh
```

然后执行测试请求：

```bash
python scripts/server/disagg/qwen/post_qwen_t2i.py
```

T2I 结果图默认保存为 `qwen_t2i_disagg.png`。

#### Qwen I2I

先启动 Transformer：

```bash
bash LightX2V/scripts/server/disagg/qwen/start_qwen_i2i_disagg_transformer.sh
```

再启动 Encoder：

```bash
bash LightX2V/scripts/server/disagg/qwen/start_qwen_i2i_disagg_encoder.sh
```

再执行测试请求：

```bash
python scripts/server/disagg/qwen/post_qwen_i2i.py
```

I2I 结果图默认保存为 `qwen_i2i_disagg.png`。执行前请先在 `scripts/server/disagg/qwen/post_qwen_i2i.py` 中确认 `IMAGE_PATH` 指向有效输入图。

> Qwen Image 的 T2I / I2I 都走图片任务接口 `/v1/tasks/image/`。与 Wan 使用 `/v1/tasks/video/` 不同，请求脚本不要混用。

### 3.3 接口约定总结


| 模型         | 任务        | 请求接口               |
| ---------- | --------- | ------------------ |
| Wan2.1     | T2V / I2V | `/v1/tasks/video/` |
| Qwen Image | T2I / I2I | `/v1/tasks/image/` |


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

