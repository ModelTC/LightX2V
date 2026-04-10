# LightX2V 分离部署：突破扩散模型推理的显存与吞吐瓶颈

By LightX2V and Mooncake

扩散模型在图像和视频生成领域不断突破能力边界——Wan、Qwen Image 等模型带来了惊人的生成质量。然而，投入生产时它们面对的现实很骨感：**模型太大，装不下**。单个 Qwen-Image-2512 推理流水线在 BF16 精度下需要约 58 GB 显存，连 A100-40GB 都会 OOM。在消费级显卡 RTX 4090（24 GB）上，一种常见的解决方案是 CPU offload，但是cpu offload也会引入额外的推理耗时。

在本文中，我们介绍 [LightX2V](https://github.com/ModelTC/LightX2V) 中的**分离部署**方案：一种三段式架构，将推理流水线拆分为 Encoder、Transformer（DiT）、Decoder 三个独立微服务。各阶段之间通过 [Mooncake](https://github.com/kvcache-ai/Mooncake) RDMA 引擎连接，结合上全新的**去中心化调度器**，该方案实现了：

- Text Encoder 加速 **32.2x**（RTX 4090，消除 offload）
- DiT 单步加速最高达到1.54x
- 8 卡吞吐量（QPS）加速最高达**1.89x**
- 网络通信开销 **< 0.2%**（Mooncake RDMA）

**目录：**

- [背景：扩散模型推理的显存墙](#背景扩散模型推理的显存墙)
- [三段式分离架构](#三段式分离架构)
- [Mooncake 引擎：RDMA 近零通信开销](#mooncake-引擎rdma-近零通信开销)
- [理论分析：最优 Encoder-Transformer 配比](#理论分析最优-encoder-transformer-配比)
- [去中心化调度](#去中心化调度)
- [性能实测](#性能实测)
- [总结](#总结)

---

## 背景：扩散模型推理的显存墙

现代扩散模型流水线由多个重量级组件构成，它们需要同时驻留在 GPU 显存中。下表列出了主要组件的参数量和显存占用：


| 组件                       | 参数量    | BF16 权重 |
| ------------------------ | ------ | ------- |
| Qwen-Image-2512 DiT      | 20.43B | 40.9 GB |
| Qwen2.5-VL（Text Encoder） | 8.29B  | 16.6 GB |
| Qwen Image VAE           | 0.13B  | 0.3 GB  |
| Wan2.1 DiT-14B           | 14.0B  | 28.0 GB |
| T5-XXL（Wan Text Encoder） | ~4.7B  | 9.4 GB  |
| CLIP ViT-H/14（Wan I2V）   | ~0.63B | 1.3 GB  |
| Wan VAE                  | ~0.10B | 0.2 GB  |


当所有组件同时加载，总显存需求迅速突破大多数 GPU 的上限：


| 配置                        | 权重+激活     | RTX 4090 (24 GB) | A100 (40 GB) | A100 (80 GB) | H100 (80 GB) |
| ------------------------- | --------- | ---------------- | ------------ | ------------ | ------------ |
| Qwen-2512 BF16 Baseline   | ~62–68 GB | OOM              | OOM          | 勉强           | OK           |
| Qwen-2512 BF16 分离 Encoder | ~18–20 GB | **OK**           | OK           | OK           | OK           |
| Wan 14B BF16 Baseline     | ~42–48 GB | OOM              | OOM          | OK           | OK           |
| Wan 14B BF16 分离 Encoder   | ~11–13 GB | **OK**           | OK           | OK           | OK           |
| Wan 14B INT8 分离 DiT       | ~17–20 GB | **OK**           | OK           | OK           | OK           |


在显存受限的 GPU 上，常规做法是启用 `cpu_offload`，在 CPU 和 GPU 之间来回搬运权重。虽然这让推理成为"可能"，但性能代价极为沉重：在 RTX 4090 上，Qwen-2512 的 Text Encoder 延迟从 **0.40 s**（分离部署，无需 offload）飙升到 **12.89 s**（baseline + offload），慢了 **32 倍**。而且模型越大，这个延迟增长的越多。

---

## 三段式分离架构

LightX2V 将单体流水线拆分为三个独立服务，每个服务仅加载各自所需的模型权重：

```
┌──────────┐  HTTP POST   ┌──────────────────┐ Phase1 RDMA  ┌───────────────┐ Phase2 RDMA  ┌──────────────┐
│  Client  │ ──────────→  │     Encoder      │ ──────────→  │  Transformer  │ ──────────→  │   Decoder    │
└──────────┘              │ (Text/Image/VAE) │              │    (DiT)      │              │ (VAE Decode) │
                          │   ~17–20 GB      │              │  ~28–41 GB    │              │   ~0.3 GB    │
                          └──────────────────┘              └───────────────┘              └──────────────┘
```

- **Encoder**（`disagg_mode="encoder"`）：加载 Text Encoder、Image Encoder（I2V/I2I 时）、VAE Encoder。通过 Mooncake Phase1 将特征张量发送给 Transformer。
- **Transformer**（`disagg_mode="transformer"`）：仅加载 DiT 模型。接收 Phase1 数据，执行去噪，通过 Mooncake Phase2 将潜空间发送给 Decoder。
- **Decoder**（`disagg_mode="decode"`）：仅加载 VAE Decoder。接收潜空间，解码为像素，保存输出。

### 实测峰值显存


| 模型                     | 模式             | GPU  | 峰值显存       |
| ---------------------- | -------------- | ---- | ---------- |
| Qwen-2512 BF16         | Baseline       | H100 | ~58 GB     |
| Qwen-2512 BF16+offload | 分离 Encoder     | H100 | **~18 GB** |
| Qwen-2512 BF16+offload | 分离 Transformer | H100 | **~40 GB** |
| Wan T2V 14B            | Baseline       | H100 | ~39 GB     |
| Wan T2V 14B            | 分离 Encoder     | H100 | ~11 GB     |
| Wan T2V 14B            | 分离 Transformer | H100 | ~28 GB     |
| Wan I2V 14B 480P       | Baseline       | H100 | ~48 GB     |
| Wan I2V 14B 480P       | 分离 Encoder     | H100 | ~13 GB     |
| Wan I2V 14B 480P       | 分离 Transformer | H100 | ~32 GB     |


通过将组件拆分到不同 GPU，各节点的显存占用降至 baseline 的几分之一。关键在于，Encoder 端可以在 RTX 4090 上**完全不用 offload** 就能运行，从而解锁 32 倍的 Text Encoder 加速。

---

## Mooncake 引擎：RDMA 近零通信开销

分离部署最大的顾虑是通信开销。LightX2V 集成了 [Mooncake 传输引擎](https://github.com/kvcache-ai/Mooncake) ，提供零拷贝 RDMA 传输和微秒级延迟。

### 端到端延迟分解

我们对 H100 上的单个 Qwen-2512 T2I 请求（50 步）做了全链路性能分解：


| 阶段                                | 延迟          | 占比       |
| --------------------------------- | ----------- | -------- |
| Encoder：Text Encoder 计算           | ~0.26 s     | 1.0%     |
| Encoder：Phase1 发送（序列化 + RDMA）     | ~0.025 s    | 0.1%     |
| Transformer：DiT 推理（50 步）          | ~25.3 s     | 96.2%    |
| Transformer：Phase2 发送（序列化 + RDMA） | ~0.024 s    | 0.09%    |
| Decoder：VAE 解码                    | ~0.31 s     | 1.2%     |
| **网络总计（Phase1 + Phase2）**         | **~0.05 s** | **0.2%** |
| **Pipeline 总计**                   | **~26.3 s** |          |


网络开销主要由序列化（约4ms）和mooncake RDNA传输（约20ms）构成。在 H100 + InfiniBand（理论带宽 400 GB/s）环境下，即使 100 MB 的数据传输也仅需亚毫秒级时间。

### 不同模型的传输数据大小


| 传输阶段                  | 数据内容                           | 典型大小    | 模型         |
| --------------------- | ------------------------------ | ------- | ---------- |
| Phase1（Wan T2V）       | context (512×4096 BF16)        | ~4 MB   | Wan2.1-14B |
| Phase1（Wan I2V）       | context + clip_out + vae_enc   | ~20 MB  | Wan2.1-14B |
| Phase1（Qwen T2I）      | prompt_embeds (4096×3584 BF16) | ~28 MB  | Qwen-2512  |
| Phase2（Wan 480P 81 帧） | latent (16×21×60×104 BF16)     | ~42 MB  | Wan2.1-14B |
| Phase2（Wan 720P 81 帧） | latent (16×21×90×160 BF16)     | ~97 MB  | Wan2.1-14B |
| Phase2（Qwen T2I 16:9） | latent (16×1×104×58 BF16)      | ~0.2 MB | Qwen-2512  |


### 跨模型网络开销


| 模型                           | Encoder | DiT 总计  | VAE Decoder | Pipeline 总计 | 网络开销    |
| ---------------------------- | ------- | ------- | ----------- | ----------- | ------- |
| Qwen-2512 T2I 50 step (H100) | 0.30 s  | 22.04 s | ~2.7 s      | ~25.0 s     | < 0.1%  |
| Wan 14B T2V 50 step (H100)   | 0.89 s  | 252.8 s | 2.38 s      | ~253.5 s    | < 0.01% |
| Wan 14B I2V 40 step (H100)   | 3.17 s  | 207.7 s | 2.22 s      | ~210.9 s    | < 0.02% |


通信开销在所有模型和配置下均可忽略不计。Mooncake 的 RDMA 集成确保了分离部署几乎不引入额外延迟。


| 特性     | **RDMA**             | **TCP**        |
| ------ | -------------------- | -------------- |
| 传输机制   | 零拷贝，绕过内核             | 走内核网络协议栈       |
| CPU 开销 | 极低                   | 较高             |
| 延迟     | 微秒级                  | 毫秒级            |
| 硬件要求   | InfiniBand / RoCE 网卡 | 任意网络           |
| 推荐场景   | 生产多机部署               | 单机验证、无 RDMA 硬件 |


---

## 理论分析：最优 Encoder-Transformer 配比

为了方便计算，目前将encoder和decoder部署在同一张GPU上，共享计算资源。

### 吞吐量模型

假设系统有 E 个 Encoder 卡和 T 个 Transformer 卡（Decoder 与 Encoder 共享 GPU，因其显存占用可忽略）。设 t_e 为 Encoder 处理单个请求的时间，t_t 为 Transformer 处理单个请求的时间。

各阶段的吞吐量为：

$$R_e = \frac{E}{t_e}, \quad R_t = \frac{T}{t_t}$$

系统吞吐量受瓶颈阶段约束：

$$R_{\text{system}} = \min(R_e, R_t)$$

为最大化资源利用率，令两阶段吞吐量相等：

$$R_e = R_t \implies \frac{E}{t_e} = \frac{T}{t_t} \implies \boxed{\frac{T}{E} = \frac{t_t}{t_e}}$$

**最优 Transformer:Encoder 比例等于二者单请求处理时间之比。**

### 分离部署如何改变比例

一个关键洞察是：分离部署本身通过加速 Encoder 阶段（消除 offload 开销）极大地改变了这个比例：


| 场景                        | Text Encoder | DiT (8 步)      | DiT:Enc 比例   |
| ------------------------- | ------------ | -------------- | ------------ |
| Baseline (4090 + offload) | 12.89 s      | 23.2 s         | **1.8 : 1**  |
| 分离 (4090)                 | 0.40 s       | 15.0 s         | **37.5 : 1** |
| Baseline (4090 + offload) | 12.89 s      | 287.5 s (50 步) | **22.3 : 1** |
| 分离 (4090)                 | 0.40 s       | 188 s (50 步)   | **470 : 1**  |


Encoder 时间缩短 32 倍后，比例变得极端化，这凸显了"尽可能多分配 Transformer GPU"策略的必要性。

### 步长对配比的影响

步长是驱动最优配比的首要因素（Qwen-2512, 4090）：


| 步长  | Tencoder | TDiT    | DiT:Encoder 比例 | 推荐配置             |
| --- | -------- | ------- | -------------- | ---------------- |
| 50  | 0.4 s    | 188 s   | 470 : 1        | 尽可能多 Transformer |
| 8   | 0.4 s    | 15.04 s | 37.6 : 1       | ~38 : 1          |
| 4   | 0.4 s    | 7.52 s  | 18.8 : 1       | ~19 : 1          |
| 2   | 0.4 s    | 3.76 s  | 9.4 : 1        | ~10 : 1          |
| 1   | 0.4 s    | 1.88 s  | 4.7 : 1        | ~5 : 1           |


### 不同规模最优配比

**8 卡**（常见单机配置）：


| 模型        | 步长  | 理论比例     | 最优配置        | 分析                                          |
| --------- | --- | -------- | ----------- | ------------------------------------------- |
| Qwen-2512 | 8   | 37.5 : 1 | **7T : 1E** | 1 Encoder @ 2.5 req/s >> 7T 瓶颈 @ 0.47 req/s |
| Qwen-2512 | 50  | 470 : 1  | **7T : 1E** | Encoder 永远不是瓶颈                              |


**16 卡**（8 步蒸馏模型）：


| 配置       | T 吞吐       | E 吞吐      | 瓶颈          | E 利用率     |
| -------- | ---------- | --------- | ----------- | --------- |
| 15T : 1E | 1.0 req/s  | 2.5 req/s | Transformer | 40%       |
| 14T : 2E | 0.93 req/s | 5.0 req/s | Transformer | 18.7%（浪费） |


结论：**15:1** 更优——增加第二个 Encoder 浪费了一张 GPU。

**32 卡**（50 步模型）：

31T : 1E 吞吐 0.165 req/s。单个 Encoder（2.5 req/s 容量）仅使用 6.6% 利用率——仍不是瓶颈。

**800 卡**（8 步蒸馏模型）——Encoder 首次成为瓶颈的规模：


| 配置             | T 吞吐            | E 吞吐           | 系统吞吐            | E 利用率     | T 利用率     |
| -------------- | --------------- | -------------- | --------------- | --------- | --------- |
| 790T : 10E     | 52.67 req/s     | 25 req/s       | 25 req/s        | 100%      | 47.5%     |
| 780T : 20E     | 52 req/s        | 50 req/s       | 50 req/s        | 100%      | 96.2%     |
| **779T : 21E** | **51.93 req/s** | **52.5 req/s** | **51.93 req/s** | **98.9%** | **99.8%** |


理论最优值为 $E = N / (1 + t_t/t_e) = 800 / 38.5 \approx 21$。

### 实用配置建议


| 规模          | 8 步蒸馏                                            | 50 步常规                  |
| ----------- | ------------------------------------------------ | ----------------------- |
| **8 卡**     | 7T : 1E                                          | 7T : 1E                 |
| **16–64 卡** | 每 30–40 卡配 1 个 Encoder                           | 每 60–100 卡配 1 个 Encoder |
| **100+ 卡**  | $E = \lceil N / (1 + t_t/t_e) \times 1.2 \rceil$ | 同公式                     |


**何时增加 Encoder：**

1. 实测 Encoder 利用率 > 80%
2. P95 延迟明显高于 P50（排队严重）
3. 并发请求数超过 Transformer 数量

否则，**始终优先增加 Transformer GPU**。

---

## 去中心化调度

标准三段式部署中，客户端需要依次向 Decoder → Transformer → Encoder 发送请求，调用方逻辑复杂且引入调度开销。LightX2V 引入了**去中心化调度器**，将其简化为单次 HTTP POST：

```
┌──────────┐  HTTP POST   ┌──────────┐ Phase1 RDMA ┌─────────────┐ Phase2 RDMA ┌──────────┐
│  Client  │ ──────────→  │ Encoder  │ ──────────→ │ Transformer │ ──────────→ │ Decoder  │
└──────────┘              │ (GPU 0)  │             │ (GPU 1/2/3) │             │ (GPU 0)  │
                          └──────────┘             └─────────────┘             └──────────┘
                                ↑                        ↑                          ↑
                          lightx2v.server          pull worker ×N              pull worker
                          HTTP port 8002
                                │
                          ┌──────────┐
                          │Controller│  ← RDMA 元数据 ring buffer（后台常驻）
                          └──────────┘
```

### 核心设计

- **Controller**：维护三组 RDMA 环形缓冲区（request / phase1 / phase2）用于元数据分发。不加载模型，不参与推理。
- **Encoder**：作为 HTTP 服务运行，执行 Text Encoder 推理后将分发元数据写入 Phase1 RDMA ring。
- **Transformer** 和 **Decoder**：作为 pull-based worker 运行，自动从各自的 RDMA ring 消费任务。
- **多 Transformer worker** 可部署在不同 GPU 上。客户端通过 `disagg_phase1_receiver_engine_rank` 指定目标 worker，支持轮询或显式路由。

### 与标准三段式对比


| 特性          | 标准三段式                                    | 去中心化调度                                 |
| ----------- | ---------------------------------------- | -------------------------------------- |
| 客户端调用       | 需分别向 Decoder → Transformer → Encoder 发请求 | **仅向 Encoder HTTP 发一次请求**              |
| Transformer | HTTP 服务，每次处理一个请求                         | Pull worker，可部署多实例并行消费                 |
| Decoder     | HTTP 服务                                  | Pull worker，自动消费 Phase2                |
| 请求分发        | 客户端显式指定所有端点                              | Encoder 写 RDMA ring，Worker 按 rank 自动拉取 |
| 结果获取        | 轮询 Decoder HTTP                          | 轮询 Encoder HTTP                        |
| 扩展性         | 固定 1:1:1 比例                              | **灵活的 N Transformer : M Encoder**      |


### 对吞吐量和尾部延迟的影响

去中心化调度器消除了顺序请求分发的开销并减少了队列竞争。在 8× RTX 4090、7:1 配比下：


| 调度方式             | QPS      | P50      | P95      | P99      |
| ---------------- | -------- | -------- | -------- | -------- |
| 集中式三段调度 (7:1)    | 0.24     | 17 s     | 25 s     | 28 s     |
| **去中心化调度 (7:1)** | **0.34** | **17 s** | **20 s** | **22 s** |


去中心化调度器相比集中式三段调度方案提升 QPS **1.42x**（相比 baseline 提升 **1.89x**），同时显著降低尾部延迟——P95 从 25 s 降至 20 s，P99 从 28 s 降至 22 s。

---

## 性能实测

### 如何运行（Qwen-2512 T2I 去中心化部署）

以 4 GPU（1 Encoder + 3 Transformer）的 Qwen-2512 T2I 去中心化部署为例，一条命令即可启动全部服务：

```bash
git clone git@github.com:ModelTC/LightX2V.git
cd LightX2V
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg_decentralized.sh
```

启动脚本会依次拉起 Controller → Encoder → Decoder → Transformer×3 共 5 个进程，默认的 GPU 分配为：


| 角色                | GPU   | 说明                          |
| ----------------- | ----- | --------------------------- |
| Controller        | CPU   | 仅维护 RDMA ring buffer，不占 GPU |
| Encoder + Decoder | GPU 0 | 共享同一张卡（两者显存占用很小）            |
| Transformer 1     | GPU 1 | DiT worker                  |
| Transformer 2     | GPU 2 | DiT worker                  |
| Transformer 3     | GPU 3 | DiT worker                  |


通过环境变量可自定义 GPU 分配和 Transformer 数量：

```bash
GPU_ENCODER=0 GPU_DECODER=0 \
GPU_TRANSFORMER_1=1 GPU_TRANSFORMER_2=2 GPU_TRANSFORMER_3=3 \
NUM_TRANSFORMERS=3 \
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg_decentralized.sh
```

服务启动后，通过 Controller 的 HTTP API 发送请求和查询结果：

```bash
# 发送生成请求
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cute cat on a table", "seed": 42, "aspect_ratio": "16:9"}'

# 查询结果状态
curl http://localhost:8080/v1/status/{room_id}
```

### 配置文件解析

去中心化部署的配置位于 `configs/disagg/qwen/`，每个角色对应一个 JSON 文件。以下是各角色配置的关键字段：

**Controller** (`qwen_image_t2i_disagg_controller.json`)：

```json
{
  "disagg_mode": "controller",
  "disagg_config": {
    "protocol": "rdma",
    "rdma_buffer_slots": 128,
    "rdma_buffer_slot_size": 4096,
    "rdma_request_handshake_port": 5566,
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_handshake_port": 5568
  }
}
```

Controller 不加载任何模型权重，仅初始化三组 RDMA ring buffer（request / phase1 / phase2）用于元数据分发。`rdma_buffer_slots` 控制队列深度，`rdma_*_handshake_port` 是各 ring 的 RDMA 握手端口。

**Encoder** (`qwen_image_t2i_disagg_encoder_decentralized.json`)：

```json
{
  "disagg_mode": "encoder",
  "disagg_config": {
    "decentralized_queue": true,
    "sender_engine_rank": 0,
    "receiver_engine_rank": 1,
    "rdma_phase1_host": "127.0.0.1",
    "rdma_phase1_handshake_port": 5567
  }
}
```

`text_encoder_type: "lightllm_kernel"` 启用 LightLLM kernel 级优化（~30% 加速）。`decentralized_queue: true` 表示使用去中心化调度模式。Encoder 推理完成后将特征元数据写入 Phase1 RDMA ring，由 Transformer worker 自行拉取。

**Transformer** (`qwen_image_t2i_disagg_transformer_decentralized.json`)：

```json
{
  "disagg_mode": "transformer",
  "disagg_config": {
    "decentralized_queue": true,
    "transformer_engine_rank": 1,
    "decoder_engine_rank": 4,
    "rdma_phase1_handshake_port": 5567,
    "rdma_phase2_handshake_port": 5568
  }
}
```

Transformer 作为 pull worker 运行，从 Phase1 ring 消费任务，完成 DiT 推理后将潜空间写入 Phase2 ring。部署多个 Transformer 时，每个 worker 使用不同的 `transformer_engine_rank`（启动脚本自动处理）。

**Decoder** (`qwen_image_t2i_disagg_decoder_decentralized.json`)：

```json
{
  "disagg_mode": "decode",
  "disagg_config": {
    "decoder_engine_rank": 2,
    "rdma_phase2_handshake_port": 5568
  }
}
```

Decoder 仅加载 VAE Decoder（~0.3 GB），从 Phase2 ring 接收潜空间并解码为最终图像。

### 显存受限 GPU 上的分阶段加速

在显存受限且需要 CPU offload 的 GPU 上，分离部署的加速最为显著。以 RTX 4090 上的 Qwen-2512（BF16 + block offload）为例：


| 组件            | Baseline（offload） | 分离部署（Encoder 无需 offload） | 加速比        |
| ------------- | ----------------- | ------------------------ | ---------- |
| Text Encoder  | 12.89 s           | 0.40 s                   | **32.22x** |
| DiT 单步（50 步）  | 5.75 s/step       | 3.76 s/step              | **1.53x**  |
| DiT 单步（8 步蒸馏） | 2.90 s/step       | 1.88 s/step              | **1.54x**  |


Text Encoder 加速来源于完全消除权重搬运——Encoder 节点仅需 ~17 GB 显存，在 4090 上无需任何 offload。DiT 加速来源于独占显存带宽：baseline 模式下 DiT 在 offload 传输期间与其他组件争抢 PCIe 和显存带宽。

Wan2.1-I2V-14B 在 RTX 4090（BF16，40 步，block offload）上的结果：


| 指标                   | Baseline (1×4090) | 分离部署 (2×4090) |
| -------------------- | ----------------- | ------------- |
| DiT 单步 (480P)        | 24.24 s/step      | 19.02 s/step  |
| DiT 单步 (720P)        | 90.71 s/step      | 62.80 s/step  |
| Text Encoder         | 2.14 s            | 0.20 s        |
| Image Encoder (480P) | 0.57 s            | 0.28 s        |


**随着模型规模增大，offload 的性能惩罚成比例增长，分离部署的优势也更加突出。** Qwen-2512（baseline 58 GB）的 Text Encoder 加速达 32 倍，而较小的 Wan 14B（baseline 39 GB）为 10 倍。

### 多 GPU 吞吐量

我们在 RTX 4090 上对 Qwen-2512 T2I 进行了 100 请求的压测，对比 baseline（每张卡加载全部组件）与不同 DiT:Encoder 配比的分离部署。

**4 卡——集中式三段调度：**


| 模式       | DiT:Enc 配比 | 并发数   | 总时间       | P50      | P95      | P99      | QPS      |
| -------- | ---------- | ----- | --------- | -------- | -------- | -------- | -------- |
| Baseline | 4:0        | 4     | 1079 s    | 41 s     | 55 s     | 61 s     | 0.092    |
| 分离       | 2:2        | 2     | 1037 s    | 35 s     | 62 s     | 63 s     | 0.096    |
| 分离       | **3:1**    | **3** | **705 s** | **35 s** | **38 s** | **39 s** | **0.15** |


**8 卡——去中心化调度：**


| 模式       | DiT:Enc 配比 | 并发数   | 总时间       | P50      | P95      | P99      | QPS      |
| -------- | ---------- | ----- | --------- | -------- | -------- | -------- | -------- |
| Baseline | 8:0        | 8     | 550 s     | 22 s     | 28 s     | 30 s     | 0.18     |
| 分离       | 4:4        | 4     | 497 s     | 20 s     | 23 s     | 25 s     | 0.20     |
| 分离       | 5:3        | 5     | 402 s     | 19 s     | 20 s     | 23 s     | 0.25     |
| 分离       | 6:2        | 6     | 335 s     | 18 s     | 20 s     | 22 s     | 0.30     |
| 分离       | **7:1**    | **7** | **294 s** | **17 s** | **20 s** | **22 s** | **0.34** |


*RTX 4090，Qwen-2512 T2I，BF16 + block offload，100 请求。分离部署行使用去中心化调度。*

在 8 卡 7:1 配比下，分离部署达到 **0.34 QPS**，对比 baseline 的 **0.18 QPS**——**1.89 倍**的吞吐提升。去中心化调度器还显著降低了尾部延迟：P95 从 28 s（baseline）降至 20 s，P99 从 30 s 降至 22 s。

### 推理参数的影响

**Prompt 长度**对 DiT:Encoder 比例影响较小，因为 Encoder 延迟在所有长度下都远低于 DiT 延迟（Qwen-2512, H100）：


| Prompt 长度   | Encoder 延迟 | DiT 4 步总计 | 比例   |
| ----------- | ---------- | --------- | ---- |
| 16 tokens   | 35 ms      | 1702 ms   | 49:1 |
| 256 tokens  | 53 ms      | 1735 ms   | 33:1 |
| 1024 tokens | 102 ms     | 1855 ms   | 18:1 |
| 4096 tokens | 81 ms      | 1877 ms   | 23:1 |


**分辨率**影响 DiT 延迟但不影响 Encoder 延迟，更大的输出会拉大比例（Qwen-2512, 4090 + offload, 50 步）：


| 分辨率       | 宽高比  | DiT 总计 | DiT 单步 |
| --------- | ---- | ------ | ------ |
| 1664×928  | 16:9 | 188 s  | 3.76 s |
| 1328×1328 | 1:1  | 204 s  | 4.07 s |
| 1472×1140 | 4:3  | 215 s  | 4.29 s |


---

## 总结

本文介绍了 LightX2V 中的分离部署方案——一种三段式架构，将 Encoder、Transformer 和 Decoder 物理拆分到独立 GPU 节点。通过集成 Mooncake RDMA 传输（< 0.2% 开销）和去中心化调度器，我们实现了：

1. **显存解耦**：各节点仅加载自身组件，使 Qwen-2512（58 GB）可在 RTX 4090（24 GB）上无 offload 运行。
2. **Encoder 大幅加速**：消除 offload 在显存受限 GPU 上带来 32 倍 Text Encoder 加速。
3. **灵活扩展**：最优 Encoder:Transformer 配比遵循 $T:E = t_t : t_e$，从 7:1（8 卡）到 779:21（800 卡）均有实用配置。
4. **生产级吞吐**：去中心化调度在 8 卡上实现 1.89 倍 QPS 提升。

随着扩散模型持续增长——20B、50B 甚至更大——单体部署与分离部署之间的差距只会越来越大。Encoder 阶段始终保持小巧，而 DiT 阶段随模型参数量增长，这使得分离部署在每一代模型中都更具说服力。
