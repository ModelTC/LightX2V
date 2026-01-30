# Text Encoder 分离部署/优化指南 (Advanced Guide)

对于大规模模型推理，Text Encoder 往往占据显存且计算相对独立。LightX2V 提供了两种先进的 Text Encoder 优化方案：**Service Mode (分离部署)** 和 **Kernel Mode (内核优化)**。

这两种方案目前针对 **Qwen-Image** 系列模型的 Text Encoder 进行过深度优化，显著降低了显存占用并提升了推理吞吐量。

## 方案对比

| 特性 | **Baseline (原始 huggingface)** | **Service Mode (分离部署)** | **Kernel Mode (内核优化)** |
| :--- | :--- | :--- | :--- |
| **部署架构** | 与主模型在同一进程 | 独立服务，通过 HTTP/SHM 通信 | 与主模型在同一进程 |
| **显存占用** | 高 (加载完整 HF 模型) | **极低** (Client 端不加载权重) | **中** (加载精简模型 + Kernel) |
| **跨请求复用** | 无 | **支持** (多客户端共享一个 Encoder) | 无 |
| **通信开销** | 无 | 有 (HTTP/SharedMemory) | 无 |
| **推理速度** | 慢 (标准 Layer) | **极快** (LightLLM 后端加速) | **快** (集成 LightLLM Kernel) |
| **适用场景** | 快速验证、小显存单卡 | **多卡/多机生产环境**、DiT 显存瓶颈 | **高性能单机推理**、追求极限速度 |

详情性能数据可参考: [Performance Benchmark](https://github.com/ModelTC/LightX2V/pull/829)

---

## 1. Service Mode (分离部署模式)

Service Mode 将 Text Encoder 作为一个独立的服务启动，基于高性能 LLM 推理框架 **LightLLM**。主模型 (LightX2V Client) 通过 API 请求获取 hidden states。

### 1.1 环境准备

Text Encoder 服务端需要使用 **LightLLM** 框架。

**服务端安装步骤:**
1. 拉取 LightLLM 代码 (指定 `return_hiddens` 分支)
```bash
git clone git@github.com:ModelTC/LightLLM.git -b return_hiddens
cd LightLLM
```

2. 配置环境
请参考 LightLLM 官方文档配置 Python 环境 (通常需要 PyTorch, CUDA, Triton 等)。
*注意：确保服务端环境支持 FlashAttention 以获得最佳性能。*

### 1.2 启动 Text Encoder 服务

使用 `lightllm.server.api_server` 启动服务。

**编写启动脚本 `start_encoder_service.sh` (参考示例):**

```bash
#!/bin/bash
# 显卡设置 (例如使用独立的卡运行 Text Encoder)
export CUDA_VISIBLE_DEVICES=1
export LOADWORKER=18

# 指向 LightLLM 代码目录
# export PYTHONPATH=/path/to/LightLLM:$PYTHONPATH

# 模型相关路径 (需替换为实际路径)
MODEL_DIR="/path/to/models/Qwen-Image-Edit-2511/text_encoder"
TOKENIZER_DIR="/path/to/models/Qwen-Image-Edit-2511/tokenizer"
PROCESSOR_DIR="/path/to/models/Qwen-Image-Edit-2511/processor"

# 设置环境变量供 LightLLM 内部使用
export LIGHTLLM_TOKENIZER_DIR=$TOKENIZER_DIR
export LIGHTLLM_PROCESSOR_DIR=$PROCESSOR_DIR
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1

python -m lightllm.server.api_server \
    --model_dir $MODEL_DIR \
    --host 0.0.0.0 \
    --port 8010 \
    --tp 1 \
    --enable_fa3 \
    --return_input_hidden_states \
    --enable_multimodal \
    --disable_dynamic_prompt_cache
```

**关键参数说明:**
*   `--return_input_hidden_states`: **必须开启**。让 LightLLM 返回 hidden states 而不是生成的 token，这是 Service Mode 的核心。
*   `--enable_multimodal`: 开启多模态支持 (处理 Qwen 的 Vision Token)。
*   `--port 8010`: 服务监听端口，需与 Client 端配置一致。
*   `--tp 1`: Tensor Parallel 度，通常 Text Encoder 单卡即可部署。
*   `--enable_fa3`: 启用 FlashAttention。
*   `--disable_dynamic_prompt_cache`: 禁用动态 Prompt Cache。

启动服务:
```bash
bash start_encoder_service.sh
```
看到类似 "Uvicorn running on http://0.0.0.0:8010" 即表示启动成功。

### 1.3 配置 LightX2V Client

在 LightX2V 端，只需修改 `config_json` 来启用 Service Mode。

**配置文件 (`configs/qwen_image/qwen_image_i2i_2511_service.json`):**

```json
{
    "text_encoder_type": "lightllm_service",
    "lightllm_config": {
        "service_url": "http://localhost:8010",
        "service_timeout": 30,
        "service_retry": 3,
        "use_shm": true
    },
    // ... 其他参数 (infer_steps, prompt_template 等) ...
}
```

**参数说明:**
*   `text_encoder_type`: 设置为 **"lightllm_service"**。
*   `service_url`: Text Encoder 服务的地址。
*   `use_shm`: **强烈推荐开启**。
    *   `true`: 启用共享内存 (Shared Memory) 通信。如果 Client 和 Server 在同一台机器 (即使不同 Docker 容器，需挂载共享内存)，数据传输将通过内存直读，**零拷贝，速度极快**。
    *   `false`: 使用 HTTP 传输 base64 编码数据。适用于跨机部署。

**运行推理:**

编写运行脚本 (`scripts/qwen_image/qwen_image_i2i_2511_service.sh`):

```bash
python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path /path/to/Qwen-Image-Edit-2511 \
    --config_json configs/qwen_image/qwen_image_i2i_2511_service.json \
    --prompt "Make the girl from Image 1 wear the black dress from Image 2..." \
    --image_path "1.png,2.png,3.png" \
    --save_result_path output.png
```

---

## 2. Kernel Mode (内核优化模式)

Kernel Mode 适合单机高性能推理场景。它不在后台启动独立服务，而是在进程内直接加载 Text Encoder，但**替换了 HuggingFace 原始的慢速算子**，集成了 LightLLM 的核心 Triton Kernel。

### 2.1 优势
*   **无需独立服务**: 简化部署运维。
*   **Triton 加速**: 使用高度优化的 FlashAttention 和 Fused RMSNorm Triton Kernel。
*   **无通信开销**: 纯进程内内存操作。

### 2.2 配置方法

只需修改 `config_json` 启用 Kernel Mode。

**配置文件 (`configs/qwen_image/qwen_image_i2i_2511_kernel.json`):**

```json
{
    "text_encoder_type": "lightllm_kernel",
    "lightllm_config": {
        "use_flash_attention_kernel": true,
        "use_rmsnorm_kernel": true
    },
    // ... 其他参数 ...
}
```

**参数说明:**
*   `text_encoder_type`: 设置为 **"lightllm_kernel"**。
*   `use_flash_attention_kernel`: 启用 FlashAttention 加速 Attention 层。 默认情况下将使用 flash_attention_2，但你也可以使用 “use_flash_attention_kernel”: “flash_attention_3”。
*   `use_rmsnorm_kernel`: 启用 Fused RMSNorm Kernel (需安装 `sgl_kernel` 或相关依赖，如未安装会自动降级)。

**运行推理:**

编写运行脚本 (`scripts/qwen_image/qwen_image_i2i_2511_kernel.sh`):

```bash
python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path /path/to/Qwen-Image-Edit-2511 \
    --config_json configs/qwen_image/qwen_image_i2i_2511_kernel.json \
    --prompt "..." \
    --image_path "..." \
    --save_result_path output.png
```

---

## 总结建议

*   **开发调试**: 默认模式 (HuggingFace) 兼容性最好。
*   **单机高性能**: 使用 **Kernel Mode**。
*   **多机/多卡/显存受限**: 使用 **Service Mode**。将 Text Encoder 部署在显存较小的卡上，主卡专注于 DiT 推理，并通过 Shared Memory 实现高效通信。
