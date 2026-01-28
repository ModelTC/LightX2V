# Text Encoder Separation/Optimization Guide (Advanced Guide)

For large-scale model inference, the Text Encoder often consumes significant memory and its computation is relatively independent. LightX2V offers two advanced Text Encoder optimization schemes: **Service Mode** and **Kernel Mode**.

These schemes have been deeply optimized for the **Qwen-Image** series Text Encoder, significantly reducing memory usage and improving inference throughput.

## Comparison

| Feature | **Baseline (Original HF)** | **Service Mode (Separated)** | **Kernel Mode (Kernel Optimized)** |
| :--- | :--- | :--- | :--- |
| **Deployment Architecture** | Same process as main model | Independent service via HTTP/SHM | Same process as main model |
| **Memory Usage** | High (Loads full HF model) | **Very Low** (Client loads no weights) | **Medium** (Loads simplified model + Kernel) |
| **Cross-Request Reuse** | No | **Supported** (Shared by multiple clients) | No |
| **Communication Overhead** | None | Yes (HTTP/SharedMemory) | None |
| **Inference Speed** | Slow (Standard Layer) | **Very Fast** (LightLLM backend acceleration) | **Fast** (Integrated LightLLM Kernels) |
| **Applicable Scenarios** | Quick validation, small memory single-card | **Multi-card/Multi-node production**, DiT memory bottleneck | **High-performance single-node**, extreme speed pursuit |

For detailed performance data, please refer to: [Performance Benchmark](https://github.com/ModelTC/LightX2V/pull/829)

---

## 1. Service Mode (Separated Deployment)

Service Mode runs the Text Encoder as an independent service based on the high-performance LLM inference framework **LightLLM**. The main model (LightX2V Client) retrieves hidden states via API requests.

### 1.1 Environment Preparation

The Text Encoder server side requires the **LightLLM** framework.

**Server Installation Steps:**
1. Clone LightLLM code (specify `return_hiddens` branch)
```bash
git clone git@github.com:ModelTC/LightLLM.git -b return_hiddens
cd LightLLM
```

2. Configure Environment
Please refer to the LightLLM official documentation to configure the Python environment (usually requires PyTorch, CUDA, Triton, etc.).
*Note: Ensure the server environment supports FlashAttention to achieve the best performance.*

### 1.2 Start Text Encoder Service

Use `lightllm.server.api_server` to start the service.

**Create start script `start_encoder_service.sh` (example):**

```bash
#!/bin/bash
# GPU settings (e.g., use a separate card for Text Encoder)
export CUDA_VISIBLE_DEVICES=1
export LOADWORKER=18

# Point to LightLLM code directory
# export PYTHONPATH=/path/to/LightLLM:$PYTHONPATH

# Model paths (replace with actual paths)
MODEL_DIR="/path/to/models/Qwen-Image-Edit-2511/text_encoder"
TOKENIZER_DIR="/path/to/models/Qwen-Image-Edit-2511/tokenizer"
PROCESSOR_DIR="/path/to/models/Qwen-Image-Edit-2511/processor"

# Set environment variables for LightLLM internal use
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

**Key Arguments Explanation:**
*   `--return_input_hidden_states`: **Must be enabled**. Instructs LightLLM to return hidden states instead of generated tokens, which is the core of Service Mode.
*   `--enable_multimodal`: Enable multimodal support (handles Qwen's Vision Token).
*   `--port 8010`: Service listening port, must match the Client configuration.
*   `--tp 1`: Tensor Parallel degree, usually 1 is sufficient for Text Encoder.
*   `--enable_fa3`: Enable FlashAttention.
*   `--disable_dynamic_prompt_cache`: Disable dynamic prompt cache.

Start the service:
```bash
bash start_encoder_service.sh
```
Seeing something like "Uvicorn running on http://0.0.0.0:8010" indicates successful startup.

### 1.3 Configure LightX2V Client

On the LightX2V side, simply modify the `config_json` to enable Service Mode.

**Configuration File (`configs/qwen_image/qwen_image_i2i_2511_service.json`):**

```json
{
    "text_encoder_type": "lightllm_service",
    "lightllm_config": {
        "service_url": "http://localhost:8010",
        "service_timeout": 30,
        "service_retry": 3,
        "use_shm": true
    },
    // ... other parameters (infer_steps, prompt_template, etc.) ...
}
```

**Parameters Explanation:**
*   `text_encoder_type`: Set to **"lightllm_service"**.
*   `service_url`: The address of the Text Encoder service.
*   `use_shm`: **Strongly Recommended**.
    *   `true`: Enable Shared Memory communication. If Client and Server are on the same machine (even in different Docker containers, provided shared memory is mounted), data transfer will happen via direct memory reading, **zero-copy, extremely fast**.
    *   `false`: Use HTTP to transfer base64 encoded data. Suitable for cross-machine deployment.

**Run Inference:**

Create a run script (`scripts/qwen_image/qwen_image_i2i_2511_service.sh`):

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

## 2. Kernel Mode (Kernel Optimization)

Kernel Mode is suitable for single-node high-performance inference scenarios. It does not start an independent service in the background, but loads the Text Encoder directly in the process, while **replacing HuggingFace's original slow operators** with LightLLM's core Triton Kernels.

### 2.1 Advantages
*   **No Independent Service**: Simplifies deployment and operations.
*   **Triton Acceleration**: Uses highly optimized FlashAttention and Fused RMSNorm Triton Kernels.
*   **No Communication Overhead**: Pure in-process memory operations.

### 2.2 Configuration

Simply modify `config_json` to enable Kernel Mode.

**Configuration File (`configs/qwen_image/qwen_image_i2i_2511_kernel.json`):**

```json
{
    "text_encoder_type": "lightllm_kernel",
    "lightllm_config": {
        "use_flash_attention_kernel": true,
        "use_rmsnorm_kernel": true
    },
    // ... other parameters ...
}
```

**Parameters Explanation:**
*   `text_encoder_type`: Set to **"lightllm_kernel"**.
*   `use_flash_attention_kernel`: Enable FlashAttention acceleration for Attention layers.
*   `use_rmsnorm_kernel`: Enable Fused RMSNorm Kernel (requires `sgl_kernel` or related dependencies; will automatically downgrade if not installed).

**Run Inference:**

Create run script (`scripts/qwen_image/qwen_image_i2i_2511_kernel.sh`):

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

## Summary and Recommendations

*   **Development/Debugging**: Default Mode (HuggingFace) for best compatibility.
*   **High-Performance Single-Node**: Use **Kernel Mode**.
*   **Multi-Node/Multi-Card/Memory Constrained**: Use **Service Mode**. Deploy the Text Encoder on a card with smaller memory, let the main card focus on DiT inference, and achieve efficient communication via Shared Memory.
