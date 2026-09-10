# MiniMax-H3 on Apple MPS

MiniMax-H3 只支持 diffusers 权重布局。本机的两个脚本均使用：

```text
/Users/yongyang/Documents/x2v/models/MiniMaxAI/diffusers/MiniMax-H3
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-*.safetensors
├── text_encoder/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── model-*.safetensors
├── tokenizer/
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model*.safetensors
└── audio_vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

文本编码器仍遵循 Transformers 的 `model.safetensors.index.json` 命名。上图展示下载模型的标准目录；DiT 权重发现沿用上游规则：枚举目录中的 `*.safetensors`，也接受单个权重文件，不依赖索引文件名或参数名前缀白名单。流式读取器只扫描文件头来记录张量位置，数据在每层需要时读取；模型仍按 diffusers 参数名取权重，不转换 raw 的 QKV/FFN、配置别名或 `video_vae/source` 布局。视频 VAE 直接从 `vae/` 加载匹配的 diffusers 张量。

AdaLN 缓存构建器使用 ModelTC/LightX2V 上游实现，直接读取 diffusers 参数。MPS 逐层磁盘读取器只负责按需加载已有张量，不承担权重格式转换。

在 LightX2V 根目录执行：

```bash
# 首次运行先生成与配置匹配的缓存；已有缓存时无需重复构建。
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
bash scripts/platforms/mps/run_minimax_h3_t2av.sh
```

默认使用 `configs/platforms/mps/minimax_h3_t2av_4step_512_22.json`：512×512、22 帧、4 步、BF16，DiT 和文本编码器逐层磁盘加载，VAE 按阶段加载。输出为 `save_results/output_lightx2v_minimax_h3_t2av.mp4`。这份 4 步配置用于本机快速验证。

MPS 配置使用独立缓存根目录 `~/.cache/lightx2v/adaln/diffusers`，避免与之前的权重混用。换权重或改动缓存相关计算后应重新生成缓存；步数和 flow shift 必须与推理配置一致。

流式权重初始化在清理 MPS 缓存前显式同步 GPU，避免 safetensors 存储释放回调与 Python GIL 相互等待。正式入口已经包含该处理，无需使用之前 bench 目录中的临时包装入口。
