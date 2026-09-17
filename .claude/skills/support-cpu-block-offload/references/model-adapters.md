# 现有模型适配案例

以下记录当前仓库的实现边界，作为定位入口，不替代新任务的源码核对与实测。通用规则见 [implementation-patterns.md](implementation-patterns.md)。不要把某个模型的量化、任务或 cache 限制变成公共后端限制。

## 选择最接近的路径

| 案例 | checkpoint 与共享范围 | 适合参考的接入问题 |
|---|---|---|
| Wan | 按 block 分文件的 FP8-vllm DiT；non-block 权重私有 | 量化 dtype／scale 语义、分块 checkpoint、共享 adapter |
| Qwen Image | Diffusers BF16 分片；transformer blocks 共享、pre/post 等私有 | index/header 一致性、跨 shard manifest、严格算子转置校验 |
| MiniMax H3 DiT | 原始 safetensors；选中的 transformer blocks 共享 | AdaLN cache 过滤、GPU 缓冲释放重建、多组件初始化顺序 |
| MiniMax H3 文本 | Qwen3-VL 文本前缀权重，包括 embedding 与选定层 | 额外共享组件、事件 slot、共享 source 恢复 |
| MiniMax H3 视频 VAE | 选中的 encoder／decoder 静态参数共享 | 原生 nn.Module、混合运行 dtype、decoder 分块与多 tile |

同一 pipeline 中“权重共享范围”和“block offload 范围”可以不同，尤其要区分 H3 视频 VAE encoder 与 decoder。

## Wan：FP8-vllm adapter

源码入口：

- [model.py](../../../../lightx2v/models/networks/wan/model.py)：`WanModel._load_shared_cpu_weights()`。
- [shared_block_weights.py](../../../../lightx2v/models/networks/wan/shared_block_weights.py)：`WanFp8VllmSharedBlockAdapter`。
- `lightx2v/models/networks/wan/weights/transformer_weights.py`：block 权重与 CUDA buffer。
- `lightx2v/models/networks/wan/infer/offload/transformer_infer.py`：`infer_with_blocks_offload()`。

调用顺序：adapter 检查配置和 `block_*.safetensors` → 构造 manifest；模型加载私有 `non_block.safetensors` → 协调 preflight 错误 → `materialize()` → `build_weight_map()` → 公共权重绑定及校验。

关键函数：

- `_discover_block_files()`、`_inspect_checkpoint()` 检查层号、schema、内容签名及运行 dtype。
- `_target_dtype()` 决定最终 CPU 存储类型；`_populate()` 复现既有加载／后处理语义。
- `materialize()` 将 scope、NUMA strict 和注册大小传给公共 coordinator。

特别保留 FP8 scale 的数值语义：当前基线会先将发布 checkpoint 的 FP32 scale 转为 inference dtype，再保存为 FP32 staging。共享填充必须复现这一舍入过程；直接把原 FP32 scale 拷入共享 FP32 tensor 会改变结果。FP8 权重本身保持 FP8。

当前 adapter 要求 `cpu_offload=true`、`offload_granularity=block`、`dit_quantized=true`、`dit_quant_scheme=fp8-vllm`，并要求推理和 sensitive dtype 相同。其 TP、lazy、在线量化、LoRA／adapter 限制由 `_validate_config()` 定义。不要由该案例推断 BF16 Wan 或全部 Wan 家族已支持共享。

启动入口在 [scripts/wan/offload](../../../../scripts/wan/offload/)，配置沿用 `configs/offload/block/`。当前 `run_wan_i2v_block_shared_offload_sp8.sh` 对应 `wan_i2v_block_shared_sp8.json`，scope 为 host；目录尚缺 NUMA 成对入口。这是历史案例的交付缺口，不代表公共 coordinator 不支持 NUMA。后续任务若补齐 Wan 接入，应添加真实 NUMA 配置与脚本并验证，不能将该 host-only 结构作为完整模板。

现有示例中的 T5、CLIP 量化和 DiT 共享是不同能力；不能把编码器量化解释为 CPU 权重共享。

## Qwen Image：Diffusers BF16 adapter

源码入口：

- [model.py](../../../../lightx2v/models/networks/qwen_image/model.py)：`_load_shared_cpu_weights()`、`_validate_shared_cpu_weights()`。
- [shared_block_weights.py](../../../../lightx2v/models/networks/qwen_image/shared_block_weights.py)：`QwenBf16SharedBlockAdapter`、`validate_qwen_shared_block_views()`。
- `lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py`：block 调度循环。

`_inspect_checkpoint()` 使用 safetensors index 与各 shard header 构建 manifest，检查层号、完整性、重复 key 与 block schema。`load_private_weights()` 只加载被选为私有的权重；`_populate()` 写入共享 blocks；`materialize()`、`build_weight_map()` 接入公共协议。

当前条件包括：T2I、未量化 BF16、`DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=BF16`、`feature_caching=NoCaching`、block offload。layered、TP、lazy、LoRA／diff／adapter 和部分 checkpoint 路径覆盖当前不受支持。不要通过自动改成 T2I 或移除 adapter 满足条件；超出边界的用户需求须单独实现或明确报告。

`validate_qwen_shared_block_views()` 在公共地址校验之外，依据算子的 `base_attrs` 验证是否采用规定的转置方向。这可发现方阵上仅比较 shape 检测不到的布局错误。

启动入口在 [scripts/qwen_image/offload](../../../../scripts/qwen_image/offload/)，配置在 `configs/qwen_image/offload/`。已有共享脚本成对使用：

```text
qwen_image_t2i_2512_block_shared_offload_host_sp8.sh
qwen_image_t2i_2512_block_shared_offload_numa_sp8.sh
```

当前示例共享 DiT blocks；编码器和 VAE 没有因此自动开启共享。扩展新组件需要独立 loader、owner、生命周期和证据。

## MiniMax H3：共享加载与多阶段内存管理

### DiT

源码入口：

- [shared_block_weights.py](../../../../lightx2v/models/networks/minimax_h3/shared_block_weights.py)：`load_h3_shared_weights()`、`load_shared_dit()`。
- [model.py](../../../../lightx2v/models/networks/minimax_h3/model.py)：共享 override、`release_block_offload_buffers()`、`ensure_block_offload_buffers()`。
- `lightx2v/models/networks/minimax_h3/weights/transformer_weights.py` 和 `infer/offload/transformer_infer.py`：权重缓冲与计算循环。

`load_h3_shared_weights()` 的参数契约：

| 参数 | 作用 |
|---|---|
| `expected` | 验证所选 checkpoint tensor 的精确名字、shape 和源 dtype |
| `runtime_dtypes` | 在最终共享 CPU 存储中采用的运行 dtype |
| `include` | 过滤当前推理不使用的 checkpoint tensor |
| `shared` | 在已选 tensor 中划分共享和私有范围 |
| `validate` | 组件特有条件，在物化前检查并协调错误 |

helper 通过 meta tensor 构建 manifest，私有权重单独加载，共享 tensor 由 leader 在 `populate()` 中直接写入目标 view。它仍是 H3 的模型级 helper；新家族优先复用公共 coordinator，不仅为文档统一就将 H3 loader 上提或扩大 API。

当前 `load_shared_dit()` 要求 block offload 与 AdaLN cache，按 `model.remove_keys` 排除不需要的 AdaLN／time 等权重，仅共享 `transformer_blocks.*`。当前路径限制 TP、量化、LoRA、lazy、compile 等组合；这是 H3 adapter 的边界。

DiT GPU buffer 可以在文本编码或 VAE 阶段释放，并在 denoise 前由 `ensure_block_offload_buffers()` 重建。重建使用 CPU source 的 shape、dtype、stride，不重载整个 checkpoint，也不关闭共享 arena。

### 文本编码器

入口为 [shared_weights.py](../../../../lightx2v/models/input_encoders/hf/minimax_h3/shared_weights.py) 的 `load_shared_text_weights()`，调度位于 `lightx2v/models/input_encoders/hf/minimax_h3/qwen3vl.py`。

按文本前缀的 expected schema 选择 embedding 和实际使用的层，以 BF16 共享加载；`backbone.load()` 后恢复 CPU source、验证并保存 `shared_cpu_weight_owner`。当前要求未量化的文本 block offload，且没有 text TP。

事件 slot 调度保护各层 GPU 缓冲，completion event 保护请求间复用。embedding 临时上 GPU 后恢复原 pin source；释放 GPU block buffer 时保留共享 CPU 权重。视觉 tower 是另一组件，其整体 offload 不能算作文本共享的一部分。

### 视频 VAE

入口为 [weights.py](../../../../lightx2v/models/video_encoders/hf/minimax_h3/weights.py) 的 `load_shared_video_vae()`、`video_vae.py` 的 `from_pretrained()`／`_activate()`，以及 [offload.py](../../../../lightx2v/models/video_encoders/hf/minimax_h3/offload.py) 的 `VideoVAEDecoderOffload`。

先在 meta 模型上取得 checkpoint expected specs，再通过 `_prepare_inference_dtypes()` 确定运行 dtype，将 FP16／FP32 等目标类型直接落实到共享存储。显式绑定 Parameter 后验证并保存 owner，避免完整 CPU 模型转换产生第二份权重。

视频 VAE 静态共享权重可以同时包含 encoder 和 decoder；当前 block offload 在 decoder 上使用两个 `NativeModuleBlockSlot`。encode 阶段整体激活 encoder 及相关投影，decode 阶段激活必要非 block 权重并逐块执行 decoder。多 tile 通过 completion event 串联，释放时恢复原 CPU source。当前 block 路径有设备、量化及 compile 等边界，修改前核对 `video_vae.py` 的检查。

音频 VAE 的整体 offload 仍是独立私有路径，不能把“视频 VAE 共享”扩大为“全部 VAE 共享”。

### Runner 与启动入口

`lightx2v/models/runners/minimax_h3/minimax_h3_runner.py` 的 `load_model()`、`init_run()`、`_offload_transformer()`、`run_main()` 决定组件加载顺序和阶段切换。所有 rank 必须按相同顺序进入各组件共享初始化，不能仅 rank 0 创建 text／VAE 共享模块。

启动入口在 [scripts/minimax_h3/offload](../../../../scripts/minimax_h3/offload/)，已有 T2AV、I2AV、L2AV、FL2AV、REF2AV 的 host／NUMA 成对脚本。兼容任务复用：

```text
configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_host_sp8.json
configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_numa_sp8.json
```

task 由 CLI 决定，不能只看配置文件名判定实际任务。普通 `scripts/minimax_h3/run_minimax_h3_i2av.sh` 也不能代替 offload 子目录的共享入口，应核对其实际引用配置。

H3 示例涉及独立组件开关：`text_encoder_shared_cpu_weights`、`video_vae_shared_cpu_weights`、对应 block offload 开关，以及 `text_encoder_release_block_offload_buffers`、`dit_release_block_offload_buffers`。新模型只加入实际闭合的组件开关。

三个目录中的实验报告是历史证据。复用其中的命令、比较方法和问题定位线索，但正式结论应由当前代码、资产、硬件和配置的运行支撑；不要复制过去的内存数值或输出 hash 作为本次结果。
