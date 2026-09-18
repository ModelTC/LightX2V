# 现有模型适配案例

简体中文 | [English](model-adapters_en.md)

以下记录当前仓库的实现边界，作为定位入口，不替代新任务的源码核对与实测。通用规则见 [implementation-patterns.md](implementation-patterns.md)。不要把某个模型的量化、任务或 cache 限制变成公共后端限制。

## 选择最接近的路径

| 案例 | checkpoint 与共享范围 | 适合参考的接入问题 |
|---|---|---|
| Wan | 按 block 分文件的 FP8-vllm DiT；non-block 权重私有 | 量化 dtype／scale 语义、分块 checkpoint、共享 adapter |
| Qwen Image | Diffusers BF16 分片；transformer blocks 共享、pre/post 等私有 | index/header 一致性、跨 shard manifest、严格算子转置校验 |
| Hunyuan Image 3.0 | 原始 BF16 indexed checkpoint；共享 arena 包含全部 storage-TP slices | TP／SP／CFG 组合、MoE 融合缓冲、异构 block slots 与逻辑 KV cache 层号 |
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

当前 adapter 要求 `cpu_offload=true`、`offload_granularity=block`、`dit_quantized=true`、`dit_quant_scheme=fp8-vllm`，并要求推理和 sensitive dtype 相同。其 TP、lazy、在线量化、LoRA／adapter 限制由 `_validate_config()` 定义。FP8 checkpoint 格式与推理 dtype 是不同概念；默认 BF16 推理不等于支持未量化 BF16 checkpoint 共享，也不能推断全部 Wan 家族已支持共享。

启动入口在 [scripts/wan/offload](../../../../scripts/wan/offload/)，统一脚本为 `run_wan_block_shared_offload.sh`，配置沿用 `configs/offload/block/wan_block_shared.json`。默认 I2V、host + 8 卡 SP8；将脚本中的 `--shared_cpu_weight_scope host` 改为 `numa` 切换模式。改卡数时同时编辑显卡列表、进程数和 JSON 并行布局。T2V 还需匹配的 T2V checkpoint、配置和输入参数。

当前脚本沿用 `base.sh` 的 BF16 默认值，`SENSITIVE_LAYER_DTYPE=None` 跟随主 dtype；外部环境可改变这些值。旧 FP16 验证记录只覆盖当时精度，默认值变更后的 BF16 完整推理须有独立证据。

现有示例中的 T5、CLIP 量化和 DiT 共享是不同能力；不能把编码器量化解释为 CPU 权重共享。

## Qwen Image：Diffusers BF16 adapter

源码入口：

- [model.py](../../../../lightx2v/models/networks/qwen_image/model.py)：`_load_shared_cpu_weights()`、`_validate_shared_cpu_weights()`。
- [shared_block_weights.py](../../../../lightx2v/models/networks/qwen_image/shared_block_weights.py)：`QwenBf16SharedBlockAdapter`、`validate_qwen_shared_block_views()`。
- `lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py`：block 调度循环。

`_inspect_checkpoint()` 使用 safetensors index 与各 shard header 构建 manifest，检查层号、完整性、重复 key 与 block schema。`load_private_weights()` 只加载被选为私有的权重；`_populate()` 写入共享 blocks；`materialize()`、`build_weight_map()` 接入公共协议。

当前条件包括：T2I、未量化 BF16、主 dtype 与 sensitive dtype 的解析结果均为 BF16、`feature_caching=NoCaching`、block offload。`base.sh` 的 `DTYPE=BF16`、`SENSITIVE_LAYER_DTYPE=None` 已满足默认精度条件，不要求字面值必须写成两个 `BF16`。layered、TP、lazy、LoRA／diff／adapter 和部分 checkpoint 路径覆盖当前不受支持。不要通过自动改成 T2I 或移除 adapter 满足条件；超出边界的用户需求须单独实现或明确报告。

`validate_qwen_shared_block_views()` 在公共地址校验之外，依据算子的 `base_attrs` 验证是否采用规定的转置方向。这可发现方阵上仅比较 shape 检测不到的布局错误。

启动入口在 [scripts/qwen_image/offload](../../../../scripts/qwen_image/offload/)，统一脚本为 `qwen_image_2512_block_shared_offload.sh`，配置为 `configs/qwen_image/offload/qwen_image_2512_block_shared.json`。默认 T2I、host + 8 卡 SP8，通过命令中的 `--shared_cpu_weight_scope` 切换 NUMA。该共享 adapter 当前不支持 I2I。

当前示例共享 DiT blocks；编码器和 VAE 没有因此自动开启共享。扩展新组件需要独立 loader、owner、生命周期和证据。

## Hunyuan Image 3.0：TP 分片与 MoE block slots

源码入口：

- [model.py](../../../../lightx2v/models/networks/hunyuan_image3/model.py)：`_validate_offload_config()`、`_load_shared_cpu_weights()`、`_init_offload_manager()`、`close_shared_cpu_weights()`。
- [shared_block_weights.py](../../../../lightx2v/models/networks/hunyuan_image3/shared_block_weights.py)：`HunyuanImage3SharedBlockAdapter`、`validate_hunyuan_shared_views()`。
- [offload.py](../../../../lightx2v/models/networks/hunyuan_image3/offload.py)：`block_signature()`、`HunyuanImage3BlockSlot`、`HunyuanImage3BlockOffload`。

adapter 从 `model.safetensors.index.json` 和 shard header 构造包含所有 storage-TP slices 的同一 manifest。不同 TP coordinate 选择自己的区域，相同 TP coordinate 的 SP／CFG rank 采用相同 CPU views。signature 包含 TP 大小、micro-shard 布局与运行 dtype。leader 填充时保留 checkpoint 转换顺序，router 最终转 FP32 前仍先经过基线加载 dtype。

GPU 缓冲按 `block_signature()` 区分布局和 MoE 语义，每个兼容 block 族分配两个 slot。expert 权重直接复制进 slot 的融合 MoE pack，不为每个逻辑层保留一份 GPU pack。计算仍使用逻辑层号访问 KV cache；不能用可复用 slot 编号代替它。ready/free 事件保护 slot，completion event 保护连续调用；关闭 GPU offload manager 后再关闭共享 owner。

启动入口在 [scripts/hunyuan_image3/offload](../../../../scripts/hunyuan_image3/offload/)，配置为 `configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json`。默认 T2I、host、8 卡 TP2 × SP2 × CFG2、FlashInfer MoE 和 KV cache；TI2I 需改 `--task` 并添加 `--image_path`。保留 JSON 的 `size` 时仍输出 1024×1024；按参考图对齐需调整尺寸配置。FlashInfer autotune cache 应区分任务和并行布局。

runner 根据 `HUNYUAN_IMAGE3_REPO_PATH` 解析上游代码目录，提供 tokenizer、VAE 和视觉模块；无需在启动脚本再追加该目录到 `PYTHONPATH`。pre/post、VAE、视觉编码器和 GPU KV cache／激活保持 rank 私有。

当前共享入口使用原始 BF16 权重；量化、LoRA、lazy loading、compile、AR CUDA Graph 与跨 GPU pipeline 不在该路径内。TP／SP／CFG 是否有效还取决于 head divisibility 与实际显存，不能将其他模型的“无 TP”限制复制过来，也不能把布局示例当作已运行证据。

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

先在 meta 模型上取得 checkpoint expected specs，再通过 `_prepare_inference_weights(use_channels_last_encoder=False)` 确定运行 dtype，将 FP16／FP32 等目标类型直接落实到共享存储。显式绑定 Parameter 后验证并保存 owner，避免完整 CPU 模型转换产生第二份权重。

视频 VAE 静态共享权重可以同时包含 encoder 和 decoder；当前 block offload 在 decoder 上使用两个 `NativeModuleBlockSlot`。encode 阶段整体激活 encoder 及相关投影，decode 阶段激活必要非 block 权重并逐块执行 decoder。多 tile 通过 completion event 串联，释放时恢复原 CPU source。共享加载要求 `vae_encoder_conv_mode="torch"`；channels-last 与 FP8 Conv3D 需要相应共享 manifest，不能直接套用。当前 block 路径还有设备、量化及 compile 等边界，修改前核对 `video_vae.py` 的检查。

音频 VAE 的整体 offload 仍是独立私有路径，不能把“视频 VAE 共享”扩大为“全部 VAE 共享”。

### Runner 与启动入口

`lightx2v/models/runners/minimax_h3/minimax_h3_runner.py` 的 `load_model()`、`init_run()`、`_offload_transformer()`、`run_main()` 决定组件加载顺序和阶段切换。所有 rank 必须按相同顺序进入各组件共享初始化，不能仅 rank 0 创建 text／VAE 共享模块。

启动入口在 [scripts/minimax_h3/offload](../../../../scripts/minimax_h3/offload/)，共享启动入口为 `run_minimax_h3_block_shared_offload.sh`，对应 `configs/minimax_h3/offload/minimax_h3_block_shared_offload.json`。通过脚本中的 `--task` 选择 `t2av/i2av/l2av/fl2av/ref2av`，并提供对应输入；默认 `t2av`、`--model-variant fl2av`、host + 8 卡 SP8。通过 `--shared_cpu_weight_scope` 切换 NUMA，显卡列表、进程数与 JSON 并行布局一起修改。

AdaLN cache 按 `--model-variant` 和实际 transformer 权重匹配，不能只按 task 名选择。`fl2av` variant 使用 `transformer/` 与 fl2av cache，专用 `ref2av` variant 使用 `transformer_ref/` 与 ref2av cache；步数、flow shifts 和 cache 目录也须一致。缓存工具已还原为上游的直接文件启动方式：

```bash
python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
  --model-variant fl2av
```

该片段需先设置项目／模型路径并加载 `scripts/base/base.sh`。若使用缓存工具自带 shell 脚本，直接编辑其中的路径、GPU、配置和 variant；当前脚本不读取 `MINIMAX_H3_MODEL_PATH`、`MINIMAX_H3_CONFIG` 或 `MINIMAX_H3_CACHE_TASK` 环境覆盖。以工具源码为准，不沿用旧包装器说明。

task 由 CLI 决定，不能只看配置文件名判定实际任务。普通 `scripts/minimax_h3/run_minimax_h3_i2av.sh` 也不能代替 offload 子目录的共享入口，应核对其实际引用配置。

H3 示例涉及独立组件开关：`text_encoder_shared_cpu_weights`、`video_vae_shared_cpu_weights`、对应 block offload 开关，以及 `text_encoder_release_block_offload_buffers`、`dit_release_block_offload_buffers`。新模型只加入实际闭合的组件开关。

历史提交或外部验证记录只能作为背景，当前 offload 目录中的 README 是使用说明，不是实验报告。正式结论应由当前代码、资产、硬件和配置的运行支撑；不要复制过去的内存数值或输出 hash 作为本次结果。
