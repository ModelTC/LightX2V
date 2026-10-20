# XPU 模型接入细节

从 [XPU 主文](../SKILL.md) 按接入阶段读取对应章节；保留权重、推理与构建实例。

## 第一步：读取原始权重，确定架构参数

```bash
python -c "
from safetensors import safe_open
import glob, os
for s in sorted(glob.glob('MODEL_PATH/**/*.safetensors', recursive=True))[:2]:
    with safe_open(s, framework='pt') as f:
        keys = list(f.keys())
    print(os.path.basename(s), len(keys), 'keys')
    for k in keys[:20]: print(' ', k)
"
```

| 需要确认 | 用途 |
|---------|------|
| block key 正则（如 `transformer_blocks\.(\d+)\.`） | converter.py `model_type_keys_map[...]["key_idx"]` |
| block 总数 | 权重类 `num_layers` |
| 需量化的子模块名（attention + FFN 的 2D weight） | converter.py `model_type_keys_map[...]["target_keys"]` |
| 文本编码器加载/forward 峰值和可用内存 | 是否需已支持的量化或 offload |
| 调度器类型（Flow Matching / DDIM） | Scheduler 实现 |

---

## 第二步：INT8/FP8 转换

### 优先扩展通用转换工具

`tools/convert/converter.py` 已支持 `wan_dit` / `h3` / `hunyuan_dit` / `wan_t5` / `wan_clip` / `wan_animate_dit` / `qwen_image_dit` / `qwen25vl_llm` / `z_image_dit` / `self_forcing` 等 `--model_type`。量化算法是 per-channel 对称 int8/fp8（见 `tools/convert/quant/quant.py`：INT8 除以 127，FP8 E4M3 除以 `torch.finfo(torch.float8_e4m3fn).max`），scale shape `(out_dim, 1)`，和运行时 `MMWeightInt8IntelXpu`/`MMWeightFp8IntelXpu` 的加载格式完全兼容。**新模型优先在这里加一个 `model_type`，而不是新写一份转换脚本。**

在 `tools/convert/converter.py` 里改 3 处：

1. **key 映射**（仅当原始 checkpoint 的 key 名和 LightX2V runtime 权重类注册的 key 名不一致时才需要）：在 `get_key_mapping_rules()` 里加一段 `forward`/`backward` 正则规则（参考其中 `wan_dit` 的写法）。如果原始 checkpoint 已经用 LightX2V runtime key（比如 MiniMax H3），像 `h3` 一样直接 `return []`。
2. **注册 model_type**：`main()` 里 `--model_type` 的 `choices=[...]` 加上新模型名。
3. **量化范围**：`main()` 里的 `model_type_keys_map` 字典加一条（用 Step 1 确认的 block key 正则位置和量化子模块名）：
   ```python
   "{model}": {
       "key_idx": 2,                       # key.split(".") 中子模块名的下标（block key 正则里那个位置）
       "target_keys": ["attn", "ffn"],      # 需要量化的 2D weight 所在子模块名
       "ignore_key": None,                  # 整条 key 都跳过加载的子串，没有就 None
       # "preserve_non_quant_dtype": True,  # 模型里混有需要保留精度的 FP32 tensor 时打开（参考 h3）
   },
   ```

**运行**（参考真实的 MiniMax H3 命令 `tools/convert/examples/convert_minimax_h3_int8_cpu.sh`）：
```bash
python tools/convert/converter.py \
    --source <原始目录或单个 safetensors> \
    --output <int8目录> \
    --output_name {model}_int8 \
    --model_type {model} \
    --quantized --linear_type int8 \
    --device cpu \
    --single_file
```
FP8 把 `--linear_type` 换成 `fp8` 即可，其余不变。需要 `lazy_load` 分块加载时，把 `--single_file` 换成 `--save_by_block`——转换器按 key 里的 `blocks\.(\d+)\.` 正则自动分组成 `block_N.safetensors` + `non_block.safetensors` + `diffusion_pytorch_model.safetensors.index.json`，`_scale` 权重会和对应权重落在同一分片。该分组只决定文件格式，不会为模型新增 lazy 支持；MiniMax H3 当前仍拒绝 lazy。

### 转换内存与产物验证

通用 converter 会持有转换前后的权重；`--save_by_block` 改变保存方式，不等于流式加载。先核对源格式、key 映射和内存预算。确需在内存受限环境转换时，阅读[流式转换参考](streaming-conversion.md)：保留了原按 block 读取模板和产物检查，并列出必须处理的 dtype、精度保留、key 映射与多 shard 限制。优先将必要能力接入现有工具。

检查实际输出文件及 index 的 `weight_map`，抽查每种算子的 shape、dtype、scale 和 non-block 权重。INT8/FP8 与 BF16/FP32 非量化 tensor 的保留规则按 model_type 的 `preserve_non_quant_dtype` 等设置判断，不能把所有非线性权重统一视作 BF16。

---

## 第三步：补齐模型文件骨架

已有原生模型只修改其实际缺口，不重新创建整套 runner/weights。下列为新家族的目录示意，按需创建：

```bash
MODEL=your_model_name

mkdir -p lightx2v/models/networks/$MODEL/weights
mkdir -p lightx2v/models/networks/$MODEL/infer/offload
touch lightx2v/models/networks/$MODEL/__init__.py
touch lightx2v/models/networks/$MODEL/weights/__init__.py
touch lightx2v/models/networks/$MODEL/infer/__init__.py
touch lightx2v/models/networks/$MODEL/infer/offload/__init__.py
mkdir -p lightx2v/models/video_encoders/hf/$MODEL
mkdir -p lightx2v/models/input_encoders/hf/$MODEL
mkdir -p lightx2v/models/runners/$MODEL
mkdir -p lightx2v/models/schedulers/$MODEL
touch lightx2v/models/runners/$MODEL/__init__.py
touch lightx2v/models/schedulers/$MODEL/__init__.py
```

| 文件 | 内容 |
|------|------|
| `networks/{model}/weights/pre_weights.py` | patch_embed、time_embed 等全局权重 |
| `networks/{model}/weights/transformer_weights.py` | block 权重 + offload buffer（见第四步） |
| `networks/{model}/infer/transformer_infer.py` | 无 offload 推理（见第五步） |
| `networks/{model}/infer/offload/transformer_infer.py` | offload 推理（见第五步） |
| `networks/{model}/infer/pre_infer.py` | patchify + position embed + time embed |
| `networks/{model}/infer/post_infer.py` | unpatchify |
| `networks/{model}/model.py` | 主模型类（见第六步） |
| `video_encoders/hf/{model}/vae.py` | 包装 diffusers AutoencoderKL* |
| `input_encoders/hf/{model}/text_encoder.py` | 包装 T5 / CLIP / Qwen 等 |
| `runners/{model}/{model}_runner.py` | Runner 注册（见第七步） |
| `schedulers/{model}/{model}_scheduler.py` | Flow Matching / DDIM 等 |

---

## 第四步：实现 `weights/transformer_weights.py`

复用公共 block offload 时，`MyModelTransformerWeights(WeightModule)` 按 manager 的接口创建以下 WeightModuleList：

| 属性名 | 数量 | create_cuda_buffer | create_cpu_buffer | 条件 |
|--------|------|--------------------|-------------------|------|
| `self.blocks` | `num_layers` | False | False | 始终 |
| `self.offload_block_cuda_buffers` | 2 | True | False | CPU block offload |
| `self.offload_block_cpu_buffers` | 2 | False | True | CPU block offload + 已实现的 `lazy_load` |

每组创建后调 `self.add_module(name, list)`。仅对已支持 lazy 的 block 类按其签名传入 `lazy_load` / `lazy_load_path`；不是所有模型都有这些参数。无 offload 时两个 buffer 属性设为 `None`。

`MyModelBlock(WeightModule)` 接收 `(block_index, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_path)`，按 Step 1 的 key 名用 `MM_WEIGHT_REGISTER[mm_type](key)` 注册 2D weight，`RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](key)` 注册 RMSNorm，按 RoPE 类的实际构造签名注册旋转位置编码（通常是 layout/compute_dtype，不能把权重 key 直接传进去）。

**mm_type 选项**：
- `"Default"` = BF16 无量化
- `"fp8-intel-xpu"` = FP8 GEMM（`onednn_w8a16_fp8`；kernel 不可用时退化为反量化+matmul，fallback）
- `"int8-intel-xpu"` = INT8 W8A8 GEMM（`onednn_w8a8_int8`，激活动态量化）

**rms_type 选项**：
- `"torch_native"` / `"torch"` = 标准 RMSNorm
- `"intel_xpu"` = XPU 优化的 RMSNorm

**rope_type 选项**：
- `"torch_real_rope"` / `"torch_complex_rope"` = 通用实现，按模型的实数/复数与 layout 契约选择
- `"minimax_h3_xpu_rope"` = MiniMax H3 XPU 优化版

**H3 的 qkv_norm_rope_type 选项**：`use_fused_qkv` 控制投影融合，`use_fused_qkv_norm_rope` 独立控制 Norm/RoPE 融合，两者默认关闭。
- `"triton"` = 默认的 Norm/RoPE 融合后端；只有开启对应融合开关才使用
- `"intel_xpu"` = XPU 专用融合后端；契约不匹配时实现回退到分步 Norm/RoPE

---

## 前置条件：编译 sycl_kernels

使用新内核（CUTE attention、XPU RMSNorm、INT8 GEMM）需要先编译 `sycl_kernels`。

当前 LightX2V 使用仓库根目录的 `lightx2v_kernel/intel_xpu/`。skill 内旧 Windows 工程保留为独立历史参考包，其接口、构建和二进制不代表当前支持。构建细节见 [wheel 集成说明](../../lightx2v-kernel-skills/intel-xpu/integration-skills/sycl-esimd-to-python-wheel/SKILL.md)。

- Linux `build.sh` 将 `XPU_TARGET=bmg|ptl-h`（默认 `bmg`）传给 CMake/CUTE，`build_version.py` 添加 `+bmg` / `+ptlh` 后缀。但独立的 `lgrf_uni/build.sh` 仍写死 legacy ESIMD SDP 的 `ptl-h` 目标，不读取该变量；wheel 后缀不代表其中所有内核均适用于该设备。
- Windows `build.bat` 当前在 ESIMD 编译命令中写死 `ptl-h`，没有读取 `XPU_TARGET`，并关闭 CUTE。仅设置环境变量不能把它变为另一硬件目标；先核对 DLL 和 CMake 目标是否与实际设备一致。
- 不按“Arc”名称猜 AOT target；先读取设备信息并核对编译器支持。按实际产物选择一个 wheel，避免通配符同时安装多个 target 的 wheel。
- oneAPI、oneDNN 和 PyTorch ABI 必须匹配。以当前项目构建文件、[环境说明](../../../../lightx2v_kernel/intel_xpu/README.md) 和目标机器实测为依据，不在 skill 中另维护固定版本推荐。

Linux 示例（CUTE 目标为 BMG；使用 legacy SDP 前还须核对或修改其独立构建目标）：
```bash
source /opt/intel/oneapi/setvars.sh
cd lightx2v_kernel/intel_xpu
XPU_TARGET=bmg ./build.sh
ls dist/*.whl
# 从输出中选择本次构建的 wheel，再用 python -m pip install <wheel> 安装
```

---

## 第五步：实现推理类

`infer/transformer_infer.py` 基础结构：`infer()` 调 `infer_func(weights.blocks, x, pre_infer_out)`，`infer_without_offload` 顺序遍历所有 block，`infer_block` 实现单 block 的 AdaLN + attn + FFN + 残差。

**新算子集成**：
```python
class MyModelTransformerInfer:
    def infer_block(self, block_weights, x, pre_infer_out):
        # INT8 W8A8 GEMM（自动在 attn/FFN 的 mm_weight.apply() 中调用，非反量化）
        # 使用新的 attention 实现
        x = self.attn(x, block_weights.attn)  # 内部使用 intel_xpu_cute_attn（Linux）或 intel_xpu_flash_attn（Windows）

        # 使用新的 RMSNorm（XPU 优化）
        x = self.norm(x, block_weights.norm)  # 内部使用 intel_xpu RMSNorm

        # 使用新的旋转位置编码
        # rope 在 pre_infer 或 attn 内调用 minimax_h3_xpu_rope
        return x
```

**注意**：
- `intel_xpu_cute_attn` 和 `intel_xpu_flash_attn` 均**仅支持双向注意力**（Bidirectional），不支持单向或因果掩码
- 当前 Windows 构建关闭 CUTE，选择契约匹配的 `intel_xpu_flash_attn` 或 `torch_sdpa`
- 对于单向注意力（如 decoder-only）的模型，使用 `torch_sdpa` 并验证调用方传递 causal/mask

offload 版仅实现模型实际支持的粒度（H3 仅 model/block；Wan 可作为 phase/lazy 的参考）。`__init__` 创建 `WeightAsyncStreamManager(offload_granularity=granularity)`，按粒度设 `infer_func`；phase+lazy 时调 `init_lazy_load(num_workers)`。

**Offload 粒度选择**：

| 粒度 | 缓冲数 | 何时用 | 特点 |
|------|--------|--------|------|
| **block** | 2 个完整 block | 标准情况 | 一次加载整个 block，双缓冲流水 |
| **phase** | 按模型 phase 定义 | block 峰值仍过高且模型已支持 | 缓冲、调度与搬运成本需实测 |

**Block offload 核心循环**（非 lazy 的 XPU 分支示意；正式实现复用现有循环）：
```python
def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
    for block_idx in range(len(blocks)):
        if self.offload_manager.need_init_first_buffer:
            self.offload_manager.init_first_buffer(blocks)
        self.offload_manager.prefetch_weights((block_idx + 1) % len(blocks), blocks)
        block = self.offload_manager.cuda_buffers[0]
        x = self.run_block(block_idx, block, x, pre_infer_out)
        self.offload_manager.swap_blocks()  # device-wide sync + swap ping/pong
    return x
```

**Phase offload**：Wan 分为 3 个 phase，Qwen Image 为 4 个；H3 不支持。已实现的 phase+lazy 路径还使用 `start_prefetch_block` / `swap_cpu_buffers` / `prefetch_phase` / `swap_phases`，应复用对应模型的顺序和依赖。

---

## 第六步：实现 `model.py`

继承 `BaseTransformerModel`，设置 `pre_weight_class` / `transformer_weight_class` 类属性。

`__init__` 关键点：
- 已支持 lazy 时按实际 checkpoint 前缀设置 `self.remove_keys`，跳过 block 权重初始加载；不能把 `transformer_blocks.` 套给所有模型
- `_init_infer_class` 按 `self.cpu_offload` 选择 offload 或普通推理类
- `_init_infer` 末尾：若 `transformer_infer` 有 `offload_manager`，调 `self._init_offload_manager()`（连接 cuda/cpu buffers，由基类提供）

---

## 第七步：创建 Runner 并注册

**创建 `runners/{model}/{model}_runner.py`**：
```python
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.request_fields import VIDEO_REQUEST_FIELDS

@RUNNER_REGISTER("{model_cls}")
class MyModelRunner(DefaultRunner):
    def load_transformer(self):
        return MyModel(self.config["model_path"], self.config, self.init_device)
    def load_text_encoder(self):
        return [MyTextEncoder(self.config)]
    supported_request_fields_by_task = {"t2v": VIDEO_REQUEST_FIELDS}

    def load_vae(self):
        return None, MyVAE(self.config)  # T2V 示例；有图像输入时还需 encoder
    def init_scheduler(self):
        self.scheduler = MyScheduler(self.config)
```

在 `lightx2v/models/runners/runner_factory.py:RUNNER_MODULES` 中添加 `model_cls → runner 模块路径`；`build_runner()` 延迟导入，CLI 的 choices 直接使用该表，不要在 `infer.py` 再添加 eager import 或重复 choices。

同时核对 `supported_request_fields_by_task`、`InputInfo` 与实际任务；上面的类仅展示关键 hook，不是可独立运行的完整模型。`configs/model_pipeline.json` 不参与当前 `build_runner()` 的原生 CLI 注册。

---

## 第八步：选择 XPU Config 并验证

从 `configs/platforms/intel_xpu/` 与 `scripts/platforms/intel_xpu/` 的匹配入口出发，不把 H3 的算子字段复制给所有模型：

| 家族 | 注意力/Norm 字段 | 位置编码与模式边界 |
|---|---|---|
| Wan | `self_attn_1_type`、`cross_attn_1_type`、必要的 `cross_attn_2_type`；`rms_norm_type`、`layer_norm_type` | 按模型选择 `torch_real_rope` 等；block/phase/lazy 分别查 weights 与 infer |
| MiniMax H3 | `attn_type`、`refiner_attn_type`、`rms_type` | `minimax_h3_xpu_rope` 是 H3 专用；CPU offload 要求 AdaLN cache，仅 model/block，`lazy_load=false` |
| 其他家族 | 阅读本家族 weights/infer 实际读取的 key | 核对 head_dim、layout、causal/mask、dtype 与算子契约 |

`intel_xpu_cute_attn` 需要 Linux CUTE 构建；`intel_xpu_flash_attn` 需要兼容的 ESIMD SDP，缺失时实现会回退到设备上的 torch SDPA。两个 XPU adapter 都没有传递 causal/mask；需要因果语义时使用 `torch_sdpa` 并确认调用方传入 `causal=True` / `attn_mask`。

以现有 Wan 1.3B 通用 SDPA 配置做入口示例（运行前填模型路径）：
```bash
export PLATFORM=intel_xpu
python -m lightx2v.infer \
    --model_cls wan2.1 --task t2v \
    --model_path /path/to/Wan2.1-T2V-1.3B \
    --config_json configs/platforms/intel_xpu/wan_t2v_1_3.json \
    --prompt "A red ball bouncing" \
    --save_result_path output/test.mp4
```

对同一配置、输入和 seed 比较 baseline；分别标注静态检查、算子测试和完整生成结果。初始化成功不证明数值、连续请求或所有 offload 模式可用。

---
