# RelayVSR 视频超分

原生接入 [RelayVSR](https://github.com/kopperx/RelayVSR) 官方大模型：约 5.03B 参数的 Wan 生成稀疏参考帧，约 31.9M 参数的 Flash-S 解码全部视频帧。

**模型内部始终计算 4×。** `--sr_ratio 4` 直接输出；`--sr_ratio 2` 将裁去 padding 的 4× 结果用 FP32、抗锯齿 bicubic 缩小，再编码。2× 不会省去 4× 模型计算，但可以降低输出搬运、编码和文件大小。

## 准备与运行

使用 LightX2V 的 CUDA 环境，包含 PyTorch、Triton、PyAV 和 FFmpeg。默认 BF16。下载官方两个文件即可，不需要转换权重：

```bash
hf download kopper/RelayVSR wan.safetensors flash.safetensors --local-dir /models/RelayVSR
```

先修改 `scripts/relayvsr/run_relayvsr_sr.sh` 顶部的 `lightx2v_path`、`model_path` 和 `video_path`，然后运行：

```bash
bash scripts/relayvsr/run_relayvsr_sr.sh
```

显卡和推理参数直接在脚本中设置，通过 `scripts/base/base.sh` 初始化环境。将 `--sr_ratio` 改为 `4` 可直接输出 4×，输出位置通过 `--save_result_path` 设置。也可以直接调用统一 CLI：

```bash
python -m lightx2v.infer \
  --model_cls relayvsr --task sr \
  --model_path /models/RelayVSR \
  --config_json configs/relayvsr/relayvsr.json \
  --video_path /videos/input.mp4 --sr_ratio 2 --seed 42 \
  --save_result_path /videos/output_x2.mp4
```

图片用 `--image_path input.png` 替换 `--video_path`，输出设为 PNG。

## 执行模式

| 配置 | 行为 |
| --- | --- |
| `relayvsr.json` | 常驻显存，eager，关闭编译和预热 |
| `relayvsr_compile.json` | Wan 和 Flash 的 Transformer block 都启用 `torch.compile`，启动时预热 |
| `relayvsr_block_offload.json` | Wan 权重放在 CPU，用 LightX2V 双缓冲和预取逐块上卡；Flash 常驻 |
| `relayvsr_sp.json` | 两卡序列并行；Wan 和 Flash 都复用 LightX2V 的 Ulysses all-to-all |

选择配置时修改脚本中的 `--config_json`，例如将其改为 `"${lightx2v_path}/configs/relayvsr/relayvsr_block_offload.json"`，即可启用 block 权重卸载。

示例脚本默认使用 `relayvsr_compile.json`，同时开启编译和启动预热。`use_compile` 和 `warmup` 是启动 JSON 参数：两者都为 `true` 时，模型加载后自动在预热过程中触发编译；只开启 `warmup` 可预热 eager；只开启 `use_compile` 则在首次推理时触发编译。需要关闭两者时，将脚本的 `--config_json` 改回 `"${lightx2v_path}/configs/relayvsr/relayvsr.json"`。编译覆盖 Wan 的 30 个 block 和 Flash 的 10 个 block，包括 Flash 条件帧与视频帧两条路径；前后处理、权重搬运和视频读写不在编译图内。

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls relayvsr --task sr --model_path /models/RelayVSR \
  --config_json configs/relayvsr/relayvsr_sp.json \
  --video_path /videos/input.mp4 --sr_ratio 2 --seed 42 \
  --save_result_path /videos/output_x2.mp4
```

Flash 只有两个 KV groups，序列并行最多支持两卡。分块线性计算可能带来 BF16 数值差异；需要逐元素对齐时用默认单卡 eager。

每份配置显式列出交叉注意力后端、卸载开关、窗口长度和编码参数，方便修改。默认使用 `torch_sdpa`、16 帧窗口和源帧率。CPU 编码使用 `libx265`、`ultrafast`、CRF 18，并设置 `hvc1` 标记；`encoder_threads` 控制 x265 工作线程池。

## 服务启动与请求

修改仓库和模型路径后，启动单卡服务：

```bash
lightx2v_path=/path/to/LightX2V
model_path=/models/RelayVSR

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
  --model_cls relayvsr --task sr \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/relayvsr/relayvsr_compile.json" \
  --host 0.0.0.0 --port 8000
```

示例通过启动配置开启编译和预热。等待模型加载、预热完成后，在另一终端检查服务状态并提交 x4 请求。输入和输出路径都应填写服务端可访问的绝对路径：

```bash
curl -fsS http://127.0.0.1:8000/v1/service/status

curl -fsS -X POST http://127.0.0.1:8000/v1/tasks/video/ \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/videos/input.mp4",
    "sr_ratio": 4,
    "save_result_path": "/videos/output_x4.mp4",
    "seed": 42
  }'
```

提交接口异步返回 `task_id`。将其填入下面的变量，查询任务状态：

```bash
task_id=REPLACE_WITH_RETURNED_TASK_ID
curl -fsS "http://127.0.0.1:8000/v1/tasks/${task_id}/status"
```

状态接口的 `status` 为 `completed` 时，视频已保存到服务端的 `save_result_path`；若为 `failed`，查看返回的 `error`。常驻服务可连续提交请求，复用已加载的模型；每个请求分别设置输入路径、输出路径和 `seed`。将 `sr_ratio` 改为 `2` 即可输出 x2。编译、预热和编码参数通过启动 JSON 配置，修改后需重启服务。
