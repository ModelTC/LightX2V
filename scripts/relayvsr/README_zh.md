# RelayVSR 视频超分

[English](README.md) | [简体中文](README_zh.md)

## 模型下载

[RelayVSR 官方 GitHub 仓库](https://github.com/kopperx/RelayVSR)

- 原版 4× 模型：[kopper/RelayVSR](https://huggingface.co/kopper/RelayVSR)
- LightX2V 训练的原生 2× 模型：[lightx2v/RelayVSR_X2](https://huggingface.co/lightx2v/RelayVSR_X2)

根据所选模型版本，将下表中对应的两个权重文件下载到同一模型目录。

## 模型入口

| `model_cls` | 权重 | 输出规则 |
| --- | --- | --- |
| `relayvsr_x4` | RelayVSR HF `wan.safetensors`、`flash.safetensors` | 计算 4×，通过 `sr_ratio=2/4` 选择 2× 或 4× 输出 |
| `relayvsr_x2` | LightX2V HF `wan_x2.safetensors`、`flash_x2.safetensors` | 计算 2×，只接受 `sr_ratio=2` |

两个入口会校验权重格式、倍率及关键结构，混用 2×/4× 权重会直接报错。
两个入口均不接受 `size`；输出尺寸由输入尺寸和 `sr_ratio` 决定。

4× 与 2× 的配置分别带 `_x4`、`_x2` 后缀，须与 `model_cls` 对应。

## 4× 用法

修改脚本中的路径后，在仓库根目录运行：

```bash
bash scripts/relayvsr/run_relayvsr_x4_sr.sh
```

x4 模型也支持 2 倍超分：将脚本中的 `--sr_ratio 4` 改为 `--sr_ratio 2` 即可。模型内部先计算 4×，再将结果缩小到 2×。

## 2× 用法

修改脚本中的路径后，在仓库根目录运行：

```bash
bash scripts/relayvsr/run_relayvsr_x2_sr.sh
```

## 执行模式

| 配置 | 行为 |
| --- | --- |
| `relayvsr_x4.json` | 权重常驻显存 |
| `relayvsr_x4_block_offload.json` | Wan 按 block 卸载权重，Flash 常驻显存 |
| `relayvsr_x4_sp.json` | 双卡序列并行 |

2× 模型提供对应的 `_x2` 配置。在脚本中通过 `--config_json` 选择执行模式。

所有配置默认开启编译和预热；需要关闭时，在所选配置中将 `use_compile` 和 `warmup` 都设为 `false`。

设置 `PROFILING_DEBUG_LEVEL=1` 或 `2` 时，RelayVSR 记录完整 `run_pipeline` 的耗时，不含模型加载和启动预热。

## 服务启动与请求

修改仓库和模型路径后，启动单卡服务：

```bash
lightx2v_path=/path/to/LightX2V
model_path=/models/RelayVSR

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.server \
  --model_cls relayvsr_x4 --task sr \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/relayvsr/relayvsr_x4.json" \
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

常驻服务可连续提交请求，复用已加载的模型；每个请求分别设置输入路径、输出路径和 `seed`。将 `sr_ratio` 改为 `2` 即可输出 x2。
