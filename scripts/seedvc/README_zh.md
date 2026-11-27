# Seed-VC V1 音频转换

输入应用侧已经准备好的内容特征、参考 mel、音色特征和可选 F0，通过 LightX2V 原生 Length Regulator、DiT、CFM 采样器和 BigVGAN 声码器生成完整 WAV/FLAC。

当前支持 V1 44.1 kHz F0 歌声模型、V1 22.05 kHz 语音模型，以及本地 `seed-vc-lora-v1` 格式的固定歌声 LoRA。V2 AR 风格转换、tiny/HiFT、实时声卡输入输出及训练不在当前接入范围内。

## 权重

权重来源：

- [Seed-VC V1 核心权重与 YAML](https://huggingface.co/Plachta/Seed-VC/tree/main)
- [44.1 kHz BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)
- [22.05 kHz BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)

使用前修改启动脚本顶部的 `lightx2v_path` 和 `model_path`，分别指向本地仓库和权重根目录；`data_dir` 默认读取同级 `lightx2v_examples` 仓库中的共享样本。BigVGAN 推理代码已包含在 LightX2V 中，原始权重无需转换。权重根目录包含 `svc/`、`vc/`，组织如下：

```text
Seed-VC/
├── svc/
│   ├── config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
│   ├── DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
│   ├── adapter.pth                 # 可选，自己的歌声 LoRA
│   └── vocoder/
│       ├── config.json
│       └── bigvgan_generator.pt
└── vc/
    ├── config_dit_mel_seed_uvit_whisper_small_wavenet.yml
    ├── DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
    └── vocoder/
        ├── config.json
        └── bigvgan_generator.pt
```

## 样例数据

样例位于 [lightx2v_examples/a2a/seedvc](https://github.com/ModelTC/lightx2v_examples/tree/main/a2a/seedvc)。在 LightX2V 根目录执行以下命令，将样例仓库放在同级目录，仅检出 Seed-VC 数据：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/ModelTC/lightx2v_examples.git ../lightx2v_examples
git -C ../lightx2v_examples sparse-checkout set a2a/seedvc
```

脚本中的 `data_dir` 为 `${lightx2v_path}/../lightx2v_examples/a2a/seedvc`。使用其他位置时修改该变量；启用预热时，同时在部署配置中设置 `warmup_sample_path`。

## 正常推理样例

以下命令在 LightX2V 根目录执行。脚本只接收功能编号，默认 1；根据编号自动选择输入和模型，功能 6 加载 LoRA。所有命令最终都输出可播放音频。

| 编号 | 功能 | 输入 sample | 与默认请求的差异 | 输出文件（位于 `save_results/seedvc/`） |
|---|---|---|---|---|
| 1 | 歌声音色转换，保留源音高 | `../lightx2v_examples/a2a/seedvc/svc.json` | 无 | `function_1.wav` |
| 2 | 歌声转换并升高两个半音 | 同上 | `semitone_shift=2` | `function_2.wav` |
| 3 | 歌声转换并自动对齐参考音高中位数 | 同上 | `auto_f0_adjust=true` | `function_3.wav` |
| 4 | 歌声转换，输出时长约为源音频的 1.2 倍 | 同上 | `length_adjust=1.2` | `function_4.wav` |
| 5 | 语音音色转换 | `../lightx2v_examples/a2a/seedvc/vc.json` | 加载 V1 语音权重 | `function_5.wav` |
| 6 | 加载 LoRA 的歌声音色转换 | `../lightx2v_examples/a2a/seedvc/svc_lora.json` | 加载 `svc/adapter.pth`，使用单依纯参考音色 | `function_6.wav` |

```bash
bash scripts/seedvc/run_seedvc.sh 1
```

```bash
bash scripts/seedvc/run_seedvc.sh 2
```

```bash
bash scripts/seedvc/run_seedvc.sh 3
```

```bash
bash scripts/seedvc/run_seedvc.sh 4
```

```bash
bash scripts/seedvc/run_seedvc.sh 5
```

```bash
bash scripts/seedvc/run_seedvc.sh 6
```

1–4、6 共用歌声模型，其中 6 加载 LoRA；5 使用语音模型。歌声和语音输出分别为 44.1 kHz 和 22.05 kHz 单声道。歌声样本会根据源音频和参考音频长度自动分段推理并重叠拼接。源音频放在 `source/`，参考音频放在 `reference/`，可直接试听其中的 FLAC 文件。

功能 1–4 使用 `svc.json`，功能 6 保留独立的 `svc_lora.json`。两份配置指向同一份 `source/svc.flac` 和 `source/svc.safetensors`；区别仅在参考音色：前者使用上游示例，后者使用 `reference/shan_yi_chun.*` 中的单依纯 20 秒参考音频及特征。功能 6 同时加载对应的 LoRA。替换其他歌手的 LoRA 时，应同时准备与其匹配的参考特征。

功能 6 需要将匹配基础权重的 adapter 放在 `svc/adapter.pth`。启动时检查 adapter 格式、基础权重 SHA256、模型配置签名及所有目标张量，随后合并到 FP32 权重；不支持每请求切换 adapter。这个 adapter 格式来自本地训练扩展，不是 PEFT，仓库不附带私人 adapter。

## 输入与推理参数

所有样例共用仓库外的 `../lightx2v_examples/a2a/seedvc/`。JSON 描述 sample，safetensors 保存特征；三份样本配置共用源数据，参考数据按音色组织。

```text
../lightx2v_examples/a2a/seedvc/
├── svc.json
├── svc_lora.json
├── vc.json
├── source/
│   ├── svc.flac
│   ├── svc.safetensors
│   ├── vc.flac
│   └── vc.safetensors
└── reference/
    ├── s2p1_svc.flac
    ├── s2p1_svc.safetensors
    ├── s2p1_vc.flac
    ├── s2p1_vc.safetensors
    ├── shan_yi_chun.flac
    └── shan_yi_chun.safetensors
```

`s2p1_svc` 和 `s2p1_vc` 是同一上游参考音频分别按歌声、语音模型规格准备的数据。

| 字段 | 形状/含义 |
|---|---|
| `source_content` / `reference_content` | `[1, T, 768]`，Whisper-small 内容特征 |
| `reference_mel` | `[1, 128, T]`（歌声）或 `[1, 80, T]`（语音） |
| `style` | `[1, 192]`，CAMPPlus 参考音色特征 |
| `source_f0` / `reference_f0` | `[1, T]`，歌声模型使用的原始 F0 |
| `source_mel_frames` | JSON 中的源音频 mel 帧数，用于长度对齐 |

JSON 还包含格式版本、模型变体、采样率、hop length、前置编码器身份和相对特征路径。Whisper 本身输出 FP16，样本以 FP16 保存该输出，读取后恢复 FP32，没有额外量化。新音频的特征提取在应用仓库完成，并遵循上游的重采样、mel 计算、参考截取和 F0 处理方式。

CLI 与 HTTP 共用这些请求参数：`sample_path`、`seed`、`diffusion_steps`（默认 30）、`cfg_rate`（默认 0.7）、`length_adjust`（默认 1）、`semitone_shift`（默认 0）和 `auto_f0_adjust`（默认 false）。`cfg_rate=0` 使用单分支；音高参数只适用于歌声权重。输出格式由 `save_result_path` 的 `.wav` / `.flac` 后缀决定。

## 部署方式：与功能选择相互独立

单卡使用 `run_seedvc.sh`，双卡使用 `run_seedvc_tp2.sh`，两者都只传功能编号 1–6。部署配置由推理命令中的 `--config_json` 指定。

| 配置 | 用途 |
|---|---|
| `seedvc.json` | 默认单卡推理，关闭编译和预热 |
| `seedvc_offload.json` | DiT block 权重放 CPU，降低显存占用；声码器保留在 GPU |
| `seedvc_tp2.json` | DiT 两卡张量并行，声码器仅在 rank 0 解码 |
| `seedvc_compile.json` | 编译 DiT block，并开启启动预热 |

CPU offload 或编译模式直接修改 `run_seedvc.sh` 中的 `--config_json`：

```text
CPU offload: --config_json ${lightx2v_path}/configs/seedvc/seedvc_offload.json
编译与预热: --config_json ${lightx2v_path}/configs/seedvc/seedvc_compile.json
恢复默认: --config_json ${lightx2v_path}/configs/seedvc/seedvc.json
```

修改后仍只传功能编号，例如：

```bash
bash scripts/seedvc/run_seedvc.sh 1
```

双卡脚本已指定 `seedvc_tp2.json`：

```bash
bash scripts/seedvc/run_seedvc_tp2.sh 1
```

6 个功能共用所选的部署配置。功能 6 自动加载 `svc/adapter.pth`，也可在部署配置中通过 `adapter_path` 指定其他 adapter，无需额外的 LoRA 配置文件。

单卡输出为 `save_results/seedvc/function_<编号>.wav`，双卡为 `save_results/seedvc/function_<编号>_tp2.wav`。需要保留不同单卡部署方式的结果时，修改脚本中的 `--save_result_path`。

单卡和 HTTP 服务使用 GPU 0，双卡使用 GPU 0、1；需要换卡时，直接修改对应启动脚本中的 `export CUDA_VISIBLE_DEVICES=...`。Seed-VC 内部固定使用 FP32 精度，无需在脚本中设置 `DTYPE`。

`use_compile` 和 `warmup` 是配置中的独立开关，默认均为 false；编译配置同时开启两者。预热默认使用同级 `lightx2v_examples/a2a/seedvc/` 下对应的样本，不保存音频。自定义数据可通过 `warmup_sample_path` 指定代表性样本，新的输入形状仍可能触发编译。

## HTTP 服务

```bash
bash scripts/seedvc/server/run_server.sh svc
```

在另一个终端提交请求并下载最终音频：

```bash
python scripts/seedvc/server/post_a2a.py \
  --sample-path "$(pwd)/../lightx2v_examples/a2a/seedvc/svc.json" \
  --output save_results/seedvc/http_svc.wav
```

默认服务地址为 `http://127.0.0.1:8018`。`sample_path` 必须是服务器可读路径；客户端使用 `--url` 指定远程服务。语音服务启动参数改为 `vc`，请求 sample 改为 `vc.json`。LoRA 服务使用 `svc_lora` 启动，并提交 `svc_lora.json`。客户端等待任务结束，从结果 API 下载音频。
