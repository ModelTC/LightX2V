<div align="center" style="font-family: charter;">
<h1>RealtimeWAM:<br>One-Step Asynchronous World Action Models</h1>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/LightX2V.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/LightX2V)&nbsp;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RealtimeWAM-yellow)](https://huggingface.co/lightx2v/RealtimeWAM)&nbsp;

</div>

### 💡 Why RealtimeWAM

* 🥇 Pioneer work: The first **one-step World Action Model (WAM)** with near-lossless performance, built on **any Mixture-of-Transformers (MoT)-based WAM architecture**.
* 🏆 Superior performance: Maintains **less than 1% average accuracy drop** compared with multi-step counterparts across diverse benchmarks, including **RoboTwin 2.0**, **LIBERO**, and **LIBERO-Plus**.
* ⚡ Extreme acceleration: With one-step distillation, asynchronous inference, CUDA Graph, and efficient kernels, achieves **24.55× end-to-end speedup** on **Fast-WAM** (**12.2 ms**) and **13.56×** on **Faster-WAM** (**16.1 ms**) on a single **NVIDIA H100**, enabling real-time action generation. Latency includes VAE encoding and excludes text encoding, which is performed once per episode.

## 💪 TODO

| Resource | RealtimeWAM<sup>*</sup> | RealtimeWAM<sup>†</sup> |
| --- | :---: | :---: |
| Checkpoints | ✅ | ✅ |
| Inference code | ✅ | ✅ |
| Evaluation code | ✅ | ✅ |
| Training code | ❌ | ❌ |

<sup>*</sup> and <sup>†</sup> denote variants based on Fast-WAM and Faster-WAM, respectively.

## ✨ Quick Start

### Environment

#### Inference

RealtimeWAM uses the LightX2V environment. You can follow the environment setup instructions in the [LightX2V README](../../README.md). For comprehensive usage instructions, please refer to the LightX2V documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**. **We highly recommend using the Docker environment, as it is the simplest and fastest way to set up the environment. For details, please refer to the Quick Start section in the documentation.** If you prefer to build from source, please refer to [**Building from Source**](../../README.md#building-from-source) in the LightX2V README.

#### Evaluation

We provide [uv](https://docs.astral.sh/uv/getting-started/installation/)-based environment setup scripts for LIBERO, LIBERO-Plus, and RoboTwin 2.0. We recommend **CUDA 12.8 and PyTorch 2.7.1**. The scripts create independent Python 3.10 environments with LightX2V and PyTorch 2.7.1, and initialize the corresponding benchmark submodules. Run the following commands from the LightX2V repository root. Install only the environment you need.

For **LIBERO and LIBERO-Plus**, install and activate their shared evaluation environment:

```bash
bash scripts/bench/robotics/install_env.sh libero
source .venvs/libero/bin/activate
```

For **RoboTwin**, first install CUDA Toolkit 12.8 and set `CUDA_HOME` to its installation path, then install and activate the evaluation environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/bench/robotics/install_env.sh robotwin
source .venvs/robotwin/bin/activate
```

Dependencies are listed in [`requirements_libero.txt`](https://github.com/chengtao-lv/LightX2V/blob/main/scripts/bench/robotics/requirements_libero.txt) and [`requirements_robotwin.txt`](https://github.com/chengtao-lv/LightX2V/blob/main/scripts/bench/robotics/requirements_robotwin.txt).

### Download Checkpoints

Download the RealtimeWAM checkpoints from [lightx2v/RealtimeWAM](https://huggingface.co/lightx2v/RealtimeWAM) on Hugging Face. The simplest option is to download the **checkpoints with the action expert's LoRA adapters already merged**.

| Checkpoint | Benchmark | Backbone | Data stats |
| --- | --- | --- | --- |
| [realtimewam_libero_fast.pt](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/realtimewam_libero_fast.pt) | LIBERO / LIBERO-Plus | Fast-WAM | [libero_dataset_stats.json](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/libero_dataset_stats.json) |
| [realtimewam_libero_faster.pt](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/realtimewam_libero_faster.pt) | LIBERO / LIBERO-Plus | Faster-WAM | [libero_dataset_stats.json](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/libero_dataset_stats.json) |
| [realtimewam_robotwin_fast.pt](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/realtimewam_robotwin_fast.pt) | RoboTwin 2.0 | Fast-WAM | [robotwin_dataset_stats.json](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/robotwin_dataset_stats.json) |
| [realtimewam_robotwin_faster.pt](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/realtimewam_robotwin_faster.pt) | RoboTwin 2.0 | Faster-WAM | [robotwin_dataset_stats.json](https://huggingface.co/lightx2v/RealtimeWAM/resolve/main/robotwin_dataset_stats.json) |

```bash
huggingface-cli download lightx2v/RealtimeWAM \
  realtimewam_libero_fast.pt \
  realtimewam_libero_faster.pt \
  realtimewam_robotwin_fast.pt \
  realtimewam_robotwin_faster.pt \
  libero_dataset_stats.json \
  robotwin_dataset_stats.json \
  --local-dir ./checkpoints/RealtimeWAM
```

### Inference

The end-to-end inference latency of **RealtimeWAM<sup>*</sup>** and **RealtimeWAM<sup>†</sup>** can be measured using the following scripts.

```bash
# RealtimeWAM*, based on Fast-WAM
bash scripts/realtimewam/run_libero_fastwam_i2va.sh

# RealtimeWAM†, based on Faster-WAM
bash scripts/realtimewam/run_libero_fasterwam_i2va.sh
```

We provide example inference inputs in [`examples/realtimewam/assets/`](assets/).

The measured latency is reported in the logs under **`RealtimeWAM End-to-End Latency (Excluding Text Encoding)`**, in milliseconds. It includes VAE encoding and video/action inference, while excluding text encoding.

Inference latency comparisons are shown below. Speedups are relative to the original model on the same GPU. The scripts above can be used to reproduce the inference latencies reported in our paper. LightX2V achieves slightly lower latency than reported in the paper.

| GPU | Fast-WAM (ms) | RealtimeWAM<sup>*</sup> (ms) | Speedup<sup>*</sup> | Faster-WAM (ms) | RealtimeWAM<sup>†</sup> (ms) | Speedup<sup>†</sup> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA H100 | 300 | **12.2** | **24.56×** | 219 | **16.1** | **13.56×** |
| NVIDIA RTX 4090D | 481 | **27.2** | **17.68×** | 362 | **44.7** | **8.11×** |

### Evaluation

#### [Optional] Prepare Benchmark Resources

Skip this step if the benchmark assets are already installed. Use the repository-pinned benchmark versions; training demonstration datasets are not required for evaluation.

* **LIBERO:** Task definitions and initial states are included in the submodule.
* **LIBERO-Plus:** Download and extract `assets.zip` from `Sylvest/LIBERO-plus` on Hugging Face into `LIBERO-plus/libero/libero/assets/`.
* **RoboTwin:** Run `bash script/_download_assets.sh` from the RoboTwin root directory to download and configure the simulation assets.

For detailed preparation instructions, refer to the original repositories: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus#-installation), and [RoboTwin](https://github.com/robotwin-Platform/robotwin).

#### Run Evaluation

Run the following commands from the LightX2V repository root. Set the Wan2.2 model path and the GPUs to use:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.2-TI2V-5B
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

The examples below evaluate **RealtimeWAM<sup>*</sup>** with one-step action denoising. Use the downloaded checkpoint and its matching dataset statistics. `CKPT_PATH`, `DATASET_STATS_PATH`, and `CONFIG_JSON` select the policy weights, normalization statistics, and inference configuration; `OUT` specifies the result directory.

For **LIBERO**, evaluate 40 tasks across four suites, with 50 trials per task. The policy starts after 30 settling steps:

```bash
source .venvs/libero/bin/activate
CKPT_PATH=./checkpoints/RealtimeWAM/realtimewam_libero_fast.pt \
DATASET_STATS_PATH=./checkpoints/RealtimeWAM/libero_dataset_stats.json \
CONFIG_JSON=configs/realtimewam/libero_fastwam_i2va.json \
OUT=evaluate_results/libero/realtimewam_fast \
bash scripts/bench/robotics/run_libero.sh model=realtimewam seed=42 \
  EVALUATION.num_trials=50 EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

For **LIBERO-Plus**, reuse the LIBERO checkpoint and statistics:

```bash
source .venvs/libero/bin/activate
CKPT_PATH=./checkpoints/RealtimeWAM/realtimewam_libero_fast.pt \
DATASET_STATS_PATH=./checkpoints/RealtimeWAM/libero_dataset_stats.json \
CONFIG_JSON=configs/realtimewam/libero_fastwam_i2va.json \
OUT=evaluate_results/libero_plus/realtimewam_fast \
bash scripts/bench/robotics/run_libero_plus.sh model=realtimewam seed=42 \
  EVALUATION.num_trials=1 EVALUATION.num_steps_wait=30 EVALUATION.replan_steps=10
```

For **RoboTwin 2.0**, evaluate 50 tasks under both clean and randomized conditions, with 100 episodes per task and condition using unseen instructions:

```bash
source .venvs/robotwin/bin/activate
CKPT_PATH=./checkpoints/RealtimeWAM/realtimewam_robotwin_fast.pt \
DATASET_STATS_PATH=./checkpoints/RealtimeWAM/robotwin_dataset_stats.json \
CONFIG_JSON=configs/realtimewam/robotwin_fastwam_i2va.json \
OUT=evaluate_results/robotwin/realtimewam_fast \
bash scripts/bench/robotics/run_robotwin.sh model=realtimewam seed=42 \
  EVALUATION.eval_num_episodes=100 EVALUATION.instruction_type=unseen EVALUATION.replan_steps=24
```

To evaluate **RealtimeWAM<sup>†</sup>**, replace the checkpoint suffix `_fast.pt` with `_faster.pt`, select the corresponding configuration below, and use a separate output directory.

| Variant | LIBERO / LIBERO-Plus config | RoboTwin config |
| --- | --- | --- |
| RealtimeWAM<sup>*</sup> | `libero_fastwam_i2va.json` | `robotwin_fastwam_i2va.json` |
| RealtimeWAM<sup>†</sup> | `libero_fasterwam_i2va.json` | `robotwin_fasterwam_i2va.json` |

All configurations are under `configs/realtimewam/` and use one action denoising step.

The evaluation scripts above can also be used to reproduce the success rates reported in our paper. Results below are success rates (%). NFE denotes the number of denoising steps for video and action generation.

| Method | Video NFE | Action NFE | RoboTwin 2.0 Clean ↑ | RoboTwin 2.0 Random ↑ | RoboTwin 2.0 Overall ↑ | LIBERO Overall ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast-WAM | 1 | 10 | 91.82 | 91.19 | 91.51 | 97.0 |
| **RealtimeWAM<sup>*</sup>** | 1 | 1 | 91.96 | 89.72 | 90.84 | 97.0 |
| Faster-WAM | 1 | 10 | 93.20 | 92.66 | 92.93 | 98.9 |
| **RealtimeWAM<sup>†</sup>** | 1 | 1 | 92.98 | 92.30 | 92.64 | 99.0 |
