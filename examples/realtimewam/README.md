<div align="center" style="font-family: charter;">
<h1>RealtimeWAM:<br>One-Step Asynchronous World Action Models</h1>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/LightX2V.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/LightX2V)&nbsp;

Chengtao Lv<sup>1*</sup>, Jinyang Du<sup>2*</sup>, Shuyi Feng<sup>1</sup>, Yang Yong<sup>3</sup>, Shiqiao Gu<sup>3</sup>,<br>
Shunzi Yang<sup>2</sup>, Ruihao Gong<sup>2</sup>📧, Shen Ren<sup>4</sup>, Tianwei Zhang<sup>1</sup>, Wenya Wang<sup>1</sup>📧

<sup>1</sup>Nanyang Technological University, <sup>2</sup>Beihang University,<br>
<sup>3</sup>Sensetime, <sup>4</sup>Continental Automotive Singapore

(* denotes equal contribution; 📧 denotes corresponding author.)

</div>

### 💡 Why RealtimeWAM

* 🥇 Pioneer work: The first **one-step World Action Model (WAM)** with near-lossless performance, built on **any Mixture-of-Transformers (MoT)-based WAM architecture**.
* 🏆 Superior performance: Maintains **less than 1% average accuracy drop** compared with multi-step counterparts across diverse benchmarks, including **RoboTwin 2.0**, **LIBERO**, and **LIBERO-Plus**.
* ⚡ Extreme acceleration: With one-step distillation, asynchronous inference, CUDA Graph, and efficient kernels, achieves **24.55× end-to-end speedup** on **Fast-WAM** (**12.2 ms**) and **18.42×** on **Faster-WAM** (**11.9 ms**) on a single **NVIDIA H100**, enabling real-time action generation.

## 💪 TODO

| Resource | RealtimeWAM<sup>*</sup> | RealtimeWAM<sup>†</sup> |
| --- | :---: | :---: |
| Inference code (with latency profiling) | ✅ | ❌ |
| Evaluation code (LIBERO, LIBERO-Plus, RoboTwin 2.0) | ✅ | ❌ |
| Checkpoints | ✅ | ✅ |
| Training code | ❌ | ❌ |

<sup>*</sup> and <sup>†</sup> denote variants based on Fast-WAM and Faster-WAM, respectively.

## ✨ Quick Start

### Environment

RealtimeWAM uses the LightX2V environment. You can follow the environment setup instructions in the [LightX2V README](../../README.md). For comprehensive usage instructions, please refer to the LightX2V documentation: **[English Docs](https://lightx2v-en.readthedocs.io/en/latest/) | [中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**. **We highly recommend using the Docker environment, as it is the simplest and fastest way to set up the environment. For details, please refer to the Quick Start section in the documentation.** If you prefer to build from source, please refer to [**Building from Source**](../../README.md#building-from-source) in the LightX2V README.

### Download Checkpoints

### Inference

Here is the corresponding inference command.

```shell
sh scripts/realtimewam/run_libero_i2va.sh
```
