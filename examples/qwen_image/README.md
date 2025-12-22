# Qwen Image Edit 2511 Examples

This directory contains usage examples for the Qwen Image Edit 2511 model, providing two different inference approaches.

## Installation

First, clone the repository and install dependencies:

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
pip install -v -e .
```

## Model Download

Before using the example scripts, you need to download the corresponding models. All models can be downloaded from the following address:

**Model Download URL:** https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning

Please ensure that the models are downloaded to the correct directory so that the scripts can load them properly.

## Usage

Navigate to the example directory:

```bash
cd examples/qwen_image/
```

### Method 1: Using Step Distillation + FP8 Quantization Model

Run the `qwen_2511_fp8.py` script, which uses a model optimized with step distillation and FP8 quantization:

```bash
python qwen_2511_fp8.py
```

This method reduces inference steps through step distillation technology, while using FP8 quantization to reduce model size and memory usage, achieving faster inference speed.

### Method 2: Using Qwen-Image-Edit-2511 Model + Distillation LoRA

Run the `qwen_2511_with_distill_lora.py` script, which uses the Qwen-Image-Edit-2511 base model with distillation LoRA:

```bash
python qwen_2511_with_distill_lora.py
```

This method uses the complete Qwen-Image-Edit-2511 model and optimizes it through distillation LoRA, improving inference efficiency while maintaining model performance.
