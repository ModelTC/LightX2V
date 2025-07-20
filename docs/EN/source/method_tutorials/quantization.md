# Model Quantization

LightX2V supports quantization inference for linear layers in `Dit`, supporting `w8a8-int8`, `w8a8-fp8`, `w8a8-fp8block`, `w8a8-mxfp8`, and `w4a4-nvfp4` matrix multiplication.


## Producing Quantized Models

Use LightX2V's convert tool to convert models into quantized models. Refer to the [documentation](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme.md).

## Loading Quantized Models for Inference

Write the path of the converted quantized weights to the `dit_quantized_ckpt` field in the [configuration file](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization).

By specifying --config_json to the specific config file, you can load the quantized model for inference.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization) are some running scripts for use.

## Advanced Quantization Features

For details, please refer to the documentation of the quantization tool [LLMC](https://github.com/ModelTC/llmc/blob/main/docs/en/source/backend/lightx2v.md)
