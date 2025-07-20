# 模型量化

lightx2v支持对`Dit`中的线性层进行量化推理，支持`w8a8-int8`, `w8a8-fp8`, `w8a8-fp8block`, `w8a8-mxfp8` 和 `w4a4-nvfp4`的矩阵乘法。


## 生产量化模型

使用LightX2V的convert工具，将模型转换成量化模型，参考[文档](https://github.com/ModelTC/lightx2v/tree/main/tools/convert/readme_zh.md)。

## 加载量化模型进行推理

将转换后的量化权重的路径，写到[配置文件](https://github.com/ModelTC/lightx2v/blob/main/configs/quantization)中的`dit_quantized_ckpt`中。

通过指定--config_json到具体的config文件，即可以加载量化模型进行推理

[这里](https://github.com/ModelTC/lightx2v/tree/main/scripts/quantization)有一些运行脚本供使用。

## 高阶量化功能

具体可参考量化工具[LLMC的文档](https://github.com/ModelTC/llmc/blob/main/docs/zh_cn/source/backend/lightx2v.md)
