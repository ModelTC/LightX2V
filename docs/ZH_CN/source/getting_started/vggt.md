# VGGT-1B BF16 推理

[官方项目](https://github.com/facebookresearch/vggt)。将 facebook/VGGT-1B 的 `model.pt` 放入模型目录。原提交使用的权重 revision 为 `860abec7937da0a4c03c41d3c269c366e82abdf9`。

```bash
python -m pip install -r scripts/vggt/requirements-mlu.txt
MODEL_PATH=/path/to/VGGT-1B INPUT_PATH=/path/to/images bash scripts/vggt/run_vggt.sh
```

通过 `PLATFORM` 和对应平台的设备可见性变量选择设备。脚本使用 BF16 和 seed 42，可通过 `SEED`、`SAVE_RESULT_PATH` 或末尾的 CLI 参数覆盖。

输入为单张图像或包含图像的目录，目录按文件名字典序读取；Python Runner 还接受有序图像路径列表。图像数量可变，至少一张；32 张是历史评测样本的选取数量，不是模型的固定输入限制。沿用原有 518 × 518 补边缩放预处理。

输出 `.npz` 包含相机位姿、内外参、深度、点图、置信度和原图边界。本集成范围为重建，不包含 tracking。模型声明和加载位于 `weights/`，计算位于 `infer/`；平台选择沿用框架，矩阵乘、注意力、LayerNorm 和 Conv2d 复用公共注册算子。
