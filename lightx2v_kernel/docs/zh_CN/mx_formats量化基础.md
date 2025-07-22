# MX-Formats量化基础

**注：下文关注于分享MX-Formats量化相对于Per-Row/Per-Column量化的区别，以及与Cutlass Block Scaled GEMMs配合使用需要满足的一些布局要求。**

### 数据格式与量化因子
目标数据格式参考：[MX-Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，需要注意的是，我们这里不需要将raw data和scale factor打包在一起

源数据格式：fp16/bf16

目标数据格式：mxfp4/6/8

量化因子数据格式：E8M0, Per-Row/Per-Column量化的量化因子一般以fp32进行存储，而E8M0与fp32数值范围一致，经过rounding后可直接存储量化因子，缺点是尾数的丢失会影响精度。

量化粒度：\[1X32\]

量化维度：以Cutlass GEMM的规范，M N K表示矩阵乘的三个维度，需要沿着K维度量化

### Rounding与Clamp
不同于软件模拟，CUDA可以通过PTX或者内置函数高性能地便捷地来完成繁琐的Rouding和Clamp操作。
例如，`cvt.rn.satfinite.e2m1x2.f32` 可以将两个fp32类型的输入，转换为​两个fp4类型的输出
Rounding模式为：`rn`，​round-to-nearest-even​
Clamp模式为：`satfinite`，钳制到目标范围内的最大有限值，​排除无穷和 NaN
更多数据类型和模式参考：[PTX cvt指令](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)

### 数据布局与量化因子布局
数据布局
- mxfp4需要两两打包为uint8
- mxfp6需要每4个打包为3个uint8，格式参考：[mxfp6 cutlass mm 格式打包](https://github.com/ModelTC/LightX2V/blob/main/lightx2v_kernel/csrc/gemm/mxfp6_quant_kernels_sm120.cu#L74)

量化因子布局
Cutlass Block Scaled GEMMs为了满足矩阵运算加速，对量化因子布局有特殊的swizzle要求
参考：[Scale Factor Layouts](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts)

### 量化方法
了解完上述后，目标数据和量化因子两者自身数值的求解，可参考[nvfp4量化基础](https://github.com/theNiemand/lightx2v/blob/main/lightx2v_kernel/docs/zh_CN/nvfp4%E9%87%8F%E5%8C%96%E5%9F%BA%E7%A1%80.md)，注意MX-Formats无需量化scale本身
