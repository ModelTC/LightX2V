# nvfp4量化基础

### 数据格式

fp的计算方式是:

`ans = (-1)^s * 2^(p-b) * (1 + d1/2 + d2/4 + d3/8 + ...)`

其中，`b = 2^(e-1) - 1`，p表示指数位的值，d1, d2, d3表示尾数位的值

对于fp4，格式是E2M1，上述的式子简化为：

`b = 2^(e-1) - 1 = 2^(2-1) - 1 = 1`

`ans = (-1)^s * 2^(p-1) * (1 + d1/2)`

举例：0101

`s=0, p=(10)=2, d1=1`

`ans = 2^0 * 2^(2-1) * (1 + 1/2) = 3`

正常的fp数据格式，还会有部分数据表示inf和nan，最大只能表示±3，特化到nvfp4，取消了inf和nan，最大可以表示±6

特殊的，其中0000表示+0，1000表示-0，0001表示0.5，1001表示-0.5

综上：

| E2M1 | 0000 | 0001 | 0010 | 0011 | 0100 | 0101 | 0110 | 0111 | 1000 | 1001 | 1010 | 1011 | 1100 | 1101 | 1110 | 1111 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ans  | +0   | 0.5  | 1.0  | 1.5  | 2.0  | 3.0  | 4.0  | 6.0  | +0   | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |


### 量化过程

**weight和act都是per group量化，group size都是16，量化scale以fp8(e4m3)格式存储**

由于量化scale要用fp8存储，需要对scale也进行放缩，所以fp4量化的过程和常见的w8a8-int8过程，有一些不同

量化过程如下：

给定一组数，记作`X`

#### 计算scale

`scale1 = max(abs(Xg)) / 6.0`

其中Xg表示一个group的数，6.0表示nvfp4的最大值

#### 量化scale

`global_scale = 6.0 * 448.0 / max(abs(X))`

`scale2 = global_scale * scale1`

即 `scale2 = 6.0 * 448.0 / max(abs(X)) * max(abs(Xg)) / 6.0`

即 `scale2 = max(abs(Xg)) / max(abs(X)) * 448.0`

此时scale2被放缩到fp8(e4m3)的范围，然后对scale2进行量化到fp8

`scale2_fp8 = quant_fp8(scale2)`

`scale2_fp8`则作为最终的矩阵乘法所需的量化scale参数

#### 量化X

`scale2_fp32 = cvt2fp32(scale2_fp8)`

`Xquant = quant_fp4(X * global_scale / scale2_fp32)`

则 `Xquant ≈ quant_fp4(X / scale1)`

#### fp4矩阵乘法

`ans = Aquant * Bquant * Ascale2 * Bscale2 / Aglobal_scale / Bglobal_scale`

即 `ans ≈ Aquant * Bquant * Aglobal_scale * Ascale1 * Bglobal_scale * Bscale1 / Aglobal_scale / Bglobal_scale`

即 `ans ≈ Aquant * Bquant * Ascale1 * Bscale1`
