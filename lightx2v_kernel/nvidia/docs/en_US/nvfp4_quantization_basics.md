# nvfp4 Quantization Basics

### Data Format

The calculation method for fp is:

`ans = (-1)^s * 2^(p-b) * (1 + d1/2 + d2/4 + d3/8 + ...)`

Where `b = 2^(e-1) - 1`, p represents the value of the exponent bits, d1, d2, d3 represent the values of the mantissa bits

For fp4, the format is E2M1, and the above formula is simplified to:

`b = 2^(e-1) - 1 = 2^(2-1) - 1 = 1`

`ans = (-1)^s * 2^(p-1) * (1 + d1/2)`

Example: 0101

`s=0, p=(10)=2, d1=1`

`ans = 2^0 * 2^(2-1) * (1 + 1/2) = 3`

In normal fp data format, some data represents inf and nan, with a maximum representation of ±3. Specialized for nvfp4, inf and nan are removed, allowing a maximum representation of ±6.

Specifically, 0000 represents +0, 1000 represents -0, 0001 represents 0.5, and 1001 represents -0.5.

In summary:

| E2M1 | 0000 | 0001 | 0010 | 0011 | 0100 | 0101 | 0110 | 0111 | 1000 | 1001 | 1010 | 1011 | 1100 | 1101 | 1110 | 1111 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ans  | +0   | 0.5  | 1.0  | 1.5  | 2.0  | 3.0  | 4.0  | 6.0  | +0   | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |


### Quantization Process

**Both weight and activation use per-group quantization, with a group size of 16, and quantization scales are stored in fp8(e4m3) format**

Since the quantization scale needs to be stored in fp8, the scale also needs to be rescaled, so the fp4 quantization process differs somewhat from the common w8a8-int8 process.

The quantization process is as follows:

Given a set of numbers, denoted as `X`

#### Calculate scale

`scale1 = max(abs(Xg)) / 6.0`

Where Xg represents a group of numbers, and 6.0 represents the maximum value of nvfp4

#### Quantize scale

`global_scale = 6.0 * 448.0 / max(abs(X))`

`scale2 = global_scale * scale1`

That is `scale2 = 6.0 * 448.0 / max(abs(X)) * max(abs(Xg)) / 6.0`

That is `scale2 = max(abs(Xg)) / max(abs(X)) * 448.0`

At this point, scale2 is rescaled to the range of fp8(e4m3), then scale2 is quantized to fp8

`scale2_fp8 = quant_fp8(scale2)`

`scale2_fp8` serves as the final quantization scale parameter required for matrix multiplication

#### Quantize X

`scale2_fp32 = cvt2fp32(scale2_fp8)`

`Xquant = quant_fp4(X * global_scale / scale2_fp32)`

Then `Xquant ≈ quant_fp4(X / scale1)`

#### fp4 Matrix Multiplication

`ans = Aquant * Bquant * Ascale2 * Bscale2 / Aglobal_scale / Bglobal_scale`

That is `ans ≈ Aquant * Bquant * Aglobal_scale * Ascale1 * Bglobal_scale * Bscale1 / Aglobal_scale / Bglobal_scale`

That is `ans ≈ Aquant * Bquant * Ascale1 * Bscale1`
