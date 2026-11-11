# 数据、权重与协议

## 数据根目录

默认根目录为本项目`data/`，可用`--data-root /path/to/data`或`EVAL_DATA_ROOT`覆盖。与机器路径无关的相对位置在`configs/mlu/paths.json`中固定。

```text
data/
  fixtures/gso/fixed_full_rgb_v1/
    object_000/
      input.png
      target_01.png ... target_04.png
      depth_00.png ... depth_04.png
      pose_00.txt ... pose_04.txt
      intrinsics_00.txt ... intrinsics_04.txt
    ...
  models/...
  deps/...                   # 仅需要相应模型依赖时
```

RGB与深度对应同一像素几何；深度为单通道PNG、单位毫米。pose是4×4 camera-to-world；intrinsics是4×4文本矩阵，左上3×3为K。数据目录包含1023个对象；ID、五视图顺序、seed和逐文件大小/SHA256以`configs/common/suite.json`为准，源清单SHA256也记录在配置中。

**需要现有冻结的GSO同步RGB/depth/pose包。** 原始GSO网格下载不包含这套渲染包；本PR不提供数据包或假定公开下载地址。请从本次基准的数据维护者取得对应包，放入上述相对路径，再运行输入检查。自行重新渲染的数据若字节不同，应建立新的suite和输入哈希，并作为不同输入版本报告，不能当作同一冻结基准。

## 权重

从以下官方/模型发布页面取得权重，保留目录内文件名。对于固定revision可使用Hugging Face客户端的`hf download REPOSITORY --revision SHA --local-dir TARGET`；TARGET为下表相对数据根目录的位置。检查实际访问授权和文件是否齐全，脚本不自动下载大权重。

| 仓库 | revision | 相对目录 |
|---|---|---|
| [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | `860abec7937da0a4c03c41d3c269c366e82abdf9` | `models/facebook--VGGT-1B/860abec7937da0a4c03c41d3c269c366e82abdf9` |

`configs/common/weight-lock.json`固定实际使用文件的字节数和SHA256。默认运行核对大小；`--check-inputs --rehash-weights`做完整哈希校验。除非有明确实验版本变更，不修改权重锁。

## 精度与时间定义

VGGT每对象输入5视图、518平方预处理。GSO深度毫米转米，中心补方形和最近邻缩放；每视图用中位数尺度对齐后计算AbsRel、RMSE（米）、δ1，先平均视图再对对象宏平均。相机c2w取逆，与w2c预测比较视图对的相对旋转/平移方向及联合AUC@5/10/20度。

每次重置Python/NumPy/torch CPU/MLU seed；开启确定性warn_only，实际重复差值独立报告。输出浮点保存FP32，整数保留原类型。差值包括MAE、RMSE、最大绝对差、相对L2、逐元素一致性；形状不同明确记录。

端到端墙钟时间从适配器处理一个请求开始，到输出张量转为CPU NumPy数组结束；包含输入读取/预处理、模型推理、模型后处理和设备到主机传输。记录适配器调用、NumPy输出转换及评测框架开销；适配器内部可分辨阶段另行记录。模型事件计时是流水线时间的子项，不能再次加到端到端总时间。模型加载、预热、NPZ写盘、离线指标及导出不进入正式请求时间。另记allocated/reserved峰值、RSS、加载/保存时间和设备监控。报告覆盖全部1023个对象，质量指标在对象间宏平均。
