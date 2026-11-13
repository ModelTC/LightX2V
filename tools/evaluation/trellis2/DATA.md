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
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | `af44b45f2e35a493886929c6d786e563ec68364d` | `models/microsoft--TRELLIS.2-4B/af44b45f2e35a493886929c6d786e563ec68364d` |
| [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | `459ee480d7545f873a0768364bc5f5443eb1ff5e` | `models/microsoft--TRELLIS-image-large/459ee480d7545f873a0768364bc5f5443eb1ff5e` |
| [camenduru/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | `3c276edd87d6f6e569ff0c4400e086807d0f3881` | `deps/camenduru--dinov3-vitl16-pretrain-lvd1689m` |
| [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | `deps/ZhengPeng7--BiRefNet` |

`configs/common/weight-lock.json`固定实际使用文件的字节数和SHA256。默认运行核对大小；`--check-inputs --rehash-weights`做完整哈希校验。除非有明确实验版本变更，不修改权重锁。

## 精度与时间定义

TRELLIS.2每对象将input.png送入512流程，三阶段各12步，完整形状/纹理体素解码。其余视图用于GT点云。生成网格按面积固定seed采4096点，真值每视图最多2048点；各自中心化/RMS归一化后做20次刚性ICP，统计对称Chamfer-L1、L2-squared和F-score@0.01/0.02/0.05。纹理按坐标交集比较并报告IoU；不同网格另报原始顶点集合距离。该本地几何协议未测烘焙纹理或渲染图质量，不能冒充官方排行榜。

每个对象只进行一次正式推理，不自动重试；推理失败保留为失败记录。每次请求重置Python/NumPy/torch CPU/MLU seed；开启确定性warn_only。输出浮点保存FP32，整数保留原类型。

端到端墙钟时间从适配器处理一个请求开始，到输出张量转为CPU NumPy数组结束；包含输入读取/预处理、模型推理、模型后处理和设备到主机传输。记录适配器调用、NumPy输出转换及评测框架开销；适配器内部可分辨阶段另行记录。三阶段采样器事件计时是模型流水线时间的子项，不能再次加到端到端总时间。模型加载、预热、NPZ写盘、离线指标和几何导出不进入正式请求时间。另记allocated/reserved峰值、RSS、加载/保存时间和设备监控。报告覆盖全部1023个对象，质量指标在对象间宏平均。

## 随机数约定

与工作区 `随机数.png` 对齐：所有1023个对象每次请求均重置为 seed=42，不随对象编号递增；Python、NumPy、torch CPU和当前设备均使用此值。TRELLIS.2 的 CD-L1 表面采样同样使用42；VGGT 的 CD-L1 沿用固定64×64像素网格，不进行随机采样。原项目推理不在本评测执行范围内。

## 主要阶段计时

评测临时包装LightX2V调用，在主要阶段前后同步设备，记录互不重叠的墙钟时间：图像编码（含背景处理）、稀疏结构去噪/解码、形状去噪/解码及网格提取、纹理去噪/解码、网格补洞。runner剩余时间包括读图、噪声/坐标准备、归一化及其他开销。另记录请求准备、输出CPU/NumPy转换和评测框架开销。阶段的总耗时占比在`performance.pipeline_stage_share_percent`中给出；`runner_inference_and_postprocess_s`为包含这些阶段的总项，不能重复相加。sampler的设备事件时间属于阶段内部数据，也不能再次叠加。同步与计时包装只用于评测，普通用户推理不安装这些包装。

MLU的`--install-deps`还会从已固定的LightX2V源码构建`lightx2v_kernel/cambricon_mlu`，使用同一镜像的torch_mlu/Neuware工具链；没有使用本机外部kernel源码。
