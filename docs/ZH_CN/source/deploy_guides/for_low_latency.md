# 低延迟场景部署

在低延迟的场景，我们会追求更快的速度，忽略显存和内存开销等问题。我们提供两套方案：

## 💡 方案一：步数蒸馏模型的推理

该方案可以参考[步数蒸馏文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/step_distill.html)

🧠 **步数蒸馏**是非常直接的视频生成模型的加速推理方案。从50步蒸馏到4步，耗时将缩短到原来的4/50。同时，该方案下，仍然可以和以下方案结合使用：
1. [高效注意力机制方案](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/attention.html)
2. [模型量化](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html)

## 💡 方案二：非步数蒸馏模型的推理

步数蒸馏需要比较大的训练资源，以及步数蒸馏后的模型，可能会出现视频动态范围变差的问题。

对于非步数蒸馏的原始模型，我们可以使用以下方案或者多种方案结合的方式进行加速：

1. [并行推理](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html) 进行多卡并行加速。
2. [特征缓存](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/cache.html) 降低实际推理步数。
3. [高效注意力机制方案](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/attention.html) 加速 Attention 的推理。
4. [模型量化](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html) 加速 Linear 层的推理。
5. [变分辨率推理](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/changing_resolution.html) 降低中间推理步的分辨率。

## 💡 使用Tiny VAE

在某些情况下，VAE部分耗时会比较大，可以使用轻量级VAE进行加速，同时也可以降低一部分显存。

```python
{
    "use_tae": true,
    "tae_path": "/path to taew2_1.pth"
}
```
taew2_1.pth 权重可以从[这里](https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_1.pth)下载


## ⚠️ 注意

有一部分的加速方案之间目前无法结合使用，我们目前正在致力于解决这一问题。

如有问题，欢迎在 [🐛 GitHub Issues](https://github.com/ModelTC/lightx2v/issues) 中进行错误报告或者功能请求
