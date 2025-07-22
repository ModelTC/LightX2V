# 特征缓存

## 缓存加速算法
- 在扩散模型的推理过程中，缓存复用是一种重要的加速算法。
- 其核心思想是在部分时间步跳过冗余计算，通过复用历史缓存结果提升推理效率。
- 算法的关键在于如何决策在哪些时间步进行缓存复用，通常基于模型状态变化或误差阈值动态判断。
- 在推理过程中，需要缓存如中间特征、残差、注意力输出等关键内容。当进入可复用时间步时，直接利用已缓存的内容，通过泰勒展开等近似方法重构当前输出，从而减少重复计算，实现高效推理。

### TeaCache
`TeaCache`的核心思想是通过对相邻时间步输入的**相对L1**距离进行累加，当累计距离达到设定阈值时，判定当前时间步不使用缓存复用；相反，当累计距离未达到设定阈值时则使用缓存复用加速推理过程。
- 具体来说，算法在每一步推理时计算当前输入与上一步输入的相对L1距离，并将其累加。
- 当累计距离未超过阈值，说明模型状态变化不明显，则直接复用最近一次缓存的内容，跳过部分冗余计算。这样可以显著减少模型的前向计算次数，提高推理速度。

实际效果上，TeaCache 在保证生成质量的前提下，实现了明显的加速。在单卡H200上，加速前后的用时与视频对比如下：

<table>
  <tr>
    <td align="center">
      加速前：58s
    </td>
    <td align="center">
      加速后：17.9s
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/1781df9b-04df-4586-b22f-5d15f8e1bff6" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/e93f91eb-3825-4866-90c2-351176263a2f" width="100%"></video>
    </td>
  </tr>
</table>


- 加速比为：**3.24**
- config：[wan_t2v_1_3b_tea_480p.json](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/teacache/wan_t2v_1_3b_tea_480p.json)
- 参考论文：[https://arxiv.org/abs/2411.19108](https://arxiv.org/abs/2411.19108)

### TaylorSeer Cache
`TaylorSeer Cache`的核心在于利用泰勒公式对缓存内容进行再次计算，作为缓存复用时间步的残差补偿。
- 具体做法是在缓存复用的时间步，不仅简单地复用历史缓存，还通过泰勒展开对当前输出进行近似重构。这样可以在减少计算量的同时，进一步提升输出的准确性。
- 泰勒展开能够有效捕捉模型状态的微小变化，使得缓存复用带来的误差得到补偿，从而在加速的同时保证生成质量。

`TaylorSeer Cache`适用于对输出精度要求较高的场景，能够在缓存复用的基础上进一步提升模型推理的表现。

<table>
  <tr>
    <td align="center">
      加速前：57.7s
    </td>
    <td align="center">
      加速后：41.3s
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/2d04005c-853b-4752-884b-29f8ea5717d2" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/270e3624-c904-468c-813e-0c65daf1594d" width="100%"></video>
    </td>
  </tr>
</table>


- 加速比为：**1.39**
- config：[wan_t2v_taylorseer](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/taylorseer/wan_t2v_taylorseer.json)
- 参考论文：[https://arxiv.org/abs/2503.06923](https://arxiv.org/abs/2503.06923)

### AdaCache
`AdaCache`的核心思想是根据指定block块中的部分缓存内容，动态调整缓存复用的步长。
- 算法会分析相邻两个时间步在特定 block 内的特征差异，根据差异大小自适应地决定下一个缓存复用的时间步间隔。
- 当模型状态变化较小时，步长自动加大，减少缓存更新频率；当状态变化较大时，步长缩小，保证输出质量。

这样可以根据实际推理过程中的动态变化，灵活调整缓存策略，实现更高效的加速和更优的生成效果。AdaCache 适合对推理速度和生成质量都有较高要求的应用场景。

<table>
  <tr>
    <td align="center">
      加速前：227s
    </td>
    <td align="center">
      加速后：83s
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/33b2206d-17e6-4433-bed7-bfa890f9fa7d" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/084dbe3d-6ff3-4afc-9a7c-453ec53b3672" width="100%"></video>
    </td>
  </tr>
</table>


- 加速比为：**2.73**
- config：[wan_i2v_ada](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/adacache/wan_i2v_ada.json)
- 参考论文：[https://arxiv.org/abs/2411.02397](https://arxiv.org/abs/2411.02397)

### CustomCache
`CustomCache`综合了`TeaCache`和`TaylorSeer Cache`的优势。
- 它结合了`TeaCache`在缓存决策上的实时性和合理性，通过动态阈值判断何时进行缓存复用.
- 同时利用`TaylorSeer`的泰勒展开方法对已缓存内容进行利用。

这样不仅能够高效地决定缓存复用的时机，还能最大程度地利用缓存内容，提升输出的准确性和生成质量。实际测试表明，`CustomCache`在多个内容生成任务上，生成的视频质量优于单独使用`TeaCache、TaylorSeer Cache`或`AdaCache`的方案，是目前综合性能最优的缓存加速算法之一。

<table>
  <tr>
    <td align="center">
      加速前：57.9s
    </td>
    <td align="center">
      加速后：16.6s
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/304ff1e8-ad1c-4013-bcf1-959ac140f67f" width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/d3fb474a-79af-4f33-b965-23d402d3cf16" width="100%"></video>
    </td>
  </tr>
</table>


- 加速比为：**3.49**
- config：[wan_t2v_custom_1_3b](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/custom/wan_t2v_custom_1_3b.json)


## 使用方式

特征缓存的config文件在[这里](https://github.com/ModelTC/lightx2v/tree/main/configs/caching)

通过指定--config_json到具体的config文件，即可以测试不同的cache算法

[这里](https://github.com/ModelTC/lightx2v/tree/main/scripts/cache)有一些运行脚本供使用。
