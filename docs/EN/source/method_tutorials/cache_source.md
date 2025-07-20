# Feature Caching

## Cache Acceleration Algorithm
- In the inference process of diffusion models, cache reuse is an important acceleration algorithm.
- The core idea is to skip redundant computations at certain time steps by reusing historical cache results to improve inference efficiency.
- The key to the algorithm is how to decide which time steps to perform cache reuse, usually based on dynamic judgment of model state changes or error thresholds.
- During inference, key content such as intermediate features, residuals, and attention outputs need to be cached. When entering reusable time steps, the cached content is directly utilized, and the current output is reconstructed through approximation methods like Taylor expansion, thereby reducing repeated calculations and achieving efficient inference.

### TeaCache
The core idea of `TeaCache` is to accumulate the **relative L1** distance between adjacent time step inputs. When the accumulated distance reaches a set threshold, it determines that the current time step should not use cache reuse; conversely, when the accumulated distance does not reach the set threshold, cache reuse is used to accelerate the inference process.
- Specifically, the algorithm calculates the relative L1 distance between the current input and the previous step input at each inference step and accumulates it.
- When the accumulated distance does not exceed the threshold, it indicates that the model state change is not obvious, so the most recently cached content is directly reused, skipping some redundant calculations. This can significantly reduce the number of forward computations of the model and improve inference speed.

In practical effects, TeaCache achieves significant acceleration while ensuring generation quality. On a single H200 card, the time consumption and video comparison before and after acceleration are as follows:

<table>
  <tr>
    <td align="center">
      Before acceleration: 58s
    </td>
    <td align="center">
      After acceleration: 17.9s
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


- Acceleration ratio: **3.24**
- Config: [wan_t2v_1_3b_tea_480p.json](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/teacache/wan_t2v_1_3b_tea_480p.json)
- Reference paper: [https://arxiv.org/abs/2411.19108](https://arxiv.org/abs/2411.19108)

### TaylorSeer Cache
The core of `TaylorSeer Cache` lies in using Taylor's formula to recalculate cached content as residual compensation for cache reuse time steps.
- The specific approach is to not only simply reuse historical cache at cache reuse time steps, but also approximately reconstruct the current output through Taylor expansion. This can further improve output accuracy while reducing computational load.
- Taylor expansion can effectively capture minor changes in model state, allowing errors caused by cache reuse to be compensated, thereby ensuring generation quality while accelerating.

`TaylorSeer Cache` is suitable for scenarios with high output accuracy requirements and can further improve model inference performance based on cache reuse.

<table>
  <tr>
    <td align="center">
      Before acceleration: 57.7s
    </td>
    <td align="center">
      After acceleration: 41.3s
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


- Acceleration ratio: **1.39**
- Config: [wan_t2v_taylorseer](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/taylorseer/wan_t2v_taylorseer.json)
- Reference paper: [https://arxiv.org/abs/2503.06923](https://arxiv.org/abs/2503.06923)

### AdaCache
The core idea of `AdaCache` is to dynamically adjust the step size of cache reuse based on partial cached content in specified block chunks.
- The algorithm analyzes feature differences between two adjacent time steps within specific blocks and adaptively determines the next cache reuse time step interval based on the difference magnitude.
- When model state changes are small, the step size automatically increases, reducing cache update frequency; when state changes are large, the step size decreases to ensure output quality.

This allows flexible adjustment of caching strategies based on dynamic changes in the actual inference process, achieving more efficient acceleration and better generation results. AdaCache is suitable for application scenarios that have high requirements for both inference speed and generation quality.

<table>
  <tr>
    <td align="center">
      Before acceleration: 227s
    </td>
    <td align="center">
      After acceleration: 83s
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


- Acceleration ratio: **2.73**
- Config: [wan_i2v_ada](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/adacache/wan_i2v_ada.json)
- Reference paper: [https://arxiv.org/abs/2411.02397](https://arxiv.org/abs/2411.02397)

### CustomCache
`CustomCache` combines the advantages of `TeaCache` and `TaylorSeer Cache`.
- It combines the real-time and reasonable cache decision-making of `TeaCache`, determining when to perform cache reuse through dynamic thresholds.
- At the same time, it utilizes `TaylorSeer`'s Taylor expansion method to make use of cached content.

This not only efficiently determines the timing of cache reuse but also maximizes the utilization of cached content, improving output accuracy and generation quality. Actual testing shows that `CustomCache` produces video quality superior to using `TeaCache`, `TaylorSeer Cache`, or `AdaCache` alone across multiple content generation tasks, making it one of the currently optimal comprehensive cache acceleration algorithms.

<table>
  <tr>
    <td align="center">
      Before acceleration: 57.9s
    </td>
    <td align="center">
      After acceleration: 16.6s
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


- Acceleration ratio: **3.49**
- Config: [wan_t2v_custom_1_3b](https://github.com/ModelTC/lightx2v/tree/main/configs/caching/custom/wan_t2v_custom_1_3b.json)


## Usage

The config files for feature caching are located [here](https://github.com/ModelTC/lightx2v/tree/main/configs/caching)

By specifying --config_json to the specific config file, you can test different cache algorithms.

[Here](https://github.com/ModelTC/lightx2v/tree/main/scripts/cache) are some running scripts for use.
