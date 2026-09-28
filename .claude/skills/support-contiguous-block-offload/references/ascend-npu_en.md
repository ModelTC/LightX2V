# Ascend NPU

English | [简体中文](ascend-npu.md)

## Entry points and the complete inference path

Set `PLATFORM=ascend_npu` before importing the project and select the target single device with `ASCEND_RT_VISIBLE_DEVICES`. Provide separate baseline and contiguous scripts/configurations while reusing the common loader, manager, and model computation.

Confirm that the device runtime works before investigating model implementation issues. Container device nodes, driver libraries, and torch_npu/PyTorch/CANN compatibility are environment prerequisites. Changing the weight layout cannot fix missing drivers. Disabling device backend autoloading or running on CPU does not validate NPU inference.

Inspect the entire pipeline using the effective configuration:

| Scope | What to check |
|---|---|
| DiT | Attention, RoPE, RMSNorm, LayerNorm, modulation, matrix operators, and checkpoint precision |
| Text encoder | Its own operators, loading precision, and component offload; checking only the DiT is insufficient |
| Scheduler | Device, dtype, positional encoding, and possible CPU fallback |
| VAE | Model loading, decoding operators, input/output precision, and component device transfers |

Prefer existing NPU implementations or PyTorch implementations available on the device, with selection controlled by configuration. A PyTorch operation on an NPU tensor is not necessarily a CPU fallback. Likewise, selecting NPU attention does not establish that the remaining components are adapted.

## Platform storage interface

[NpuDevice / NpuBlockOffload](../../../../lightx2v_platform/base/ascend_npu.py) provides memory operations through the same [backend interface](../../../../lightx2v_platform/base/offload.py) as CUDA.

- The current contiguous layout creates typed views over byte buffers. Call the backend's `prepare()` to request ND storage before constructing the relevant device buffers. Use the common loading flow rather than repeating device settings inside models.
- Some versions expose only a setter for `torch.npu.config.allow_internal_format`. Record the requested setting and actual tensor format where useful, without relying on a nonexistent getter. Check available interfaces in the target environment.
- The NPU backend's `validate_checkpoint=True` also validates original checkpoint precision for ordinary block baseline runs, keeping baseline and contiguous paths consistent in what they accept.
- Use the actual device module for copies, streams, events, and synchronization. Do not hardcode `torch.cuda` in new common functions.

`NpuDevice.copy_to_cpu()` handles noncontiguous CPU destinations by first completing D2H into a contiguous CPU source, then copying on CPU according to the destination strides. Do not simplify this to a direct asynchronous write into a transposed host view; that previously corrupted offloaded weights during round trips. This fix belongs in the platform copy layer. CUDA should not be forced to perform the same extra copy.

Related implementations: [common weight helpers](../../../../lightx2v/common/ops/utils.py), [MM template](../../../../lightx2v_platform/ops/mm/template.py), and [norm template](../../../../lightx2v_platform/ops/norm/norm_template.py). The platform norm templates provide storage descriptions, attribute mappings, and binding for ordinary floating weights; NPU RMSNorm/LayerNorm inherit these without duplicating them. Specialized storage rules such as quantization remain in the corresponding [MM operator](../../../../lightx2v_platform/ops/mm/ascend_npu/mm_weight.py).

## Two existing configuration choices

| Operator | Wan NPU example | Qwen NPU example |
|---|---|---|
| Attention | All three attention configuration entries use `npu_flash_attn` | `attn_type=npu_flash_attn` |
| RoPE | `npu_rope` | `torch_real_rope` |
| RMSNorm / LayerNorm | `npu_rms_norm` / `npu_layer_norm` | `torch` / `torch` |
| Modulation | `torch` | `torch` |
| Text encoder | `t5_rms_norm_type=torch` | Qwen2.5-VL's own path |

Choose according to current source code and operator semantics, rather than mechanically switching to every operator whose name starts with `npu_`. Check real/complex RoPE representation, layout, rotation dimensions, and computation precision. A registered name does not guarantee compatibility with the target model's input contract.

## Hardware validation and interpreting results

First validate weight loading and repeated H2D/D2H round trips, especially transposed matrices and CPU source values. Then validate successive steps, CFG branches, and complete generation. When both variants fail, investigate their common loading, computation, and decoding paths before attributing the failure to contiguous layout.

Keep the source version and effective configuration traceable. Historical Wan 910B success does not establish correctness after refactoring, and the existence of a Qwen NPU launcher does not establish successful generation.

Without an NPU, check configuration, import boundaries, storage contracts, and CUDA regressions, clearly labeling simulated or substitute-device tests. Provide commands, required weights, and expected outputs for the target machine, and retain an unverified status until hardware validation is complete. Preserve evidence as described in [Validation](validation_en.md); passing mocks does not establish completed 910B support.
