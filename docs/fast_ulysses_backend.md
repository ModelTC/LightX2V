# Fast Ulysses Backend

LightX2V provides an optional `fast_ulysses` sequence-parallel attention backend that wraps the migrated fast-ulysses NVSHMEM all-to-all op around the existing LightX2V attention module.

This PR only migrates the A2A path from `https://github.com/triple-mu/fast-ulysses`. Upstream fused QK/RoPE/RMSNorm ops are intentionally not included.

## Install

Build the optional native package only on machines that have NVSHMEM:

```bash
NVSHMEM_HOME=/path/to/nvshmem pip install ./lightx2v_fast_ulysses
```

`NVSHMEM_HOME` must contain `include/nvshmem.h` and `lib/cmake/nvshmem`.

On the local validation host:

```bash
export NVSHMEM_HOME=/data1/lyxu18/workspace/nvshem
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
pip install -v ./lightx2v_fast_ulysses --no-build-isolation
```

## Configure

Use the existing sequence-parallel attention selector:

```json
{
  "parallel": {
    "seq_p_attn_type": "fast_ulysses"
  }
}
```

The fast path targets the validated single-node pure-image/self-attention path with fp16 or bf16 tensors. Unsupported paths such as fp8/fp4 communication, head parallelism, GQA, `q_only_img`, mixed text/image split attention, missing native extension, or cross-node sequence-parallel groups automatically fall back to the existing `ulysses` backend.

## Validation

Measured on one H800 node with GPUs 4/5/6/7, `seq_p_size=4`, `cfg_p_size=1`, bf16, and flash_attn3. The fast run used the native A2A path without fallback.

| Scope | Shape / metric | `ulysses` | `fast_ulysses` | Speedup |
| --- | --- | ---: | ---: | ---: |
| Ulysses attention layer | `N=27280,H=40,D=128`, warmup=10, iters=50 | 7518.400 us | 6762.496 us | 1.112x |
| Wan I2V DiT steady state | average step time excluding first step | 1.601850 s | 1.540803 s | 1.040x |

Full single-run e2e wall time is not used as the primary performance claim because model loading, cache state, and first-step native initialization add noise that is unrelated to the attention backend.

## Third-Party Handling

The fast-ulysses source used by this backend is migrated under `lightx2v_fast_ulysses/` and attributed to `https://github.com/triple-mu/fast-ulysses`.

NVSHMEM remains an external dependency and is not a LightX2V top-level submodule. If cloning a LightX2V tree that does contain third-party submodules, use the usual recursive commands:

```bash
git clone --recursive <repo>
git submodule update --init --recursive
```

This matches the existing LightX2V pattern for heavy CUDA dependencies such as CUTLASS: keep them external or fetch them in build images instead of making normal Python installs depend on them.
