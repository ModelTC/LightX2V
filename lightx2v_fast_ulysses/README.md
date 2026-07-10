# lightx2v-fast-ulysses

Optional NVSHMEM-backed A2A extension for LightX2V's `fast_ulysses` attention backend.

Only the fast-ulysses all-to-all path is migrated here. Upstream fused QK/RoPE/RMSNorm ops are out of scope for this package.

Build:

```bash
NVSHMEM_HOME=/path/to/nvshmem pip install ./lightx2v_fast_ulysses
```

Local validation path:

```bash
NVSHMEM_HOME=/data1/workspace/nvshem \
LD_LIBRARY_PATH=/data1/workspace/nvshem/lib:$LD_LIBRARY_PATH \
pip install -v ./lightx2v_fast_ulysses --no-build-isolation
```

If cloning LightX2V with third-party submodules, use:

```bash
git clone --recursive <repo>
git submodule update --init --recursive
```

`fast_ulysses` source is migrated here from https://github.com/triple-mu/fast-ulysses; NVSHMEM is still an external dependency.
