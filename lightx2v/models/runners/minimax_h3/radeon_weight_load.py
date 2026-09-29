"""Fast weight residency for MiniMax-H3 multi-GPU runs (``radeon_gfx1201_h3``).

- CPU-offloaded weights stay zero-copy safetensors mmap views (no per-tensor pinned copies); every offloaded H2D
  (DiT blocks, text-encoder layers, video VAE) goes through a 2-slot pinned staging ring filled by a parallel CPU
  copy (RADEON_CORESW_STAGE_THREADS, default 16) and an async DMA, issued after the current block's kernels.
- Video VAE FP32->FP16 weight prepare runs with RADEON_CORESW_LOAD_THREADS (default 16) intra-op threads.
- Idle eviction (sequence parallel): at init_run the video VAE goes back to its host copy and the text encoder
  releases its block-offload buffers, so the DiT steps never hit allocator OOM retries (seen to corrupt resident
  block weights -> NaN at SP2); the VAE is re-uploaded on its next use. Same bytes everywhere.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from lightx2v.models.networks.minimax_h3.infer.radeon_sp import prefetch_gate

_KEPT = []
_EVICT = [False]
_POOL = []
_ALIGN = 4096
_CHUNK = 32 << 20


def _pool():
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(int(os.environ.get("RADEON_CORESW_STAGE_THREADS", "16")), thread_name_prefix="stage"))
    return _POOL[0]


_STATS = {"calls": 0, "bytes": 0, "seconds": 0.0}


def _stage(slot, tensors):
    """Parallel byte copy of CPU tensors into the pinned uint8 slot; name -> pinned view (same dtype/shape/stride)."""
    t0 = time.perf_counter()
    jobs, views, off = [], {}, 0
    for name, t in tensors.items():
        n = _span_bytes(t)
        src = torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage(), t.storage_offset() * t.element_size(), (n,), (1,))
        jobs.extend((off + o, src[o : o + min(_CHUNK, n - o)]) for o in range(0, n, _CHUNK))
        views[name] = torch.empty(0, dtype=t.dtype).set_(slot.untyped_storage(), off // t.element_size(), t.shape, t.stride())
        off += -(-n // _ALIGN) * _ALIGN
    list(_pool().map(_copy_job, [(slot, off, src) for off, src in jobs]))
    _STATS["calls"] += 1
    _STATS["bytes"] += off
    _STATS["seconds"] += time.perf_counter() - t0
    if os.environ.get("RADEON_CORESW_STAGE_STATS") == "1" and _STATS["calls"] % 16 == 0:
        print(f"[radeon_weight_load][rank={os.environ.get('RANK', '0')}] staged {_STATS['calls']} blocks {_STATS['bytes'] / 1e9:.1f} GB in {_STATS['seconds']:.2f} s host", file=sys.stderr, flush=True)
    return views


def _copy_job(job):
    slot, off, src = job
    with torch.inference_mode():  # worker threads do not inherit the caller's mode; the slot may be an inference tensor
        slot[off : off + src.numel()].copy_(src)


_SLOTS = {"buf": [None, None], "done": [None, None], "thread": None}


def _prealloc(nbytes, device):
    torch.cuda.set_device(device)
    for k in range(2):
        if _SLOTS["buf"][k] is None:
            _SLOTS["buf"][k] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def _slot(k, need):
    """Shared pinned slot k (text encoder and DiT rings take turns); waits for its last DMA."""
    if _SLOTS["thread"] is not None:
        _SLOTS["thread"].join()
        _SLOTS["thread"] = None
    if _SLOTS["done"][k] is not None:
        _SLOTS["done"][k].synchronize()
    if _SLOTS["buf"][k] is None or _SLOTS["buf"][k].numel() < need:
        _SLOTS["buf"][k] = torch.empty(need, dtype=torch.uint8, pin_memory=True)
    return _SLOTS["buf"][k]


class _Ring:
    """2-slot pinned staging ring; a slot is refilled only after its previous DMA completed."""

    def __init__(self):
        self.k = 0

    def load(self, target, state, stream, *args):
        cpu = {k: t for k, t in state.items() if isinstance(t, torch.Tensor) and t.device.type == "cpu" and not t.is_pinned()}
        k = self.k
        slot = _slot(k, sum(-(-_span_bytes(t) // _ALIGN) * _ALIGN for t in cpu.values()))
        state = dict(state)
        state.update(_stage(slot, cpu))
        with torch.cuda.stream(stream):
            target.load_state_dict(state, *args)
            event = torch.cuda.Event()
            event.record(stream)
        _SLOTS["done"][k], self.k = event, 1 - k


def install_runner(module):
    """Allocate the two pinned staging slots on a background thread while the models load (hipHostMalloc of
    2 x ~1 GiB takes ~0.9 s and otherwise lands inside the text encoder and the first DiT block)."""
    cls = module.MiniMaxH3Runner
    native = cls.load_model

    def load_model(self, *args, **kwargs):
        if _SLOTS["thread"] is None and _SLOTS["buf"][0] is None:
            nbytes = int(os.environ.get("RADEON_CORESW_STAGE_SLOT_MB", "1024")) << 20
            _SLOTS["thread"] = threading.Thread(target=_prealloc, args=(nbytes, torch.cuda.current_device()), daemon=True)
            _SLOTS["thread"].start()
        return native(self, *args, **kwargs)

    cls.load_model = load_model


def _no_pin(tensor, transpose=False, dtype=None):
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    _KEPT.append(tensor)
    return tensor.t() if transpose else tensor


def _keep(tensor):
    _KEPT.append(tensor)
    return tensor


def prefault(threads=16):
    """Map every kept weight page now (parallel), instead of page-faulting inside H2D copies later."""
    t0 = time.time()
    seen, spans = set(), []
    for t in _KEPT:
        if t.device.type != "cpu" or t.numel() == 0:
            continue
        st = t.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        if key not in seen:
            seen.add(key)
            spans.append(torch.empty(0, dtype=torch.uint8).set_(st))
    step = 64 << 20
    jobs = [u[o : o + step] for u in spans for o in range(0, u.numel(), step)]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda c: int(c[::4096].sum()), jobs))
    _KEPT.clear()
    print(f"[weight_load] prefaulted {sum(u.numel() for u in spans) / 1e9:.1f} GB in {time.time() - t0:.1f} s", flush=True)


def install_ops_utils(module):
    module.create_pin_tensor = _no_pin


def install_tensor(module):
    module.DefaultTensor._create_cpu_pin_tensor = lambda self, tensor: _keep(tensor)


def install_embedding(module):
    for cls in vars(module).values():
        if isinstance(cls, type) and "_create_cpu_pin_tensor" in vars(cls):
            cls._create_cpu_pin_tensor = lambda self, tensor: _keep(tensor)


def install_offload_infer(module):
    cls = module.MiniMaxH3OffloadTransformerInfer

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        dev = module.torch_device_module
        manager = self.offload_manager
        if hasattr(manager, "cpu_buffers"):
            raise RuntimeError("radeon_weight_load expects block offload straight from the CPU weight modules")
        ring = manager.__dict__.setdefault("_wl_ring", _Ring())
        manager.compute_stream.wait_stream(dev.current_stream())
        num_blocks = len(blocks)
        for block_index in range(num_blocks):
            if manager.need_init_first_buffer:
                ring.load(manager.cuda_buffers[0], blocks[0].state_dict(), manager.init_stream, 0, None)
                manager.init_stream.synchronize()
                manager.need_init_first_buffer = False
            self.block_idx = block_index
            with dev.stream(manager.compute_stream):
                hidden_states = self.run_block(block_index, manager.cuda_buffers[0], hidden_states, pre_infer_out)
            # staged (pinned) copy: host work is the parallel memcpy, the DMA is async on the load stream
            nxt = (block_index + 1) % num_blocks
            gate = prefetch_gate() if pre_infer_out.sequence_parallel_state is not None else None
            if gate is not None:  # keep the DMA off the SP Q/K/V SDMA pulls at the block start
                manager.cuda_load_stream.wait_event(gate)
            ring.load(manager.cuda_buffers[1], blocks[nxt].state_dict(), manager.cuda_load_stream, nxt, None)
            manager.swap_blocks()
        return hidden_states

    cls.infer_with_blocks_offload = infer_with_blocks_offload


def _span_bytes(t):
    if t.numel() == 0:
        return 0
    return (1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))) * t.element_size()


def install_event_manager(module):
    """Slot offload (text encoder): stage each layer from the mmap views into a 2-slot pinned ring with a CPU
    copy, then DMA from pinned memory (a cold pageable H2D copy runs at only ~2.6 GB/s)."""
    cls = module.EventSlotWeightAsyncStreamManager
    native = cls._load_block_to_buffer
    align = 4096

    def _load_block_to_buffer(self, target_buffer, block_idx, blocks, adapter_block_idx):
        if getattr(self, "block_slabs", {}).get(block_idx) is not None or hasattr(self, "cpu_buffers") or blocks is None:
            return native(self, target_buffer, block_idx, blocks, adapter_block_idx)
        sd = blocks[block_idx].state_dict()
        cpu = {k: t for k, t in sd.items() if isinstance(t, torch.Tensor) and t.device.type == "cpu" and not t.is_pinned()}
        if not cpu:
            return native(self, target_buffer, block_idx, blocks, adapter_block_idx)
        k = self.__dict__.get("_wl_k", 0)
        sd.update(_stage(_slot(k, sum(-(-_span_bytes(t) // align) * align for t in cpu.values())), cpu))
        target_buffer.load_state_dict(sd, block_idx, adapter_block_idx)
        ev = module.torch_device_module.Event()
        ev.record(self.cuda_load_stream)
        _SLOTS["done"][k] = ev
        self._wl_k = 1 - k

    cls._load_block_to_buffer = _load_block_to_buffer


def _upload(tensors, device):
    """{id(t): device copy} for CPU tensors: contiguous ones through the 2-slot pinned ring (parallel memcpy + async
    DMA on the current stream), others with a native copy. Returns (copies, staged bytes) once the copies are done."""
    cap = max(int(os.environ.get("RADEON_CORESW_STAGE_SLOT_MB", "1024")) << 20, _ALIGN)
    stream = torch.cuda.current_stream(device)
    moved, groups, cur, size = {}, [], [], 0
    for t in tensors:
        n = -(-_span_bytes(t) // _ALIGN) * _ALIGN
        if id(t) in moved or not t.is_contiguous():
            moved.setdefault(id(t), None)
            continue
        moved[id(t)] = None
        if cur and size + n > cap:
            groups.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += n
    if cur:
        groups.append(cur)
    k, nbytes = 0, 0
    for group in groups:
        need = sum(-(-_span_bytes(t) // _ALIGN) * _ALIGN for t in group)
        views = _stage(_slot(k, need), {i: t for i, t in enumerate(group)})
        with torch.cuda.stream(stream):
            for i, t in enumerate(group):
                g = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=device)
                g.copy_(views[i], non_blocking=True)
                moved[id(t)] = g
            event = torch.cuda.Event()
            event.record(stream)
        _SLOTS["done"][k], k = event, 1 - k
        nbytes += need
    for t in tensors:
        if moved[id(t)] is None:
            moved[id(t)] = t.to(device)
    stream.synchronize()
    return moved, nbytes


def _evict_enabled():
    return _EVICT[0]


def _to_device_staged(model, device):
    """Move every CPU parameter/buffer of `model` to `device` through the 2-slot pinned ring (parallel memcpy +
    async DMA on the current stream) instead of per-tensor pageable copies; tied tensors stay tied. With
    RADEON_CORESW_IDLE_EVICT=1 the host tensors are kept for _evict_model/_restore_model."""
    t0 = time.perf_counter()
    todo = []
    for mod in model.modules():
        for kind in ("_parameters", "_buffers"):
            for name, t in getattr(mod, kind).items():
                if t is not None and t.device.type == "cpu":
                    todo.append((mod, kind, name, t))
    moved, nbytes = _upload([item[3] for item in todo], device)
    params = {}
    for mod, kind, name, t in todo:
        g = moved[id(t)]
        if kind == "_parameters":
            if id(t) not in params:
                params[id(t)] = torch.nn.Parameter(g, requires_grad=t.requires_grad)
            mod._parameters[name] = params[id(t)]
        else:
            mod._buffers[name] = g
    if _evict_enabled():
        model._wl_host, model._wl_resident = todo, True
    print(f"[radeon_weight_load][rank={os.environ.get('RANK', '0')}] staged {len(todo)} tensors {nbytes / 1e9:.2f} GB to {device} in {time.perf_counter() - t0:.2f} s", file=sys.stderr, flush=True)


def _evict_model(model):
    """Point the parameters/buffers of a _to_device_staged model back at their host tensors; returns freed bytes."""
    if not getattr(model, "_wl_resident", False):
        return 0
    freed, seen = 0, set()
    for mod, kind, name, t in model._wl_host:
        cur = getattr(mod, kind)[name]
        if cur.is_cuda and cur.untyped_storage().data_ptr() not in seen:
            seen.add(cur.untyped_storage().data_ptr())
            freed += cur.untyped_storage().nbytes()
        if kind == "_parameters":
            cur.data = t
        else:
            mod._buffers[name] = t
    model._wl_resident = False
    return freed


def _restore_model(model, device):
    t0 = time.perf_counter()
    moved, nbytes = _upload([item[3] for item in model._wl_host], device)
    for mod, kind, name, t in model._wl_host:
        if kind == "_parameters":
            mod._parameters[name].data = moved[id(t)]
        else:
            mod._buffers[name] = moved[id(t)]
    model._wl_resident = True
    retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
    during = retries - _STATS.get("retries_at_evict", retries)
    print(
        f"[radeon_weight_load][rank={os.environ.get('RANK', '0')}] restored video VAE {nbytes / 1e9:.2f} GB in "
        f"{time.perf_counter() - t0:.2f} s; allocator retries since eviction {during}" + (" WARNING: OOM retries while the DiT ran" if during else ""),
        file=sys.stderr,
        flush=True,
    )


def _evict_idle(runner):
    """init_run: free the video VAE weights and the text encoder's layer buffers before the DiT steps."""
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    freed = 0
    vae = getattr(runner, "video_vae", None)
    if vae is not None:
        freed += _evict_model(vae)
    for encoder in getattr(runner, "text_encoders", None) or []:
        model = getattr(encoder, "text_encoder", None)
        buffers = getattr(model, "offload_cuda_buffers", None)
        if buffers is not None and hasattr(model, "release_block_offload_buffers"):
            freed += sum(t.untyped_storage().nbytes() for t in _cuda_tensors(buffers))
            model.release_block_offload_buffers()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _STATS["retries_at_evict"] = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    print(
        f"[radeon_weight_load][rank={os.environ.get('RANK', '0')}] idle evict {freed / 2**30:.2f} GiB in {time.perf_counter() - t0:.2f} s, device free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB",
        file=sys.stderr,
        flush=True,
    )


def _cuda_tensors(obj, seen=None):
    """CUDA tensor attributes reachable through WeightModule _modules/_parameters (one per storage)."""
    seen = set() if seen is None else seen
    out = []
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    for attr, value in vars(obj).items() if hasattr(obj, "__dict__") else ():
        if isinstance(value, torch.Tensor):
            if value.is_cuda and ("ptr", value.untyped_storage().data_ptr()) not in seen:
                seen.add(("ptr", value.untyped_storage().data_ptr()))
                out.append(value)
        elif attr in ("_modules", "_parameters") and isinstance(value, dict):
            for child in value.values():
                if child is not None and not isinstance(child, torch.Tensor):
                    out.extend(_cuda_tensors(child, seen))
    return out


def install_residency(module):
    _EVICT[0] = True
    """Runner hook (RADEON_CORESW_IDLE_EVICT=1): evict idle VAE/text-encoder GPU memory when the DiT starts."""
    cls = module.MiniMaxH3Runner
    native = cls.init_run

    def init_run(self, *args, **kwargs):
        _evict_idle(self)
        return native(self, *args, **kwargs)

    cls.init_run = init_run


def install_video_vae(module):
    """Video VAE from_pretrained: weight prepare with RADEON_CORESW_LOAD_THREADS threads, then staged H2D (the
    native model.to(device) that follows finds the weights already on the device)."""
    cls = module.MiniMaxH3VideoVAE
    native = cls._prepare_inference_weights

    def _prepare_inference_weights(self, *args, **kwargs):
        threads, t0 = torch.get_num_threads(), time.perf_counter()
        torch.set_num_threads(int(os.environ.get("RADEON_CORESW_LOAD_THREADS", "16")))
        try:
            native(self, *args, **kwargs)
        finally:
            torch.set_num_threads(threads)
        print(f"[radeon_weight_load][rank={os.environ.get('RANK', '0')}] video VAE weight prepare {time.perf_counter() - t0:.2f} s", file=sys.stderr, flush=True)
        if not self.cpu_offload and self.execution_device.type == "cuda":
            _to_device_staged(self, self.execution_device)

    cls._prepare_inference_weights = _prepare_inference_weights
    native_activate = cls._activate

    def _activate(self, *args, **kwargs):
        if getattr(self, "_wl_resident", True) is False:
            _restore_model(self, self.execution_device)
        return native_activate(self, *args, **kwargs)

    cls._activate = _activate
