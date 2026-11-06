"""Ulysses sequence parallelism (SP2/4/8) for the MiniMax-H3 DiT block on gfx1201 (``radeon_gfx1201_h3``).

Per block and rank (epoch e, no host sync): Q/K/V projections run per destination head shard straight into
IPC-shared send slots; peers pull their shard with SDMA copies (hipMemcpyAsync on IPC memory, fine-grained
flag page + stream write/wait-value ops), run the fused norm/RoPE/Sage prepare and the ASM attention core
into an IPC-shared output, and pull back their rows in chunks overlapped with the out-projection GEMM.
The rest of the block uses aiter gfx1201 HIP ops (rms_modulate, gated_residual, swiglu).
Knobs (environment): RADEON_CORESW_SP_O_CHUNKS (3), RADEON_CORESW_SP_PULL_ORDER (1),
RADEON_CORESW_SP_DOWN_SPLIT (2), RADEON_CORESW_SP_PREFETCH_GATE (core|post|none),
RADEON_CORESW_SP_FULLGEMM (0, validation only), RADEON_CORESW_RMS_MOD / RADEON_CORESW_SWIGLU (1).
Opt-in lossy Sol sparse attention core (``SOL``, installed by radeon_runtime with RADEON_CORESW_SOL=1).
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import record_function

HEAD_DIM = 128
_exchanges = {}
_engines = {}
# flag page: one uint32 slot per peer rank in each region; a slot is written only by that peer
QKV_READY, QKV_PULLED, O_READY, O_PULLED = 0, 256, 512, 768
sys.path.insert(0, str(Path(__file__).resolve().parent))


class SolSchedule:
    """Which attention calls use the Sol sparse core (aiter.ops.gfx1201.sol_attention).

    Scheduler steps < dense_steps and blocks < dense_layers keep the dense core (Sol-H3 ``dense_steps`` /
    ``dense_layers``). ``begin_step`` is called once per denoising step with the scheduler step and the number of
    leading condition tokens (text, references, target audio): the Sol-H3 exact sink, computed exactly both as keys
    and as queries. ``tau`` is the Sol-Attn threshold in standard deviations (higher is sparser).
    """

    def __init__(self, dense_steps, dense_layers, tau):
        self.dense_steps, self.dense_layers, self.tau = dense_steps, dense_layers, tau
        self.step, self.layer, self.prefix = 0, 0, 0

    def begin_step(self, step, prefix):
        self.step, self.layer, self.prefix = step, 0, prefix

    def attention(self, prepared, out, total):
        """Run the Sol core into ``out`` and return True, or return False for the dense core."""
        layer, self.layer = self.layer, self.layer + 1
        if self.step < self.dense_steps or layer < self.dense_layers:
            return False
        from aiter.ops.gfx1201.sol_attention import sol_attention

        sol_attention(prepared, out, total, self.tau, self.prefix)
        return True


SOL = None


def split_sizes(total, world):
    base, extra = divmod(total, world)
    return tuple(base + (rank < extra) for rank in range(world))


# hipBLASLt solutions for the SP2 (seq_p=2) BF16 GEMMs (per-destination Q/K/V, FFN in-projection), keyed by (N, K) of
# torch.mm(x[M, K], w[K, N]) with M >= 16384: same FP32-accumulation error as the heuristic choice, 1.5-4% faster in
# the block (tooling/benchmark/sp2_extreme_v1)
_SP2_GEMM = {(3584, 5376): 106882, (28672, 5376): 107517}
_hipb = []


def _mm(inputs, weight, out=None, world=1):
    """torch.mm(inputs, weight, out=out), with the tuned hipBLASLt solution for SP2 shapes (RADEON_CORESW_SP_TUNED_GEMM)."""
    index = _SP2_GEMM.get((weight.shape[1], weight.shape[0])) if world == 2 and inputs.shape[0] >= 16384 else None
    if index is None or os.environ.get("RADEON_CORESW_SP_TUNED_GEMM", "1") != "1" or inputs.dtype != torch.bfloat16:
        return torch.mm(inputs, weight, out=out)
    if not _hipb:
        from aiter.ops.gradlib import _hipb_mm, hipb_create_extension

        hipb_create_extension()
        _hipb.append(_hipb_mm)
    if out is None:
        out = torch.empty((inputs.shape[0], weight.shape[1]), device=inputs.device, dtype=inputs.dtype)
    _hipb[0](inputs, weight, index, out, None, None, None, None, None, None)
    return out


class PairwiseExchange:
    """Ulysses all-to-all as W-1 rounds of pairwise send/recv; XOR partners when W is a power of two.

    Each round is its own batched P2P call on the group communicator, so rounds run in order on one
    NCCL stream without blocking the compute stream until wait()."""

    def __init__(self, group, sizes):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if len(sizes) != self.world or min(sizes) < 1:
            raise ValueError(f"Expected one positive row count per rank, got {sizes}")
        self.sizes = tuple(sizes)
        self.offsets = tuple(sum(self.sizes[:rank]) for rank in range(self.world))
        self.total = sum(self.sizes)
        xor = self.world & (self.world - 1) == 0
        self.rounds = [(self.rank ^ k, self.rank ^ k) if xor else ((self.rank + k) % self.world, (self.rank - k) % self.world) for k in range(1, self.world)]
        self.global_ranks = [dist.get_global_rank(group, rank) for rank in range(self.world)]

    def _rows(self, tensor, rank):
        return tensor[self.offsets[rank] : self.offsets[rank] + self.sizes[rank]]

    def _round(self, send, send_to, recv, recv_from):
        return dist.batch_isend_irecv([dist.P2POp(dist.isend, send, self.global_ranks[send_to], self.group), dist.P2POp(dist.irecv, recv, self.global_ranks[recv_from], self.group)])

    def seq_to_head(self, local):
        """[S_r, W*C] local rows of all heads -> [S, C] all rows of this rank's head shard (valid after wait)."""
        rows, width = local.shape[0], local.shape[1] // self.world
        if local.ndim != 2 or rows != self.sizes[self.rank] or local.shape[1] != width * self.world:
            raise ValueError(f"Expected local rows [{self.sizes[self.rank]}, W*C], got {tuple(local.shape)}")
        packed = local.view(rows, self.world, width).transpose(0, 1).contiguous()
        full = torch.empty((self.total, width), device=local.device, dtype=local.dtype)
        self._rows(full, self.rank).copy_(packed[self.rank])
        works = [work for send, recv in self.rounds for work in self._round(packed[send], send, self._rows(full, recv), recv)]
        return full, (packed, works)

    def head_to_seq(self, full):
        """[S, C] all rows of this rank's head shard -> packed [W, S_r, C] (valid after wait; see unpack)."""
        if full.ndim != 2 or full.shape[0] != self.total or not full.is_contiguous():
            raise ValueError(f"Expected contiguous [{self.total}, C], got {tuple(full.shape)}")
        packed = torch.empty((self.world, self.sizes[self.rank], full.shape[1]), device=full.device, dtype=full.dtype)
        packed[self.rank].copy_(self._rows(full, self.rank))
        works = [work for send, recv in self.rounds for work in self._round(self._rows(full, send), send, packed[recv], recv)]
        return packed, (full, works)

    @staticmethod
    def wait(pending):
        for work in pending[1]:
            work.wait()

    @staticmethod
    def unpack(packed):
        return packed.transpose(0, 1).reshape(packed.shape[1], packed.shape[0] * packed.shape[2])


_comm_groups = {}


def comm_group(group):
    """Dedicated copy of the seq_p group whose RCCL stream has high priority (RADEON_CORESW_SP_HIPRIO=0 disables)."""
    if os.environ.get("RADEON_CORESW_SP_HIPRIO", "0") != "1":
        return group
    if id(group) not in _comm_groups:
        options = dist.ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        ranks = [dist.get_global_rank(group, rank) for rank in range(dist.get_world_size(group))]
        _comm_groups[id(group)] = dist.new_group(ranks, pg_options=options, use_local_synchronization=True)
    return _comm_groups[id(group)]


def get_exchange(group, sizes):
    key = (id(group), tuple(sizes))
    if key not in _exchanges:
        _exchanges[key] = PairwiseExchange(comm_group(group), sizes)
    return _exchanges[key]


def pre_process(model, pre_infer_out):
    from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState

    group = model.seq_p_group
    rank = dist.get_rank(group)
    exchange = get_exchange(group, split_sizes(pre_infer_out.hidden_states.shape[0], dist.get_world_size(group)))
    start, stop = exchange.offsets[rank], exchange.offsets[rank] + exchange.sizes[rank]
    rotary_emb = tuple(tensor.contiguous() for tensor in pre_infer_out.rotary_emb)
    state = MiniMaxH3SequenceParallelState(
        aux_length=0,
        main_shard_length=stop - start,
        timestep_indices=pre_infer_out.timestep_indices,
        adaln_indices=pre_infer_out.adaln_indices,
        rotary_emb=rotary_emb,
    )
    state.exchange = exchange
    pre_infer_out.sequence_parallel_state = state
    pre_infer_out.hidden_states = pre_infer_out.hidden_states[start:stop].contiguous()
    pre_infer_out.timestep_indices = pre_infer_out.timestep_indices[start:stop].contiguous()
    pre_infer_out.adaln_indices = pre_infer_out.adaln_indices[start:stop].contiguous()
    pre_infer_out.rotary_emb = tuple(tensor[start:stop] for tensor in rotary_emb)
    return pre_infer_out


def post_process(model, output, pre_infer_out):
    state = pre_infer_out.sequence_parallel_state
    exchange = getattr(state, "exchange", None)
    if exchange is None or output.shape[0] != exchange.sizes[exchange.rank]:
        raise RuntimeError("MiniMax-H3 SP metadata is missing or the local row count changed")
    rows = max(exchange.sizes)
    local = output.contiguous()
    if local.shape[0] < rows:
        local = torch.cat((local, local.new_zeros((rows - local.shape[0], *local.shape[1:]))))
    gathered = local.new_empty((exchange.world * rows, *local.shape[1:]))
    dist.all_gather_into_tensor(gathered, local, group=exchange.group)
    if min(exchange.sizes) != rows:
        gathered = torch.cat([gathered[rank * rows : rank * rows + size] for rank, size in enumerate(exchange.sizes)])
    pre_infer_out.timestep_indices = state.timestep_indices
    pre_infer_out.adaln_indices = state.adaln_indices
    pre_infer_out.rotary_emb = state.rotary_emb
    pre_infer_out.sequence_parallel_state = None
    return gathered


def _norm_weights(owner, weights, state, heads):
    if not getattr(owner, "_radeon_sp_checked", False):
        from aiter.ops.gfx1201 import norm_rope_prepare as prepare

        from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightNative
        from lightx2v.common.ops.rope.torch_rope import TorchRealRope

        if owner.tp_size != 1 or owner.head_dim != HEAD_DIM or heads * state.exchange.world != owner.num_heads:
            raise ValueError("Radeon SP requires tensor_p_size=1, D128 and heads divisible by seq_p_size")
        if heads not in prepare.HEADS:
            raise ValueError(f"Radeon SP supports {prepare.HEADS} heads per rank, got {heads}")
        if owner.use_fused_qkv and weights.has_fused_qkv:
            raise ValueError("Radeon SP requires separate Q/K/V projections (use_fused_qkv=false)")
        if any(type(norm) is not RMSWeightNative or norm.eps != 1e-5 for norm in (weights.norm_q, weights.norm_k)):
            raise ValueError("Radeon SP requires torch_native QK RMSNorm with eps=1e-5")
        if type(weights.rope) is not TorchRealRope or weights.rope.layout != "split_half" or weights.rope.compute_dtype != torch.float32:
            raise ValueError("Radeon SP requires FP32 split-half TorchRealRope")
        if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1201"):
            raise ValueError("Radeon SP requires gfx1201")
        owner._radeon_sp_checked = True
        print(
            f"========== [RADEON OP][rank={os.environ.get('RANK', '0')}][ACTIVE] SP Ulysses: seq_p={state.exchange.world} "
            f"rows={state.exchange.sizes} heads/rank={heads}; SDMA IPC pulls + fused prepare + ASM core ==========",
            file=sys.stderr,
            flush=True,
        )
    return weights.norm_q._get_actual_weight(), weights.norm_k._get_actual_weight()


class SdmaEngine:
    """Per-rank exchange state: IPC send slots / core output, local receive buffers, flag page, copy streams."""

    def __init__(self, exchange, heads, device):
        from . import radeon_hip_ipc as hip

        self.hip, self.x, self.heads, self.device = hip, exchange, heads, device
        self.world, self.rank, self.total = exchange.world, exchange.rank, exchange.total
        self.rows, self.shard = exchange.sizes[exchange.rank], heads * HEAD_DIM
        self.padded = (self.total + 31) // 32 * 32
        w, r, sh, total = self.world, self.rows, self.shard, self.total

        def buffer(elements, shape):
            ptr = hip.malloc(elements * 2)  # not the expandable-segment allocator: IPC + large SDMA copies need hipMalloc
            return ptr, hip.tensor(ptr, elements * 2, device).view(torch.bfloat16).view(shape)

        self.send_ptr, self.send = buffer(3 * w * r * sh, (3, w, r, sh))  # [qkv][dest][my rows][shard]
        self.out_ptr, self.out = buffer(self.padded * sh, (1, self.padded, heads, HEAD_DIM))  # core output
        self.full_ptr, self.full = buffer(3 * total * sh, (3, total, sh))  # [qkv][all rows][shard]
        self.stage_ptr, self.stage = buffer(w * r * sh, (w, r, sh))  # [src][my rows][shard]
        self.flags = hip.malloc(4096, hip.FINEGRAINED)
        hip.check(hip.lib().hipMemset(self.flags, 0, 4096), "hipMemset")
        torch.cuda.synchronize(device)
        infos = [None] * w
        dist.all_gather_object(infos, (hip.ipc_handle(self.send_ptr), hip.ipc_handle(self.out_ptr), hip.ipc_handle(self.flags)), group=exchange.group)
        peers = [p for p in range(w) if p != self.rank]
        self.peer_send, self.peer_out, self.peer_flags = ({p: hip.ipc_open(*infos[p][i]) for p in peers} for i in range(3))
        power_of_two = w & (w - 1) == 0
        self.order = [self.rank ^ k if power_of_two else (self.rank + k) % w for k in range(1, w)]
        self.streams = {p: torch.cuda.Stream(device=device) for p in self.order}
        self.qkv_pulled = {p: torch.cuda.Event() for p in self.order}
        self.out_pulled = {}  # (src, chunk) -> event
        self.prepared, self.unpacked = torch.cuda.Event(), torch.cuda.Event()
        self.core_started = torch.cuda.Event(enable_timing=True)
        self.core_released = hip.release_event()
        self.o_ready = {p: torch.cuda.Event(enable_timing=True) for p in self.order}  # when each peer's O was readable
        self.pull_order = list(self.order)
        self.epoch = 0
        dist.barrier(group=exchange.group)

    def _mine(self, region, peer):
        return self.flags + 4 * (region + peer)

    def _theirs(self, peer, region):
        return self.peer_flags[peer] + 4 * (region + self.rank)

    def _chunks(self, rows):
        count = max(1, int(os.environ.get("RADEON_CORESW_SP_O_CHUNKS", "3")))
        step = rows if count == 1 else (-(-rows // count) + 255) // 256 * 256
        return [(r, min(r + step, rows)) for r in range(0, rows, step)]

    def _update_pull_order(self):
        """Pull peers' O in the order it became ready in the previous epoch (all P2P pulls run serialized on one SDMA
        ring, so a late peer at the head blocks every later pull). Local timing events of the previous epoch only (the
        block-end sync makes them complete); RADEON_CORESW_SP_PULL_ORDER=0 keeps the XOR order. Schedule only."""
        if self.epoch < 3 or os.environ.get("RADEON_CORESW_SP_PULL_ORDER", "1") != "1":
            return
        if not all(event.query() for event in [self.core_started, *self.o_ready.values()]):
            return
        ready = {p: self.core_started.elapsed_time(self.o_ready[p]) for p in self.order}
        self.pull_order = sorted(self.order, key=lambda p: ready[p])

    def _pulled(self, src, chunk):
        return self.out_pulled.setdefault((src, chunk), torch.cuda.Event())

    def attention(self, inputs, weights, query_weight, key_weight, cosine, sine, project=None):
        """project: optional [K, N] out-projection weight view (MMWeight._get_actual_weight()); if given, returns
        torch.mm(unpacked attention output, project) computed per row chunk, else the unpacked output."""
        from aiter.ops.gfx1201.asm_attention import launch_hip_sage_core
        from aiter.ops.gfx1201.norm_rope_prepare import norm_rope_prepare_sage

        hip, x, me, w, sh, rows = self.hip, self.x, self.rank, self.world, self.shard, self.rows
        self.epoch += 1
        self._update_pull_order()
        epoch, stream = self.epoch, torch.cuda.current_stream(self.device)
        own = slice(x.offsets[me], x.offsets[me] + rows)
        full_gemm = os.environ.get("RADEON_CORESW_SP_FULLGEMM") == "1"
        projected = [torch.mm(inputs, weight) for weight in weights] if full_gemm else None
        for dest in self.order + [me]:
            if dest != me:
                hip.wait_value(stream.cuda_stream, self._mine(QKV_PULLED, dest), epoch - 1)
            for index, weight in enumerate(weights):
                out = self.send[index, dest] if dest != me else self.full[index, own]
                if full_gemm:
                    out.copy_(projected[index][:, dest * sh : (dest + 1) * sh])
                else:
                    _mm(inputs, weight[:, dest * sh : (dest + 1) * sh], out, w)
            if dest != me:
                hip.write_value(stream.cuda_stream, self._theirs(dest, QKV_READY), epoch)
        del projected
        chunk = rows * sh * 2
        for src in self.order:
            copy = self.streams[src]
            copy.wait_event(self.prepared)  # previous epoch's prepare has read the full buffers
            hip.wait_value(copy.cuda_stream, self._mine(QKV_READY, src), epoch)
            src_rows = x.sizes[src]
            for index in range(3):
                hip.copy(self.full_ptr + (index * self.total + x.offsets[src]) * sh * 2, self.peer_send[src] + (index * w + me) * src_rows * sh * 2, src_rows * sh * 2, copy.cuda_stream)
            hip.write_value(copy.cuda_stream, self._theirs(src, QKV_PULLED), epoch)
            self.qkv_pulled[src].record(copy)
        for src in self.order:
            stream.wait_event(self.qkv_pulled[src])
        query, key, value = (self.full[index].view(self.total, self.heads, HEAD_DIM) for index in range(3))
        prepared = norm_rope_prepare_sage(query, key, value, query_weight, key_weight, cosine, sine)
        self.prepared.record(stream)
        for dest in self.order:
            hip.wait_value(stream.cuda_stream, self._mine(O_PULLED, dest), epoch - 1)
        query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale = prepared
        self.core_started.record(stream)
        if SOL is None or not SOL.attention(prepared, self.out, self.total):
            launch_hip_sage_core(query_int8, key_int8, value_fp8, query_scale, key_scale, value_scale, self.out, 1, self.padded, self.total, self.heads)
        del prepared
        # peers pull ``self.out`` over SDMA once O_READY lands; without a system-scope release the core's stores can
        # still sit in L2 (observed with the Sol core: nondeterministic stale rows)
        hip.record(self.core_released, stream.cuda_stream)
        for dest in self.order:
            hip.write_value(stream.cuda_stream, self._theirs(dest, O_READY), epoch)
        chunks = self._chunks(rows)
        for src in self.order:
            copy = self.streams[src]
            copy.wait_event(self.unpacked)  # previous epoch's unpack has read the stage buffers
            hip.wait_value(copy.cuda_stream, self._mine(O_READY, src), epoch)
            self.o_ready[src].record(copy)
        # chunk-major submission: all P2P pulls of this rank execute serialized in submission order on one SDMA
        # ring (measured), so every peer's chunk 0 lands before any chunk 1 and the first GEMM starts early
        for index, (first, last) in enumerate(chunks):
            for src in self.pull_order:
                copy = self.streams[src]
                hip.copy(self.stage_ptr + (src * rows + first) * sh * 2, self.peer_out[src] + (x.offsets[me] + first) * sh * 2, (last - first) * sh * 2, copy.cuda_stream)
                self._pulled(src, index).record(copy)
        for src in self.order:
            hip.write_value(self.streams[src].cuda_stream, self._theirs(src, O_PULLED), epoch)
        output = torch.empty((rows, w * sh), device=self.device, dtype=torch.bfloat16)
        blocks = output.view(rows, w, sh)
        own_rows = self.out[0, own].reshape(rows, sh)
        projected = None
        if project is not None:
            projected = torch.empty((rows, project.shape[1]), device=self.device, dtype=torch.bfloat16)
        for index, (first, last) in enumerate(chunks):
            blocks[first:last, me].copy_(own_rows[first:last])
            for src in self.order:
                stream.wait_event(self._pulled(src, index))
                blocks[first:last, src].copy_(self.stage[src, first:last])
            if project is not None:
                _mm(output[first:last], project, projected[first:last], w)
        self.unpacked.record(stream)
        return output if project is None else projected


def prefetch_gate():
    """Event the offload loop's next-block weight DMA should wait for (RADEON_CORESW_SP_PREFETCH_GATE: core = start of
    the attention core, post = end of the attention path, none = no wait); keeps the DMA off the Q/K/V SDMA pulls."""
    gate = os.environ.get("RADEON_CORESW_SP_PREFETCH_GATE", "core")
    if gate == "none" or not _engines:
        return None
    engine = next(iter(_engines.values()))
    return engine.core_started if gate == "core" else engine.unpacked


def get_engine(exchange, heads, device):
    key = (id(exchange), heads, str(device))
    if key not in _engines:
        _engines[key] = SdmaEngine(exchange, heads, device)
    return _engines[key]


def _projection_weights(attn):
    modules = (attn.to_q, attn.to_k, attn.to_v)
    if any(getattr(m, "has_lora_branch", False) or getattr(m, "has_diff", False) or getattr(m, "bias", None) is not None for m in modules):
        raise ValueError("Radeon SP requires plain (no LoRA/diff/bias) Q/K/V projections")
    return tuple(m._get_actual_weight() if hasattr(m, "_get_actual_weight") else m.weight for m in modules)


def _down_projection(linear, inputs):
    """FFN out-projection as column blocks of one output (RADEON_CORESW_SP_DOWN_SPLIT column blocks)."""
    parts = int(os.environ.get("RADEON_CORESW_SP_DOWN_SPLIT", "2"))
    if parts <= 1 or type(linear).__name__ != "MMWeight" or linear.has_lora_branch or getattr(linear, "bias", None) is not None:
        return linear.apply(inputs)
    weight = linear._get_actual_weight()
    columns = weight.shape[1]
    step = (-(-columns // parts) + 127) // 128 * 128
    output = torch.empty((inputs.shape[0], columns), device=inputs.device, dtype=inputs.dtype)
    for first in range(0, columns, step):
        _mm(inputs, weight[:, first : first + step], output[:, first : first + step])
    return output


def _norm_modulate(norm, hidden_states, scale, shift, indices, modulate):
    """RMSNorm + adaLN modulation; RADEON_CORESW_RMS_MOD=1 -> aiter gfx1201 fused HIP kernel."""
    if os.environ.get("RADEON_CORESW_RMS_MOD", "1") == "1" and type(norm).__name__ == "RMSWeightNative":
        from aiter.ops.gfx1201.h3_ops import rms_modulate

        return rms_modulate(hidden_states, norm._get_actual_weight(), scale, shift, indices, norm.eps)
    return modulate(norm.apply(hidden_states), scale, shift, indices)


def infer_block(owner, weights, hidden_states, pre_infer_out, modulation):
    from aiter.ops.gfx1201.h3_ops import gated_residual
    from aiter.ops.gfx1201.modulation import modulate

    state = pre_infer_out.sequence_parallel_state
    exchange = state.exchange
    heads = owner.num_heads // exchange.world
    attn = weights.attn
    query_weight, key_weight = _norm_weights(owner, attn, state, heads)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    indices = pre_infer_out.adaln_indices
    with record_function("radeon_sp.norm_modulation.attn"):
        normed = _norm_modulate(weights.norm1, hidden_states, scale_msa, shift_msa, indices, modulate)
    to_out = attn.to_out
    direct = type(to_out).__name__ == "MMWeight" and not to_out.has_lora_branch and getattr(to_out, "bias", None) is None
    with record_function("radeon_sp.attention"):
        output = get_engine(exchange, heads, normed.device).attention(
            normed, _projection_weights(attn), query_weight, key_weight, *state.rotary_emb, project=to_out._get_actual_weight() if direct else None
        )
    del normed
    with record_function("radeon_sp.out_projection_residual"):
        hidden_states = gated_residual(hidden_states, output if direct else to_out.apply(output), gate_msa, indices)
    del output
    with record_function("radeon_sp.norm_modulation.ffn"):
        normed = _norm_modulate(weights.norm2, hidden_states, scale_mlp, shift_mlp, indices, modulate)
    with record_function("radeon_sp.ffn_residual"):
        if os.environ.get("RADEON_CORESW_SWIGLU", "1") == "1":
            from aiter.ops.gfx1201.h3_ops import swiglu

            up = weights.ff.in_proj
            if type(up).__name__ == "MMWeight" and not up.has_lora_branch and getattr(up, "bias", None) is None:
                up = _mm(normed, up._get_actual_weight(), world=exchange.world)
            else:
                up = up.apply(normed)
            ffn = _down_projection(weights.ff.out_proj, swiglu(up))
        else:
            ffn = owner._ff(weights.ff, normed)
        return gated_residual(hidden_states, ffn, gate_mlp, indices)
