from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v_platform.base.global_var import AI_DEVICE


_ATTENTION_STATE_NAMES = (
    "self_attn_cu_seqlens_qkv",
    "cross_attn_cu_seqlens_q",
    "cross_attn_cu_seqlens_kv",
    "cross_attn_cu_seqlens_kv_img",
)


@dataclass
class _GraphEvents:
    start: Any
    load_ready: Tuple[Any, ...]
    block_done: Tuple[Any, ...]


@dataclass
class _GraphEntry:
    key: tuple
    label: str
    graph: Any
    pool: Any
    static_pre_infer_out: WanPreInferModuleOutput
    static_output: torch.Tensor
    captured_output: torch.Tensor
    attention_state: Dict[str, Optional[torch.Tensor]]
    events: _GraphEvents


class WanOffloadCudaGraphRunner:
    """Capture the fixed Wan DiT block/offload schedule for one active shape."""

    def __init__(self, transformer_infer, offload_manager, warmup_iters=3):
        if AI_DEVICE != "cuda":
            raise ValueError("Wan CUDA Graph inference requires PLATFORM=cuda")
        if warmup_iters < 1:
            raise ValueError("Wan CUDA Graph requires at least one warmup iteration")
        self.transformer_infer = transformer_infer
        self.offload_manager = offload_manager
        self.warmup_iters = warmup_iters
        self._entry = None
        self.capture_count = 0
        self.replay_count = 0

    @property
    def raw_cuda_graph_exec(self):
        if self._entry is None or not hasattr(self._entry.graph, "raw_cuda_graph_exec"):
            return None
        return self._entry.graph.raw_cuda_graph_exec()

    @torch.no_grad()
    def run(self, blocks, x, pre_infer_out):
        self._validate_inputs(blocks, x, pre_infer_out)
        key = self._shape_key(blocks, x, pre_infer_out)
        if self._entry is None or self._entry.key != key:
            self._replace_entry(blocks, x, pre_infer_out, key)

        entry = self._entry
        current_stream = torch.cuda.current_stream(device=x.device)
        compute_stream = self.offload_manager.compute_stream
        compute_stream.wait_stream(current_stream)

        with torch.cuda.stream(compute_stream):
            self._copy_inputs(entry.static_pre_infer_out, x, pre_infer_out)
            self._restore_attention_state(entry.attention_state)
            entry.graph.replay()

        current_stream.wait_stream(compute_stream)
        self.replay_count += 1
        logger.info(f"[CUDA Graph] replay {entry.label}, count={self.replay_count}, key={key}")
        return entry.static_output

    def _replace_entry(self, blocks, x, pre_infer_out, key):
        if self._entry is not None:
            logger.info(f"[CUDA Graph] input shape changed; recapturing Wan graph: {self._entry.key} -> {key}")
            self.offload_manager.compute_stream.synchronize()
            self.offload_manager.cuda_load_stream.synchronize()
            self._entry = None

        try:
            self._entry = self._capture(blocks, x, pre_infer_out, key)
        except Exception as exc:
            raise RuntimeError(f"Failed to capture Wan block-offload CUDA Graph for key={key}") from exc

    def _capture(self, blocks, x, pre_infer_out, key):
        self._validate_pinned_block_weights(blocks)
        label = self._expert_label()
        logger.info(f"[CUDA Graph] capturing {label} Wan DiT blocks, key={key}")

        current_stream = torch.cuda.current_stream(device=x.device)
        compute_stream = self.offload_manager.compute_stream
        compute_stream.wait_stream(current_stream)

        static_pre_infer_out = self._create_static_pre_infer_out(x, pre_infer_out)
        static_output = torch.empty_like(x)
        events = self._create_events(len(blocks))

        with torch.cuda.stream(compute_stream):
            for _ in range(self.warmup_iters):
                self._copy_inputs(static_pre_infer_out, x, pre_infer_out)
                self.transformer_infer.cos_sin = static_pre_infer_out.cos_sin
                warmup_output = self._enqueue_blocks(blocks, static_pre_infer_out, events)

        compute_stream.synchronize()
        del warmup_output

        # Warmup creates constant cu_seqlens outside capture. The graph entry
        # retains them because Wan resets its Python-side attention state per step.
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(compute_stream):
            self._copy_inputs(static_pre_infer_out, x, pre_infer_out)
            self.transformer_infer.cos_sin = static_pre_infer_out.cos_sin
            with torch.cuda.graph(graph, pool=pool, stream=compute_stream, capture_error_mode="relaxed"):
                captured_output = self._enqueue_blocks(blocks, static_pre_infer_out, events)
                static_output.copy_(captured_output)

        attention_state = self._capture_attention_state()
        self.capture_count += 1
        logger.info(f"[CUDA Graph] captured {label} Wan DiT blocks, capture_count={self.capture_count}")
        return _GraphEntry(
            key=key,
            label=label,
            graph=graph,
            pool=pool,
            static_pre_infer_out=static_pre_infer_out,
            static_output=static_output,
            captured_output=captured_output,
            attention_state=attention_state,
            events=events,
        )

    def _enqueue_blocks(self, blocks, pre_infer_out, events):
        block_count = len(blocks)
        compute_stream = self.offload_manager.compute_stream
        load_stream = self.offload_manager.cuda_load_stream

        events.start.record(compute_stream)
        load_stream.wait_event(events.start)
        with torch.cuda.stream(load_stream):
            initial_count = min(2, block_count)
            for block_idx in range(initial_count):
                self.offload_manager.load_block_to_buffer(block_idx, block_idx, blocks)
                events.load_ready[block_idx].record(load_stream)

        x = pre_infer_out.x
        for block_idx in range(block_count):
            buffer_idx = block_idx % 2
            compute_stream.wait_event(events.load_ready[block_idx])
            self.transformer_infer.block_idx = block_idx
            x = self.transformer_infer.infer_block(
                self.offload_manager.cuda_buffers[buffer_idx],
                x,
                pre_infer_out,
            )
            events.block_done[block_idx].record(compute_stream)

            next_block_idx = block_idx + 2
            if next_block_idx < block_count:
                load_stream.wait_event(events.block_done[block_idx])
                with torch.cuda.stream(load_stream):
                    self.offload_manager.load_block_to_buffer(buffer_idx, next_block_idx, blocks)
                    events.load_ready[next_block_idx].record(load_stream)

        return x

    def _create_static_pre_infer_out(self, x, pre_infer_out):
        return WanPreInferModuleOutput(
            embed=torch.empty_like(pre_infer_out.embed),
            grid_sizes=GridOutput(
                tensor=torch.empty_like(pre_infer_out.grid_sizes.tensor),
                tuple=tuple(pre_infer_out.grid_sizes.tuple),
            ),
            x=torch.empty_like(x),
            embed0=torch.empty_like(pre_infer_out.embed0),
            context=torch.empty_like(pre_infer_out.context),
            cos_sin=torch.empty_like(pre_infer_out.cos_sin),
            valid_token_len=pre_infer_out.valid_token_len,
            valid_latent_num=pre_infer_out.valid_latent_num,
            adapter_args=dict(pre_infer_out.adapter_args),
            conditional_dict=dict(pre_infer_out.conditional_dict),
        )

    def _copy_inputs(self, static_pre_infer_out, x, pre_infer_out):
        static_pre_infer_out.x.copy_(x)
        static_pre_infer_out.embed.copy_(pre_infer_out.embed)
        static_pre_infer_out.embed0.copy_(pre_infer_out.embed0)
        static_pre_infer_out.context.copy_(pre_infer_out.context)
        static_pre_infer_out.cos_sin.copy_(pre_infer_out.cos_sin)
        static_pre_infer_out.grid_sizes.tensor.copy_(pre_infer_out.grid_sizes.tensor)
        self.transformer_infer.cos_sin = static_pre_infer_out.cos_sin

    def _create_events(self, block_count):
        return _GraphEvents(
            start=torch.cuda.Event(enable_timing=False),
            load_ready=tuple(torch.cuda.Event(enable_timing=False) for _ in range(block_count)),
            block_done=tuple(torch.cuda.Event(enable_timing=False) for _ in range(block_count)),
        )

    def _capture_attention_state(self):
        return {name: getattr(self.transformer_infer, name, None) for name in _ATTENTION_STATE_NAMES}

    def _restore_attention_state(self, state):
        for name, value in state.items():
            setattr(self.transformer_infer, name, value)

    def _validate_inputs(self, blocks, x, pre_infer_out):
        if len(blocks) == 0:
            raise ValueError("Wan CUDA Graph requires at least one transformer block")
        if len(self.offload_manager.cuda_buffers) != 2:
            raise ValueError("Wan CUDA Graph block offload requires exactly two CUDA weight buffers")
        if self.transformer_infer.has_post_adapter:
            raise ValueError("Wan CUDA Graph does not support post adapters")
        if any(value is not None for value in pre_infer_out.adapter_args.values()):
            raise ValueError("Wan CUDA Graph does not support dynamic adapter inputs")
        if pre_infer_out.conditional_dict:
            raise ValueError("Wan CUDA Graph does not support conditional_dict inputs")
        if pre_infer_out.cos_sin is None:
            raise ValueError("Wan CUDA Graph requires a precomputed cos_sin tensor")

        tensors = {
            "x": x,
            "embed": pre_infer_out.embed,
            "embed0": pre_infer_out.embed0,
            "context": pre_infer_out.context,
            "cos_sin": pre_infer_out.cos_sin,
            "grid_sizes": pre_infer_out.grid_sizes.tensor,
        }
        for name, tensor in tensors.items():
            if tensor.device.type != "cuda":
                raise ValueError(f"Wan CUDA Graph input {name} must be on CUDA, got {tensor.device}")

    def _validate_pinned_block_weights(self, blocks):
        pageable = []
        for block_idx in range(len(blocks)):
            for name, tensor in self.offload_manager.get_block_state_dict(block_idx, blocks).items():
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and not tensor.is_pinned():
                    pageable.append(f"block={block_idx}:{name}")
                    if len(pageable) == 8:
                        break
            if len(pageable) == 8:
                break
        if pageable:
            details = ", ".join(pageable)
            raise ValueError(f"Wan CUDA Graph requires pinned CPU offload weights; pageable tensors: {details}")

    def _shape_key(self, blocks, x, pre_infer_out):
        return (
            len(blocks),
            tuple(x.shape),
            str(x.dtype),
            tuple(pre_infer_out.embed.shape),
            str(pre_infer_out.embed.dtype),
            tuple(pre_infer_out.embed0.shape),
            str(pre_infer_out.embed0.dtype),
            tuple(pre_infer_out.context.shape),
            str(pre_infer_out.context.dtype),
            tuple(pre_infer_out.cos_sin.shape),
            str(pre_infer_out.cos_sin.dtype),
            tuple(pre_infer_out.grid_sizes.tuple),
            str(x.device),
        )

    def _expert_label(self):
        step_index = getattr(self.transformer_infer.scheduler, "step_index", -1)
        boundary = self.transformer_infer.config.get("boundary_step_index", 2)
        return "high-noise" if step_index < boundary else "low-noise"
