"""
DisaggMixin: Mooncake-based disaggregation communication mixin for Runners.

Provides send/receive capabilities for encoder outputs over RDMA/TCP,
allowing Encoder and Transformer roles to run on separate devices/machines.

Decentralized queue mode (disagg_config.decentralized_queue=true) follows PR #964:
phase1 / phase2 RDMA meta rings (see RDMABuffer) for dispatching jobs across torch
HTTP encoder and pull-based transformer/decoder workers.

LIMITATION: RDMABuffer multi-consumer ordering uses a read-modify-write FAA shim
(rdma_client.rdma_faa) that is not a true remote atomic across processes. Use for
controlled benchmarks / single-controller deployments; for production multi-consumer
correctness prefer IBV_WR_ATOMIC_FETCH_AND_ADD or single-consumer transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from lightx2v.disagg.conn import (
    DataArgs,
    DataManager,
    DataPoll,
    DataReceiver,
    DataSender,
    DisaggregationMode,
    DisaggregationPhase,
)
from lightx2v.disagg.rdma_buffer import RDMABuffer, RDMABufferDescriptor
from lightx2v.disagg.rdma_client import RDMAClient
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

logger = logging.getLogger(__name__)


def _estimate_encoder_buffer_sizes(config) -> List[int]:
    """Estimate upper-bound byte sizes for each RDMA buffer slot."""
    text_len = int(config.get("text_len", 512))
    enable_cfg = bool(config.get("enable_cfg", False))
    use_image_encoder = bool(config.get("use_image_encoder", True))
    task = config.get("task", "i2v")

    text_dim = int(config.get("text_encoder_dim", 4096))
    clip_dim = int(config.get("clip_embed_dim", 1024))
    z_dim = int(config.get("vae_z_dim", 16))

    vae_stride = config.get("vae_stride", (4, 8, 8))
    stride_t, stride_h, stride_w = int(vae_stride[0]), int(vae_stride[1]), int(vae_stride[2])

    target_video_length = int(config.get("target_video_length", 81))
    target_height = int(config.get("target_height", 480))
    target_width = int(config.get("target_width", 832))

    t_prime = 1 + (target_video_length - 1) // stride_t
    h_prime = int(math.ceil(target_height / stride_h))
    w_prime = int(math.ceil(target_width / stride_w))

    bytes_per_elem = torch.tensor([], dtype=torch.float32).element_size()
    int_bytes_per_elem = torch.tensor([], dtype=torch.int64).element_size()

    buffer_sizes = []
    # context
    context_bytes = text_len * text_dim * bytes_per_elem
    buffer_sizes.append(context_bytes)
    # context_null (if cfg enabled)
    if enable_cfg:
        buffer_sizes.append(context_bytes)
    # clip + vae (if i2v task)
    if task in ("i2v", "flf2v", "animate", "s2v", "rs2v", "i2i"):
        if use_image_encoder:
            buffer_sizes.append(clip_dim * bytes_per_elem)
        vae_bytes = (z_dim + 4) * t_prime * h_prime * w_prime * bytes_per_elem
        buffer_sizes.append(vae_bytes)
    # latent_shape buf
    buffer_sizes.append(10 * int_bytes_per_elem)
    # metadata
    buffer_sizes.append(4096)

    return buffer_sizes


def _buffer_view(buf: torch.Tensor, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    """Create a typed view over a raw uint8 buffer without copying."""
    view = torch.empty(0, dtype=dtype, device=buf.device)
    view.set_(buf.untyped_storage(), 0, shape)
    return view


def _sha256_tensor(tensor: Optional[torch.Tensor]) -> Optional[str]:
    if tensor is None:
        return None
    data_tensor = tensor.detach()
    if data_tensor.dtype == torch.bfloat16:
        data_tensor = data_tensor.to(torch.float32)
    data = data_tensor.contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class DisaggMixin:
    """Mixin that adds Mooncake disaggregation capabilities to a Runner."""

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def init_disagg(self, config):
        """Initialize Mooncake communication based on ``disagg_mode``.

        Supported modes:
        - "encoder"     : Phase 1 sender  (Encoder → Transformer)
        - "transformer" : Phase 1 receiver + optional Phase 2 sender (→ Decoder)
        - "decode"      : Phase 2 receiver (Transformer → Decoder)

        When ``disagg_config.decentralized_queue`` is true, Mooncake sessions are
        created per request (or per dispatch packet); RDMA meta rings connect to
        the controller process (see ControllerService).
        """
        disagg_cfg = config.get("disagg_config", {})
        self._disagg_mode = config.get("disagg_mode")  # "encoder" | "transformer" | "decode" | None
        self._disagg_bootstrap_addr = disagg_cfg.get("bootstrap_addr", "127.0.0.1")
        self._disagg_bootstrap_room = int(disagg_cfg.get("bootstrap_room", 0))
        self._disagg_sender_rank = int(disagg_cfg.get("sender_engine_rank", 0))
        self._disagg_receiver_rank = int(disagg_cfg.get("receiver_engine_rank", 1))
        self._disagg_data_mgr: Optional[DataManager] = None
        self._disagg_sender: Optional[DataSender] = None
        self._disagg_receiver: Optional[DataReceiver] = None
        self._disagg_rdma_buffers: List[torch.Tensor] = []

        # Phase 2 attributes (Transformer → Decoder)
        self._disagg_p2_data_mgr: Optional[DataManager] = None
        self._disagg_p2_sender: Optional[DataSender] = None
        self._disagg_p2_receiver: Optional[DataReceiver] = None
        self._disagg_p2_rdma_buffers: List[torch.Tensor] = []

        self._disagg_decentralized = bool(disagg_cfg.get("decentralized_queue", False))
        self._disagg_cfg = disagg_cfg
        self._disagg_phase1_queue: Optional[RDMABuffer] = None
        self._disagg_phase2_queue: Optional[RDMABuffer] = None
        self._disagg_phase1_client: Optional[RDMAClient] = None
        self._disagg_phase2_client: Optional[RDMAClient] = None
        self._disagg_reconnect_cooldown_s = 5.0
        self._disagg_phase1_last_retry_ts = 0.0
        self._disagg_phase2_last_retry_ts = 0.0
        self._disagg_active_encoder_room: Optional[int] = None
        self._disagg_active_transformer_room: Optional[int] = None
        self._disagg_active_decoder_room: Optional[int] = None

        if self._disagg_mode == "encoder":
            if self._disagg_decentralized:
                self._disagg_data_mgr = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.ENCODE)
                self._ensure_disagg_phase1_queue_producer(disagg_cfg)
                logger.info("[Disagg] Encoder decentralized queue mode: per-request Mooncake session.")
            else:
                buffer_sizes = _estimate_encoder_buffer_sizes(config)
                self._disagg_alloc_buffers(buffer_sizes)
                data_ptrs = [buf.data_ptr() for buf in self._disagg_rdma_buffers]
                data_lens = [buf.numel() for buf in self._disagg_rdma_buffers]
                data_args = DataArgs(
                    sender_engine_rank=self._disagg_sender_rank,
                    receiver_engine_rank=self._disagg_receiver_rank,
                    data_ptrs=data_ptrs,
                    data_lens=data_lens,
                    data_item_lens=data_lens,
                    ib_device=None,
                )
                self._disagg_data_mgr = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.ENCODE)
                self._disagg_data_mgr.init(data_args, self._disagg_bootstrap_room)
                self._disagg_sender = DataSender(
                    self._disagg_data_mgr,
                    self._disagg_bootstrap_addr,
                    self._disagg_bootstrap_room,
                )

        elif self._disagg_mode == "transformer":
            if self._disagg_decentralized:
                self._disagg_data_mgr = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.TRANSFORMER)
                self._disagg_p2_data_mgr = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.TRANSFORMER)
                self._ensure_disagg_phase1_queue_consumer(disagg_cfg)
                if disagg_cfg.get("decoder_engine_rank") is not None:
                    self._ensure_disagg_phase2_queue_producer(disagg_cfg)
                logger.info("[Disagg] Transformer decentralized queue mode: dispatch via phase1/phase2 rings.")
            else:
                # Phase 1: receive encoder outputs
                buffer_sizes = _estimate_encoder_buffer_sizes(config)
                self._disagg_alloc_buffers(buffer_sizes)
                data_ptrs = [buf.data_ptr() for buf in self._disagg_rdma_buffers]
                data_lens = [buf.numel() for buf in self._disagg_rdma_buffers]
                data_args = DataArgs(
                    sender_engine_rank=self._disagg_sender_rank,
                    receiver_engine_rank=self._disagg_receiver_rank,
                    data_ptrs=data_ptrs,
                    data_lens=data_lens,
                    data_item_lens=data_lens,
                    ib_device=None,
                )
                self._disagg_data_mgr = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.TRANSFORMER)
                self._disagg_data_mgr.init(data_args, self._disagg_bootstrap_room)
                self._disagg_receiver = DataReceiver(
                    self._disagg_data_mgr,
                    self._disagg_bootstrap_addr,
                    self._disagg_bootstrap_room,
                )
                self._disagg_receiver.init()

                # Phase 2 (optional): send latents to decoder
                if disagg_cfg.get("decoder_engine_rank") is not None:
                    self._init_phase2_transformer_sender(config, disagg_cfg)

        elif self._disagg_mode == "decode":
            if self._disagg_decentralized:
                self._disagg_p2_data_mgr = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.DECODE)
                self._ensure_disagg_phase2_queue_consumer(disagg_cfg)
                logger.info("[Disagg] Decoder decentralized queue mode: phase2 dispatch ring.")
            else:
                # Phase 2: receive latents from transformer
                self._init_phase2_decoder_receiver(config, disagg_cfg)

    def _disagg_alloc_buffers(self, buffer_sizes: List[int]):
        self._disagg_rdma_buffers = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
            self._disagg_rdma_buffers.append(buf)

    def _disagg_alloc_p2_buffers(self, buffer_sizes: List[int]):
        self._disagg_p2_rdma_buffers = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
            self._disagg_p2_rdma_buffers.append(buf)

    def _init_phase2_transformer_sender(self, config, disagg_cfg):
        """Setup Phase 2 sender for Transformer role (send latents to Decoder)."""
        from lightx2v.disagg.utils import estimate_transformer_buffer_sizes

        p2_transformer_rank = int(disagg_cfg.get("receiver_engine_rank", 1))
        p2_decoder_rank = int(disagg_cfg.get("decoder_engine_rank", 2))
        p2_bootstrap_addr = disagg_cfg.get("bootstrap_addr", "127.0.0.1")
        p2_bootstrap_room = int(disagg_cfg.get("decoder_bootstrap_room", 1))

        buffer_sizes = estimate_transformer_buffer_sizes(config)
        self._disagg_alloc_p2_buffers(buffer_sizes)
        data_ptrs = [buf.data_ptr() for buf in self._disagg_p2_rdma_buffers]
        data_lens = [buf.numel() for buf in self._disagg_p2_rdma_buffers]
        data_args = DataArgs(
            sender_engine_rank=p2_transformer_rank,
            receiver_engine_rank=p2_decoder_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self._disagg_p2_data_mgr = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.TRANSFORMER)
        self._disagg_p2_data_mgr.init(data_args, p2_bootstrap_room)
        self._disagg_p2_sender = DataSender(self._disagg_p2_data_mgr, p2_bootstrap_addr, p2_bootstrap_room)
        logger.info(f"[Disagg] Phase2 sender initialized (rank {p2_transformer_rank} → {p2_decoder_rank}, room={p2_bootstrap_room})")

    def _init_phase2_decoder_receiver(self, config, disagg_cfg):
        """Setup Phase 2 receiver for Decoder role (receive latents from Transformer)."""
        from lightx2v.disagg.utils import estimate_transformer_buffer_sizes

        p2_transformer_rank = int(disagg_cfg.get("sender_engine_rank", 1))
        p2_decoder_rank = int(disagg_cfg.get("receiver_engine_rank", 2))
        p2_bootstrap_addr = disagg_cfg.get("bootstrap_addr", "127.0.0.1")
        p2_bootstrap_room = int(disagg_cfg.get("bootstrap_room", 1))

        buffer_sizes = estimate_transformer_buffer_sizes(config)
        self._disagg_alloc_p2_buffers(buffer_sizes)
        data_ptrs = [buf.data_ptr() for buf in self._disagg_p2_rdma_buffers]
        data_lens = [buf.numel() for buf in self._disagg_p2_rdma_buffers]
        data_args = DataArgs(
            sender_engine_rank=p2_transformer_rank,
            receiver_engine_rank=p2_decoder_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self._disagg_p2_data_mgr = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.DECODE)
        self._disagg_p2_data_mgr.init(data_args, p2_bootstrap_room)
        self._disagg_p2_receiver = DataReceiver(self._disagg_p2_data_mgr, p2_bootstrap_addr, p2_bootstrap_room)
        self._disagg_p2_receiver.init()
        logger.info(f"[Disagg] Phase2 receiver initialized (rank {p2_transformer_rank} → {p2_decoder_rank}, room={p2_bootstrap_room})")

    # ------------------------------------------------------------------ #
    #  Decentralized RDMA meta queues (PR #964 style)
    # ------------------------------------------------------------------ #

    def _disagg_json_safe_value(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [self._disagg_json_safe_value(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._disagg_json_safe_value(v) for k, v in obj.items()}
        return str(obj)

    def _disagg_build_request_config_snapshot(self) -> Dict[str, Any]:
        """Payload for phase1/phase2 ring: per-request fields for workers."""
        disagg_cfg = dict(self.config.get("disagg_config", {}) or {})
        room = int(
            self.config.get("data_bootstrap_room", disagg_cfg.get("bootstrap_room", self._disagg_bootstrap_room))
        )
        payload: Dict[str, Any] = {
            "data_bootstrap_room": room,
            "task": self.config.get("task"),
            "model_cls": self.config.get("model_cls"),
        }
        for key in (
            "seed",
            "infer_steps",
            "aspect_ratio",
            "enable_cfg",
            "sample_guide_scale",
            "target_height",
            "target_width",
            "text_len",
            "controller_result_host",
            "controller_result_port",
        ):
            if key in self.config and self.config.get(key) is not None:
                payload[key] = self._disagg_json_safe_value(self.config.get(key))

        ii = getattr(self, "input_info", None)
        if ii is not None:
            if getattr(ii, "prompt", None):
                payload["prompt"] = ii.prompt
            if getattr(ii, "negative_prompt", None) is not None:
                payload["negative_prompt"] = ii.negative_prompt
            if getattr(ii, "save_result_path", None):
                payload["save_result_path"] = ii.save_result_path
                payload["save_path"] = ii.save_result_path
            if getattr(ii, "target_shape", None) is not None:
                payload["target_shape"] = self._disagg_json_safe_value(ii.target_shape)
            if getattr(ii, "aspect_ratio", None) and "aspect_ratio" not in payload:
                payload["aspect_ratio"] = ii.aspect_ratio
            if getattr(ii, "seed", None) is not None and "seed" not in payload:
                payload["seed"] = ii.seed

        dpr = self.config.get("disagg_phase1_receiver_engine_rank")
        if dpr is not None:
            try:
                payload["disagg_phase1_receiver_engine_rank"] = int(dpr)
            except (TypeError, ValueError):
                pass

        return {k: v for k, v in payload.items() if v is not None}

    def _disagg_connect_queue_client(
        self,
        disagg_cfg: dict,
        *,
        phase: str,
    ) -> None:
        host_key = "rdma_phase1_host" if phase == "phase1" else "rdma_phase2_host"
        port_key = "rdma_phase1_handshake_port" if phase == "phase1" else "rdma_phase2_handshake_port"
        host = str(disagg_cfg.get(host_key, "127.0.0.1"))
        port = int(disagg_cfg.get(port_key, 5567 if phase == "phase1" else 5568))
        slots = int(disagg_cfg.get("rdma_buffer_slots", 128))
        slot_size = int(disagg_cfg.get("rdma_buffer_slot_size", 4096))

        client_attr = "_disagg_phase1_client" if phase == "phase1" else "_disagg_phase2_client"
        buf_attr = "_disagg_phase1_queue" if phase == "phase1" else "_disagg_phase2_queue"
        if getattr(self, buf_attr, None) is not None:
            return
        client: Optional[RDMAClient] = getattr(self, client_attr)

        if client is None:
            client = RDMAClient(local_buffer_size=slot_size)
            setattr(self, client_attr, client)
        client.connect_to_server(host, port)
        remote_info = client.remote_info
        base_addr = int(remote_info["addr"])
        descriptor = RDMABufferDescriptor(
            slot_addr=base_addr + 16,
            slot_bytes=slots * slot_size,
            slot_size=slot_size,
            buffer_size=slots,
            head_addr=base_addr,
            tail_addr=base_addr + 8,
            rkey=int(remote_info.get("rkey", 0)),
        )
        setattr(
            self,
            buf_attr,
            RDMABuffer(role="client", rdma_client=client, remote=descriptor),
        )

    def _ensure_disagg_phase1_queue_producer(self, disagg_cfg: dict) -> None:
        try:
            self._disagg_connect_queue_client(disagg_cfg, phase="phase1")
        except Exception as exc:
            logger.warning("[Disagg] phase1 queue producer connect failed: %s", exc)

    def _ensure_disagg_phase1_queue_consumer(self, disagg_cfg: dict) -> None:
        try:
            self._disagg_connect_queue_client(disagg_cfg, phase="phase1")
        except Exception as exc:
            logger.warning("[Disagg] phase1 queue consumer connect failed: %s", exc)

    def _ensure_disagg_phase2_queue_producer(self, disagg_cfg: dict) -> None:
        try:
            self._disagg_connect_queue_client(disagg_cfg, phase="phase2")
        except Exception as exc:
            logger.warning("[Disagg] phase2 queue producer connect failed: %s", exc)

    def _ensure_disagg_phase2_queue_consumer(self, disagg_cfg: dict) -> None:
        try:
            self._disagg_connect_queue_client(disagg_cfg, phase="phase2")
        except Exception as exc:
            logger.warning("[Disagg] phase2 queue consumer connect failed: %s", exc)

    def disagg_try_consume_phase1(self) -> Optional[Dict[str, Any]]:
        """Non-blocking consume for transformer worker loop."""
        if not getattr(self, "_disagg_decentralized", False):
            return None
        if not self._disagg_phase1_queue:
            now = time.monotonic()
            if now - self._disagg_phase1_last_retry_ts < self._disagg_reconnect_cooldown_s:
                return None
            self._disagg_phase1_last_retry_ts = now
            self._ensure_disagg_phase1_queue_consumer(self._disagg_cfg)
            if not self._disagg_phase1_queue:
                return None
            logger.info("[Disagg] phase1 queue consumer connected (lazy retry succeeded)")
        try:
            return self._disagg_phase1_queue.consume()
        except Exception:
            logger.exception("[Disagg] phase1 consume failed")
            return None

    def disagg_try_consume_phase2(self) -> Optional[Dict[str, Any]]:
        """Non-blocking consume for decoder worker loop."""
        if not getattr(self, "_disagg_decentralized", False):
            return None
        if not self._disagg_phase2_queue:
            now = time.monotonic()
            if now - self._disagg_phase2_last_retry_ts < self._disagg_reconnect_cooldown_s:
                return None
            self._disagg_phase2_last_retry_ts = now
            self._ensure_disagg_phase2_queue_consumer(self._disagg_cfg)
            if not self._disagg_phase2_queue:
                return None
            logger.info("[Disagg] phase2 queue consumer connected (lazy retry succeeded)")
        try:
            return self._disagg_phase2_queue.consume()
        except Exception:
            logger.exception("[Disagg] phase2 consume failed")
            return None

    def disagg_transformer_prepare_dispatch(self, packet: Dict[str, Any]) -> None:
        """After phase1 consume: init Mooncake P1/P2 and publish phase2 meta for decoder."""
        if not self._disagg_decentralized:
            raise RuntimeError("disagg_transformer_prepare_dispatch requires decentralized_queue mode")

        req = dict(packet.get("request_config") or {})
        enc_addr = str(packet.get("encoder_node_address", "127.0.0.1"))

        with self.config.temporarily_unlocked():
            self.config.update(req)

        disagg_cfg = self.config.get("disagg_config", {})
        room = int(self.config.get("data_bootstrap_room", disagg_cfg.get("bootstrap_room", 0)))

        self.disagg_transformer_teardown_session()

        self._disagg_sender_rank = int(disagg_cfg.get("sender_engine_rank", self._disagg_sender_rank))
        pkt_recv_rank = req.get("disagg_phase1_receiver_engine_rank")
        if pkt_recv_rank is not None:
            self._disagg_receiver_rank = int(pkt_recv_rank)
        else:
            self._disagg_receiver_rank = int(disagg_cfg.get("receiver_engine_rank", self._disagg_receiver_rank))

        buffer_sizes = _estimate_encoder_buffer_sizes(self.config)
        self._disagg_alloc_buffers(buffer_sizes)
        data_ptrs = [buf.data_ptr() for buf in self._disagg_rdma_buffers]
        data_lens = [buf.numel() for buf in self._disagg_rdma_buffers]
        data_args = DataArgs(
            sender_engine_rank=self._disagg_sender_rank,
            receiver_engine_rank=self._disagg_receiver_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self._disagg_data_mgr.init(data_args, room)
        self._disagg_receiver = DataReceiver(self._disagg_data_mgr, enc_addr, room)
        self._disagg_receiver.init()

        from lightx2v.disagg.utils import estimate_transformer_buffer_sizes

        if disagg_cfg.get("decoder_engine_rank") is None:
            raise RuntimeError("decentralized transformer requires decoder_engine_rank in disagg_config")

        p2_transformer_rank = int(self._disagg_receiver_rank)
        p2_decoder_rank = int(disagg_cfg.get("decoder_engine_rank", 2))
        p2_bootstrap_addr = str(disagg_cfg.get("bootstrap_addr", "127.0.0.1"))

        buffer_sizes_p2 = estimate_transformer_buffer_sizes(self.config)
        self._disagg_alloc_p2_buffers(buffer_sizes_p2)
        p2_ptrs = [buf.data_ptr() for buf in self._disagg_p2_rdma_buffers]
        p2_lens = [buf.numel() for buf in self._disagg_p2_rdma_buffers]
        p2_args = DataArgs(
            sender_engine_rank=p2_transformer_rank,
            receiver_engine_rank=p2_decoder_rank,
            data_ptrs=p2_ptrs,
            data_lens=p2_lens,
            data_item_lens=p2_lens,
            ib_device=None,
        )
        self._disagg_p2_data_mgr.init(p2_args, room)
        self._disagg_p2_sender = DataSender(self._disagg_p2_data_mgr, p2_bootstrap_addr, room)

        if self._disagg_phase2_queue is None:
            self._ensure_disagg_phase2_queue_producer(self._disagg_cfg)
        if self._disagg_phase2_queue is None:
            raise RuntimeError("phase2 meta queue not connected; check Controller and rdma_phase2_* config")

        merged_req = {**self._disagg_build_request_config_snapshot(), **req}
        dc_out = {**dict(disagg_cfg)}
        dc_out["sender_engine_rank"] = int(self._disagg_receiver_rank)
        dc_out["receiver_engine_rank"] = int(disagg_cfg.get("decoder_engine_rank", 4))
        dc_out["bootstrap_room"] = room
        merged_req["disagg_config"] = dc_out
        phase2_meta = {
            "request_config": merged_req,
            "transformer_node_address": self._disagg_p2_data_mgr.get_localhost(),
            "transformer_session_id": self._disagg_p2_data_mgr.get_session_id(),
        }
        self._disagg_phase2_queue.produce(phase2_meta)
        self._disagg_active_transformer_room = room
        logger.info("[Disagg] Transformer dispatch prepared for room=%s", room)

    def disagg_transformer_teardown_session(self) -> None:
        room = self._disagg_active_transformer_room
        if room is None:
            self._disagg_rdma_buffers = []
            self._disagg_p2_rdma_buffers = []
            self._disagg_receiver = None
            self._disagg_p2_sender = None
            return
        try:
            if self._disagg_data_mgr is not None and room in self._disagg_data_mgr.data_args:
                self._disagg_data_mgr.remove(room)
        except Exception:
            logger.exception("[Disagg] transformer phase1 remove failed room=%s", room)
        try:
            if self._disagg_p2_data_mgr is not None and room in self._disagg_p2_data_mgr.data_args:
                self._disagg_p2_data_mgr.remove(room)
        except Exception:
            logger.exception("[Disagg] transformer phase2 remove failed room=%s", room)
        self._disagg_rdma_buffers = []
        self._disagg_p2_rdma_buffers = []
        self._disagg_receiver = None
        self._disagg_p2_sender = None
        self._disagg_active_transformer_room = None

    def disagg_decoder_prepare_dispatch(self, packet: Dict[str, Any]) -> None:
        """After phase2 consume: init Mooncake P2 receiver from transformer address."""
        if not self._disagg_decentralized:
            raise RuntimeError("disagg_decoder_prepare_dispatch requires decentralized_queue mode")

        req = dict(packet.get("request_config") or {})
        trans_addr = str(packet.get("transformer_node_address", "127.0.0.1"))

        with self.config.temporarily_unlocked():
            self.config.update(req)

        disagg_cfg = self.config.get("disagg_config", {})
        room = int(self.config.get("data_bootstrap_room", disagg_cfg.get("bootstrap_room", 0)))

        self.disagg_decoder_teardown_session()

        p2_transformer_rank = int(disagg_cfg.get("sender_engine_rank", 1))
        p2_decoder_rank = int(disagg_cfg.get("receiver_engine_rank", 2))

        from lightx2v.disagg.utils import estimate_transformer_buffer_sizes

        buffer_sizes = estimate_transformer_buffer_sizes(self.config)
        self._disagg_alloc_p2_buffers(buffer_sizes)
        data_ptrs = [buf.data_ptr() for buf in self._disagg_p2_rdma_buffers]
        data_lens = [buf.numel() for buf in self._disagg_p2_rdma_buffers]
        data_args = DataArgs(
            sender_engine_rank=p2_transformer_rank,
            receiver_engine_rank=p2_decoder_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self._disagg_p2_data_mgr.init(data_args, room)
        self._disagg_p2_receiver = DataReceiver(self._disagg_p2_data_mgr, trans_addr, room)
        self._disagg_p2_receiver.init()
        self._disagg_active_decoder_room = room
        logger.info("[Disagg] Decoder dispatch prepared for room=%s", room)

    def disagg_decoder_teardown_session(self) -> None:
        room = self._disagg_active_decoder_room
        if room is None:
            self._disagg_p2_rdma_buffers = []
            self._disagg_p2_receiver = None
            return
        try:
            if self._disagg_p2_data_mgr is not None and room in self._disagg_p2_data_mgr.data_args:
                self._disagg_p2_data_mgr.remove(room)
        except Exception:
            logger.exception("[Disagg] decoder phase2 remove failed room=%s", room)
        self._disagg_p2_rdma_buffers = []
        self._disagg_p2_receiver = None
        self._disagg_active_decoder_room = None

    def _disagg_encoder_teardown_room(self, room: int) -> None:
        try:
            if self._disagg_data_mgr is not None and room in self._disagg_data_mgr.data_args:
                self._disagg_data_mgr.remove(room)
        except Exception:
            logger.exception("[Disagg] encoder teardown failed room=%s", room)
        self._disagg_sender = None
        self._disagg_rdma_buffers = []
        if self._disagg_active_encoder_room == room:
            self._disagg_active_encoder_room = None

    def _disagg_encoder_setup_room(self, room: int) -> None:
        if self._disagg_active_encoder_room == room and self._disagg_sender is not None:
            return
        if self._disagg_active_encoder_room is not None and self._disagg_active_encoder_room != room:
            self._disagg_encoder_teardown_room(self._disagg_active_encoder_room)

        recv_rank = int(
            self.config.get(
                "disagg_phase1_receiver_engine_rank",
                self.config.get("disagg_config", {}).get("receiver_engine_rank", self._disagg_receiver_rank),
            )
        )

        buffer_sizes = _estimate_encoder_buffer_sizes(self.config)
        self._disagg_alloc_buffers(buffer_sizes)
        data_ptrs = [buf.data_ptr() for buf in self._disagg_rdma_buffers]
        data_lens = [buf.numel() for buf in self._disagg_rdma_buffers]
        data_args = DataArgs(
            sender_engine_rank=self._disagg_sender_rank,
            receiver_engine_rank=recv_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self._disagg_data_mgr.init(data_args, room)
        self._disagg_sender = DataSender(self._disagg_data_mgr, self._disagg_bootstrap_addr, room)
        self._disagg_active_encoder_room = room

    def _disagg_produce_phase1_for_encoder(self) -> None:
        if self._disagg_phase1_queue is None:
            raise RuntimeError("phase1 meta queue not connected")
        req = self._disagg_build_request_config_snapshot()
        room = int(req.get("data_bootstrap_room", self._disagg_bootstrap_room))
        phase1_meta = {
            "request_config": req,
            "encoder_node_address": self._disagg_data_mgr.get_localhost(),
            "encoder_session_id": self._disagg_data_mgr.get_session_id(),
        }
        self._disagg_phase1_queue.produce(phase1_meta)
        logger.info("[Disagg] Encoder published phase1 dispatch for room=%s", room)

    # ------------------------------------------------------------------ #
    #  Encoder role: serialize and send
    # ------------------------------------------------------------------ #

    def send_encoder_outputs(self, inputs: dict, latent_shape: list):
        """Serialize encoder outputs into RDMA buffers and send via Mooncake."""
        config = self.config
        disagg_cfg = config.get("disagg_config", {})

        if getattr(self, "_disagg_decentralized", False):
            room = int(config.get("data_bootstrap_room", disagg_cfg.get("bootstrap_room", 0)))
            self._ensure_disagg_phase1_queue_producer(disagg_cfg)
            if self._disagg_phase1_queue is None:
                raise RuntimeError("[Disagg] decentralized encoder could not connect phase1 queue")
            self._disagg_encoder_setup_room(room)
            self._disagg_produce_phase1_for_encoder()
        text_encoder_output = inputs["text_encoder_output"]
        image_encoder_output = inputs.get("image_encoder_output")

        # Support both Wan2.1 and QwenImage keys
        context = text_encoder_output.get("context", text_encoder_output.get("prompt_embeds"))
        context_null = text_encoder_output.get("context_null", text_encoder_output.get("negative_prompt_embeds"))

        # In QwenImage I2I, image_info is part of text_encoder_output, we serialize it to meta
        image_info = text_encoder_output.get("image_info", None)
        if image_info:
            clean_info = {}
            for k, v in image_info.items():
                if k == "vae_image_list":
                    continue
                if isinstance(v, torch.Tensor):
                    clean_info[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    clean_info[k] = [t.tolist() for t in v]
                else:
                    clean_info[k] = v
            image_info = clean_info

        clip_encoder_out = None
        vae_encoder_out = None
        if image_encoder_output is not None:
            if isinstance(image_encoder_output, dict):
                clip_encoder_out = image_encoder_output.get("clip_encoder_out")
                vae_encoder_out = image_encoder_output.get("vae_encoder_out")
            elif isinstance(image_encoder_output, list) and len(image_encoder_output) > 0:
                # For QwenImage I2I, it's a list of dicts/tensors
                item = image_encoder_output[0]
                vae_encoder_out = item.get("image_latents", item) if isinstance(item, dict) else item

        text_len = int(config.get("text_len", 512))
        text_dim = int(config.get("text_encoder_dim", 4096))
        clip_dim = int(config.get("clip_embed_dim", 1024))
        z_dim = int(config.get("vae_z_dim", 16))

        vae_stride = config.get("vae_stride", (4, 8, 8))
        stride_t, stride_h, stride_w = int(vae_stride[0]), int(vae_stride[1]), int(vae_stride[2])
        target_video_length = int(config.get("target_video_length", 81))
        target_height = int(config.get("target_height", 480))
        target_width = int(config.get("target_width", 832))

        t_prime = 1 + (target_video_length - 1) // stride_t
        h_prime = int(math.ceil(target_height / stride_h))
        w_prime = int(math.ceil(target_width / stride_w))

        task = config.get("task")
        enable_cfg = bool(config.get("enable_cfg", False))
        use_image_encoder = bool(config.get("use_image_encoder", True))

        buffer_index = 0

        # context
        context_flat = context.reshape(-1)
        context_buf = _buffer_view(self._disagg_rdma_buffers[buffer_index], GET_DTYPE(), (self._disagg_rdma_buffers[buffer_index].numel() // torch.tensor([], dtype=GET_DTYPE()).element_size(),))
        context_buf[: context_flat.numel()].copy_(context_flat)
        buffer_index += 1

        # context_null
        if enable_cfg and context_null is not None:
            context_null_flat = context_null.reshape(-1)
            context_null_buf = _buffer_view(
                self._disagg_rdma_buffers[buffer_index], GET_DTYPE(), (self._disagg_rdma_buffers[buffer_index].numel() // torch.tensor([], dtype=GET_DTYPE()).element_size(),)
            )
            context_null_buf[: context_null_flat.numel()].copy_(context_null_flat)
            buffer_index += 1
        elif enable_cfg:  # if enable_cfg is True but context_null is None (e.g. QwenImage empty neg_prompt)
            buffer_index += 1

        # clip + vae (for i2v-like tasks)
        if task in ("i2v", "flf2v", "animate", "s2v", "rs2v", "i2i"):
            if use_image_encoder:
                clip_buf = _buffer_view(self._disagg_rdma_buffers[buffer_index], GET_DTYPE(), (clip_dim,))
                if clip_encoder_out is not None:
                    clip_encoder_out_flat = clip_encoder_out.reshape(-1)
                    clip_buf[: clip_encoder_out_flat.numel()].copy_(clip_encoder_out_flat)
                else:
                    clip_buf.zero_()
                buffer_index += 1

            vae_buf = _buffer_view(
                self._disagg_rdma_buffers[buffer_index],
                GET_DTYPE(),
                (z_dim + 4, t_prime, h_prime, w_prime),
            )
            vae_buf.zero_()
            if vae_encoder_out is not None:
                src_flat = vae_encoder_out.reshape(-1)
                vae_buf.view(-1)[: src_flat.numel()].copy_(src_flat)
            buffer_index += 1

        # latent_shape
        latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
        latent_buf = _buffer_view(self._disagg_rdma_buffers[buffer_index], torch.int64, (10,))
        latent_buf.zero_()
        if latent_tensor.numel() > 0:
            latent_buf[: latent_tensor.numel()].copy_(latent_tensor)
        buffer_index += 1

        # meta includes shapes, hashes, and image_info (for QwenImage)
        meta = {
            "version": 1,
            "task": task,
            "context_shape": list(context.shape),
            "context_hash": _sha256_tensor(context),
            "context_null_shape": list(context_null.shape) if context_null is not None else None,
            "context_null_hash": _sha256_tensor(context_null),
            "clip_shape": list(clip_encoder_out.shape) if clip_encoder_out is not None else None,
            "clip_hash": _sha256_tensor(clip_encoder_out),
            "vae_shape": list(vae_encoder_out.shape) if vae_encoder_out is not None else None,
            "vae_hash": _sha256_tensor(vae_encoder_out),
            "latent_shape": list(latent_shape),
            "latent_hash": _sha256_tensor(latent_tensor),
            "image_info": image_info,
        }
        import numpy as _np

        class _NativeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, _np.integer):
                    return int(obj)
                if isinstance(obj, _np.floating):
                    return float(obj)
                return super().default(obj)

        meta_bytes = json.dumps(meta, cls=_NativeEncoder, ensure_ascii=True).encode("utf-8")
        meta_buf = _buffer_view(self._disagg_rdma_buffers[buffer_index], torch.uint8, (self._disagg_rdma_buffers[buffer_index].numel(),))
        if len(meta_bytes) > meta_buf.numel():
            raise ValueError("metadata buffer too small for hash/shape payload")
        meta_buf.zero_()
        meta_buf[: len(meta_bytes)].copy_(torch.from_numpy(np.frombuffer(meta_bytes, dtype=np.uint8).copy()))

        # Send
        torch.cuda.synchronize()
        buffer_ptrs = [buf.data_ptr() for buf in self._disagg_rdma_buffers]
        self._disagg_sender.send(buffer_ptrs)

        # Wait for transfer completion
        while True:
            status = self._disagg_sender.poll()
            if status == DataPoll.Success:
                logger.info("Disagg: encoder outputs sent successfully.")
                break
            time.sleep(0.01)

        if getattr(self, "_disagg_decentralized", False):
            self._disagg_encoder_teardown_room(
                int(config.get("data_bootstrap_room", disagg_cfg.get("bootstrap_room", 0)))
            )

    # ------------------------------------------------------------------ #
    #  Transformer role: receive and deserialize
    # ------------------------------------------------------------------ #

    def receive_encoder_outputs(self) -> dict:
        """Poll for data from Encoder and reconstruct standard inputs dict."""
        config = self.config

        # Wait for data
        while True:
            status = self._disagg_receiver.poll()
            if status == DataPoll.Success:
                logger.info("Disagg: encoder outputs received successfully.")
                break
            time.sleep(0.01)

        # Immediately snapshot all RDMA destination buffers after poll() returns.
        # Without this, a concurrent Encoder send for the next request can overwrite
        # these shared buffers before we finish reading the current request's data,
        # causing the meta hash (read first) to mismatch the tensor data (read later).
        received_buffers = [buf.detach().clone() for buf in self._disagg_rdma_buffers]

        # Re-register for the next request immediately after snapshotting.
        # The bootstrap_room status is never automatically reset between requests:
        # without this call, the next request's poll() returns Success instantly
        # using stale buffer content from this request.
        # Calling init() resets request_status[room] = WaitingForInput and notifies
        # the Encoder sender that we are ready to receive the next transfer.
        if not getattr(self, "_disagg_decentralized", False):
            self._disagg_receiver.init()

        text_len = int(config.get("text_len", 512))
        text_dim = int(config.get("text_encoder_dim", 4096))
        clip_dim = int(config.get("clip_embed_dim", 1024))
        z_dim = int(config.get("vae_z_dim", 16))

        vae_stride = config.get("vae_stride", (4, 8, 8))
        target_video_length = int(config.get("target_video_length", 81))
        target_height = int(config.get("target_height", 480))
        target_width = int(config.get("target_width", 832))

        t_prime = 1 + (target_video_length - 1) // int(vae_stride[0])
        h_prime = int(math.ceil(target_height / int(vae_stride[1])))
        w_prime = int(math.ceil(target_width / int(vae_stride[2])))

        enable_cfg = bool(config.get("enable_cfg", False))
        use_image_encoder = bool(config.get("use_image_encoder", True))

        buffer_index = 0

        # Parse metadata first (last buffer)
        meta_buf = received_buffers[-1]
        meta_raw = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
        meta_str = meta_raw.split(b"\x00", 1)[0].decode("utf-8") if meta_raw else ""
        meta = json.loads(meta_str) if meta_str else {}

        task = meta.get("task", config.get("task", "i2v"))

        # context
        context_shape = tuple(meta.get("context_shape") or (1, text_len, text_dim))
        context_buf_flat = _buffer_view(received_buffers[buffer_index], GET_DTYPE(), (received_buffers[buffer_index].numel() // torch.tensor([], dtype=GET_DTYPE()).element_size(),))
        context = context_buf_flat[: math.prod(context_shape)].view(context_shape).to(AI_DEVICE).clone()
        buffer_index += 1

        # context_null
        context_null = None
        if enable_cfg:
            context_null_shape = meta.get("context_null_shape")
            if context_null_shape is not None:
                context_null_shape = tuple(context_null_shape)
                context_null_buf_flat = _buffer_view(received_buffers[buffer_index], GET_DTYPE(), (received_buffers[buffer_index].numel() // torch.tensor([], dtype=GET_DTYPE()).element_size(),))
                context_null = context_null_buf_flat[: math.prod(context_null_shape)].view(context_null_shape).to(AI_DEVICE).clone()
            buffer_index += 1

        # Restore appropriately depending on model
        text_encoder_output = {}
        if config.get("model_cls", "") in ["qwen_image", "qwen2.5_vl"]:
            text_encoder_output["prompt_embeds"] = context
            if context_null is not None:
                text_encoder_output["negative_prompt_embeds"] = context_null
            if meta.get("image_info"):
                text_encoder_output["image_info"] = meta["image_info"]
        else:
            text_encoder_output["context"] = context
            text_encoder_output["context_null"] = context_null

        # clip + vae
        clip_encoder_out = None
        vae_encoder_out = None
        image_encoder_output = None

        if task in ("i2v", "flf2v", "animate", "s2v", "rs2v", "i2i"):
            if use_image_encoder:
                clip_shape = tuple(meta.get("clip_shape") or (clip_dim,))
                clip_encoder_out = _buffer_view(received_buffers[buffer_index], GET_DTYPE(), clip_shape).to(AI_DEVICE).clone()
                buffer_index += 1

            # vae_encoder_out
            vae_shape = tuple(meta.get("vae_shape") or (z_dim + 4, t_prime, h_prime, w_prime))
            vae_encoder_out_padded = _buffer_view(received_buffers[buffer_index], GET_DTYPE(), vae_shape).to(AI_DEVICE).clone()
            buffer_index += 1

            # latent_shape
            latent_shape_buf = received_buffers[buffer_index]
            buffer_index += 1
            if meta and meta.get("latent_shape") is not None:
                latent_shape = meta.get("latent_shape")
            else:
                latent_shape = _buffer_view(latent_shape_buf, torch.int64, (10,)).tolist()

            # Trim vae to actual latent dimensions if not i2i
            if task == "i2i":
                vae_encoder_out = vae_encoder_out_padded
            else:
                if vae_encoder_out_padded.ndim == 3:
                    valid_c, valid_h, valid_w = latent_shape[2], latent_shape[3], latent_shape[4]
                    vae_encoder_out = vae_encoder_out_padded[:valid_c, :valid_h, :valid_w].clone()
                elif vae_encoder_out_padded.ndim == 4:
                    valid_t, valid_h, valid_w = latent_shape[1], latent_shape[2], latent_shape[3]
                    vae_encoder_out = vae_encoder_out_padded[:, :valid_t, :valid_h, :valid_w].clone()
                else:
                    vae_encoder_out = vae_encoder_out_padded

            if task == "i2i":
                image_encoder_output = [{"image_latents": vae_encoder_out}]
            else:
                image_encoder_output = {"clip_encoder_out": clip_encoder_out, "vae_encoder_out": vae_encoder_out}
        else:
            # T2V — only latent_shape
            latent_shape_buf = received_buffers[buffer_index]
            buffer_index += 1
            if meta and meta.get("latent_shape") is not None:
                latent_shape = meta.get("latent_shape")
            else:
                latent_shape = _buffer_view(latent_shape_buf, torch.int64, (10,)).tolist()

        # Integrity checks
        if meta:
            self._disagg_verify_integrity(meta, context, context_null, clip_encoder_out, vae_encoder_out, latent_shape, enable_cfg, task)

        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
            "latent_shape": latent_shape,
        }

    # ------------------------------------------------------------------ #
    #  Integrity verification
    # ------------------------------------------------------------------ #

    def _disagg_verify_integrity(self, meta, context, context_null, clip_encoder_out, vae_encoder_out, latent_shape, enable_cfg, task):
        """Verify SHA256 hashes of transferred tensors."""
        if meta.get("context_hash") is not None:
            if _sha256_tensor(context) != meta["context_hash"]:
                raise ValueError("Disagg: context hash mismatch")

        if enable_cfg and meta.get("context_null_hash") is not None:
            if _sha256_tensor(context_null) != meta["context_null_hash"]:
                raise ValueError("Disagg: context_null hash mismatch")

        if task in ("i2v", "flf2v", "animate", "s2v", "rs2v", "i2i"):
            if meta.get("clip_hash") is not None and clip_encoder_out is not None:
                if _sha256_tensor(clip_encoder_out) != meta["clip_hash"]:
                    raise ValueError("Disagg: clip hash mismatch")
            if meta.get("vae_hash") is not None and vae_encoder_out is not None:
                if _sha256_tensor(vae_encoder_out) != meta["vae_hash"]:
                    logger.error(f"[Disagg] VAE actual shape: {vae_encoder_out.shape if vae_encoder_out is not None else None}")
                    logger.error(f"[Disagg] VAE expected shape: {meta.get('vae_shape')}")
                    logger.error(f"[Disagg] VAE expected hash: {meta.get('vae_hash')} vs actual res: {_sha256_tensor(vae_encoder_out)}")
                    raise ValueError("Disagg: vae hash mismatch")

        if meta.get("latent_hash") is not None:
            latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
            if _sha256_tensor(latent_tensor) != meta["latent_hash"]:
                raise ValueError("Disagg: latent_shape hash mismatch")

        logger.info("Disagg: all integrity checks passed.")

    # ------------------------------------------------------------------ #
    #  Transformer role: send latents to Decoder (Phase 2)
    # ------------------------------------------------------------------ #

    def send_transformer_outputs(self, latents: torch.Tensor):
        """Serialize DiT latents into Phase 2 RDMA buffer and send via Mooncake."""
        if self._disagg_p2_sender is None:
            raise RuntimeError("[Disagg] Phase2 sender is not initialized. Check decoder_engine_rank in disagg_config.")
        if len(self._disagg_p2_rdma_buffers) < 2:
            raise RuntimeError("[Disagg] Phase2 RDMA buffers require [latents, meta] entries.")

        latents_to_send = latents.detach().to(GET_DTYPE()).contiguous()
        latents_nbytes = latents_to_send.numel() * latents_to_send.element_size()
        latents_buf = self._disagg_p2_rdma_buffers[0]
        if latents_nbytes > latents_buf.numel():
            raise ValueError(f"[Disagg] Latents buffer too small: need={latents_nbytes}, capacity={latents_buf.numel()}")

        latents_buf.zero_()
        latents_view = _buffer_view(latents_buf, latents_to_send.dtype, tuple(latents_to_send.shape))
        latents_view.copy_(latents_to_send)

        import numpy as _np

        # Include pixel-space dimensions so the Decoder can reconstruct auto_height/width
        # correctly even when latents are in packed (sequence) format (e.g. QwenImage).
        _input_info = getattr(self, "input_info", None)
        latents_meta = {
            "version": 1,
            "latents_shape": list(latents_to_send.shape),
            "latents_dtype": str(latents_to_send.dtype),
            "latents_hash": _sha256_tensor(latents_to_send),
            "auto_height": getattr(_input_info, "auto_height", None),
            "auto_width": getattr(_input_info, "auto_width", None),
        }
        meta_bytes = json.dumps(latents_meta, ensure_ascii=True).encode("utf-8")
        meta_buf = self._disagg_p2_rdma_buffers[1]
        meta_view = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),))
        if len(meta_bytes) > meta_view.numel():
            raise ValueError("[Disagg] Phase2 metadata buffer too small for latents meta payload")
        meta_view.zero_()
        meta_view[: len(meta_bytes)].copy_(torch.from_numpy(_np.frombuffer(meta_bytes, dtype=_np.uint8).copy()))

        torch.cuda.synchronize()
        buffer_ptrs = [buf.data_ptr() for buf in self._disagg_p2_rdma_buffers]
        self._disagg_p2_sender.send(buffer_ptrs)
        while True:
            status = self._disagg_p2_sender.poll()
            if status == DataPoll.Success:
                logger.info("[Disagg] Transformer latents sent to Decoder successfully.")
                break
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    #  Decoder role: receive latents from Transformer (Phase 2)
    # ------------------------------------------------------------------ #

    def receive_transformer_outputs(self) -> torch.Tensor:
        """Poll Phase 2 and reconstruct latents tensor from RDMA buffer."""
        if self._disagg_p2_receiver is None:
            raise RuntimeError("[Disagg] Phase2 receiver is not initialized.")
        if len(self._disagg_p2_rdma_buffers) < 2:
            raise RuntimeError("[Disagg] Phase2 RDMA buffers require [latents, meta] entries.")

        while True:
            status = self._disagg_p2_receiver.poll()
            if status == DataPoll.Success:
                logger.info("[Disagg] Decoder received latents from Transformer successfully.")
                break
            time.sleep(0.01)

        # Immediately snapshot all Phase2 RDMA destination buffers after poll() returns.
        # Without this, a concurrent Transformer send for the next request can overwrite
        # these shared buffers before we finish reading, causing hash mismatches.
        received_p2_buffers = [buf.detach().clone() for buf in self._disagg_p2_rdma_buffers]

        # Re-register for the next request: resets request_status[room] = WaitingForInput
        # and notifies the Transformer sender to accept the next Phase 2 transfer.
        # Without this, the next request's poll() returns Success instantly with stale
        # data, and the Transformer hangs forever in send_transformer_outputs().
        if not getattr(self, "_disagg_decentralized", False):
            self._disagg_p2_receiver.init()

        meta_buf = received_p2_buffers[1]
        meta_raw = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
        meta_str = meta_raw.split(b"\x00", 1)[0].decode("utf-8") if meta_raw else ""
        if not meta_str:
            raise ValueError("[Disagg] Missing latents metadata from transformer (Phase 2)")
        meta = json.loads(meta_str)

        latents_shape_val = meta.get("latents_shape")
        if not isinstance(latents_shape_val, list) or len(latents_shape_val) < 1:
            raise ValueError(f"[Disagg] Invalid latents_shape in Phase 2 metadata: {latents_shape_val}")
        latent_shape = tuple(int(v) for v in latents_shape_val)

        dtype_map = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        latents_dtype = dtype_map.get(meta.get("latents_dtype"), GET_DTYPE())

        latents = _buffer_view(received_p2_buffers[0], latents_dtype, latent_shape)
        if meta.get("latents_hash") is not None and _sha256_tensor(latents) != meta.get("latents_hash"):
            raise ValueError("[Disagg] Latents hash mismatch between transformer and decoder")
        latents = latents.to(AI_DEVICE).contiguous()
        logger.info(f"[Disagg] Phase2 latents restored: shape={latent_shape}, dtype={latents_dtype}")
        # Store the Phase 2 metadata so the caller (e.g. QwenImageRunner decode mode) can
        # access pixel-space dimensions (auto_height/auto_width) that are not recoverable
        # from the packed latent tensor shape alone.
        self._p2_receive_meta = meta
        return latents

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def release_disagg(self):
        """Release RDMA buffers (Phase 1 and Phase 2) and deregister from transfer engine."""
        if getattr(self, "_disagg_decentralized", False):
            try:
                self.disagg_transformer_teardown_session()
            except Exception:
                pass
            try:
                self.disagg_decoder_teardown_session()
            except Exception:
                pass
            if self._disagg_active_encoder_room is not None:
                try:
                    self._disagg_encoder_teardown_room(self._disagg_active_encoder_room)
                except Exception:
                    pass
            self._disagg_phase1_queue = None
            self._disagg_phase2_queue = None
            self._disagg_phase1_client = None
            self._disagg_phase2_client = None
            if self._disagg_data_mgr is not None:
                try:
                    self._disagg_data_mgr.release()
                except Exception:
                    pass
                self._disagg_data_mgr = None
            if self._disagg_p2_data_mgr is not None:
                try:
                    self._disagg_p2_data_mgr.release()
                except Exception:
                    pass
                self._disagg_p2_data_mgr = None
            torch.cuda.empty_cache()
            return

        if self._disagg_rdma_buffers:
            for buf in self._disagg_rdma_buffers:
                if self._disagg_data_mgr is not None:
                    try:
                        self._disagg_data_mgr.engine.deregister(buf.data_ptr())
                    except Exception:
                        pass
            self._disagg_rdma_buffers = []
        if self._disagg_p2_rdma_buffers:
            for buf in self._disagg_p2_rdma_buffers:
                if self._disagg_p2_data_mgr is not None:
                    try:
                        self._disagg_p2_data_mgr.engine.deregister(buf.data_ptr())
                    except Exception:
                        pass
            self._disagg_p2_rdma_buffers = []
        torch.cuda.empty_cache()