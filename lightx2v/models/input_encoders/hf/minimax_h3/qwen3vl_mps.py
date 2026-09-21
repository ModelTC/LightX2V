"""MPS text-only Qwen3-VL inference with two disk-prefetched weight buffers."""

import gc

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.modules.weight_module import WeightModuleList
from lightx2v.common.offload.mps_manager import MpsSharedWeightAsyncStreamManager, host_view
from lightx2v.common.ops.utils import resolve_block_name
from lightx2v.models.networks.minimax_h3.checkpoint import MiniMaxH3ShardCheckpoint
from lightx2v.utils.envs import GET_DTYPE

from . import qwen3vl


class _Qwen3VLMpsTextBackboneWeights(qwen3vl._Qwen3VLTextBackboneWeights):
    def __init__(self, config, text_config, num_layers=qwen3vl.MINIMAX_H3_TEXT_ENCODER_LAYER, attn_type="torch_sdpa"):
        super().__init__(config, text_config, num_layers=num_layers, attn_type=attn_type)
        self.streaming_checkpoint = None

    @property
    def device(self):
        return torch.device("mps")

    @property
    def dtype(self):
        return GET_DTYPE()

    def init_disk_streaming(self, text_encoder_path=None, weight_map=None):
        if self.streaming_checkpoint is None:
            self.streaming_checkpoint = MiniMaxH3ShardCheckpoint(text_encoder_path, weight_map)
        if self.offload_cuda_buffers is not None:
            return

        buffers = WeightModuleList(
            qwen3vl._Qwen3VLDecoderLayerWeights(
                0,
                self.config,
                self.text_config,
                self.attn_type,
                create_cuda_buffer=True,
            )
            for _ in range(2)
        )
        for layer in buffers:
            tensors = {}
            for module in layer.weight_modules():
                for name, _, _ in module.base_attrs:
                    dtype, shape, _, _ = self.streaming_checkpoint.tensor_metadata(name)
                    if dtype != GET_DTYPE():
                        raise ValueError(f"Text encoder prefetch requires matching file/inference dtypes: {name}")
                    tensors[name] = torch.empty(shape, dtype=dtype, device="mps")
            layer.load(tensors)
            layer.shared_host_tensors = {}
            for module in layer.weight_modules():
                for name, attr, transpose in module.base_attrs:
                    buffer = getattr(module, f"{attr}_cuda_buffer")
                    setattr(module, attr, buffer)
                    # Keep the file layout for I/O and the existing transposed view for GEMM.
                    layer.shared_host_tensors[name] = host_view(buffer.t() if transpose else buffer)
        self.offload_cuda_buffers = buffers
        self.offload_manager = MpsSharedWeightAsyncStreamManager()
        self.offload_manager.init_cuda_buffer(blocks_cuda_buffer=buffers)

    def load_block_into(self, layer, layer_index):
        """Read the next layer into idle shared storage; called by the prefetch worker."""
        destinations = {resolve_block_name(name, layer_index): tensor for name, tensor in layer.shared_host_tensors.items()}
        self.streaming_checkpoint.load_tensors_into(destinations)

    def _forward_streaming_embedding(self, input_ids):
        embedding_name = self.embed_tokens.weight_name
        tensors = self.streaming_checkpoint.load_tensors((embedding_name,))
        # Transfer only the selected token vectors, not the full vocabulary.
        hidden_states = F.embedding(input_ids.cpu(), tensors[embedding_name])
        return hidden_states.to("mps")

    def forward(self, input_ids, position_ids=None):
        hidden_states = self._forward_streaming_embedding(input_ids)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)
        self.init_disk_streaming()
        manager = self.offload_manager
        try:
            # Restart at layer zero for every prompt, including after a failed request.
            manager.init_cuda_buffer(blocks_cuda_buffer=self.offload_cuda_buffers)
            manager.init_first_buffer(self)
            for layer_index in range(self.num_layers):
                has_next = layer_index + 1 < self.num_layers
                layer = manager.cuda_buffers[0]
                if has_next:
                    manager.prefetch_weights(layer_index + 1, self)
                hidden_states = layer.forward(hidden_states, position_embeddings)
                if has_next:
                    manager.swap_blocks()
                else:
                    torch.mps.synchronize()
        except Exception:
            manager.close()
            raise
        return hidden_states

    def release_disk_streaming_buffer(self):
        if self.offload_cuda_buffers is None:
            return
        self.offload_manager.close()
        self.offload_manager = None
        for layer in self.offload_cuda_buffers:
            layer.shared_host_tensors.clear()
            for module in layer.weight_modules():
                for _, attr, _ in module.base_attrs:
                    setattr(module, attr, None)
                    setattr(module, f"{attr}_cuda_buffer", None)
        self.offload_cuda_buffers = None
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        logger.info("MiniMax-H3 Qwen3-VL released its disk-streaming layer buffers")

    def to_cpu(self, non_blocking=False):
        self.release_disk_streaming_buffer()
        return self


class MiniMaxH3MpsQwen3VLTextEncoder(qwen3vl.MiniMaxH3Qwen3VLTextEncoder):
    def __init__(self, config):
        self.config = config
        self.local_files_only = config.get("local_files_only", True)
        self.release_block_offload_buffers = bool(config.get("text_encoder_release_block_offload_buffers", False))
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None
        self.vision_encoder = None
        if config.get("text_encoder_load_on_init", True):
            self.load()

    def load(self):
        self.load_tokenizer()
        self.load_text_encoder()
        return self

    def load_text_encoder(self):
        if self.text_encoder is not None:
            return self.text_encoder
        text_encoder_path = self._component_path("text_encoder_path", "text_encoder")
        text_config = self._read_text_config(text_encoder_path)
        self._validate_text_config(text_config)
        text_encoder = _Qwen3VLMpsTextBackboneWeights(
            self.config,
            text_config,
            num_layers=qwen3vl.MINIMAX_H3_TEXT_ENCODER_LAYER,
            attn_type=self._resolve_attn_type(self.config),
        )
        weight_map, _ = self._preflight_native_checkpoint(text_encoder, text_encoder_path, text_config)
        text_encoder.init_disk_streaming(text_encoder_path, weight_map)
        self.text_encoder = text_encoder
        return self.text_encoder

    def unload_text_encoder(self):
        if self.text_encoder is not None:
            self.text_encoder.release_disk_streaming_buffer()
        super().unload_text_encoder()

    @torch.inference_mode()
    def infer(self, prompt, image_list=None, references=None):
        if image_list or references is not None:
            raise NotImplementedError("MiniMax-H3 Qwen3-VL MPS disk streaming supports text-only t2av prompts.")
        self._ensure_loaded()
        try:
            input_ids = self._prepare_t2av_input_ids(prompt, "cpu")
            prompt_embeds = self.text_encoder.forward(input_ids)
            expected_shape = (input_ids.shape[0], qwen3vl.MINIMAX_H3_TEXT_HIDDEN_SIZE)
            if tuple(prompt_embeds.shape) != expected_shape:
                raise RuntimeError(f"MiniMax-H3 expected conditioner hidden shape {expected_shape}, but native Qwen3-VL returned {tuple(prompt_embeds.shape)}")
            prompt_embeds = prompt_embeds.to(dtype=GET_DTYPE()).contiguous()
            return {
                "prompt_embeds": prompt_embeds,
                "text_token_tags": torch.full((input_ids.shape[0],), qwen3vl.MINIMAX_H3_TEXT_TAG, dtype=torch.long, device=prompt_embeds.device),
            }
        finally:
            if self.release_block_offload_buffers and self.text_encoder is not None:
                self.text_encoder.release_disk_streaming_buffer()
