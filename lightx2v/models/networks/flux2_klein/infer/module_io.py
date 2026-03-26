from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Flux2KleinPreInferModuleOutput:
    """Output from Flux2Klein pre-inference stage.
    
    Precisely mirrors Flux2KleinPipeline.__call__ preprocessing:
    - Embedded image latents via x_embedder
    - Embedded text embeddings via context_embedder
    - Timestep embeddings via timestep_embedder MLP
    - Position IDs for text and image (txt_ids, img_ids)
    - Rotary embeddings for attention
    - Optional I2I conditioning
    """
    # Core embeddings (all embedded/processed via weights)
    hidden_states: torch.Tensor  # x_embedder output [L, D]
    encoder_hidden_states: torch.Tensor  # context_embedder output [L, D]
    timestep: torch.Tensor  # timestep_embedder MLP output [D]
    
    # Position coordinates (from pipeline's prepare_latents/encode_prompt)
    txt_ids: Optional[torch.Tensor] = None  # [B, text_seq_len, 4]
    img_ids: Optional[torch.Tensor] = None  # [B, image_seq_len, 4]
    
    # Rotary position embeddings (from scheduler)
    image_rotary_emb: Optional[tuple] = None  # (cos, sin) tuple
    
    # Optional I2I conditioning
    input_image_latents: Optional[torch.Tensor] = None
    output_seq_len: Optional[int] = None

