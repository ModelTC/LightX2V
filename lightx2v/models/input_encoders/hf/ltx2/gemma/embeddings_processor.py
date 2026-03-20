import torch
from torch import nn

from lightx2v.models.input_encoders.hf.ltx2.gemma.embeddings_connector import Embeddings1DConnector


def _to_binary_mask(encoded: torch.Tensor, encoded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert connector output mask to binary mask and apply to encoded tensor."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * binary_mask
    return encoded, binary_mask


class EmbeddingsProcessor(nn.Module):
    """Wraps video connector + optional audio connector.
    Returns (video_encoded, audio_encoded | None, binary_mask).
    """

    def __init__(self, video_connector: Embeddings1DConnector, audio_connector: Embeddings1DConnector | None = None):
        super().__init__()
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.audio_connector is not None and audio_features is None:
            raise ValueError("Audio connector is configured but no audio features were provided.")
        if self.audio_connector is None and audio_features is not None:
            raise ValueError("Audio features were provided but no audio connector is configured.")

        video_encoded, video_mask = self.video_connector(video_features, additive_attention_mask)
        video_encoded, binary_mask = _to_binary_mask(video_encoded, video_mask)

        audio_encoded = None
        if self.audio_connector is not None:
            audio_encoded, _ = self.audio_connector(audio_features, additive_attention_mask)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)
