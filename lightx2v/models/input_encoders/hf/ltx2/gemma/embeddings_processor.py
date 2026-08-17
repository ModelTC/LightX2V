import torch
from torch import nn

from lightx2v.models.input_encoders.hf.ltx2.gemma.embeddings_connector import Embeddings1DConnector


def _to_binary_mask(encoded: torch.Tensor, encoded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert connector output mask to binary mask and apply to encoded tensor."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * binary_mask
    return encoded, binary_mask


def _compute_right_pad_order(additive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stably move valid tokens before padding for the embeddings connector."""
    binary = (additive_mask[:, 0, 0, :] >= 0).to(torch.int32)
    sort_idx = torch.argsort(binary, dim=-1, descending=True, stable=True)
    new_binary = torch.gather(binary, 1, sort_idx)
    new_additive = (new_binary.to(additive_mask.dtype) - 1) * torch.finfo(additive_mask.dtype).max
    return sort_idx, new_additive[:, None, None, :]


def _apply_right_pad_order(features: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(features, 1, sort_idx.unsqueeze(-1).expand_as(features))


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

        # Gemma tokenization is left-padded, while the connector replaces a
        # right-padded suffix with learned registers.  Normalize the layout in
        # the same stable order as current ltx-core and reuse it for audio.
        sort_idx, mask_for_connector = _compute_right_pad_order(additive_attention_mask)
        video_features = _apply_right_pad_order(video_features, sort_idx)
        video_encoded, video_mask = self.video_connector(video_features, mask_for_connector)
        video_encoded, binary_mask = _to_binary_mask(video_encoded, video_mask)

        audio_encoded = None
        if self.audio_connector is not None:
            audio_features = _apply_right_pad_order(audio_features, sort_idx)
            audio_encoded, _ = self.audio_connector(audio_features, mask_for_connector)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)
