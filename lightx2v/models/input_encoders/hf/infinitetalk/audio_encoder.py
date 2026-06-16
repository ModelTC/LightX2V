import numpy as np
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from lightx2v.models.input_encoders.hf.infinitetalk.wav2vec2 import Wav2Vec2Model


class InfiniteTalkAudioEncoder:
    def __init__(self, model_id, device="cpu", fps=25, sample_rate=16000):
        self.model_id = model_id
        self.device = torch.device(device)
        self.fps = fps
        self.sample_rate = sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, local_files_only=True)
        self.model = Wav2Vec2Model.from_pretrained(model_id, local_files_only=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, speech_array):
        speech_array = np.asarray(speech_array, dtype=np.float32)
        audio_duration = len(speech_array) / self.sample_rate
        video_length = int(audio_duration * self.fps)
        audio_feature = np.squeeze(self.feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device).unsqueeze(0)
        embeddings = self.model(audio_feature, seq_len=video_length, output_hidden_states=True)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        return audio_emb.cpu().detach()
