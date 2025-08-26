import torch
from transformers import AutoFeatureExtractor, AutoModel

from lightx2v.utils.envs import *


class SekoAudioEncoderModel:
    def __init__(self, model_path, audio_sr, cpu_offload):
        self.model_path = model_path
        self.audio_sr = audio_sr
        self.cpu_offload = cpu_offload
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.load()

    def load(self):
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.audio_feature_encoder = AutoModel.from_pretrained(self.model_path)
        self.audio_feature_encoder.to(self.device)
        self.audio_feature_encoder.eval()
        self.audio_feature_encoder.to(GET_DTYPE())

    def to_cpu(self):
        self.audio_feature_encoder = self.audio_feature_encoder.to("cpu")

    def to_cuda(self):
        self.audio_feature_encoder = self.audio_feature_encoder.to("cuda")

    @torch.no_grad()
    def infer(self, audio_segment):
        audio_feat = self.audio_feature_extractor(audio_segment, sampling_rate=self.audio_sr, return_tensors="pt").input_values.cuda().to(dtype=GET_DTYPE())
        if self.cpu_offload:
            self.audio_feature_encoder = self.audio_feature_encoder.to("cuda")
        audio_feat = self.audio_feature_encoder(audio_feat, return_dict=True).last_hidden_state
        if self.cpu_offload:
            self.audio_feature_encoder = self.audio_feature_encoder.to("cpu")
        return audio_feat
