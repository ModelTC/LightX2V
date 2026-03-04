import sys
sys.path.append("C:/Users/ASUS/Desktop/LightX2V")

from lightx2v.models.runners.mova.mova_runner import MOVARunner

class DummyInputInfo:
    def __init__(self, prompt, seed, save_result_path, target_shape, video_latent_shape, audio_latent_shape):
        self.prompt = prompt
        self.seed = seed
        self.save_result_path = save_result_path
        self.target_shape = target_shape
        self.video_latent_shape = video_latent_shape
        self.audio_latent_shape = audio_latent_shape
        self.negative_prompt = None

config = {
    "task": "t2va",
    "model_cls": "mova",
    "model_path": "dummy",
    "infer_steps": 5,
    "cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "target_video_length": 17,
    "target_audio_length": 48000,
    "video_shift": 5.0,
    "audio_shift": 5.0,
    "text_dim": 4096,
}

input_info = DummyInputInfo(
    prompt="test",
    seed=42,
    save_result_path="./dummy.mp4",
    target_shape=[352, 640],
    video_latent_shape=[16, 5, 30, 40],
    audio_latent_shape=[8, 150],
)

runner = MOVARunner(config)
runner.input_info = input_info
runner.inputs = {"text_encoder_output": runner.run_text_encoder(input_info)}
runner.load_model()
runner.run_main()
print("🎉 Full flow test passed!")