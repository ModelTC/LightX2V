from lightx2v.models.runners.flux2.flux2_runner import Flux2BaseRunner


def test_flux2_tp_load_model_keeps_text_encoder_and_vae_off_non_dit_path(monkeypatch):
    runner = Flux2BaseRunner.__new__(Flux2BaseRunner)
    runner.config = {"tensor_parallel": True}

    monkeypatch.setattr("lightx2v.models.runners.flux2.flux2_runner.dist.is_initialized", lambda: True)

    calls = []
    runner.load_text_encoder = lambda: calls.append("text_encoder")
    runner.load_vae = lambda: calls.append("vae")

    def load_transformer():
        calls.append("transformer")
        return "transformer_model"

    runner.load_transformer = load_transformer

    runner.load_model()

    assert calls == ["transformer"]
    assert runner.text_encoders is None
    assert runner.vae is None
    assert runner.model == "transformer_model"
