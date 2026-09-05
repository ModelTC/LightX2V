import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
CONFIG_PATH = REPO_ROOT / "configs/platforms/mps/minimax_h3_t2av.json"
LAUNCHER_PATH = REPO_ROOT / "scripts/platforms/mps/run_minimax_h3_t2av.sh"


def _load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def runner_module(monkeypatch):
    helper_path = REPO_ROOT / "tests/models/minimax_h3/test_runner_mps_low_memory.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_runner_low_memory_test_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper._load_runner_module(monkeypatch)


def test_mps_minimax_h3_config_enables_low_memory_streaming():
    config = _load_config()

    assert config["cpu_offload"] is True
    assert config["offload_granularity"] == "block"
    assert config["dit_disk_streaming"] is True
    assert config["text_encoder_cpu_offload"] is True
    assert config["text_encoder_offload_granularity"] == "block"
    assert config["text_encoder_disk_streaming"] is True
    assert config["text_encoder_host_pinned"] is False
    assert config["text_encoder_release_block_offload_buffers"] is True
    assert config["vae_cpu_offload"] is True
    assert config["lazy_load"] is False
    assert config["unload_modules"] is False
    assert config["warmup"] is False
    assert config["attn_type"] == "torch_sdpa"
    assert config["mps_sdpa_query_chunk_size"] == 512
    assert config["rms_type"] == "torch_native"
    assert config["rope_type"] == "torch_real_rope"
    assert config["vae_attn_type"] == "torch_sdpa"
    assert config["dit_quantized"] is False
    assert config["dit_quant_scheme"] == "Default"
    assert config["text_encoder_quantized"] is False
    assert config["video_vae_quantized"] is False
    assert config["tensor_parallel"] is False
    assert config["use_compile"] is False
    assert config["vae_use_compile"] is False
    assert "dit_original_ckpt" not in config


def test_real_mps_config_triggers_runner_low_memory_load_model(runner_module):
    config = _load_config()
    config.update({"task": "t2av", "model_path": "/tmp/minimax-h3"})
    runner = object.__new__(runner_module.MiniMaxH3Runner)
    runner.config = config
    calls = []
    runner.load_transformer = lambda: calls.append("transformer") or object()
    runner.load_text_encoder = lambda: calls.append("text_encoder") or [object()]
    runner.load_vae = lambda: calls.append("vae") or (object(), object())

    assert runner._is_mps_low_memory_streaming() is True
    runner.load_model()

    assert calls == ["transformer", "text_encoder"]
    assert runner.video_vae is None
    assert runner.audio_vae is None


def test_mps_launcher_has_safe_static_defaults():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "export PLATFORM=mps" in text
    assert "export DTYPE=BF16" in text
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in text
    assert "CUDA" not in text
    assert "PYTORCH_CUDA_ALLOC_CONF" not in text


def test_mps_launcher_fails_fast_without_model_path():
    env = os.environ.copy()
    env.pop("MODEL_PATH", None)
    result = subprocess.run(
        [str(LAUNCHER_PATH)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "MODEL_PATH must point to the MiniMax-H3 model directory." in result.stdout
