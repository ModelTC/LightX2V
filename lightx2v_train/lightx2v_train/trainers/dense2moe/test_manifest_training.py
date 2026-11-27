import json
from collections import Counter
from itertools import islice

import torch

from lightx2v_train.data.h3_selfgen_manifest import PDMDManifest, PhaseSampler
from lightx2v_train.trainers.dense2moe.data import ManifestAVDataset
from lightx2v_train.trainers.dense2moe.runtime import AVSample
from lightx2v_train.trainers.dense2moe.trainer import Dense2MoETrainer


def test_cursor_uses_new_pool_at_update_501_and_resumes(tmp_path):
    rows = [{"sample_id": i, "num_frames": 124, "task": ("t2av", "i2av")[i % 2],
             "teacher_video": None if i < 20931 else "v", "teacher_video_latent": "v.pt",
             "teacher_audio_latent": "a.pt"} for i in range(48373)]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(rows))
    dataset = PDMDManifest(path)
    counts = Counter()
    for rank in range(32):
        sampler = PhaseSampler(dataset, 500, rank=rank, num_replicas=32)
        sampler.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=5)
        ids = list(islice(sampler, 418))
        assert max(ids[:417]) < dataset.prompt_count <= ids[417]
        counts.update(ids)
        sampler.configure_from_microbatch_offset(start_microbatch=416, gradient_accumulation_iters=1, fake_update_ratio=5)
        assert list(islice(sampler, 2)) == ids[-2:]
    assert max(counts.values()) == 1


def test_i2av_paired_layout_and_inverse_cfg(tmp_path):
    positive = {"prompt_embeds": torch.ones(1, 3, 4), "text_token_tags": torch.tensor([0, 1, 1]),
                "condition_video_latents": torch.randn(2, 96)}
    torch.save({"conditioning": {"positive": positive}}, tmp_path / "condition.pt")
    torch.save({"latents": torch.randn(24, 37, 2, 4), "normalized": True}, tmp_path / "video.pt")
    torch.save(torch.randn(2, 32, 207), tmp_path / "audio.pt")
    row = {"sample_id": "sample", "task": "i2av", "num_frames": 124, "height": 32, "width": 64,
           "teacher_video": "video.mp4", "teacher_audio_latent": str(tmp_path / "audio.pt"),
           "condition": str(tmp_path / "condition.pt"), "teacher_video_latent": str(tmp_path / "video.pt")}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([row]))
    dataset = ManifestAVDataset(path, 42, tmp_path)
    condition, latents = dataset.load(0, "cpu")
    sample = AVSample(condition, row, "cpu")
    assert all(latents[k].shape == sample.shapes[k] for k in latents)
    negative = {"prompt_embeds": torch.zeros(1, 2, 4), "text_token_tags": torch.tensor([0, 1])}
    torch.save(negative, tmp_path / dataset.negative_name(row))
    trainer = Dense2MoETrainer.__new__(Dense2MoETrainer)
    trainer.cfg_scale, trainer.device, trainer.dataset = 3.0, "cpu", dataset
    sigmas = {"video": 0.5, "audio": 0.3}
    bias = torch.tensor(1.0, requires_grad=True)
    def model(**kwargs):
        torch.testing.assert_close(kwargs["hidden_states"][:, :2], sample.keyframe)
        offset = bias * kwargs["encoder_hidden_states"].mean()
        return kwargs["hidden_states"] + offset, kwargs["audio_hidden_states"] + offset
    uncond = trainer.unconditional_velocity(model, sample, 0, latents, sigmas)
    pred = trainer.flow_prediction(sample.forward(model, latents, sigmas), uncond)
    for name, value in pred.items():
        assert not uncond[name].requires_grad
        torch.testing.assert_close(value, latents[name] + 1 / 3)
    pred["video"].mean().backward()
    torch.testing.assert_close(bias.grad, torch.tensor(1 / 3))


def test_pdmd_entrypoint_reads_cached_condition_only(tmp_path, monkeypatch):
    from pathlib import Path

    from lightx2v_train.runtime.config import load_config
    from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
    from lightx2v_train.utils.registry import build_data
    from lightx2v_train.utils.utils import is_train_cache_dataset

    manifest = tmp_path / "dataset.json"
    for key in ("H3_MODEL_PATH", "H3_STUDENT_INIT", "H3_PDMD_OUTPUT"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("H3_MANIFEST", str(manifest))
    condition = tmp_path / "condition.pt"
    torch.save({"conditioning": {"positive": {"keyframe_anchors": torch.tensor([0])}}}, condition)
    row = {"sample_id": "one", "task": "i2av", "training_prompt": "one", "num_frames": 124,
           "height": 768, "width": 1344, "condition": str(condition), "image": None,
           "teacher_video": None, "teacher_video_latent": None, "teacher_audio_latent": None}
    manifest.write_text(json.dumps([row]))
    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "configs/train/dmd/minimax_h3_manifest_pdmd4_fsdp32.yaml")
    assert is_train_cache_dataset(cfg)
    validate_official_pdmd_config(cfg)
    cfg["data"]["train"]["num_workers"] = 0
    loader = build_data(cfg, "train")
    loader.sampler.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=5)
    item = next(iter(loader))
    assert item["meta"]["target_height"] == 768
    assert item["conditioning"]["positive"]["keyframe_anchors"].item() == 0


def test_calibration_resumes_scores_and_keeps_two_snapshots(tmp_path, monkeypatch):
    trainer = Dense2MoETrainer.__new__(Dense2MoETrainer)
    trainer.rank, trainer.stage, trainer.output = 0, "calibrate", tmp_path
    trainer.config = {"save_limit": 2, "save_every": 100}
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    for step in (100, 200, 256):
        trainer.checkpoint(None, None, step, {"3": torch.tensor([2., 7.]) * step})
    root = tmp_path / "checkpoints"
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [
        "calibrate-00000200", "calibrate-00000256",
    ]
    restored = {}
    assert trainer.resume(None, None, restored) == 256
    torch.testing.assert_close(restored["3"], torch.tensor([2., 7.]) * 256)


def test_dmad_recipe_only_registers_paired_records(tmp_path, monkeypatch):
    from pathlib import Path
    from lightx2v_train.runtime.config import load_config
    from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
    from lightx2v_train.utils.registry import build_data

    base = dict(task="t2av", num_frames=124, teacher_video="v", teacher_video_latent="v.pt", teacher_audio_latent="a.pt")
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps([base, dict(base, task="i2av"), dict(base, teacher_video=None), dict(base, teacher_audio_latent=None)]))
    for key in ("H3_MODEL_PATH", "H3_PDMD_OUTPUT", "H3_STUDENT_INIT"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("H3_MANIFEST", str(manifest))
    cfg = load_config(Path(__file__).resolve().parents[3] / "configs/train/dmd/minimax_h3_manifest_pdmd4_dmad_fsdp32.yaml")
    validate_official_pdmd_config(cfg)
    loader = build_data(cfg, "train")
    assert len(loader.dataset) == 2 and loader.dataset.prompt_count == 0
    assert sorted(iter(loader.sampler)) == [0, 1]
    dmd = cfg["training"]["dmd"]
    model = cfg["model"]["capabilities"]["distribution_matching"]
    assert (model["video_flow_shift"], model["audio_flow_shift"]) == (dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"]) == (12, 3)
    assert dmd["fake_update_ratio"] == 1 and dmd["gan"]["start_after_updates"] == 0
