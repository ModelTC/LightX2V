"""Paired image-reference GAN positives, native geometry and exact resume."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lightx2v_train.trainers.dmd import dmd2_gan_data as data

GEOMETRIES = [(124, 64, 96), (124, 96, 64)]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def paired_cache(tmp_path):
    rows = []
    for index in range(6):
        frames, height, width = GEOMETRIES[index // 3]
        count = 1 + index % 2
        row = dict(
            task="ref2av",
            source_id=f"real-{index}",
            caption=f"own Ref caption {index}",
            cache_fingerprint=f"condition-{index}",
            latent_fingerprint=f"AV-{index}",
            target_num_frames=frames,
            target_height=height,
            target_width=width,
            fps=24,
            reference_image_count=count,
            reference_video_count=0,
            reference_audio_count=0,
            references=[dict(kind="image", path=f"image-{index}-{ref}.png") for ref in range(count)],
            vae_normalized=True,
            video_latent_shape=[24, 37, height // 16, width // 16],
            audio_latent_shape=[2, 32, 207],
            condition_path=f"condition-{index}.pt",
            video_latent_path=f"video-{index}.pt",
            audio_latent_path=f"audio-{index}.pt",
        )
        references = [
            dict(
                kind="image",
                normalized=True,
                num_latent_frames=1,
                latent_height=4,
                latent_width=6,
                video_latents=torch.full((6, 96), index + ref / 10),
            )
            for ref in range(count)
        ]
        torch.save(
            dict(
                cache_schema_version=1,
                cache_stage="complete",
                cache_fingerprint=row["cache_fingerprint"],
                source_id=row["source_id"],
                conditioning=dict(
                    positive=dict(
                        task="ref2av",
                        prompt_embeds=torch.full((1, 3, 5120), float(index)),
                        text_token_tags=torch.tensor([0, 1, 1]),
                        references=references,
                        target_num_frames=frames,
                        target_height=height,
                        target_width=width,
                    )
                ),
            ),
            tmp_path / row["condition_path"],
        )
        torch.save(dict(normalized=True, latent_fingerprint=row["latent_fingerprint"], latents=torch.full(row["video_latent_shape"], float(index))), tmp_path / row["video_latent_path"])
        torch.save(torch.full(row["audio_latent_shape"], float(index)), tmp_path / row["audio_latent_path"])
        rows.append(row)
    manifest = tmp_path / "metadata.jsonl"
    write_rows(manifest, rows)
    return manifest, rows


def stream(path, **kwargs):
    return data.DMD2GanRealStream(path, task="ref2av", geometry=None, seed=42, **kwargs)


class Capability:
    """Model-side shape/condition contract, including prepared reference objects."""

    def latent_shape(self, sample, generation_shapes, broadcast):
        self.geometry = tuple(sample["meta"][key] for key in ("target_num_frames", "target_height", "target_width"))
        assert generation_shapes == [list(self.geometry)]
        return broadcast(self.geometry)

    def encode_conditions(self, sample, negative_prompt, guidance_scale, broadcast):
        assert (negative_prompt, guidance_scale) == ("", 1.0)
        positive = sample["conditioning"]["positive"]
        self.caption = sample["conditioning"]["prompt"]
        self.references = positive["references"]
        self.tags = positive["text_token_tags"]
        condition = dict(positive)
        condition["references"] = tuple(SimpleNamespace(kind="image", num_audio_latents=0) for _ in self.references)
        condition["condition_video_latents"] = torch.cat([ref["video_latents"] for ref in self.references])
        condition["condition_audio_latents"] = None
        return broadcast(condition), None

    def extract_turbo_latents(self, sample, shape):
        assert shape == self.geometry
        return sample["inputs"]["video_latents"]["latents"], sample["inputs"]["audio_latents"]


def test_preflight_accepts_native_ref_pairs_without_importing_torch_or_loading_payloads(paired_cache):
    manifest, _ = paired_cache
    module = Path(data.__file__)
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules; assert len(m['load_gan_real_records'](sys.argv[2],task='ref2av',geometry=None)[1])==6"
    subprocess.run([sys.executable, "-c", script, str(module), str(manifest)], check=True)
    with patch("torch.load", side_effect=AssertionError("Preflight must not read tensor payloads")):
        actual_manifest, rows, fingerprint = data.load_gan_real_records(manifest, task="ref2av", geometry=None)
        recipe = stream(manifest).recipe()
    assert actual_manifest == manifest and len(rows) == 6
    assert recipe["task"] == "ref2av" and recipe["condition_schema_version"] == 1
    assert recipe["geometries"] == [list(value) for value in GEOMETRIES]
    assert recipe["geometry_rows"] == [dict(geometry=list(value), rows=3) for value in GEOMETRIES]
    assert recipe["metadata_fingerprint"] == fingerprint


def test_geometry_pools_resume_exactly_and_never_substitute_another_shape(paired_cache):
    manifest, _ = paired_cache
    streams = [stream(manifest, rank=rank, world_size=3) for rank in range(3)]
    for geometry in GEOMETRIES:
        for cursor in range(5):
            indices = [item.index_at(cursor, geometry=geometry) for item in streams]
            assert len(set(indices)) == 3
            assert all(tuple(streams[0].rows[index][key] for key in ("target_num_frames", "target_height", "target_width")) == geometry for index in indices)
        expected = [streams[1].index_at(cursor, geometry=geometry) for cursor in range(30)]
        resumed = stream(manifest, rank=1, world_size=3)
        assert expected[7:] == [resumed.index_at(cursor, geometry=geometry) for cursor in range(7, 30)]
    with pytest.raises(ValueError, match="current rollout geometry"):
        streams[0].sample(0)
    with pytest.raises(ValueError, match="no positive targets matching"):
        streams[0].sample(0, geometry=(124, 64, 64))
    fixed = data.load_gan_real_records(manifest, task="ref2av", geometry=GEOMETRIES[0])[1]
    assert len(fixed) == 3


def test_positive_keeps_own_ref_caption_order_vision_tags_and_full_normalized_av(paired_cache):
    manifest, rows = paired_cache
    real = stream(manifest)
    geometry = GEOMETRIES[1]
    index = real.index_at(3, geometry=geometry)
    with patch("torch.load", wraps=torch.load) as loader:
        sample = real.sample(3, geometry=geometry)
    assert loader.call_count == 1 and sample["inputs"] == {}
    capability = Capability()
    with patch("torch.load", wraps=torch.load) as loader:
        (video, audio), condition = real.load(capability, 3, geometry=geometry)
    assert loader.call_count == 3
    assert capability.caption == rows[index]["caption"]
    assert [float(ref["video_latents"][0, 0]) for ref in capability.references] == pytest.approx([index + i / 10 for i in range(rows[index]["reference_image_count"])])
    assert torch.equal(capability.tags, torch.tensor([0, 1, 1]))
    assert condition["task"] == "ref2av" and all(ref.kind == "image" for ref in condition["references"])
    assert list(video.shape) == rows[index]["video_latent_shape"]
    assert list(audio.shape) == [2, 32, 207]
    assert (video == index).all() and (audio == index).all()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"reference_video_count": 1}, "image-only"),
        ({"reference_audio_count": 1}, "image-only"),
        ({"references": [dict(kind="video")]}, "image-only"),
        ({"vae_normalized": False}, "normalized"),
        ({"audio_latent_path": ""}, "audio_latent_path"),
        ({"video_latent_shape": [24, 1, 4, 6]}, "geometry"),
        ({"latent_fingerprint": ""}, "latent_fingerprint"),
    ],
)
def test_ref_real_metadata_rejects_condition_only_or_non_image_positive_streams(paired_cache, updates, match):
    manifest, rows = paired_cache
    write_rows(manifest, [dict(rows[0], **updates)])
    with pytest.raises(ValueError, match=match):
        data.load_gan_real_records(manifest, task="ref2av", geometry=None)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("source", "source_id"),
        ("schema", "schema1"),
        ("stage", "complete"),
        ("video_reference", "image-only"),
        ("reference_audio", "image-only"),
        ("reference_normalization", "normalized"),
        ("no_vision_tags", "vision token"),
        ("wrong_geometry", "target_height"),
        ("negative", "negative"),
    ],
)
def test_ref_condition_identity_and_actual_reference_contents_are_checked(paired_cache, problem, match):
    manifest, rows = paired_cache
    write_rows(manifest, [rows[0]])
    path = manifest.parent / rows[0]["condition_path"]
    payload = torch.load(path, weights_only=True)
    positive = payload["conditioning"]["positive"]
    if problem == "source":
        payload["source_id"] = "other-source"
    elif problem == "schema":
        payload["cache_schema_version"] = 2
    elif problem == "stage":
        payload["cache_stage"] = "text"
    elif problem == "video_reference":
        positive["references"][0]["kind"] = "video"
    elif problem == "reference_audio":
        positive["references"][0]["audio_latents"] = torch.zeros(2, 32)
    elif problem == "reference_normalization":
        positive["references"][0]["normalized"] = False
    elif problem == "no_vision_tags":
        positive["text_token_tags"] = torch.ones(3, dtype=torch.long)
    elif problem == "wrong_geometry":
        positive["target_height"] = 96
    else:
        payload["negative"] = {"prompt_embeds": torch.zeros(2, 3)}
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        stream(manifest).sample(0)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("normalization", "normalization"),
        ("fingerprint", "latent_fingerprint"),
        ("video_shape", "video target"),
        ("audio_shape", "audio target"),
        ("audio_nonfinite", "audio target"),
    ],
)
def test_paired_clean_av_payloads_are_verified_when_consumed(paired_cache, problem, match):
    manifest, rows = paired_cache
    write_rows(manifest, [rows[0]])
    path = manifest.parent / rows[0]["video_latent_path"]
    video = torch.load(path, weights_only=True)
    if problem == "normalization":
        video["normalized"] = False
    elif problem == "fingerprint":
        video["latent_fingerprint"] = "another-target"
    elif problem == "video_shape":
        video["latents"] = torch.zeros(24, 1, 4, 6)
    torch.save(video, path)
    audio_path = manifest.parent / rows[0]["audio_latent_path"]
    if problem == "audio_shape":
        torch.save(torch.zeros(2, 32, 1), audio_path)
    elif problem == "audio_nonfinite":
        torch.save(torch.full([2, 32, 207], float("nan")), audio_path)
    with pytest.raises(ValueError, match=match):
        stream(manifest).load(Capability(), 0, geometry=GEOMETRIES[0])


def test_ref_alias_and_reference_count_alias_preserve_identity(paired_cache):
    manifest, rows = paired_cache
    row = dict(rows[0], task="ref2va", ref_image_count=rows[0]["reference_image_count"])
    del row["reference_image_count"]
    write_rows(manifest, [row])
    real = stream(manifest)
    assert real.rows[0]["task"] == "ref2av" and real.rows[0]["reference_image_count"] == 1
    before = real.recipe()["metadata_fingerprint"]
    row["caption"] += " changed"
    write_rows(manifest, [row])
    assert stream(manifest).recipe()["metadata_fingerprint"] != before
