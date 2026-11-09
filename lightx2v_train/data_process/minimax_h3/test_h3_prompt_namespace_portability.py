"""Prompt namespace publication on shared mounts without flock support."""

import errno
import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_minimax_h3_prompt_latents as builder
import prepare_minimax_h3_gan_prompts as coordinator


def read_namespace(output):
    return json.loads((output / "preprocess_config.json").read_text())


def test_namespace_creation_ignores_legacy_lock_and_never_calls_flock(tmp_path, monkeypatch):
    legacy_lock = tmp_path / ".preprocess.lock"
    legacy_lock.write_text("previous flock-based run")
    flock = Mock(side_effect=BlockingIOError(errno.EAGAIN, "AFS flock unavailable"))
    monkeypatch.setattr(fcntl, "flock", flock)

    builder.initialize_cache_namespace(tmp_path, {"recipe": "first"}, allow_replace=False)

    assert read_namespace(tmp_path)["preprocess_config"] == {"recipe": "first"}
    assert legacy_lock.read_text() == "previous flock-based run"
    assert not (tmp_path / ".preprocess.init-lock").exists()
    flock.assert_not_called()


@pytest.mark.parametrize("same_descriptor", [True, False])
def test_concurrent_namespaces_publish_once_without_clobber(tmp_path, monkeypatch, same_descriptor):
    reservation = tmp_path / ".preprocess.init-lock"
    contended = Event()
    start = Barrier(2)
    original_mkdir = Path.mkdir
    original_write = builder.atomic_write_json
    publications = []

    def track_contention(path, *args, **kwargs):
        try:
            return original_mkdir(path, *args, **kwargs)
        except FileExistsError:
            if path == reservation:
                contended.set()
            raise

    def publish_after_contention(path, payload):
        assert contended.wait(5), "The competing initializer did not attempt the reservation"
        publications.append(payload)
        original_write(path, payload)

    monkeypatch.setattr(Path, "mkdir", track_contention)
    monkeypatch.setattr(builder, "atomic_write_json", publish_after_contention)
    descriptors = [{"recipe": "first"}, {"recipe": "first" if same_descriptor else "second"}]

    def initialize(descriptor):
        start.wait(timeout=5)
        builder.initialize_cache_namespace(tmp_path, descriptor, allow_replace=False)
        return descriptor

    successes, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize, descriptor) for descriptor in descriptors]
        for future in futures:
            try:
                successes.append(future.result(timeout=10))
            except RuntimeError as error:
                errors.append(error)

    assert len(publications) == 1
    assert len(successes) == (2 if same_descriptor else 1)
    assert len(errors) == (0 if same_descriptor else 1)
    if errors:
        assert "different input/model/geometry" in str(errors[0])
    assert read_namespace(tmp_path)["preprocess_config"] == successes[0]
    assert not reservation.exists()


def test_matching_namespace_is_read_only_even_with_leftover_reservation(tmp_path, monkeypatch):
    descriptor = {"recipe": "published"}
    builder.initialize_cache_namespace(tmp_path, descriptor, allow_replace=False)
    identity = tmp_path / "preprocess_config.json"
    original = identity.read_bytes()
    reservation = tmp_path / ".preprocess.init-lock"
    reservation.mkdir()
    mkdir = Mock(side_effect=AssertionError("Matching namespace must not acquire a reservation"))
    write = Mock(side_effect=AssertionError("Matching namespace must not be rewritten"))
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(builder, "atomic_write_json", write)

    builder.initialize_cache_namespace(tmp_path, descriptor, allow_replace=False)

    assert identity.read_bytes() == original
    assert reservation.is_dir()
    mkdir.assert_not_called()
    write.assert_not_called()


def test_unpublished_reservation_times_out_without_stealing_it(tmp_path, monkeypatch):
    reservation = tmp_path / ".preprocess.init-lock"
    reservation.mkdir()
    monkeypatch.setattr(builder, "CACHE_INIT_TIMEOUT", 0)

    with pytest.raises(TimeoutError, match="Timed out") as error:
        builder.initialize_cache_namespace(tmp_path, {"recipe": "first"}, allow_replace=False)

    assert str(reservation) in str(error.value)
    assert reservation.is_dir()
    assert list(tmp_path.iterdir()) == [reservation]


def test_replacement_requires_reservation_and_explicit_permission(tmp_path, monkeypatch):
    first, second = {"recipe": "first"}, {"recipe": "second"}
    builder.initialize_cache_namespace(tmp_path, first, allow_replace=False)
    reservation = tmp_path / ".preprocess.init-lock"
    with pytest.raises(RuntimeError, match="different input/model/geometry"):
        builder.initialize_cache_namespace(tmp_path, second, allow_replace=False)
    reservation.mkdir()
    monkeypatch.setattr(builder, "CACHE_INIT_TIMEOUT", 0)

    with pytest.raises(TimeoutError, match="Timed out"):
        builder.initialize_cache_namespace(tmp_path, second, allow_replace=True)
    assert read_namespace(tmp_path)["preprocess_config"] == first
    assert reservation.is_dir()

    reservation.rmdir()
    builder.initialize_cache_namespace(tmp_path, second, allow_replace=True)
    assert read_namespace(tmp_path)["preprocess_config"] == second
    assert not reservation.exists()


def test_failed_publication_releases_only_its_owned_reservation(tmp_path, monkeypatch):
    publish = Mock(side_effect=OSError("injected namespace publication failure"))
    monkeypatch.setattr(builder, "atomic_write_json", publish)

    with pytest.raises(OSError, match="injected namespace publication failure"):
        builder.initialize_cache_namespace(tmp_path, {"recipe": "first"}, allow_replace=False)

    publish.assert_called_once()
    assert list(tmp_path.iterdir()) == []


def write_payload(format, path, value):
    if format == "json":
        builder.atomic_write_json(path, value)
    elif format == "jsonl":
        builder.atomic_write_jsonl(path, [value])
    else:
        builder.atomic_torch_save(value, path)


@pytest.mark.parametrize("format", ["json", "jsonl", "pt"])
def test_atomic_writers_use_distinct_temporary_paths_with_identical_pids(tmp_path, monkeypatch, format):
    destination = tmp_path / f"shared.{format}"
    ready_to_publish = Barrier(2)
    temporary_paths = []
    original_replace = builder.os.replace

    def simultaneous_replace(source, target):
        temporary_paths.append(Path(source))
        ready_to_publish.wait(timeout=5)
        original_replace(source, target)

    monkeypatch.setattr(builder.os, "getpid", lambda: 12345)
    monkeypatch.setattr(builder.os, "replace", simultaneous_replace)
    payloads = [{"writer": "node-one"}, {"writer": "node-two"}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_payload, format, destination, payload) for payload in payloads]
        for future in futures:
            future.result(timeout=10)

    assert len(set(temporary_paths)) == 2
    assert all(path.parent == tmp_path for path in temporary_paths)
    if format == "pt":
        actual = torch.load(destination, weights_only=False, map_location="cpu")
    else:
        actual = json.loads(destination.read_text())
    assert actual in payloads
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("format", ["json", "jsonl", "pt"])
def test_failed_atomic_publication_preserves_target_and_cleans_temporary_file(tmp_path, monkeypatch, format):
    destination = tmp_path / f"shared.{format}"
    destination.write_bytes(b"existing cache")
    monkeypatch.setattr(builder.os, "replace", Mock(side_effect=OSError("injected rename failure")))

    with pytest.raises(OSError, match="injected rename failure"):
        write_payload(format, destination, {"writer": "replacement"})

    assert destination.read_bytes() == b"existing cache"
    assert list(tmp_path.iterdir()) == [destination]


def test_rank_zero_initializes_before_broadcast(tmp_path, monkeypatch):
    output = tmp_path / "new-cache"
    descriptor = {"recipe": "first"}
    initialize = Mock()
    monkeypatch.setattr(builder, "initialize_cache_namespace", initialize)

    def broadcast(status, src):
        assert src == 0
        assert status == [None]
        assert output.is_dir()
        initialize.assert_called_once_with(output, descriptor, allow_replace=False)

    broadcast_mock = Mock(side_effect=broadcast)
    monkeypatch.setattr(coordinator.dist, "broadcast_object_list", broadcast_mock)

    coordinator.initialize_shared_namespace(output, descriptor, rank=0)

    broadcast_mock.assert_called_once()


def test_nonzero_rank_only_receives_namespace_status(tmp_path, monkeypatch):
    output = tmp_path / "not-created-by-worker"
    initialize = Mock(side_effect=AssertionError("Worker must not initialize the namespace"))
    broadcast = Mock()
    monkeypatch.setattr(builder, "initialize_cache_namespace", initialize)
    monkeypatch.setattr(coordinator.dist, "broadcast_object_list", broadcast)

    coordinator.initialize_shared_namespace(output, {"recipe": "first"}, rank=17)

    initialize.assert_not_called()
    broadcast.assert_called_once_with([None], src=0)
    assert not output.exists()


def test_rank_zero_initialization_failure_is_broadcast_to_other_ranks(tmp_path, monkeypatch):
    output = tmp_path / "cache"
    initialize = Mock(side_effect=OSError("injected namespace failure"))
    monkeypatch.setattr(builder, "initialize_cache_namespace", initialize)
    published = []

    def broadcast(status, src):
        assert src == 0
        if status[0] is not None:
            published[:] = status
        else:
            assert published
            status[:] = published

    monkeypatch.setattr(coordinator.dist, "broadcast_object_list", broadcast)
    errors = []
    for rank in (0, 1, 17):
        with pytest.raises(RuntimeError, match="injected namespace failure") as error:
            coordinator.initialize_shared_namespace(output, {"recipe": "first"}, rank=rank)
        errors.append(str(error.value))

    initialize.assert_called_once()
    assert len(set(errors)) == 1
    assert "OSError" in published[0]
