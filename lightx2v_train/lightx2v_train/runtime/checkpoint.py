import os
import shutil


def _is_complete_checkpoint(checkpoint_dir):
    return any(
        os.path.isfile(os.path.join(checkpoint_dir, marker))
        for marker in (
            "_SUCCESS",
            "trainer_state.pt",
            "training_state.pt",
        )
    )


def _completed_checkpoint_names(output_dir):
    return [name for name in os.listdir(output_dir) if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name)) and _is_complete_checkpoint(os.path.join(output_dir, name))]


def prune_checkpoints(output_dir, total_limit):
    if total_limit is None:
        return
    if not os.path.exists(output_dir):
        return

    checkpoints = _completed_checkpoint_names(output_dir)
    checkpoints = sorted(checkpoints, key=lambda name: parse_checkpoint_iteration(name))
    if len(checkpoints) < total_limit:
        return

    for name in checkpoints[: len(checkpoints) - total_limit + 1]:
        shutil.rmtree(os.path.join(output_dir, name))


def parse_checkpoint_iteration(checkpoint_path):
    return int(os.path.basename(checkpoint_path).split("-")[-1])


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None, 0

    checkpoints = _completed_checkpoint_names(output_dir)
    if not checkpoints:
        return None, 0

    checkpoints = sorted(checkpoints, key=lambda name: parse_checkpoint_iteration(name))
    latest = checkpoints[-1]
    return os.path.join(output_dir, latest), parse_checkpoint_iteration(latest)
