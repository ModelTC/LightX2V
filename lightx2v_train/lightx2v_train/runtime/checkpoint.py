import os
import shutil


def prune_checkpoints(output_dir, total_limit):
    if total_limit is None:
        return
    if not os.path.exists(output_dir):
        return

    checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-")]
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

    checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-")]
    if not checkpoints:
        return None, 0

    checkpoints = sorted(checkpoints, key=lambda name: parse_checkpoint_iteration(name))
    latest = checkpoints[-1]
    return os.path.join(output_dir, latest), parse_checkpoint_iteration(latest)
