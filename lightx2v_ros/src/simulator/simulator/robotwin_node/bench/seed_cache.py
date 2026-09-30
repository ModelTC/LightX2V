"""Legacy-compatible RoboTwin validated seed lists, not policy-success seeds."""

import fcntl
import json
import warnings
from itertools import pairwise
from pathlib import Path

from simulator.sim.bench.results import atomic_json


def load_seeds(path):
    path = Path(path)
    if not path.is_file():
        return []
    try:
        seeds = json.loads(path.read_text())
        if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("expected a list of nonnegative integer seeds")
        if any(right <= left for left, right in pairwise(seeds)):
            raise ValueError("seeds must be strictly increasing")
        return seeds
    except (OSError, ValueError) as exc:
        warnings.warn(f"Ignoring unreadable or invalid seed cache {path}: {exc}", stacklevel=2)
        return []


class SeedCache:
    def __init__(self, cfg, task):
        evaluation = cfg["EVALUATION"]
        self.path = None
        self.replay = []
        if evaluation["reuse_seed_cache"]:
            self.path = Path(evaluation["seed_cache_dir"]) / task["phase"] / f"{task['task_name']}_seed.json"
            self.replay = load_seeds(self.path)
            print(f"Seed cache: {self.path} ({len(self.replay)} seeds); cached seeds override the initial seed but are expert-revalidated", flush=True)
        self.position = 0
        self.next_fresh = 100000 * (1 + cfg["seed"])
        self.current_hit = False
        self.rejected = set()
        self.accepted = set()

    def next_candidate(self, fallback):
        if self.path is None:
            self.current_hit = False
            return fallback
        self.current_hit = self.position < len(self.replay)
        if self.current_hit:
            seed = self.replay[self.position]
            self.position += 1
        else:
            seed = self.next_fresh
        self.next_fresh = seed + 1
        return seed

    def reject(self, seed):
        self.rejected.add(seed)

    def record_validated(self, seed):
        """Call after expert success AND successful policy-environment setup."""
        if self.path is None:
            return
        self.accepted.add(seed)
        self.rejected.discard(seed)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Merge under a lock so separate managers sharing a cache don't lose seeds.
            with self.path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                seeds = sorted((set(load_seeds(self.path)) - self.rejected) | self.accepted)
                atomic_json(self.path, seeds)
        except OSError as exc:
            # Legacy evaluation treats optional cache persistence as best-effort.
            warnings.warn(f"Failed to write seed cache {self.path}: {exc}", stacklevel=2)
            return
        print(f"Saved {len(seeds)} validated seeds to {self.path}", flush=True)


def find_validated_seed(environment, cache, max_attempts):
    """Replay candidates using the same real expert checks as fresh seeds."""
    last_error = None
    for _ in range(max(1, max_attempts)):
        environment.seed = cache.next_candidate(environment.seed)
        try:
            environment._setup_demo()
            info = environment.env.play_once()
            solvable = bool(environment.env.plan_success) and bool(environment.env.check_success())
            environment._close_task_env()
            if solvable:
                return info
            environment._log(f"seed {environment.seed}: expert cannot solve this layout; trying next seed")
        except Exception as exc:  # noqa: BLE001 - legacy expert retries; bounded and logged
            last_error = exc
            environment._close_task_env()
            environment._log(f"seed {environment.seed}: expert check raised {exc!r}; trying next seed")
        cache.reject(environment.seed)
        environment.seed += 1
    raise RuntimeError(f"no expert-solvable seed found after {max_attempts} attempts; last error: {last_error}")
