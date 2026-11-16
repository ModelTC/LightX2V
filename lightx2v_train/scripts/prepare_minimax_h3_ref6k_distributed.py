#!/usr/bin/env python3
"""Publish the shared Ref6k manifest once, then validate the training recipe."""

import argparse
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    # One CPU process per ACP node. Only global rank zero publishes shared AFS files.
    dist.init_process_group("gloo", timeout=timedelta(hours=3))
    status = [None]
    try:
        if dist.get_rank() == 0:
            try:
                command = [sys.executable, "-u", str(scripts / "minimax_h3_ref6k_prepare.py")]
                for flag, key in (
                    ("teacher-export-dir", "H3_TEACHER_EXPORT"),
                    ("raw-jsonl", "H3_RAW_JSONL"),
                    ("media-root", "H3_MEDIA_ROOT"),
                    ("selfgen-metadata", "H3_SELFGEN_METADATA"),
                    ("output-dir", "H3_REF6K_DATA"),
                ):
                    command.extend([f"--{flag}", os.environ[key]])
                print(f"[ref6k] Preparing shared manifest: {os.environ['H3_REF6K_DATA']} (existing latents, no encoding)", flush=True)
                subprocess.run(command, check=True)
                print("[ref6k] Checking source EMA, original bases and PDMD + GAN recipe", flush=True)
                subprocess.run([sys.executable, "-u", str(scripts / "minimax_h3_ref6k_pdmd_gan_preflight.py"), args.config], check=True)
            except Exception as error:
                status[0] = f"Ref6k preparation failed: {error}"
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(status[0])
        print(f"[ref6k] node rank={dist.get_rank()}: shared manifest and CM initialization verified", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
