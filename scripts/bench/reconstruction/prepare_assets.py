"""Download pinned weights or reconstruct the frozen latency inputs."""

import argparse
import shutil
import zipfile
from pathlib import Path

from fixtures import load_requests, sha256, unique_images
from huggingface_hub import hf_hub_download, snapshot_download

REVISIONS = {
    "vggt": ("facebook/VGGT-1B", "860abec7937da0a4c03c41d3c269c366e82abdf9"),
    "trellis2": ("microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d"),
    "base": ("microsoft/TRELLIS-image-large", "459ee480d7545f873a0768364bc5f5443eb1ff5e"),
    "dinov3": ("camenduru/dinov3-vitl16-pretrain-lvd1689m", "3c276edd87d6f6e569ff0c4400e086807d0f3881"),
    "rembg": ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"),
}
ETH3D_REVISION = "64154bdca815ef7712f5f2621e231bac7dc99d59"
GSO_ARCHIVE_SHA256 = "d5451221671f1e6cde403d226a3473762f9f45a322c7154aba806f7243fb1ba9"


def prepare_weights(model, output):
    repo, revision = REVISIONS[model]
    patterns = ["model.pt"] if model == "vggt" else ["pipeline.json", "ckpts/*512_bf16.*", "ckpts/ss_flow_img_dit_1_3B_64_bf16.*", "ckpts/*dec_next_dc_f16c32_fp16.*"]
    patterns += ["LICENSE*", "README.md"]
    snapshot_download(repo, revision=revision, local_dir=output, allow_patterns=patterns)
    if model == "trellis2":
        for component in ("base", "dinov3", "rembg"):
            repo, revision = REVISIONS[component]
            patterns = ["ckpts/ss_dec_conv3d_16l8_fp16.*"] if component == "base" else ["*.json", "*.safetensors", "*.py"]
            patterns += ["LICENSE*", "README.md"]
            snapshot_download(repo, revision=revision, local_dir=output / component, allow_patterns=patterns)


def prepare_inputs(model, output, archive):
    requests = load_requests(Path(__file__).with_name(f"{model}_requests.json"))
    files = unique_images(requests)
    if model == "trellis2" and archive is None:
        raise ValueError("Supply --gso-archive from https://drive.google.com/uc?id=1tKHhH-L1viCvTuBO1xg--B_ioK7JUrrE (IBRNet Google Scanned Objects renderings)")
    if archive is not None and sha256(archive) != GSO_ARCHIVE_SHA256:
        raise ValueError("GSO archive SHA256 does not match the frozen source")
    for relative, image in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != image["sha256"]:
            if model == "vggt":
                source = hf_hub_download("Livioni/eth3d_omnivggt", relative.removeprefix("vggt/"), repo_type="dataset", revision=ETH3D_REVISION)
                shutil.copyfile(source, target)
            else:
                with zipfile.ZipFile(archive) as source, source.open(image["archive_member"]) as stream, target.open("wb") as destination:
                    shutil.copyfileobj(stream, destination)
        if target.stat().st_size != image["size_bytes"] or sha256(target) != image["sha256"]:
            raise ValueError(f"Input bytes do not match the frozen manifest: {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["weights", "inputs"])
    parser.add_argument("model", choices=["vggt", "trellis2"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gso-archive", type=Path)
    args = parser.parse_args()
    if args.action == "weights":
        prepare_weights(args.model, args.output)
    else:
        prepare_inputs(args.model, args.output, args.gso_archive)
