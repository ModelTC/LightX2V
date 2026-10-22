"""Make native SageAttention2 launches follow PyTorch's current CUDA stream."""

import argparse
import re
from pathlib import Path

SOURCE_FILES = (
    "csrc/fused/fused.cu",
    "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
)


def current_stream_launch(match):
    arguments = [argument.strip() for argument in match.group(1).split(",")]
    if len(arguments) == 2:
        arguments.append("0")
    if len(arguments) == 3:
        arguments.append("at::cuda::getCurrentCUDAStream()")
    return "<<<" + ", ".join(arguments) + ">>>"


def patch_source(source):
    patched = re.sub(r"<<<([^<>]*?)>>>", current_stream_launch, source)
    if patched != source and "#include <ATen/cuda/CUDAContext.h>" not in patched:
        position = patched.index("#include")
        patched = patched[:position] + "#include <ATen/cuda/CUDAContext.h>\n" + patched[position:]
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SageAttention source checkout")
    args = parser.parse_args()
    for relative in SOURCE_FILES:
        path = args.source / relative
        source = path.read_text()
        patched = patch_source(source)
        if patched != source:
            path.write_text(patched)
            print(f"Updated {path}")
        else:
            print(f"Already uses explicit streams: {path}")


if __name__ == "__main__":
    main()
