

from .mmsr_block import NaMMSRTransformerBlock


nadit_blocks = {
    "mmdit_sr": NaMMSRTransformerBlock,
}


def get_nablock(block_type: str):
    if block_type in nadit_blocks:
        return nadit_blocks[block_type]
    raise NotImplementedError(f"{block_type} is not supported")
