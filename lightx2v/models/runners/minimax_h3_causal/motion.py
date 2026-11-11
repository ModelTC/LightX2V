"""Resolve frame-based motion cues to H3 v0.1 DiT chunks."""

from lightx2v.models.networks.minimax_h3.packing import FRAMES_PER_CHUNK


def prepare_motions(motions, *, prompt, num_frames):
    """Return chunk-to-prompt mappings, rounding frame cues up to a boundary."""
    if motions is None:
        motions = [{"frame": 0, "text": prompt}]
    if not isinstance(motions, list) or not motions:
        raise ValueError("motions must be a nonempty list of {frame, text}")
    if any(not isinstance(motion, dict) or type(motion.get("frame")) is not int or not isinstance(motion.get("text"), str) or not motion["text"].strip() for motion in motions):
        raise ValueError("motions must contain integer frame indices and nonempty text")
    frames = [motion["frame"] for motion in motions]
    if frames[0] != 0 or frames != sorted(set(frames)) or frames[-1] >= num_frames:
        raise ValueError("motions must start at frame 0, increase strictly, and fit the output duration")

    # A full DiT chunk spans 17 output frames. Keep the requested frame indices in
    # InputInfo; only this execution schedule uses the next chunk boundary.
    chunks = [(frame + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK for frame in frames]
    if len(set(chunks)) != len(chunks):
        raise ValueError("motions must align to distinct 17-frame chunk boundaries")
    if chunks[-1] * FRAMES_PER_CHUNK >= num_frames:
        raise ValueError("The last motion aligns to a chunk boundary beyond the output duration")
    return {chunk: motion["text"] for chunk, motion in zip(chunks, motions)}
