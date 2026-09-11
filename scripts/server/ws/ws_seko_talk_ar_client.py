#!/usr/bin/env python3
"""Minimal protobuf websocket client for seko_talk_ar live inference."""

import argparse
import os
import subprocess
import tempfile
import wave

from lightx2v.server.ws.protocol import (
    AudioChunk,
    ClientMessage,
    Empty,
    ServerEvent,
    ServerMessage,
    Start,
)


def read_wav_pcm16_mono(path: str, target_sr: int = 16000) -> bytes:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"expected 16-bit wav, got sampwidth={sampwidth}")
    if channels != 1:
        raise ValueError(f"expected mono wav, got channels={channels}")
    if sr != target_sr:
        raise ValueError(f"expected {target_sr} Hz wav, got {sr}")
    return frames


def _try_recv(ws, chunks, size, timeout=0.01):
    try:
        raw = ws.recv(timeout=timeout)
    except TimeoutError:
        return ""
    except Exception as e:
        if "close" in type(e).__name__.lower() or "close" in str(e).lower():
            return "closed"
        raise
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    msg = ServerMessage.parse(bytes(raw))
    which = msg.which()
    if which == "error":
        raise RuntimeError(f"server error {msg.error.code}: {msg.error.message}")
    if which == "video_start":
        size[0] = msg.video_start.video_width
        size[1] = msg.video_start.video_height
        print(f"video_start {size[0]}x{size[1]}")
    elif which == "event":
        name = {ServerEvent.START_SPEAKING: "START_SPEAKING", ServerEvent.STOP_SPEAKING: "STOP_SPEAKING"}.get(msg.event.type, str(msg.event.type))
        print(f"event {name} {msg.event.message}")
        if msg.event.type == ServerEvent.STOP_SPEAKING:
            return "stop_speaking"
    elif which == "video":
        chunks.append(msg.video)
        print(f"video seq={msg.video.sequence} audio={len(msg.video.audio_data)} video={len(msg.video.video_data)}")
    return ""


def _mux_raw(output, width, height, fps, video_bytes, audio_bytes):
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ws_raw_") as tmp:
        rgb_path = os.path.join(tmp, "video.rgb")
        pcm_path = os.path.join(tmp, "audio.pcm")
        with open(rgb_path, "wb") as f:
            f.write(video_bytes)
        with open(pcm_path, "wb") as f:
            f.write(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            pcm_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            rgb_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            output,
        ]
        subprocess.check_call(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="LightX2V seko_talk_ar websocket live client")
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765/v1/live")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True, help="16-bit mono 16kHz wav")
    parser.add_argument("--output", type=str, default="save_results/seko_talk_ar_ws.mp4")
    parser.add_argument("--request_id", type=str, default="ws-client")
    parser.add_argument("--chunk_bytes", type=int, default=3200, help="PCM bytes per AudioChunk (100ms at 16k mono s16le)")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aspect_ratio", type=str, default="16:9", choices=["1:1", "16:9", "9:16"])
    parser.add_argument("--chunk_frames", type=int, default=8)
    return parser.parse_args()


def main():
    try:
        from websockets.sync.client import connect
    except ImportError as e:
        raise SystemExit("pip install websockets") from e

    args = parse_args()
    with open(args.image, "rb") as f:
        image_data = f.read()
    pcm = read_wav_pcm16_mono(args.audio)
    ext = os.path.splitext(args.image)[1].lstrip(".") or "jpg"
    audio_secs = len(pcm) / 16000.0 / 2

    start = Start(
        request_id=args.request_id,
        audio_format="pcm_s16le/16000/1",
        image_fromat=ext,
        image_data=[image_data],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        aspect_ratio=args.aspect_ratio,
        chunk_frames=args.chunk_frames,
    )

    chunks = []
    size = [0, 0]
    with connect(args.url, max_size=None) as ws:
        ws.send(ClientMessage(start=start).serialize())
        ws.send(ClientMessage(audio_start=Empty()).serialize())
        seq = 1
        for i in range(0, len(pcm), args.chunk_bytes):
            ws.send(ClientMessage(audio=AudioChunk(sequence=seq, data=pcm[i : i + args.chunk_bytes])).serialize())
            seq += 1
        ws.send(ClientMessage(audio_end=Empty()).serialize())
        print(f"audio sent {audio_secs:.2f}s, recv video")

        reason = ""
        while not reason:
            reason = _try_recv(ws, chunks, size, timeout=1.0)
        print(f"recv done ({reason}), mux {len(chunks)} chunks then disconnect")

    video_bytes = b"".join(c.video_data for c in chunks)
    audio_bytes = b"".join(c.audio_data for c in chunks)
    if size[0] <= 0 or size[1] <= 0 or not video_bytes:
        raise SystemExit("no video received")
    _mux_raw(args.output, size[0], size[1], args.fps, video_bytes, audio_bytes)
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
