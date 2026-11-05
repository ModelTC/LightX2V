#!/usr/bin/env python3
"""Interrupt/resume test client for seko_talk_ar websocket live inference.

Flow:
  1. Start + send full audio once
  2. Wait 10s (keep draining server messages)
  3. Send Interrupt (clear audio buffer), then send a duration=-1 SetPrompt and the action SetPrompt back to back
  4. Wait 15s (keep draining)
  5. Send full audio again
  6. Exit after receiving STOP_SPEAKING twice, then disconnect and mux output
"""

import argparse
import os
import subprocess
import tempfile
import time
import wave

from lightx2v.server.ws.protocol import (
    AudioChunk,
    ClientMessage,
    Empty,
    Interrupt,
    ServerEvent,
    ServerMessage,
    SetPrompt,
    Start,
)

DEFAULT_GLOBAL_PROMPT = "保持状态，不要乱晃。"
DEFAULT_ACTION_PROMPT = "<Subject 1> crosses both wrists in front of her chest into a clear X and holds the crossed wrists toward the camera. Her elbows stay bent and her feet stay planted. The feet stay planted and the torso does not shift, lean, or walk."


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
        name = {ServerEvent.START_SPEAKING: "START_SPEAKING", ServerEvent.STOP_SPEAKING: "STOP_SPEAKING", ServerEvent.PROMPT_UPDATE: "PROMPT_UPDATE"}.get(msg.event.type, str(msg.event.type))
        print(f"event {name} {msg.event.message}")
        if msg.event.type == ServerEvent.STOP_SPEAKING:
            return "stop_speaking"
    elif which == "video":
        chunks.append(msg.video)
        print(f"video seq={msg.video.sequence} audio={len(msg.video.audio_data)} video={len(msg.video.video_data)}")
    return ""


def _mux_raw(output, width, height, fps, video_bytes, audio_bytes, sample_rate=16000):
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ws_raw_") as tmp:
        h264_path = os.path.join(tmp, "video.h264")
        pcm_path = os.path.join(tmp, "audio.pcm")
        with open(h264_path, "wb") as f:
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
            str(int(sample_rate)),
            "-ac",
            "1",
            "-i",
            pcm_path,
            "-r",
            str(fps),
            "-f",
            "h264",
            "-i",
            h264_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            output,
        ]
        subprocess.check_call(cmd)


def send_full_audio(ws, pcm: bytes, chunk_bytes: int, seq: int) -> int:
    ws.send(ClientMessage(audio_start=Empty()).serialize())
    for i in range(0, len(pcm), chunk_bytes):
        ws.send(ClientMessage(audio=AudioChunk(sequence=seq, data=pcm[i : i + chunk_bytes])).serialize())
        seq += 1
    ws.send(ClientMessage(audio_end=Empty()).serialize())
    return seq


def drain_for(ws, chunks, size, seconds: float, stop_counter: list) -> None:
    deadline = time.monotonic() + seconds
    print(f"drain for {seconds}s ...")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        reason = _try_recv(ws, chunks, size, timeout=min(1.0, remaining))
        if reason == "stop_speaking":
            stop_counter[0] += 1
            print(f"STOP_SPEAKING count={stop_counter[0]}")
        elif reason == "closed":
            raise RuntimeError("websocket closed while waiting")


def recv_until_stop_count(ws, chunks, size, target: int, stop_counter: list) -> None:
    print(f"waiting for STOP_SPEAKING x{target} (current={stop_counter[0]}) ...")
    while stop_counter[0] < target:
        reason = _try_recv(ws, chunks, size, timeout=1.0)
        if reason == "stop_speaking":
            stop_counter[0] += 1
            print(f"STOP_SPEAKING count={stop_counter[0]}")
        elif reason == "closed":
            raise RuntimeError("websocket closed before enough STOP_SPEAKING events")


def parse_args():
    parser = argparse.ArgumentParser(description="LightX2V seko_talk_ar interrupt/resume websocket test client")
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765/v1/live")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True, help="16-bit mono 16kHz wav")
    parser.add_argument("--output", type=str, default="save_results/seko_talk_ar_ws_interrupt.mp4")
    parser.add_argument("--request_id", type=str, default="ws-interrupt-client")
    parser.add_argument("--chunk_bytes", type=int, default=3200, help="PCM bytes per AudioChunk (100ms at 16k mono s16le)")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--output_audio_sr", type=int, default=16000, help="s16le sample rate of received audio (16000 seko, 24000 H3)")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aspect_ratio", type=str, default="16:9", choices=["1:1", "16:9", "9:16"])
    parser.add_argument("--resolution", type=str, default="", help="output resolution key, e.g. 480p or 720p")
    parser.add_argument("--chunk_frames", type=int, default=8)
    parser.add_argument("--action_prompt", type=str, default=DEFAULT_ACTION_PROMPT)
    parser.add_argument("--action_duration", type=float, default=0.0, help="SetPrompt duration seconds; 0 omit (server default 3), -1 hold, or >0")
    parser.add_argument("--wait_before_interrupt", type=float, default=10.0)
    parser.add_argument("--wait_after_interrupt", type=float, default=15.0)
    parser.add_argument("--stop_speaking_count", type=int, default=2)
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
    stream_config = {}
    if args.resolution:
        stream_config["resolution"] = args.resolution

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
        config_json=stream_config,
    )

    chunks = []
    size = [0, 0]
    stop_counter = [0]
    seq = 1

    with connect(args.url, max_size=None) as ws:
        ws.send(ClientMessage(start=start).serialize())
        print("Start sent")

        seq = send_full_audio(ws, pcm, args.chunk_bytes, seq)
        print(f"round-1 audio sent ({audio_secs:.2f}s), seq now {seq}")

        drain_for(ws, chunks, size, 10, stop_counter)

        drain_for(ws, chunks, size, args.wait_before_interrupt, stop_counter)

        ws.send(ClientMessage(interrupt=Interrupt(reason="test interrupt")).serialize())
        print("Interrupt sent")

        drain_for(ws, chunks, size, 3, stop_counter)

        ws.send(ClientMessage(set_prompt=SetPrompt(prompt=DEFAULT_GLOBAL_PROMPT, duration=-1)).serialize())
        print(f"SetPrompt sent: {DEFAULT_GLOBAL_PROMPT} duration=-1")
        ws.send(ClientMessage(set_prompt=SetPrompt(prompt=args.action_prompt, duration=args.action_duration)).serialize())
        print(f"SetPrompt sent: {args.action_prompt} duration={args.action_duration}")

        drain_for(ws, chunks, size, args.wait_after_interrupt, stop_counter)

        seq = send_full_audio(ws, pcm, args.chunk_bytes, seq)
        print(f"round-2 audio sent ({audio_secs:.2f}s), seq now {seq}")

        recv_until_stop_count(ws, chunks, size, args.stop_speaking_count, stop_counter)
        print(f"done: STOP_SPEAKING x{stop_counter[0]}, disconnect")

    video_bytes = b"".join(c.video_data for c in chunks)
    audio_bytes = b"".join(c.audio_data for c in chunks)
    if size[0] <= 0 or size[1] <= 0 or not video_bytes:
        raise SystemExit("no video received")
    _mux_raw(args.output, size[0], size[1], args.fps, video_bytes, audio_bytes, sample_rate=args.output_audio_sr)
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
