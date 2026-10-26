"""Persistent HPSv3/VideoAlign inference in an isolated Transformers 4 environment."""

import argparse
import json
import logging
import math
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

from lightx2v_rl.rewards.common import (
    HPS_METRICS,
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    VIDEOALIGN_METRICS,
    decode_request,
    positive_fps,
    validate_metrics,
    video_to_tchw,
)

LOGGER = logging.getLogger(__name__)


class ScoreEngine:
    def __init__(self, hpsv3=None, videoalign=None):
        if hpsv3 is None and videoalign is None:
            raise ValueError("Enable at least one reward model")
        self.hpsv3, self.videoalign = hpsv3, videoalign

    def load(self):
        for backend in (self.hpsv3, self.videoalign):
            if backend is not None:
                backend.load()

    def health(self):
        return {
            "version": PROTOCOL_VERSION,
            "status": "ready",
            "metrics": list((HPS_METRICS if self.hpsv3 is not None else ()) + (VIDEOALIGN_METRICS if self.videoalign is not None else ())),
        }

    def score(self, video, prompt, metrics, source_fps):
        video = video_to_tchw(video)
        metrics = validate_metrics(metrics)
        source_fps = positive_fps(source_fps)
        hps = [name for name in metrics if name in HPS_METRICS]
        alignment = [name for name in metrics if name in VIDEOALIGN_METRICS]
        if hps and self.hpsv3 is None:
            raise ValueError("HPSv3 is not enabled in this reward server")
        if alignment and self.videoalign is None:
            raise ValueError("VideoAlign is not enabled in this reward server")
        scores = {}
        if hps:
            scores.update(self.hpsv3.score(video, prompt, variants=hps))
        if alignment:
            scores.update(self.videoalign.score(video, prompt, source_fps, dimensions=alignment))
        result = {name: float(scores[name]) for name in metrics}
        if any(not math.isfinite(value) for value in result.values()):
            raise RuntimeError("Reward model produced nonfinite scores")
        return result


def make_server(engine, host="127.0.0.1", port=8791):
    """Single inference request at a time bounds GPU/host memory usage.

    Use one upstream CPU reward worker initially. There is no video codec or
    pickle deserialization on this route; NPZ transports original uint8 pixels.
    """

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def respond(self, status, result):
            payload = json.dumps(result, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, engine.health())
            else:
                self.respond(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/score":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                if self.headers.get_content_type() != "application/x-npz":
                    raise ValueError("Content-Type must be application/x-npz")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Content-Length must be between 1 and 512 MiB")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Incomplete reward request body")
                video, prompt, metrics, source_fps = decode_request(payload)
            except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile, EOFError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            try:
                scores = engine.score(video, prompt, metrics, source_fps)
                self.respond(200, {"version": PROTOCOL_VERSION, "scores": scores})
            except Exception as exc:
                LOGGER.exception("Reward inference failed")
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format, *args):
            LOGGER.info("%s: " + format, self.address_string(), *args)

    class RewardServer(HTTPServer):
        request_queue_size = 64
        allow_reuse_address = True

    return RewardServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--models", default="hpsv3,videoalign", help="Comma-separated hpsv3,videoalign subset")
    parser.add_argument("--hps-device", default="cuda:0")
    parser.add_argument("--videoalign-device", default="cuda:1")
    parser.add_argument("--hps-frame-chunk-size", type=int, default=16, help="0 scores all frames in one batch")
    parser.add_argument("--videoalign-sample-fps", type=float, default=4.0)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = args.models.split(",")
    if not models or any(name not in {"hpsv3", "videoalign"} for name in models) or len(set(models)) != len(models):
        parser.error("--models must be hpsv3, videoalign, or hpsv3,videoalign")
    if args.hps_frame_chunk_size < 0:
        parser.error("--hps-frame-chunk-size must be >= 0")
    from lightx2v_rl.rewards.assets import resolve_reward_assets

    try:
        assets = resolve_reward_assets(args.assets_root)
        hps = alignment = None
        if "hpsv3" in models:
            from lightx2v_rl.rewards.hpsv3 import HPSv3Scorer

            hps = HPSv3Scorer(
                assets["hpsv3"]["base_model"],
                assets["hpsv3"]["checkpoint"],
                device=args.hps_device,
                frame_chunk_size=args.hps_frame_chunk_size or None,
                attn_implementation=args.attention,
            )
        if "videoalign" in models:
            from lightx2v_rl.rewards.videoalign import VideoAlignReward

            alignment = VideoAlignReward(
                assets["videoalign"]["checkpoint_dir"],
                assets["videoalign"]["base_model"],
                device=args.videoalign_device,
                sample_fps=args.videoalign_sample_fps,
                attention=args.attention,
            )
        engine = ScoreEngine(hps, alignment)
        LOGGER.info("Loading local reward models: %s", ", ".join(models))
        engine.load()
        with make_server(engine, args.host, args.port) as server:
            LOGGER.info("Reward server ready at http://%s:%s: %s", *server.server_address, engine.health())
            server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Reward server stopped")
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(1, f"Reward server startup failed: {exc}\n")


if __name__ == "__main__":
    main()
