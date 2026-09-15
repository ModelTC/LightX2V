import argparse
import time

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.workload import (
    DisaggLoadShape,
    build_payload,
    current_stage,
    load_base_config,
    load_stage_specs,
    send_workload_end_signal,
    start_workload_clock,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dynamic disagg workload user and push configs to Controller")
    parser.add_argument("--controller_host", type=str, default="127.0.0.1")
    parser.add_argument("--controller_request_port", type=int, default=REQUEST_POLLING_PORT - 2)
    parser.add_argument("--max_requests", type=int, default=0, help="0 means no hard cap")
    parser.add_argument("--sleep_min_ms", type=float, default=5.0, help="minimum loop sleep in ms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str)
    parser.add_argument("--image_path", type=str)
    return parser


def main():
    args = _build_parser().parse_args()

    req_mgr = ReqManager()
    stages = load_stage_specs()
    base_config = load_base_config()
    request_data = {"seed": args.seed, "prompt": args.prompt, "negative_prompt": args.negative_prompt, "image_path": args.image_path}
    base_config.update({key: value for key, value in request_data.items() if value is not None})
    shape = DisaggLoadShape()

    start_workload_clock()

    sent = 0
    last_tick_ts = 0.0

    while True:
        tick = shape.tick()
        if tick is None:
            break

        _, spawn_rate = tick
        spawn_rate = max(float(spawn_rate), 0.1)

        stage = current_stage(stages)
        payload = build_payload(base_config, stage, sent)
        req_mgr.send(args.controller_host, args.controller_request_port, payload)
        sent += 1

        now = time.time()
        if now - last_tick_ts >= 1.0:
            print(f"stage={stage.name} spawn_rate={spawn_rate:.3f} req/s sent={sent}")
            last_tick_ts = now

        if args.max_requests > 0 and sent >= args.max_requests:
            break

        time.sleep(max(1.0 / spawn_rate, args.sleep_min_ms / 1000.0))

    send_workload_end_signal()
    print(f"workload finished: sent={sent}, end signal sent")


if __name__ == "__main__":
    main()
