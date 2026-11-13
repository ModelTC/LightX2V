#!/usr/bin/env python3
"""RefA2AV：上传参考图/声音，用文本驱动生成，保存 WS 片段或持续推流。"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from websockets.sync.client import connect

# 直接加载生成的协议类，客户端无需导入模型包或安装 GPU 运行时。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lightx2v/server/ws/pb"))
from lightx2v_v1_pb2 import ClientMessage, ServerEvent, ServerMessage, SetPrompt, Start

FPS = 24
OUTPUT_AUDIO_SR = 24000
EXAMPLE_NUM_FRAMES = 957

# Start.prompt：人物、场景、镜头等全局描述；请与参考图保持一致。
GLOBAL_PROMPT = (
    "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her "
    "dark blue off-shoulder lace top against a pale paneled wall. "
    "She is a friendly virtual host with an upbeat, sweet, expressive female voice "
    "matching <Audio 1>. The camera and lighting remain stable."
)

# Start.config_json.idle_prompt：初始 motion，也是临时动作到期后的恢复状态。
# 这里不放台词，避免等待期间持续说话；可直接修改或使用 --idle_prompt 覆盖。
IDLE_PROMPT = (
    "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her "
    "dark blue off-shoulder lace top against the pale paneled wall. "
    "She faces the camera with a relaxed expression, her arms resting naturally. "
    "She breathes gently, keeps her lips closed, and remains silent. No dialogue."
)

# 参考用户提供的 957 帧示例。frame 用于推导每段的持续时间，不作为 WS 精确切换帧号。
# 每段用 SetPrompt 临时生效；等恢复 idle 后才继续，段间也会额外保留 idle。
MOTION_SEQUENCE = [
    {
        "frame": 0,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Lifting her right arm high beside her cheeks and shoulders with the open palm facing the lens, <Subject 1> (S1) swings her hand in three wide, energetic side-to-side arcs while swaying her upper body and beaming at the camera, speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>: <d>[Chinese] 欢迎来到直播间！</d>",
    },
    {
        "frame": 85,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Bringing both forearms up across at upper-chest and collarbone height to form a bold, unmistakable X sign in front of the camera while shaking her head emphatically from side to side, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, warns clearly: <d>[Chinese] 今天状态超好。</d>",
    },
    {
        "frame": 170,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Bringing both hands wide apart and clapping her palms together at upper-chest and collarbone height in three big, spirited rhythmic claps with lively shoulder bounce, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, calls out: <d>[Chinese] 全是实测干货。</d>",
    },
    {
        "frame": 255,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Turning her upper torso noticeably to the side and sweeping her open arm in a wide, dramatic arc toward the display area beside her in the frame while looking back at the lens, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, presents: <d>[Chinese] 千万别踩雷区。</d>",
    },
    {
        "frame": 340,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Lifting both shoulders high in an exaggerated, theatrical shrug while spreading both upturned palms wide apart at upper-chest and collarbone height with raised eyebrows, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, laughs: <d>[Chinese] 这套妆容超美。</d>",
    },
    {
        "frame": 425,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Leaning her upper body noticeably closer to the lens and cupping her right hand wide behind her right ear with an eager, expectant grin, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, asks: <d>[Chinese] 感谢大家支持。</d>",
    },
    {
        "frame": 510,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Extending both hands forward toward the lens at upper-chest and collarbone height with palms facing up and curling all fingers inward twice in a bold 'come closer' beckoning wave, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, invites the viewer: <d>[Chinese] 咱们互动起来。</d>",
    },
    {
        "frame": 595,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Touching her right fingertips lightly beneath her chin and sweeping the whole right arm forward and outward toward the lens in a grand flying-kiss arc without covering her mouth, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, bids farewell: <d>[Chinese] 快戳下方链接。</d>",
    },
    {
        "frame": 680,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Clenching both hands into tight fists high beside her cheeks and shoulders and pumping both arms vigorously up and down twice with lifted shoulders and an excited grin, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, cheers loudly: <d>[Chinese] 马上给你们排。</d>",
    },
    {
        "frame": 765,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Raising both arms high beside her cheeks and shoulders and curving both hands inward above her head to form a large, unmistakable overhead heart arch with a sweet head tilt, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, says warmly: <d>[Chinese] 惊喜福利驾到。</d>",
    },
    {
        "frame": 850,
        "text": "[Shot 1] A static vertical medium close-up frames <Subject 1> indoors in her dark blue off-shoulder lace top against the pale paneled wall. Snapping her right hand up to her right brow in a sharp, high-contrast two-finger salute with squared shoulders and a decisive nod, <Subject 1> (S1), speaking in an upbeat, sweet, and expressive female voice matching <Audio 1>, declares: <d>[Chinese] 咱们下期再见！</d>",
    },
]


def motion_duration(args, motions, index):
    """与 WS 示例共用每段动作的生成预算。"""
    if args.duration is not None:
        return args.duration
    if args.motion:
        return 6.0
    end_frame = motions[index + 1]["frame"] if index + 1 < len(motions) else EXAMPLE_NUM_FRAMES
    # duration 在 protobuf 中是 float32；稍减一点，避免 85/24 被向上取整成 6 chunks。
    return (end_frame - motions[index]["frame"]) / FPS - 1e-4


def receive_message(ws, timeout):
    raw = ws.recv(timeout=timeout)
    if not isinstance(raw, bytes):
        return None
    message = ServerMessage.FromString(raw)
    kind = message.WhichOneof("body")
    if kind == "error":
        error = message.error
        raise RuntimeError(f"request_id={error.request_id} command_id={error.command_id} {error.code}: {error.message}")
    if kind == "event":
        event = message.event
        print(f"{ServerEvent.Type.Name(event.type)} request_id={event.request_id} command_id={event.command_id} {event.message}")
    return message


def stream_sequence(ws, args, request_id):
    """URL 输出不回传 WS 媒体或动作事件，按墙钟间隔循环发送，保持接收错误。"""
    motions = [{"frame": 0, "text": args.motion}] if args.motion else MOTION_SEQUENCE
    started_at = time.monotonic()
    next_motion_at = started_at + args.idle_seconds
    finish_at = started_at + args.seconds if args.seconds is not None else None
    sent = 0
    while True:
        now = time.monotonic()
        if finish_at is not None and now >= finish_at:
            print("已达到 --seconds 推流时长上限")
            return
        if now >= next_motion_at:
            index = sent % len(motions)
            cue = motions[index]
            duration = motion_duration(args, motions, index)
            command_id = f"{request_id}-motion-{sent:06d}"
            command = SetPrompt(prompt=cue["text"], duration=duration, command_id=command_id)
            ws.send(ClientMessage(set_prompt=command).SerializeToString())
            sent += 1
            next_motion_at = time.monotonic() + duration + args.idle_seconds
            print(f"发送第 {sent} 段 motion（序列 {index + 1}/{len(motions)}）duration={duration:.4f}s command_id={command_id}")
        wake_at = min(next_motion_at, finish_at) if finish_at is not None else next_motion_at
        timeout = max(0.001, min(args.timeout, wake_at - time.monotonic()))
        try:
            receive_message(ws, timeout)
        except TimeoutError:
            # 推流时 WS 可以一直没有消息，超时后继续检查下一次发送时间。
            pass


def play_sequence(ws, args, request_id, video_file, audio_file):
    """单个 recv 循环接收音视频，同时依应用/恢复事件顺序发送 motion。"""
    motions = [{"frame": 0, "text": args.motion}] if args.motion else MOTION_SEQUENCE
    audio_size = video_size = 0
    video_started = False
    index = 0
    current_id = None
    state = "idle"
    next_motion_at = args.idle_seconds
    finish_at = None
    deadline = None
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"等待 {current_id} 的 {state} 事件超时")
        message = receive_message(ws, args.timeout)
        if message is None:
            continue
        kind = message.WhichOneof("body")
        if kind == "event":
            event = message.event
            if event.request_id == request_id and event.type == ServerEvent.PROMPT_UPDATE:
                if state == "wait_applied" and event.command_id == current_id:
                    state = "wait_restore"
                    deadline = time.monotonic() + args.timeout + duration
                elif state == "wait_restore" and event.command_id == "" and event.message == args.idle_prompt:
                    print(f"motion {index + 1}/{len(motions)} 已恢复 idle")
                    index += 1
                    current_id = None
                    deadline = None
                    state = "idle"
                    received_seconds = audio_size / (OUTPUT_AUDIO_SR * 2)
                    if index == len(motions):
                        # 恢复事件先于画面播放，继续接收一段尾部，观察 idle 并排出缓冲。
                        finish_at = received_seconds + args.tail_seconds
                    else:
                        next_motion_at = received_seconds + args.idle_seconds
        elif kind == "video_start":
            video_started = True
            print(f"video_start {message.video_start.video_width}x{message.video_start.video_height}")
        elif kind == "video":
            video_file.write(message.video.video_data)
            audio_file.write(message.video.audio_data)
            video_size += len(message.video.video_data)
            audio_size += len(message.video.audio_data)
            print(f"video seq={message.video.sequence} received_audio={audio_size / (OUTPUT_AUDIO_SR * 2):.2f}s")

        received_seconds = audio_size / (OUTPUT_AUDIO_SR * 2)
        if args.seconds is not None and received_seconds >= args.seconds:
            print("已达到 --seconds 录制上限")
            break
        if finish_at is not None and received_seconds >= finish_at:
            break
        if video_started and current_id is None and index < len(motions) and received_seconds >= next_motion_at:
            cue = motions[index]
            duration = motion_duration(args, motions, index)
            current_id = f"motion-{index:02d}-frame-{cue['frame']}"
            state = "wait_applied"
            deadline = time.monotonic() + args.timeout
            command = SetPrompt(prompt=cue["text"], duration=duration, command_id=current_id)
            ws.send(ClientMessage(set_prompt=command).SerializeToString())
            print(f"发送 motion {index + 1}/{len(motions)} reference_frame={cue['frame']} duration={duration:.4f}s command_id={current_id}")
    return audio_size, video_size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/live")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference_audio", required=True, help="完整 PCM WAV 声音参考，仅在 Start 上传一次")
    parser.add_argument("--output", default="save_results/refa2av_ws.mp4")
    parser.add_argument("--output_url", default="", help="RTMP 或 HTTP(S) WHIP 推流地址；指定后循环发送 motion，Ctrl-C 停止，不保存本地文件")
    parser.add_argument("--prompt", default=GLOBAL_PROMPT, help="Start 的全局人物/场景描述")
    parser.add_argument("--idle_prompt", default=IDLE_PROMPT, help="初始及每段动作结束后的静默 motion")
    parser.add_argument("--motion", help="可选：使用单条 motion；不指定时使用完整 11 段序列，URL 模式会循环发送")
    parser.add_argument("--duration", type=float, help="可选：统一覆盖每段生成预算；必须 > 0，以便自动恢复 idle")
    parser.add_argument("--seconds", type=float, help="可选时长上限：WS 模式按收到的音频计时，URL 模式按墙钟计时；URL 默认持续推流")
    parser.add_argument("--idle_seconds", type=float, default=1, help="启动及段间的等待秒数；WS 模式按音频计时，URL 模式按墙钟计时")
    parser.add_argument("--tail_seconds", type=float, default=3, help="最后恢复 idle 后继续接收多少秒音频")
    parser.add_argument("--timeout", type=float, default=120, help="网络接收及每个控制阶段的超时，墙钟秒")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--aspect_ratio", default="9:16")
    args = parser.parse_args(argv)
    if args.output_url and not args.output_url.startswith(("http://", "https://", "rtmp://")):
        parser.error("output_url 必须是 RTMP 地址或 HTTP(S) WHIP 发布地址")
    if (args.seconds is not None and args.seconds <= 0) or (args.duration is not None and args.duration <= 0):
        parser.error("seconds、duration 必须 > 0")
    if args.idle_seconds < 0 or args.tail_seconds < 0 or args.timeout <= 0:
        parser.error("idle_seconds、tail_seconds 必须 >= 0；timeout 必须 > 0")
    image = Path(args.image)
    request_id = "refa2av-" + uuid4().hex
    config = {"resolution": args.resolution, "idle_prompt": args.idle_prompt}
    if args.output_url:
        config["output_url"] = args.output_url
    start = Start(
        request_id=request_id,
        image_fromat=image.suffix.lstrip("."),
        image_data=[image.read_bytes()],
        audio_data=[Path(args.reference_audio).read_bytes()],
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        chunk_frames=8,
        config_json=config,
    )
    # 不发送 audio_start/audio/audio_end：声音参考只在 Start 中上传一次。
    if args.output_url:
        try:
            with connect(args.url, max_size=None) as ws:
                ws.send(ClientMessage(start=start).SerializeToString())
                print(f"Start request_id={request_id}; URL 推流模式，循环发送 motion，Ctrl-C 停止")
                stream_sequence(ws, args, request_id)
        except KeyboardInterrupt:
            print("停止推流并关闭 WS 会话")
        return

    # 服务端必须使用 --task refa2av；默认输出为 24 FPS、24 kHz 单声道 PCM。
    # 临时文件放在输出目录，避免依赖系统 /tmp 空间。
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="refa2av_ws_", dir=output.parent) as tmp:
        video_path, audio_path = Path(tmp) / "video.h264", Path(tmp) / "audio.pcm"
        with video_path.open("wb") as video_file, audio_path.open("wb") as audio_file:
            try:
                with connect(args.url, max_size=None) as ws:
                    ws.send(ClientMessage(start=start).SerializeToString())
                    print(f"Start request_id={request_id}; motion 初始状态为 idle_prompt")
                    audio_size, video_size = play_sequence(ws, args, request_id, video_file, audio_file)
            except KeyboardInterrupt:
                print("停止接收并保存已收到的片段")
                audio_size, video_size = audio_file.tell(), video_file.tell()
        if not video_size or not audio_size:
            raise SystemExit("没有收到可保存的音视频")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(OUTPUT_AUDIO_SR),
                "-ac",
                "1",
                "-i",
                str(audio_path),
                "-r",
                str(FPS),
                "-f",
                "h264",
                "-i",
                str(video_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(output),
            ],
            check=True,
        )
        print(f"已保存 {output}")


if __name__ == "__main__":
    main()
