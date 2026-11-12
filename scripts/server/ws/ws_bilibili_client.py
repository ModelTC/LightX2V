#!/usr/bin/env python3
"""通过 Start 启动 B 站直播输入，并保持 WebSocket 会话。"""

import argparse
import json
import os
from pathlib import Path

from websockets.sync.client import connect

from lightx2v.server.ws.protocol import ClientMessage, ServerMessage, Start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/live")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_url", default=os.getenv("BILIBILI_RTMP_URL", ""))
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--aspect_ratio", default="9:16")
    parser.add_argument("--prompt", default="A person speaks naturally to the camera, with occasional clear gestures.")
    parser.add_argument("--prompt_file", help="UTF-8 file containing the H3 visual prompt")
    parser.add_argument("--config_json", help="Optional JSON file with additional Start.config_json settings")
    args = parser.parse_args()
    if not args.output_url.startswith("rtmp://"):
        parser.error("--output_url or BILIBILI_RTMP_URL must be a full rtmp:// URL")
    # 此 config 通过 Start.config_json 下发，服务端将其作为 stream_config 使用。
    # LLM/TTS 的 API Key 在服务端配置，不通过客户端传递。
    config = {
        # 主播对话模型在这里改；当前示例通过 SenseNova 调用 DeepSeek。
        # 优先于服务端 SENSENOVA_MODEL；删除此项则使用服务端设置。
        "llm_model": "sensenova-6.8-flash-lite",
        "silence_duration": 60,
        "chat_history_max_length": 20,
        "event_batch_seconds": 2,
        "max_dm": 100,
        # 旧版补边示例（已注释）：扩大画布，在画面左侧添加说明区域。
        # 切回补边时，用这段配置替换下面的 overlay 配置。
        # "padding_config": {
        #     "mode": "padding",
        #     "target_ratio": -1.7333,
        #     "start_texts": [
        #         "我是一个AI虚拟主播",
        #         "--------------------------------",
        #         "【聊天问答】发送弹幕与我互动",
        #         "【更换装扮】如：主播戴个帽子",
        #         "【更换背景】如：换个森林背景",
        #         "【点播歌曲】即将支持",
        #         "--------------------------------",
        #         "视频生成项目：",
        #         "github.com/ModelTC/LightX2V",
        #         "视频生成网站：",
        #         "x2v.light-ai.top",
        #     ],
        #     "font_size": 20,
        #     "spacing_line": 10,
        # },
        # 内嵌文字示例（当前启用）：推流前叠加半透明说明框，输出分辨率不变。
        # 内嵌文字也复用 padding_config，通过 mode="overlay" 区分；不使用 target_ratio。
        "padding_config": {
            "mode": "overlay",
            # 说明框左上角坐标，单位为原视频画面的像素。
            "x": 0,
            "y": 0,
            # 宽度不足时自动换行；省略 height 时按行数自动计算高度。
            "width": 272,
            # "height": 380,  # 可选：固定说明框高度，超出高度的文字行会截断。
            # 仅控制背景不透明度：0 为透明，1 为不透明；文字保持不透明。
            "bg_opacity": 0.55,
            "bg_color": [0, 0, 0],
            "text_color": [255, 255, 255],
            # 每个列表项单独起一行，也可在字符串内用 \n 分行。
            "start_texts": [
                # "我是AI虚拟主播",
                "发送弹幕参与互动",
                "可换装、换背景、做动作",
            ],
            "font_size": 22,
            "spacing_line": 10,
            "start_x": 5,  # 说明框内部的左右边距。
            "start_y": 5,  # 说明框内部的上下边距。
            # 使用容器内的中文字体；Dockerfile_ws_deploy 已包含该字体的安装配置。
            # "font_path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        },
    }
    if args.config_json:
        overrides = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            parser.error("--config_json must contain a JSON object")
        config.update(overrides)
    config.update({"input_url": "https://bilibili", "output_url": args.output_url, "resolution": args.resolution})
    # 协议字段是 map<string, string>；数值和嵌套对象统一编码成字符串。
    config = {key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) for key, value in config.items()}
    image = Path(args.image)
    start = Start(
        request_id="bilibili-live",
        audio_format="pcm_s16le/16000/1",
        image_fromat=image.suffix.lstrip("."),
        image_data=[image.read_bytes()],
        prompt=Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt,
        aspect_ratio=args.aspect_ratio,
        config_json=config,
    )
    try:
        with connect(args.url, max_size=None) as websocket:
            websocket.send(ClientMessage(start=start).serialize())
            print("已发送 Start。音频由服务端生成，请保持连接；按 Ctrl-C 结束直播会话。")
            for raw in websocket:
                if not isinstance(raw, bytes):
                    continue
                message = ServerMessage.parse(raw)
                if message.which() == "error":
                    raise SystemExit("%s: %s" % (message.error.code, message.error.message))
    except KeyboardInterrupt:
        print("直播会话已停止。")


if __name__ == "__main__":
    main()
