# -*- coding: utf-8 -*-

import asyncio
import io
import json
import os
import struct
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Optional

import websockets
from loguru import logger
from pydub import AudioSegment


# Protocol definitions (from podcasts_protocols)
class MsgType(IntEnum):
    """Message type enumeration"""

    Invalid = 0
    FullClientRequest = 0b1
    AudioOnlyClient = 0b10
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111
    ServerACK = AudioOnlyServer


class MsgTypeFlagBits(IntEnum):
    """Message type flag bits"""

    NoSeq = 0
    PositiveSeq = 0b1
    LastNoSeq = 0b10
    NegativeSeq = 0b11
    WithEvent = 0b100


class VersionBits(IntEnum):
    """Version bits"""

    Version1 = 1


class HeaderSizeBits(IntEnum):
    """Header size bits"""

    HeaderSize4 = 1
    HeaderSize8 = 2
    HeaderSize12 = 3
    HeaderSize16 = 4


class SerializationBits(IntEnum):
    """Serialization method bits"""

    Raw = 0
    JSON = 0b1
    Thrift = 0b11
    Custom = 0b1111


class CompressionBits(IntEnum):
    """Compression method bits"""

    None_ = 0
    Gzip = 0b1
    Custom = 0b1111


class EventType(IntEnum):
    """Event type enumeration"""

    None_ = 0
    StartConnection = 1
    StartTask = 1
    FinishConnection = 2
    FinishTask = 2
    ConnectionStarted = 50
    TaskStarted = 50
    ConnectionFailed = 51
    TaskFailed = 51
    ConnectionFinished = 52
    TaskFinished = 52
    StartSession = 100
    CancelSession = 101
    FinishSession = 102
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    UsageResponse = 154
    ChargeData = 154
    TaskRequest = 200
    UpdateConfig = 201
    AudioMuted = 250
    SayHello = 300
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359
    PodcastRoundStart = 360
    PodcastRoundResponse = 361
    PodcastRoundEnd = 362
    PodcastEnd = 363


@dataclass
class Message:
    """Message object"""

    version: VersionBits = VersionBits.Version1
    header_size: HeaderSizeBits = HeaderSizeBits.HeaderSize4
    type: MsgType = MsgType.Invalid
    flag: MsgTypeFlagBits = MsgTypeFlagBits.NoSeq
    serialization: SerializationBits = SerializationBits.JSON
    compression: CompressionBits = CompressionBits.None_
    event: EventType = EventType.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0
    payload: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        """Create message object from bytes"""
        if len(data) < 3:
            raise ValueError(f"Data too short: expected at least 3 bytes, got {len(data)}")
        type_and_flag = data[1]
        msg_type = MsgType(type_and_flag >> 4)
        flag = MsgTypeFlagBits(type_and_flag & 0b00001111)
        msg = cls(type=msg_type, flag=flag)
        msg.unmarshal(data)
        return msg

    def marshal(self) -> bytes:
        """Serialize message to bytes"""
        buffer = io.BytesIO()
        header = [
            (self.version << 4) | self.header_size,
            (self.type << 4) | self.flag,
            (self.serialization << 4) | self.compression,
        ]
        header_size = 4 * self.header_size
        if padding := header_size - len(header):
            header.extend([0] * padding)
        buffer.write(bytes(header))
        writers = self._get_writers()
        for writer in writers:
            writer(buffer)
        return buffer.getvalue()

    def unmarshal(self, data: bytes) -> None:
        """Deserialize message from bytes"""
        buffer = io.BytesIO(data)
        version_and_header_size = buffer.read(1)[0]
        self.version = VersionBits(version_and_header_size >> 4)
        self.header_size = HeaderSizeBits(version_and_header_size & 0b00001111)
        buffer.read(1)
        serialization_compression = buffer.read(1)[0]
        self.serialization = SerializationBits(serialization_compression >> 4)
        self.compression = CompressionBits(serialization_compression & 0b00001111)
        header_size = 4 * self.header_size
        read_size = 3
        if padding_size := header_size - read_size:
            buffer.read(padding_size)
        readers = self._get_readers()
        for reader in readers:
            reader(buffer)
        remaining = buffer.read()
        if remaining:
            raise ValueError(f"Unexpected data after message: {remaining}")

    def _get_writers(self) -> List[Callable[[io.BytesIO], None]]:
        """Get list of writer functions"""
        writers = []
        if self.flag == MsgTypeFlagBits.WithEvent:
            writers.extend([self._write_event, self._write_session_id])
        if self.type in [MsgType.FullClientRequest, MsgType.FullServerResponse, MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                writers.append(self._write_sequence)
        elif self.type == MsgType.Error:
            writers.append(self._write_error_code)
        else:
            raise ValueError(f"Unsupported message type: {self.type}")
        writers.append(self._write_payload)
        return writers

    def _get_readers(self) -> List[Callable[[io.BytesIO], None]]:
        """Get list of reader functions"""
        readers = []
        if self.type in [MsgType.FullClientRequest, MsgType.FullServerResponse, MsgType.FrontEndResultServer, MsgType.AudioOnlyClient, MsgType.AudioOnlyServer]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                readers.append(self._read_sequence)
        elif self.type == MsgType.Error:
            readers.append(self._read_error_code)
        if self.flag == MsgTypeFlagBits.WithEvent:
            readers.extend([self._read_event, self._read_session_id, self._read_connect_id])
        readers.append(self._read_payload)
        return readers

    def _write_event(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">i", self.event))

    def _write_session_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.StartConnection, EventType.FinishConnection, EventType.ConnectionStarted, EventType.ConnectionFailed]:
            return
        session_id_bytes = self.session_id.encode("utf-8")
        size = len(session_id_bytes)
        if size > 0xFFFFFFFF:
            raise ValueError(f"Session ID size ({size}) exceeds max(uint32)")
        buffer.write(struct.pack(">I", size))
        if size > 0:
            buffer.write(session_id_bytes)

    def _write_sequence(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">i", self.sequence))

    def _write_error_code(self, buffer: io.BytesIO) -> None:
        buffer.write(struct.pack(">I", self.error_code))

    def _write_payload(self, buffer: io.BytesIO) -> None:
        size = len(self.payload)
        if size > 0xFFFFFFFF:
            raise ValueError(f"Payload size ({size}) exceeds max(uint32)")
        buffer.write(struct.pack(">I", size))
        buffer.write(self.payload)

    def _read_event(self, buffer: io.BytesIO) -> None:
        event_bytes = buffer.read(4)
        if event_bytes:
            self.event = EventType(struct.unpack(">i", event_bytes)[0])

    def _read_session_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.StartConnection, EventType.FinishConnection, EventType.ConnectionStarted, EventType.ConnectionFailed, EventType.ConnectionFinished]:
            return
        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                session_id_bytes = buffer.read(size)
                if len(session_id_bytes) == size:
                    self.session_id = session_id_bytes.decode("utf-8")

    def _read_connect_id(self, buffer: io.BytesIO) -> None:
        if self.event in [EventType.ConnectionStarted, EventType.ConnectionFailed, EventType.ConnectionFinished]:
            size_bytes = buffer.read(4)
            if size_bytes:
                size = struct.unpack(">I", size_bytes)[0]
                if size > 0:
                    self.connect_id = buffer.read(size).decode("utf-8")

    def _read_sequence(self, buffer: io.BytesIO) -> None:
        sequence_bytes = buffer.read(4)
        if sequence_bytes:
            self.sequence = struct.unpack(">i", sequence_bytes)[0]

    def _read_error_code(self, buffer: io.BytesIO) -> None:
        error_code_bytes = buffer.read(4)
        if error_code_bytes:
            self.error_code = struct.unpack(">I", error_code_bytes)[0]

    def _read_payload(self, buffer: io.BytesIO) -> None:
        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                self.payload = buffer.read(size)


async def receive_message(websocket: websockets.WebSocketClientProtocol) -> Message:
    """Receive message from websocket"""
    try:
        data = await websocket.recv()
        if isinstance(data, str):
            raise ValueError(f"Unexpected text message: {data}")
        elif isinstance(data, bytes):
            msg = Message.from_bytes(data)
            # logger.debug(f"Received: {msg}")
            return msg
        else:
            raise ValueError(f"Unexpected message type: {type(data)}")
    except Exception as e:
        logger.error(f"Failed to receive message: {e}")
        raise


async def wait_for_event(websocket: websockets.WebSocketClientProtocol, msg_type: MsgType, event_type: EventType) -> Message:
    """Wait for specific event"""
    while True:
        msg = await receive_message(websocket)
        if msg.type != msg_type or msg.event != event_type:
            raise ValueError(f"Unexpected message: {msg}")
        if msg.type == msg_type and msg.event == event_type:
            return msg


async def start_connection(websocket: websockets.WebSocketClientProtocol) -> None:
    """Start connection"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.StartConnection
    msg.payload = b"{}"
    logger.debug(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def finish_connection(websocket: websockets.WebSocketClientProtocol) -> None:
    """Finish connection"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.FinishConnection
    msg.payload = b"{}"
    logger.debug(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def start_session(websocket: websockets.WebSocketClientProtocol, payload: bytes, session_id: str) -> None:
    """Start session"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.StartSession
    msg.session_id = session_id
    msg.payload = payload
    logger.debug(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def finish_session(websocket: websockets.WebSocketClientProtocol, session_id: str) -> None:
    """Finish session"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.FinishSession
    msg.session_id = session_id
    msg.payload = b"{}"
    logger.debug(f"Sending: {msg}")
    await websocket.send(msg.marshal())


class PodcastRoundPostProcessor:
    def __init__(self, session_id, data_manager):
        self.session_id = session_id
        self.data_manager = data_manager

        self.temp_merged_audio_name = "merged_audio.mp3"
        self.output_merged_audio_name = f"{session_id}-merged_audio.mp3"
        self.subtitle_timestamps = []  # 记录字幕时间戳
        self.current_audio_duration = 0.0  # 当前音频时长
        self.merged_audio = None  # 用于存储合并的音频对象
        self.merged_audio_bytes = None

    async def init(self):
        if self.data_manager:
            await self.data_manager.create_podcast_temp_session_dir(self.session_id)

    async def postprocess_round(self, current_round, voice, audio, podcast_texts):
        text = ""
        if podcast_texts:
            text = podcast_texts[-1].get("text", "")
        logger.debug(f"Processing round: {current_round}, voice: {voice}, text: {text}, audio: {len(audio)} bytes")

        new_segment = AudioSegment.from_mp3(io.BytesIO(bytes(audio)))
        round_duration = len(new_segment) / 1000.0

        if self.merged_audio is None:
            self.merged_audio = new_segment
        else:
            self.merged_audio = self.merged_audio + new_segment

        # 保存合并后的音频到临时文件（用于前端实时访问）
        merged_io = io.BytesIO()
        self.merged_audio.export(merged_io, format="mp3")
        self.merged_audio_bytes = merged_io.getvalue()
        if self.data_manager:
            await self.data_manager.save_podcast_temp_session_file(self.session_id, self.temp_merged_audio_name, self.merged_audio_bytes)
        merged_file_size = len(self.merged_audio_bytes)

        # 记录字幕时间戳
        self.subtitle_timestamps.append(
            {
                "start": self.current_audio_duration,
                "end": self.current_audio_duration + round_duration,
                "text": text,
                "speaker": voice,
            }
        )
        self.current_audio_duration += round_duration
        logger.debug(f"Merged audio updated: {merged_file_size} bytes, duration: {self.current_audio_duration:.2f}s")

        return {
            "url": f"/api/v1/podcast/audio?session_id={self.session_id}&filename={self.temp_merged_audio_name}",
            "size": merged_file_size,
            "duration": self.current_audio_duration,
            "round": current_round,
            "text": text,
            "speaker": voice,
        }

    async def postprocess_final(self):
        if self.data_manager:
            await self.data_manager.save_podcast_output_file(self.output_merged_audio_name, self.merged_audio_bytes)
        return {
            "subtitles": self.subtitle_timestamps,
            "audio_name": self.output_merged_audio_name,
        }

    async def cleanup(self):
        if self.data_manager:
            await self.data_manager.clear_podcast_temp_session_dir(self.session_id)
            self.data_manager = None


class VolcEnginePodcastClient:
    """
    VolcEngine Podcast客户端

    支持多种播客类型:
        - action=0: 文本转播客
        - action=3: NLP文本转播客
        - action=4: 提示词生成播客
    """

    def __init__(self):
        self.endpoint = "wss://openspeech.bytedance.com/api/v3/sami/podcasttts"
        self.appid = os.getenv("VOLCENGINE_PODCAST_APPID")
        self.access_token = os.getenv("VOLCENGINE_PODCAST_ACCESS_TOKEN")
        self.app_key = "aGjiRDfUWi"
        self.proxy = os.getenv("HTTPS_PROXY", None)
        if self.proxy:
            logger.info(f"volcengine podcast use proxy: {self.proxy}")

    async def podcast_request(
        self,
        session_id: str,
        data_manager=None,
        text: str = "",
        input_url: str = "",
        prompt_text: str = "",
        nlp_texts: str = "",
        action: int = 0,
        resource_id: str = "volc.service_type.10050",
        encoding: str = "mp3",
        input_id: str = "test_podcast",
        speaker_info: str = '{"random_order":false}',
        use_head_music: bool = False,
        use_tail_music: bool = False,
        only_nlp_text: bool = False,
        return_audio_url: bool = False,
        skip_round_audio_save: bool = False,
        on_round_complete: Optional[Callable] = None,
    ):
        """
        执行播客请求

        Args:
            text: 输入文本 (action=0时使用)
            input_url: Web URL或文件URL (action=0时使用)
            prompt_text: 提示词文本 (action=4时必须)
            nlp_texts: NLP文本 (action=3时必须)
            action: 播客类型 (0/3/4)
            resource_id: 音频资源ID
            encoding: 音频格式 (mp3/wav)
            input_id: 唯一输入标识
            speaker_info: 播客说话人信息
            use_head_music: 是否使用开头音乐
            use_tail_music: 是否使用结尾音乐
            only_nlp_text: 是否只返回播客文本
            return_audio_url: 是否返回音频URL
            skip_round_audio_save: 是否跳过单轮音频保存
            output_dir: 输出目录
            on_round_complete: 轮次完成回调函数
        """
        if not self.appid or not self.access_token:
            logger.error("APP ID or Access Key is required")
            return None, None

        headers = {
            "X-Api-App-Id": self.appid,
            "X-Api-App-Key": self.app_key,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        is_podcast_round_end = True
        audio_received = False
        last_round_id = -1
        task_id = ""
        websocket = None
        retry_num = 5
        audio = bytearray()
        voice = ""
        current_round = 0
        podcast_texts = []

        post_processor = PodcastRoundPostProcessor(session_id, data_manager)
        await post_processor.init()

        try:
            while retry_num > 0:
                # 建立WebSocket连接
                websocket = await websockets.connect(self.endpoint, additional_headers=headers)
                logger.debug(f"WebSocket connected: {websocket.response.headers}")

                # 构建请求参数
                if input_url:
                    req_params = {
                        "input_id": input_id,
                        "nlp_texts": json.loads(nlp_texts) if nlp_texts else None,
                        "prompt_text": prompt_text,
                        "action": action,
                        "use_head_music": use_head_music,
                        "use_tail_music": use_tail_music,
                        "input_info": {
                            "input_url": input_url,
                            "return_audio_url": return_audio_url,
                            "only_nlp_text": only_nlp_text,
                        },
                        "speaker_info": json.loads(speaker_info) if speaker_info else None,
                        "audio_config": {"format": encoding, "sample_rate": 24000, "speech_rate": 0},
                    }
                else:
                    req_params = {
                        "input_id": input_id,
                        "input_text": text,
                        "nlp_texts": json.loads(nlp_texts) if nlp_texts else None,
                        "prompt_text": prompt_text,
                        "action": action,
                        "use_head_music": use_head_music,
                        "use_tail_music": use_tail_music,
                        "input_info": {
                            "input_url": input_url,
                            "return_audio_url": return_audio_url,
                            "only_nlp_text": only_nlp_text,
                        },
                        "speaker_info": json.loads(speaker_info) if speaker_info else None,
                        "audio_config": {"format": encoding, "sample_rate": 24000, "speech_rate": 0},
                    }

                logger.debug(f"Request params: {json.dumps(req_params, indent=2, ensure_ascii=False)}")

                if not is_podcast_round_end:
                    req_params["retry_info"] = {"retry_task_id": task_id, "last_finished_round_id": last_round_id}

                # Start connection
                await start_connection(websocket)
                await wait_for_event(websocket, MsgType.FullServerResponse, EventType.ConnectionStarted)

                session_id = str(uuid.uuid4())
                if not task_id:
                    task_id = session_id

                # Start session
                await start_session(websocket, json.dumps(req_params).encode(), session_id)
                await wait_for_event(websocket, MsgType.FullServerResponse, EventType.SessionStarted)

                # Finish session
                await finish_session(websocket, session_id)

                while True:
                    msg = await receive_message(websocket)

                    # 音频数据块
                    if msg.type == MsgType.AudioOnlyServer and msg.event == EventType.PodcastRoundResponse:
                        if not audio_received and audio:
                            audio_received = True
                        audio.extend(msg.payload)

                    # 错误信息
                    elif msg.type == MsgType.Error:
                        raise RuntimeError(f"Server error: {msg.payload.decode()}")

                    elif msg.type == MsgType.FullServerResponse:
                        # 播客 round 开始
                        if msg.event == EventType.PodcastRoundStart:
                            data = json.loads(msg.payload.decode())
                            if data.get("text"):
                                filtered_payload = {"text": data.get("text"), "speaker": data.get("speaker")}
                                podcast_texts.append(filtered_payload)
                            voice = data.get("speaker")
                            current_round = data.get("round_id")
                            if current_round == -1:
                                voice = "head_music"
                            if current_round == 9999:
                                voice = "tail_music"
                            is_podcast_round_end = False
                            logger.debug(f"New round started: {data}")

                        # 播客 round 结束
                        if msg.event == EventType.PodcastRoundEnd:
                            data = json.loads(msg.payload.decode())
                            logger.debug(f"Podcast round end: {data}")
                            if data.get("is_error"):
                                break
                            is_podcast_round_end = True
                            last_round_id = current_round
                            if audio:
                                round_info = await post_processor.postprocess_round(current_round, voice, audio, podcast_texts)
                                if on_round_complete:
                                    await on_round_complete(round_info)
                                audio.clear()

                        # 播客结束
                        if msg.event == EventType.PodcastEnd:
                            data = json.loads(msg.payload.decode())
                            logger.info(f"Podcast end: {data}")

                    # 会话结束
                    if msg.event == EventType.SessionFinished:
                        break

                if not audio_received and not only_nlp_text:
                    raise RuntimeError("No audio data received")

                # 保持连接
                await finish_connection(websocket)
                await wait_for_event(websocket, MsgType.FullServerResponse, EventType.ConnectionFinished)

                # 播客结束, 保存最终音频文件
                if is_podcast_round_end:
                    podcast_info = await post_processor.postprocess_final()
                    return podcast_info
                else:
                    logger.error(f"Current podcast not finished, resuming from round {last_round_id}")
                    retry_num -= 1
                    await asyncio.sleep(1)
                    if websocket:
                        await websocket.close()

        finally:
            await post_processor.cleanup()
            if websocket:
                await websocket.close()
        return None


async def test(args):
    """
    Podcast测试函数

    Args:
        args: dict, 包含所有podcast参数
    """
    client = VolcEnginePodcastClient()

    # 设置默认参数
    params = {
        "text": "",
        "input_url": "https://zhuanlan.zhihu.com/p/607822576",
        "prompt_text": "",
        "nlp_texts": "",
        "action": 0,
        "resource_id": "volc.service_type.10050",
        "encoding": "mp3",
        "input_id": "test_podcast",
        "speaker_info": '{"random_order":false}',
        "use_head_music": False,
        "use_tail_music": False,
        "only_nlp_text": False,
        "return_audio_url": True,
        "skip_round_audio_save": False,
        "output_dir": "output",
    }

    # 覆盖默认参数
    if args:
        params.update(args)

    await client.podcast_request(**params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="", help="Input text Use when action in [0]")
    parser.add_argument("--input_url", default="", help="Web url or file url Use when action in [0]")
    parser.add_argument("--prompt_text", default="", help="Input Prompt Text must not empty when action in [4]")
    parser.add_argument("--nlp_texts", default="", help="Input NLP Texts must not empty when action in [3]")
    parser.add_argument("--resource_id", default="volc.service_type.10050", help="Audio Resource ID")
    parser.add_argument("--encoding", default="mp3", choices=["mp3", "wav"], help="Audio format")
    parser.add_argument("--input_id", default="test_podcast", help="Unique input identifier")
    parser.add_argument("--speaker_info", default='{"random_order":false}', help="Podcast Speaker Info")
    parser.add_argument("--use_head_music", default=False, action="store_true", help="Enable head music")
    parser.add_argument("--use_tail_music", default=False, action="store_true", help="Enable tail music")
    parser.add_argument("--only_nlp_text", default=False, action="store_true", help="Enable only podcast text when action in [0, 4]")
    parser.add_argument("--return_audio_url", default=False, action="store_true", help="Enable return audio url that can download")
    parser.add_argument("--action", default=0, type=int, choices=[0, 3, 4], help="different podcast type")
    parser.add_argument("--skip_round_audio_save", default=False, action="store_true", help="skip round audio save")
    parser.add_argument("--output_dir", default="output", help="Output directory")

    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None and not (isinstance(v, bool) and not v)}

    asyncio.run(test(kwargs))
