# -*- coding: utf-8 -*-

import asyncio
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Optional, Tuple

import aiohttp
import numpy as np
import soundfile as sf
from aiohttp import ClientWebSocketResponse

# Protobuf imports
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from loguru import logger

# ============================================================================
# Generated protocol buffer code (from tts.proto)
# ============================================================================
_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\ttts.proto\x12\x03tts"\x8a\x01\n\rSubtitleEntry\x12\x15\n\rstart_time_ms\x18\x01 \x01(\r\x12\x13\n\x0b\x65nd_time_ms\x18\x02 \x01(\r\x12\x0f\n\x07speaker\x18\x03 \x01(\t\x12\r\n\x05style\x18\x04 \x01(\t\x12\x1f\n\x08language\x18\x05 \x01(\x0e\x32\r.tts.Language\x12\x0c\n\x04text\x18\x06 \x01(\t"\x88\x01\n\nAudioChunk\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\x12\x17\n\x0f\x61udio_chunk_seq\x18\x02 \x01(\x05\x12\x15\n\ris_last_chunk\x18\x03 \x01(\x08\x12\x0c\n\x04text\x18\x04 \x01(\t\x12\x14\n\x0c\x61udio_format\x18\x05 \x01(\t\x12\x12\n\ndisable_ns\x18\x06 \x01(\x08"\x84\x04\n\nTtsRequest\x12-\n\x0cmessage_type\x18\x01 \x01(\x0e\x32\x17.tts.RequestMessageType\x12\x0e\n\x06\x61pp_id\x18\x02 \x01(\t\x12\x15\n\rapp_signature\x18\x03 \x01(\t\x12\x0c\n\x04text\x18\x04 \x01(\t\x12\x16\n\x0etext_chunk_seq\x18\x05 \x01(\x05\x12\x1a\n\x12is_last_text_chunk\x18\x06 \x01(\x08\x12 \n\ttext_type\x18\x07 \x01(\x0e\x32\r.tts.TextType\x12\x0f\n\x07speaker\x18\x08 \x01(\t\x12\x1f\n\x08language\x18\t \x01(\x0e\x32\r.tts.Language\x12\r\n\x05style\x18\n \x01(\t\x12\r\n\x05speed\x18\x0b \x01(\x02\x12\x0e\n\x06volume\x18\x0c \x01(\x02\x12\r\n\x05pitch\x18\r \x01(\x02\x12\x15\n\rstream_output\x18\x0e \x01(\x08\x12\x19\n\x11\x61udio_sample_rate\x18\x0f \x01(\x05\x12*\n\x0e\x61udio_encoding\x18\x10 \x01(\x0e\x32\x12.tts.AudioEncoding\x12\x18\n\x10output_subtitles\x18\x11 \x01(\x08\x12\x12\n\nsession_id\x18\x12 \x01(\t\x12%\n\x0cupload_audio\x18\x13 \x01(\x0b\x32\x0f.tts.AudioChunk\x12\x1a\n\x12pronunciation_dict\x18\x14 \x03(\t"\xe7\x02\n\x0bTtsResponse\x12$\n\x0bstatus_code\x18\x01 \x01(\x0e\x32\x0f.tts.StatusCode\x12\x14\n\x0c\x65rror_detail\x18\x02 \x01(\t\x12\x14\n\x0ctime_cost_ms\x18\x03 \x01(\r\x12*\n\x0e\x61udio_encoding\x18\x04 \x01(\x0e\x32\x12.tts.AudioEncoding\x12\x17\n\x0f\x61udio_chunk_seq\x18\x05 \x01(\x05\x12\x12\n\naudio_data\x18\x06 \x01(\x0c\x12\x1b\n\x13is_last_audio_chunk\x18\x07 \x01(\x08\x12\x12\n\nsession_id\x18\x08 \x01(\t\x12%\n\tsubtitles\x18\t \x03(\x0b\x32\x12.tts.SubtitleEntry\x12\x0f\n\x07speaker\x18\n \x01(\t\x12\x1a\n\x12request_char_count\x18\x0b \x01(\r\x12(\n\rerror_subcode\x18\x0c \x01(\x0e\x32\x11.tts.ErrorSubCode*\xa9\x01\n\x12RequestMessageType\x12\x1c\n\x18\x43LIENT_SYNTHESIS_REQUEST\x10\x00\x12\x19\n\x15\x43LIENT_FINISH_REQUEST\x10\x01\x12\x1d\n\x19\x43LIENT_UPLOAD_CLONE_AUDIO\x10\x02\x12\x1c\n\x18\x43LIENT_QUERY_CLONE_AUDIO\x10\x03\x12\x1d\n\x19\x43LIENT_DELETE_CLONE_AUDIO\x10\x04*\x1f\n\x08TextType\x12\t\n\x05PLAIN\x10\x00\x12\x08\n\x04SSML\x10\x01*A\n\x08Language\x12\t\n\x05ZH_CN\x10\x00\x12\t\n\x05\x45N_US\x10\x01\x12\x11\n\rZH_CN_SICHUAN\x10\x02\x12\x0c\n\x08ZH_CN_HK\x10\x03**\n\rAudioEncoding\x12\x07\n\x03PCM\x10\x00\x12\x07\n\x03WAV\x10\x01\x12\x07\n\x03MP3\x10\x02*\xa7\x01\n\nStatusCode\x12\x0b\n\x07SUCCESS\x10\x00\x12\t\n\x05\x45RROR\x10\x01\x12\x0b\n\x07TIMEOUT\x10\x02\x12\x13\n\x0fINVALID_REQUEST\x10\x03\x12\x12\n\x0eINTERNAL_ERROR\x10\x04\x12\x18\n\x14UPLOAD_AUDIO_SUCCESS\x10\x05\x12\x17\n\x13QUERY_AUDIO_SUCCESS\x10\x06\x12\x18\n\x14\x44\x45LETE_AUDIO_SUCCESS\x10\x07*\xe1\x02\n\x0c\x45rrorSubCode\x12\x0c\n\x08\x45RR_NONE\x10\x00\x12\x16\n\x12\x45RR_BASE_FILE_READ\x10\x65\x12\x17\n\x13\x45RR_BASE_FILE_WRITE\x10\x66\x12\x1c\n\x18\x45RR_BASE_INVALID_SEQ_NUM\x10g\x12\x1e\n\x1a\x45RR_BASE_SPEAKER_NOT_FOUND\x10h\x12\x14\n\x0f\x45RR_AC_INTERNAL\x10\xc9\x01\x12\x16\n\x11\x45RR_AC_LONG_AUDIO\x10\xca\x01\x12\x15\n\x10\x45RR_AC_LONG_TEXT\x10\xcb\x01\x12\x1f\n\x1a\x45RR_AC_AUDIO_TEXT_MISMATCH\x10\xcc\x01\x12 \n\x1b\x45RR_AC_UNAUTHORIZED_SPEAKER\x10\xcd\x01\x12\x1b\n\x16\x45RR_AC_INVALID_SPEAKER\x10\xce\x01\x12\x17\n\x12\x45RR_AC_SHORT_AUDIO\x10\xcf\x01\x12\x16\n\x11\x45RR_AC_SHORT_TEXT\x10\xd0\x01\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "tts_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._options = None
    _globals["_REQUESTMESSAGETYPE"]._serialized_start = 1180
    _globals["_REQUESTMESSAGETYPE"]._serialized_end = 1349
    _globals["_TEXTTYPE"]._serialized_start = 1351
    _globals["_TEXTTYPE"]._serialized_end = 1382
    _globals["_LANGUAGE"]._serialized_start = 1384
    _globals["_LANGUAGE"]._serialized_end = 1449
    _globals["_AUDIOENCODING"]._serialized_start = 1451
    _globals["_AUDIOENCODING"]._serialized_end = 1493
    _globals["_STATUSCODE"]._serialized_start = 1496
    _globals["_STATUSCODE"]._serialized_end = 1663
    _globals["_ERRORSUBCODE"]._serialized_start = 1666
    _globals["_ERRORSUBCODE"]._serialized_end = 2019
    _globals["_SUBTITLEENTRY"]._serialized_start = 19
    _globals["_SUBTITLEENTRY"]._serialized_end = 157
    _globals["_AUDIOCHUNK"]._serialized_start = 160
    _globals["_AUDIOCHUNK"]._serialized_end = 296
    _globals["_TTSREQUEST"]._serialized_start = 299
    _globals["_TTSREQUEST"]._serialized_end = 815
    _globals["_TTSRESPONSE"]._serialized_start = 818
    _globals["_TTSRESPONSE"]._serialized_end = 1177

# Import protobuf classes for easier access
# These are created by the protobuf builder above and added to _globals
# ============================================================================
# Get protobuf classes from _globals (they are created by the builder)
SubtitleEntry = _globals.get("SubtitleEntry")
AudioChunk = _globals.get("AudioChunk")
TtsRequest = _globals.get("TtsRequest")
TtsResponse = _globals.get("TtsResponse")
RequestMessageType = _globals.get("RequestMessageType")
TextType = _globals.get("TextType")
Language = _globals.get("Language")
AudioEncoding = _globals.get("AudioEncoding")
StatusCode = _globals.get("StatusCode")
ErrorSubCode = _globals.get("ErrorSubCode")

# Verify that all required classes are available
if not all([SubtitleEntry, AudioChunk, TtsRequest, TtsResponse, RequestMessageType, TextType, Language, AudioEncoding, StatusCode, ErrorSubCode]):
    raise RuntimeError("Failed to load protobuf classes. Please check protobuf installation.")
# ============================================================================

# Configuration parameters
RECEIVE_TIMEOUT = 30  # Receive timeout (seconds)

# Language mapping
lang_id2str_mapping = {
    Language.ZH_CN: "ZH_CN",
    Language.ZH_CN_SICHUAN: "ZH_CN_SICHUAN",
    Language.ZH_CN_HK: "ZH_CN_HK",
    Language.EN_US: "EN_US",
}

lang_str2id_mapping = {v: k for k, v in lang_id2str_mapping.items()}

# Audio encoding mapping
codec_id2str_mapping = {
    AudioEncoding.PCM: "pcm",
    AudioEncoding.WAV: "wav",
    AudioEncoding.MP3: "mp3",
}

codec_str2id_mapping = {v: k for k, v in codec_id2str_mapping.items()}


def parse_response(protocol_type: int, data: bytes) -> TtsResponse:
    try:
        response = TtsResponse()
        response.ParseFromString(data)
        return response
    except Exception as e:
        raise ValueError(f"Failed to parse response: {str(e)}")


def create_synthesis_request(
    message_type,
    text: str,
    text_chunk_seq: int = 0,
    is_last_text_chunk: bool = False,
    app_id: str = "",
    app_signature: str = "",
    text_type: TextType = TextType.PLAIN,
    speaker: str = "default",
    language: Language = Language.ZH_CN,
    style: str = "",
    speed: float = 1,
    volume: float = 0,
    pitch: float = 0,
    stream_output: bool = True,
    audio_sample_rate: int = 24000,
    audio_encoding: AudioEncoding = AudioEncoding.PCM,
    output_subtitles: bool = False,
    session_id: str = "",
    upload_data: Optional[AudioChunk] = None,
) -> TtsRequest:
    request = TtsRequest()
    request.message_type = message_type
    request.app_id = app_id
    request.text = text
    request.text_chunk_seq = text_chunk_seq
    request.is_last_text_chunk = is_last_text_chunk
    request.text_type = text_type
    request.speaker = speaker
    request.language = language
    request.style = style
    request.speed = speed
    request.volume = volume
    request.pitch = pitch
    request.stream_output = stream_output
    request.audio_sample_rate = audio_sample_rate
    request.audio_encoding = audio_encoding
    request.output_subtitles = output_subtitles
    request.session_id = session_id

    if upload_data is not None:
        request.upload_audio.CopyFrom(upload_data)

    return request


def serialize_request(request: TtsRequest) -> bytes:
    request_bytes = request.SerializeToString()
    request_length = struct.pack("!I", len(request_bytes))
    full_request = b"\x01" + request_length + request_bytes
    return full_request


async def receive_full_message(websocket: ClientWebSocketResponse) -> Tuple[int, bytes]:
    try:
        # Receive data
        message = await asyncio.wait_for(websocket.receive_bytes(), timeout=RECEIVE_TIMEOUT)

        if len(message) < 5:
            raise ValueError("Invalid response: too short")

        protocol_type = message[0]
        if protocol_type != 0x01:
            raise ValueError("Unsupported protocol type")

        protocol_length = struct.unpack("!I", message[1:5])[0]

        data = message[5:]
        if len(data) != protocol_length:
            logger.info(f"Length error {protocol_length}, got {len(data)}")
            # If data is incomplete, continue receiving
            while len(data) < protocol_length:
                try:
                    chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=RECEIVE_TIMEOUT)
                    if not chunk:
                        raise ValueError("Got disconnected or empty data")
                    data += chunk
                    logger.info(f"Received additional {len(chunk)} bytes, total {len(data)}/{protocol_length}")
                except asyncio.TimeoutError:
                    raise ValueError(f"Timeout while receiving message. Got {len(data)}/{protocol_length} bytes")

        return protocol_type, data

    except asyncio.TimeoutError:
        raise ValueError(f"Response timed out after {RECEIVE_TIMEOUT} seconds")
    except aiohttp.WSServerHandshakeError as e:
        # WebSocket handshake error, may contain error information
        error_msg = f"WebSocket handshake error: {str(e)}"
        if hasattr(e, "message") and e.message:
            error_msg = e.message
        raise ValueError(error_msg)
    except Exception as e:
        error_str = str(e)
        # Check if it's a WebSocket close message error
        if "1009" in error_str:
            raise ValueError("Audio file too large or format not supported. Please use WAV/MP3 audio file (max size limit).")
        elif "1000" in error_str or "WSMsgType" in error_str:
            # WebSocket close message, try to extract error information
            if "1009" in error_str:
                raise ValueError("Message too large. Audio file may be too big or in unsupported format.")
            else:
                raise ValueError(f"WebSocket connection closed: {error_str}")
        raise ValueError(f"Error receiving data: {str(e)}")


class SenseTimeTTSClient:
    """
    SenseTime TTS Client

    Parameter ranges:
        - speed: 0.5~2.0 (1.0 is normal speed)
        - volume: -12~12 dB (0 is normal volume)
        - pitch: -24~24 halftone (0 is normal pitch)
    """

    def __init__(self, url=None, app_id=None, apikey=None):
        self.url = url or os.getenv("SENSETIME_TTS_URL")
        self.app_id = app_id or os.getenv("SENSETIME_APP_ID")
        self.apikey = apikey or os.getenv("SENSETIME_APIKEY")
        if not self.apikey:
            raise ValueError("SENSETIME_APIKEY is not set")
        if not self.app_id:
            raise ValueError("SENSETIME_APP_ID is not set")
        if not self.url:
            raise ValueError("SENSETIME_TTS_URL is not set")

    async def _receive_loop(self, websocket, session_id, params, result_dict):
        """Continuously receive server responses in a loop"""
        is_running = True
        data = b""
        seq = -1
        subtitles = []
        first_latency = None

        try:
            while is_running:
                try:
                    ptype, data_bytes = await receive_full_message(websocket)
                    response = parse_response(ptype, data_bytes)

                    if response.status_code == StatusCode.SUCCESS:
                        chunk_seq = response.audio_chunk_seq
                        is_last_chunk = response.is_last_audio_chunk
                        stream = params.get("stream_output", True)

                        # Check sequence number
                        valid = chunk_seq == seq + 1
                        seq = chunk_seq
                        if not valid:
                            logger.warning(f"Session {session_id} Invalid seq")
                            is_running = False
                            break

                        if chunk_seq == 0:
                            start_time = result_dict.get("start_time")
                            if start_time is not None:
                                first_latency = (time.time() - start_time) * 1000
                                logger.info(f"Session {session_id} stream({int(stream)}) Got first package, cost(ms): {first_latency:.3f}")

                        if response.audio_data:
                            data += response.audio_data

                        logger.info(f"Audio seq:{chunk_seq},is_last:{is_last_chunk} data length: {len(response.audio_data)} bytes")

                        if response.subtitles:
                            for subtitle in response.subtitles:
                                start_time_ms = subtitle.start_time_ms
                                end_time_ms = subtitle.end_time_ms
                                fmt_sub = f"  {subtitle.text} ({start_time_ms}-{end_time_ms}ms)"
                                subtitles.append(fmt_sub)

                        if response.is_last_audio_chunk:
                            start_time = result_dict.get("start_time")
                            whole_cost = time.time() - start_time if start_time else 0
                            if len(data) > 0:
                                sample_rate = params.get("sample_rate", 24000)
                                duration = len(data) / 2 / sample_rate
                                rtf = whole_cost / duration if duration > 0 else 0

                                if len(subtitles) > 0:
                                    joint_sub = "\t".join(subtitles)
                                    logger.info(f"Session {session_id} subtile:{joint_sub}")

                                out_info = f"spk {params.get('speaker', 'default')} "
                                out_info += f"stream {int(stream)} "
                                if first_latency is not None:
                                    out_info += f"latency {first_latency:.3f} ms "
                                out_info += f"cost {whole_cost:.3f} secs "
                                if params.get("audio_format") == "pcm":
                                    out_info += f"duration {duration:.3f} secs "
                                    out_info += f"RTF {rtf:.3f}"

                                logger.info(f"Session {session_id} done, {out_info}")

                            result_dict["audio_data"] = data
                            result_dict["subtitles"] = subtitles
                            result_dict["success"] = True
                            is_running = False
                        elif response.status_code == StatusCode.INTERNAL_ERROR:
                            error_msg = response.error_detail if response.error_detail else "Internal error"
                            logger.error(f"INTERNAL_ERROR in response: {error_msg}")
                            result_dict["error"] = error_msg
                            result_dict["success"] = False
                            is_running = False
                            break
                        elif response.status_code == StatusCode.ERROR:
                            error_msg = response.error_detail if response.error_detail else "Unknown error"
                            logger.error(f"ERROR in response: {error_msg}")
                            result_dict["error"] = error_msg
                            result_dict["success"] = False
                            is_running = False
                            break
                    elif response.status_code == StatusCode.UPLOAD_AUDIO_SUCCESS:
                        if response.speaker == "":
                            logger.error("ERROR: Got none speaker for UPLOAD_AUDIO_SUCCESS")
                            result_dict["error"] = "Got none speaker for UPLOAD_AUDIO_SUCCESS"
                        else:
                            logger.info(f"OK, Got speaker id {response.speaker} session id {response.session_id}")
                            result_dict["speaker"] = response.speaker
                            result_dict["session_id"] = response.session_id
                            result_dict["success"] = True
                        is_running = False
                        break
                    elif response.status_code == StatusCode.QUERY_AUDIO_SUCCESS:
                        logger.info(f"Query speaker {response.speaker} successful")
                        result_dict["speaker"] = response.speaker
                        result_dict["success"] = True
                        is_running = False
                        break
                    elif response.status_code == StatusCode.DELETE_AUDIO_SUCCESS:
                        logger.info(f"Delete speaker {response.speaker} successful")
                        result_dict["success"] = True
                        is_running = False
                        break
                    else:
                        # Handle other error status codes, return error details directly
                        error_msg = response.error_detail if response.error_detail else "Unknown error"
                        logger.error(f"Error in response: {error_msg}")
                        result_dict["error"] = error_msg
                        result_dict["success"] = False
                        is_running = False
                        break
                except asyncio.CancelledError:
                    logger.info("Receive loop cancelled")
                    is_running = False
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")
                    result_dict["error"] = str(e)
                    break
        except Exception as e:
            logger.error(f"Receive loop terminated: {e}")
            result_dict["error"] = str(e)

        logger.info("Exit receive loop.")

    async def tts_request(
        self,
        text,
        speaker="M20",
        style="正常",
        speed=1.0,
        volume=0,
        pitch=0,
        language="ZH_CN",
        output="tts_output.wav",
        sample_rate=24000,
        audio_format="wav",
        stream_output=True,
        output_subtitles=False,
    ):
        """
        Execute TTS request

        Args:
            text: Text to convert
            speaker: Speaker, common values include "M20", "F12", "zhili", "nvguo59", or ID returned by audioclone
            style: Speaker style, common values include "正常" (normal), "高兴" (happy), "愤怒" (angry), etc.
            speed: Speech rate (0.5~2.0, 1.0 is normal speed)
            volume: Volume (-12~12 dB, 0 is normal volume)
            pitch: Pitch (-24~24 halftone, 0 is normal pitch)
            language: Language, options: "ZH_CN", "ZH_CN_SICHUAN", "ZH_CN_HK", "EN_US"
            output: Output file path
            sample_rate: Sample rate, options: 8000, 16000, 24000, 32000, 48000
            audio_format: Audio format, options: "pcm", "wav", "mp3"
            stream_output: Whether to stream output
            output_subtitles: Whether to output subtitles
        """
        # Validate parameter ranges
        if not (0.5 <= speed <= 2.0):
            logger.warning(f"speed {speed} is out of valid range [0.5, 2.0], using default value 1.0")
            speed = 1.0

        if not (-12 <= volume <= 12):
            logger.warning(f"volume {volume} is out of valid range [-12, 12], using default value 0")
            volume = 0

        if not (-24 <= pitch <= 24):
            logger.warning(f"pitch {pitch} is out of valid range [-24, 24], using default value 0")
            pitch = 0

        if language not in lang_str2id_mapping:
            logger.warning(f"language {language} is invalid, using default value ZH_CN")
            language = "ZH_CN"

        if audio_format not in codec_str2id_mapping:
            logger.warning(f"audio_format {audio_format} is invalid, using default value pcm")
            audio_format = "pcm"

        logger.info(f"Connecting to {self.url}...")
        headers = {"apikey": self.apikey} if self.url.startswith("wss:") else None

        result_dict = {"success": False, "audio_data": None, "subtitles": [], "error": None}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.ws_connect(self.url) as websocket:
                    logger.info("WebSocket connection established")

                    session_id = str(uuid.uuid4())
                    params = {
                        "speaker": speaker,
                        "style": style,
                        "speed": speed,
                        "volume": volume,
                        "pitch": pitch,
                        "language": language,
                        "sample_rate": sample_rate,
                        "audio_format": audio_format,
                        "stream_output": stream_output,
                        "output_subtitles": output_subtitles,
                    }

                    # Set start time (before sending request)
                    start_time = time.time()
                    result_dict["start_time"] = start_time

                    # Start receive loop
                    receive_task = asyncio.create_task(self._receive_loop(websocket, session_id, params, result_dict))

                    # Simulate streaming: send character by character
                    for i, chunk in enumerate(text):
                        if not receive_task.done():
                            is_last = i == len(text) - 1
                            request = create_synthesis_request(
                                message_type=RequestMessageType.CLIENT_SYNTHESIS_REQUEST,
                                app_id=self.app_id,
                                text=chunk,
                                text_chunk_seq=i,
                                is_last_text_chunk=is_last,
                                session_id=session_id,
                                speaker=speaker,
                                style=style,
                                speed=speed,
                                output_subtitles=output_subtitles,
                                audio_sample_rate=sample_rate,
                                language=lang_str2id_mapping[language],
                                volume=volume,
                                audio_encoding=codec_str2id_mapping[audio_format],
                                stream_output=stream_output,
                                pitch=pitch,
                            )
                            full_request = serialize_request(request)
                            await websocket.send_bytes(full_request)

                    # Wait for receive task to complete
                    await receive_task

                    if result_dict["success"] and result_dict["audio_data"]:
                        audio_data = result_dict["audio_data"]

                        # Save audio file
                        if audio_format == "pcm":
                            if not output.endswith(".wav"):
                                output += ".wav"
                            audio_np = np.frombuffer(audio_data, dtype=np.int16)
                            sf.write(output, audio_np, samplerate=sample_rate, subtype="PCM_16")
                        else:
                            if not output.endswith(f".{audio_format}"):
                                output += f".{audio_format}"
                            with open(output, "wb") as fp:
                                fp.write(audio_data)

                        logger.info(f"audio saved to {output}, audio size: {len(audio_data) / 1024:.2f} KB")
                        os.chmod(output, 0o644)
                        return True
                    else:
                        error_msg = result_dict.get("error", "Unknown error")
                        logger.warning(f"SenseTimeTTSClient tts request failed: {error_msg}")
                        return False

        except Exception as e:
            logger.warning(f"SenseTimeTTSClient tts request failed: {e}")
            return False

    async def upload_audio_clone(
        self,
        audio_path,
        audio_text,
        disable_ns=False,
    ):
        """
        Upload audio for voice cloning

        Args:
            audio_path: Audio file path
            audio_text: Text corresponding to the audio
            disable_ns: Whether to disable audio noise reduction processing

        Returns:
            tuple: (success: bool, result: str)
                - success: True indicates success, False indicates failure
                - result: Returns speaker_id on success, error message string on failure
        """
        logger.info(f"Connecting to {self.url}...")
        headers = {"apikey": self.apikey} if self.url.startswith("wss:") else None

        result_dict = {"success": False, "speaker": None, "session_id": None, "error": None}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.ws_connect(self.url) as websocket:
                    logger.info("WebSocket connection established")

                    session_id = str(uuid.uuid4())

                    # Start receive loop
                    receive_task = asyncio.create_task(self._receive_loop(websocket, session_id, {}, result_dict))

                    # Read and send audio
                    # Check file format, if it's a video file (e.g., MP4), extract audio first
                    tmp_audio_path = None
                    original_audio_path = audio_path

                    try:
                        file_ext = os.path.splitext(audio_path)[1].lower()
                        if file_ext in [".mp4", ".mov", ".avi", ".mkv", ".flv"]:
                            # Video file, need to extract audio first
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                                tmp_audio_path = tmp_audio.name

                            try:
                                # Use ffmpeg to extract audio
                                cmd = ["ffmpeg", "-i", audio_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", tmp_audio_path]
                                proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
                                try:
                                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                                except asyncio.TimeoutError:
                                    proc.kill()
                                    await proc.wait()
                                    raise ValueError("Audio extraction timeout. Video file may be too large.")
                                if proc.returncode != 0:
                                    raise ValueError(f"Failed to extract audio from video: {stderr.decode(errors='ignore')}")
                                logger.info(f"Extracted audio from video file to {tmp_audio_path}")
                                audio_path = tmp_audio_path
                            except subprocess.TimeoutError:
                                raise ValueError("Audio extraction timeout. Video file may be too large.")
                            except FileNotFoundError:
                                raise ValueError("ffmpeg not found. Please install ffmpeg to process video files.")
                            except Exception as e:
                                raise ValueError(f"Failed to extract audio: {str(e)}")

                        with open(audio_path, "rb") as fp:
                            audio_bytes = fp.read()

                        # Check file size (recommended not to exceed 10MB)
                        if len(audio_bytes) > 10 * 1024 * 1024:
                            logger.warning(f"Audio file size ({len(audio_bytes) / 1024 / 1024:.2f} MB) may be too large")

                        audio_chunk = AudioChunk()
                        audio_chunk.audio_data = audio_bytes
                        audio_chunk.audio_chunk_seq = 0
                        audio_chunk.is_last_chunk = 1
                        audio_chunk.text = audio_text
                        audio_chunk.disable_ns = disable_ns
                    finally:
                        # Clean up temporary files
                        if tmp_audio_path and os.path.exists(tmp_audio_path):
                            try:
                                os.unlink(tmp_audio_path)
                            except Exception:
                                pass

                    request = create_synthesis_request(
                        message_type=RequestMessageType.CLIENT_UPLOAD_CLONE_AUDIO,
                        app_id=self.app_id,
                        text="",
                        session_id=session_id,
                        upload_data=audio_chunk,
                    )
                    full_request = serialize_request(request)
                    await websocket.send_bytes(full_request)
                    logger.info(f"Sent audio chunk for cloning")

                    # Wait for receive task to complete
                    await receive_task

                    if result_dict["success"]:
                        speaker_id = result_dict.get("speaker")
                        logger.info(f"SenseTimeTTSClient upload audio clone successful, speaker: {speaker_id}")
                        return True, speaker_id
                    else:
                        # Return error message string directly
                        error_msg = result_dict.get("error", "Unknown error")
                        logger.warning(f"SenseTimeTTSClient upload audio clone failed: {error_msg}")
                        return False, error_msg

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"SenseTimeTTSClient upload audio clone failed: {error_msg}")
            return False, error_msg

    async def query_speaker(self, speaker):
        """
        Query if the specified speaker exists

        Args:
            speaker: speaker ID
        """
        logger.info(f"Connecting to {self.url}...")
        headers = {"apikey": self.apikey} if self.url.startswith("wss:") else None

        result_dict = {"success": False, "speaker": None, "error": None}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.ws_connect(self.url) as websocket:
                    logger.info("WebSocket connection established")

                    session_id = str(uuid.uuid4())

                    # Start receive loop
                    receive_task = asyncio.create_task(self._receive_loop(websocket, session_id, {}, result_dict))

                    # Send query request
                    request = create_synthesis_request(
                        message_type=RequestMessageType.CLIENT_QUERY_CLONE_AUDIO,
                        app_id=self.app_id,
                        text="",
                        session_id=session_id,
                        speaker=speaker,
                    )
                    full_request = serialize_request(request)
                    await websocket.send_bytes(full_request)
                    logger.info(f"Sent query for speaker {speaker}")

                    # Wait for receive task to complete
                    await receive_task

                    if result_dict["success"]:
                        logger.info(f"SenseTimeTTSClient query speaker successful")
                        return True
                    else:
                        error_msg = result_dict.get("error", "Unknown error")
                        logger.warning(f"SenseTimeTTSClient query speaker failed: {error_msg}")
                        return False

        except Exception as e:
            logger.warning(f"SenseTimeTTSClient query speaker failed: {e}")
            return False

    async def delete_speaker(self, speaker):
        """
        Delete the specified speaker

        Args:
            speaker: speaker ID
        """
        logger.info(f"Connecting to {self.url}...")
        headers = {"apikey": self.apikey} if self.url.startswith("wss:") else None

        result_dict = {"success": False, "error": None}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.ws_connect(self.url) as websocket:
                    logger.info("WebSocket connection established")

                    session_id = str(uuid.uuid4())

                    # Start receive loop
                    receive_task = asyncio.create_task(self._receive_loop(websocket, session_id, {}, result_dict))

                    # Send delete request
                    request = create_synthesis_request(
                        message_type=RequestMessageType.CLIENT_DELETE_CLONE_AUDIO,
                        app_id=self.app_id,
                        text="",
                        session_id=session_id,
                        speaker=speaker,
                    )
                    full_request = serialize_request(request)
                    await websocket.send_bytes(full_request)
                    logger.info(f"Sent delete request for speaker {speaker}")

                    # Wait for receive task to complete
                    await receive_task

                    if result_dict["success"]:
                        logger.info(f"SenseTimeTTSClient delete speaker successful")
                        return True
                    else:
                        error_msg = result_dict.get("error", "Unknown error")
                        logger.warning(f"SenseTimeTTSClient delete speaker failed: {error_msg}")
                        return False

        except Exception as e:
            logger.warning(f"SenseTimeTTSClient delete speaker failed: {e}")
            return False


async def test(args):
    """
    TTS test function

    Args:
        args: list, e.g. [text, speaker, style, speed, volume, pitch, language, output, sample_rate, audio_format, stream_output, output_subtitles]
              Provide as many as needed, from left to right.

    Parameter ranges:
        - speed: 0.5~2.0 (1.0 is normal speed)
        - volume: -12~12 dB (0 is normal volume)
        - pitch: -24~24 halftone (0 is normal pitch)
    """
    client = SenseTimeTTSClient()
    # Set default parameters
    params = {
        "text": "今天天气真不错，阳光明媚，微风轻拂，让人心情愉悦。",
        "speaker": "M20",
        "style": "正常",
        "speed": 1.0,
        "volume": 0,
        "pitch": 0,
        "language": "ZH_CN",
        "output": "tts_output.wav",
        "sample_rate": 24000,
        "audio_format": "pcm",
        "stream_output": True,
        "output_subtitles": False,
    }
    keys = list(params.keys())
    # Override default parameters
    for i, arg in enumerate(args):
        if i < len(keys):
            # Type conversion
            if keys[i] in ["sample_rate"]:
                params[keys[i]] = int(arg)
            elif keys[i] in ["stream_output", "output_subtitles"]:
                # Support multiple boolean inputs
                params[keys[i]] = str(arg).lower() in ("1", "true", "yes", "on")
            elif keys[i] in ["speed", "volume", "pitch"]:
                params[keys[i]] = float(arg)
            else:
                params[keys[i]] = arg

    await client.tts_request(
        params["text"],
        params["speaker"],
        params["style"],
        params["speed"],
        params["volume"],
        params["pitch"],
        params["language"],
        params["output"],
        params["sample_rate"],
        params["audio_format"],
        params["stream_output"],
        params["output_subtitles"],
    )


async def test_audio_clone(args):
    """
    Voice cloning test function

    Args:
        args: list, e.g. [audio_path, audio_text, disable_ns]
              Provide as many as needed, from left to right.

    Parameters:
        - audio_path: Audio file path (required)
        - audio_text: Text corresponding to the audio (required)
        - disable_ns: Whether to disable audio noise reduction processing, default False (optional, supports "1", "true", "yes", "on" for True)
    """
    client = SenseTimeTTSClient()
    # Set default parameters
    params = {
        "audio_path": "",
        "audio_text": "",
        "disable_ns": False,
    }
    keys = list(params.keys())
    # Override default parameters
    for i, arg in enumerate(args):
        if i < len(keys):
            # Type conversion
            if keys[i] == "disable_ns":
                # Support multiple boolean inputs
                params[keys[i]] = str(arg).lower() in ("1", "true", "yes", "on")
            else:
                params[keys[i]] = arg

    # Validate required parameters
    if not params["audio_path"]:
        logger.error("audio_path is required for audio clone test")
        return
    if not params["audio_text"]:
        logger.error("audio_text is required for audio clone test")
        return

    # Check if file exists
    if not os.path.exists(params["audio_path"]):
        logger.error(f"Audio file not found: {params['audio_path']}")
        return

    success, result = await client.upload_audio_clone(
        params["audio_path"],
        params["audio_text"],
        params["disable_ns"],
    )

    if success:
        logger.info(f"Audio clone successful! Speaker ID: {result}")
    else:
        logger.warning(f"Audio clone failed: {result}")


if __name__ == "__main__":
    # Support two test modes: regular TTS test and voice cloning test
    if len(sys.argv) > 1 and sys.argv[1] == "clone":
        # Voice cloning test mode: python sensetime_tts.py clone [audio_path] [audio_text] [disable_ns]
        asyncio.run(test_audio_clone(sys.argv[2:]))
    else:
        # Regular TTS test mode: python sensetime_tts.py [text] [speaker] ...
        asyncio.run(test(sys.argv[1:]))
