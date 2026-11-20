"""Streaming LLM providers sharing conversation history and live tools."""

import asyncio
import json
import time
from collections import deque
from contextlib import aclosing
from typing import AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit

import aiohttp
from loguru import logger

from .config import BilibiliConfig
from .logging_utils import log_preview

ToolHandler = Callable[[str, dict], Awaitable[dict]]


def tool_definitions(images_enabled: bool) -> list[dict]:
    def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        }

    tools = [
        function(
            "host_avatar_perform_action",
            "观众要求挥手、比心、交叉手臂等肢体动作时，生成英文动作描述和持续秒数，调用本工具提交执行。同轮多个动作请合并为一个连贯段落，只调用一次动作工具。",
            {
                "action_prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "只写动作段落，以 <Subject 1> 为主体，描述清晰可见的动作、手臂位置和结束时回到自然姿态。"
                        "保持身份、服装、背景和镜头不变，不虚构道具；不输出完整六段式画面提示词、[Action 1] 标题或说话内容。"
                        "例如：<Subject 1> raises one hand beside their face and waves clearly toward the camera, "
                        "then lowers the hand to a relaxed position. The torso stays steady and the camera remains fixed."
                    ),
                },
                "duration": {"type": "number", "minimum": 0.1, "maximum": 30, "description": "动作在生成视频中的持续秒数，通常为 3 秒；根据动作复杂度调整，必须提供。"},
            },
            ["action_prompt", "duration"],
        ),
    ]
    if images_enabled:
        tools.extend(
            [
                function(
                    "change_host_avatar",
                    "按观众要求更换装扮或背景，将同一次调整合并为一次调用。返回message供自然短话承接；本轮已回应过则不再追加口播，不播报内部流程或宣称已经换好。",
                    {"prompt": {"type": "string"}},
                    ["prompt"],
                ),
                function("change_origin_host_avatar", "仅在观众要求还原时，换回最开始的造型和背景。返回message供自然短话承接；本轮已回应过则不再追加口播，不播报内部流程或宣称已经恢复好。", {}, []),
            ]
        )
    return tools


class StreamingLLMClient:
    label = "LLM"

    def __init__(self, session: aiohttp.ClientSession, config: BilibiliConfig):
        self.session = session
        self.config = config
        # Store complete turns so pruning never leaves orphaned tool messages.
        self.history: deque[list[dict]] = deque(maxlen=config.history_turns)

    @property
    def request_url(self) -> str:
        return self.config.llm_base_url.rstrip("/") + self.api_path

    async def stream_reply(self, prompt: str, tools: list[dict], execute: ToolHandler) -> AsyncIterator[str]:
        turn = [{"role": "user", "content": prompt}]
        prefix = [{"role": "system", "content": self.config.system_prompt}]
        for previous in self.history:
            prefix.extend(previous)
        provider = log_preview(getattr(self.config, "llm_provider", self.label), self.config)
        for round_index in range(self.config.max_tool_rounds + 1):
            started = time.monotonic()
            first_text = False
            text_chars = 0
            output = None
            logger.info(
                "[LLM] request provider={} model={} round={} input_messages={} tools={}",
                provider,
                log_preview(self.config.llm_model, self.config),
                round_index + 1,
                len(prefix + turn),
                len(tools),
            )
            async with aclosing(self._stream_response(prefix + turn, tools)) as stream:
                async for event in stream:
                    if "delta" in event:
                        delta = event["delta"]
                        if not isinstance(delta, str):
                            raise RuntimeError("%s returned a non-text delta" % self.label)
                        if delta:
                            text_chars += len(delta)
                            if not first_text:
                                first_text = True
                                logger.info("[LLM] first text provider={} round={} latency_ms={:.0f}", provider, round_index + 1, (time.monotonic() - started) * 1000)
                        yield delta
                    else:
                        output = event["output"]
            if output is None:
                raise RuntimeError("%s stream ended without a completed response" % self.label)
            # Each transport returns complete canonical items. Never execute
            # partially streamed tool arguments or retain an unfinished turn.
            calls = [item for item in output if item["type"] == "function_call"]
            if any(call.get("status", "completed") != "completed" for call in calls):
                raise RuntimeError("%s returned an incomplete tool call" % self.label)
            logger.info(
                "[LLM] complete provider={} round={} elapsed_ms={:.0f} text_chars={} tools={}",
                provider,
                round_index + 1,
                (time.monotonic() - started) * 1000,
                text_chars,
                len(calls),
            )
            turn.extend(output)
            if not calls:
                self.history.append(turn)
                return
            if round_index == self.config.max_tool_rounds:
                raise RuntimeError("%s exceeded the maximum number of tool rounds" % self.label)
            for call in calls:
                tool_started = time.monotonic()
                name = log_preview(call.get("name", ""), self.config)
                logger.info("[LLM] tool call provider={} name={} arguments={}", provider, name, log_preview(call.get("arguments", ""), self.config))
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    result = await execute(call["name"], arguments)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    result = {"ok": False, "error": str(error)}
                logger.info(
                    "[LLM] tool result provider={} name={} elapsed_ms={:.0f} result={}",
                    provider,
                    name,
                    (time.monotonic() - tool_started) * 1000,
                    log_preview(result, self.config),
                )
                turn.append({"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result, ensure_ascii=False)})

    async def _check_response(self, response) -> None:
        if response.status == 200:
            return
        raw = await response.text()
        if self.config.llm_api_key:
            raw = raw.replace(self.config.llm_api_key, "<redacted>")
        try:
            body = json.loads(raw)
            error = body.get("error", body)
            info = {key: error[key] for key in ("code", "type", "message") if key in error} if isinstance(error, dict) else {"message": str(error)}
        except (AttributeError, ValueError):
            info = {"message": raw[:300]}
        request_id = response.headers.get("x-request-id")
        if request_id:
            info["request_id"] = request_id
        raise RuntimeError("%s request failed (HTTP %s; %s)" % (self.label, response.status, log_preview(info, self.config, 1000)))

    async def _sse_events(self, response):
        event_type = ""
        async for raw in response.content:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event_type = line[6:].strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            event = json.loads(data)
            if not isinstance(event, dict):
                raise RuntimeError("%s returned an invalid stream event" % self.label)
            yield event.get("type", event_type), event


class ResponsesLLMClient(StreamingLLMClient):
    api_path = "/responses"
    request_options = {}

    async def _stream_response(self, items: list[dict], tools: list[dict]):
        payload = {"model": self.config.llm_model, "input": items, "stream": True, "store": False, "max_output_tokens": 512}
        payload.update(self.request_options)
        if tools:
            payload["tools"] = tools
        async with self.session.post(
            self.request_url,
            headers={"Authorization": "Bearer " + self.config.llm_api_key},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            await self._check_response(response)
            async for kind, event in self._sse_events(response):
                if kind in ("response.output_text.delta", "response.message_text.delta", "response.text.delta"):
                    yield {"delta": event.get("delta", "")}
                elif kind in ("response.completed", "response.done"):
                    completed = event["response"]
                    if completed.get("status") != "completed":
                        raise RuntimeError("%s stream ended without a completed response" % self.label)
                    yield {"output": completed["output"]}
                    return
                elif kind in ("error", "response.failed", "response.incomplete"):
                    raise RuntimeError("%s stream failed (%s)" % (self.label, kind))
        raise RuntimeError("%s stream ended without a completed response" % self.label)


class VolcLLMClient(ResponsesLLMClient):
    label = "Volc LLM"
    request_options = {"thinking": {"type": "disabled"}}


class DeepSeekLLMClient(ResponsesLLMClient):
    label = "DeepSeek LLM"
    request_options = {"reasoning": {"effort": "none"}}


class SenseNovaLLMClient(StreamingLLMClient):
    """SenseNova Token Plan uses OpenAI-compatible Chat Completions."""

    label = "SenseNova LLM"
    api_path = "/chat/completions"

    @property
    def request_url(self) -> str:
        base = self.config.llm_base_url.rstrip("/")
        if not urlsplit(base).path:
            base += "/v1"
        return base + self.api_path

    @staticmethod
    def _chat_messages(items: list[dict]) -> list[dict]:
        messages = []
        for item in items:
            kind = item.get("type")
            if kind == "function_call":
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": None})
                messages[-1].setdefault("tool_calls", []).append({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}})
            elif kind == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
            elif kind == "message":
                text = "".join(part.get("text", "") for part in item.get("content", []) if part.get("type") == "output_text")
                message = {"role": item.get("role", "assistant"), "content": text or None}
                if item.get("reasoning_content"):
                    message["reasoning_content"] = item["reasoning_content"]
                messages.append(message)
            else:
                messages.append({"role": item["role"], "content": item["content"]})
        return messages

    async def _stream_response(self, items: list[dict], tools: list[dict]):
        payload = {"model": self.config.llm_model, "messages": self._chat_messages(items), "stream": True, "max_tokens": 2048}
        # DeepSeek and Lite can disable thinking; SenseNova Flash requires it.
        if self.config.llm_model in {"deepseek-flash", "sensenova-6.8-flash-lite"}:
            payload["reasoning_effort"] = "none"
        elif self.config.llm_model == "sensenova-6.8-flash":
            payload.update(reasoning_effort="minimal", max_tokens=4096)
        if tools:
            payload["tools"] = [{"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}} for tool in tools]
        text, reasoning, calls, finish_reason = "", "", {}, None
        async with self.session.post(
            self.request_url,
            headers={"Authorization": "Bearer " + self.config.llm_api_key},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            await self._check_response(response)
            async for kind, event in self._sse_events(response):
                if kind == "error" or event.get("error"):
                    raise RuntimeError("%s stream failed (%s)" % (self.label, log_preview(event.get("error", event), self.config, 500)))
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text += delta["content"]
                    yield {"delta": delta["content"]}
                # DeepSeek tool calls need reasoning in history, never in speech.
                if self.config.llm_model == "deepseek-flash" and tools:
                    reasoning += delta.get("reasoning_content") or ""
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index", 0)
                    call = calls.setdefault(index, {"type": "function_call", "call_id": "", "name": "", "arguments": ""})
                    call["call_id"] += fragment.get("id") or ""
                    function = fragment.get("function") or {}
                    call["name"] += function.get("name") or ""
                    call["arguments"] += function.get("arguments") or ""
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    if finish_reason not in {"stop", "tool_calls"}:
                        raise RuntimeError("%s stream ended with finish_reason=%s" % (self.label, finish_reason))
        if finish_reason is None:
            raise RuntimeError("%s stream ended without a completed response" % self.label)
        if finish_reason == "tool_calls" and not calls:
            raise RuntimeError("%s completed without its tool calls" % self.label)
        output = []
        if text or reasoning:
            message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}] if text else []}
            if reasoning:
                message["reasoning_content"] = reasoning
            output.append(message)
        for index in sorted(calls):
            call = calls[index]
            if not call["call_id"] or not call["name"]:
                raise RuntimeError("%s returned an incomplete tool call" % self.label)
            output.append(call)
        yield {"output": output}


def create_llm_client(session: aiohttp.ClientSession, config: BilibiliConfig) -> StreamingLLMClient:
    clients = {"volcengine": VolcLLMClient, "deepseek": DeepSeekLLMClient, "sensenova": SenseNovaLLMClient}
    if config.llm_provider not in clients:
        raise ValueError("BILIBILI_LLM_PROVIDER must be volcengine, deepseek or sensenova")
    return clients[config.llm_provider](session, config)
