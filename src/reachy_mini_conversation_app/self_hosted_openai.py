from __future__ import annotations
import io
import json
import math
import time
import wave
import uuid
import asyncio
import logging
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Awaitable, Callable, AsyncIterator
from typing import Any
from collections import deque

import gradio as gr
import httpx
import numpy as np
import websockets
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config, set_custom_profile, get_default_voice_for_backend
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.tool_constants import SystemTool
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_active_tool_specs,
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager, ToolNotification


logger = logging.getLogger(__name__)

_SYSTEM_TOOL_NAMES = {tool.value for tool in SystemTool}


def _rms_dbfs(audio: NDArray[np.int16]) -> float:
    x = audio.astype(np.float32, copy=False) / 32768.0
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x * x, dtype=np.float32) + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0), trust_env=False)


def _websocket_endpoint(base_url: str, path: str) -> str:
    endpoint = _endpoint(base_url, path)
    parsed = urlsplit(endpoint)
    if parsed.scheme == "http":
        return urlunsplit(("ws", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    if parsed.scheme == "https":
        return urlunsplit(("wss", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return endpoint


def _to_mono_int16(frame: NDArray[Any], input_sample_rate: int, target_sample_rate: int) -> NDArray[np.int16]:
    audio = np.asarray(frame)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] <= audio.shape[1]:
            audio = np.mean(audio, axis=0)
        elif audio.shape[1] <= 8:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.reshape(-1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1)

    if input_sample_rate != target_sample_rate and audio.size:
        target_len = int(audio.size * target_sample_rate / input_sample_rate)
        if target_len <= 0:
            return np.zeros(0, dtype=np.int16)
        audio = resample(audio, target_len)

    return audio_to_int16(audio).reshape(-1)


def _wav_bytes(pcm: NDArray[np.int16], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def _decode_wav_bytes(data: bytes) -> tuple[int, NDArray[np.int16]]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=False)
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        vals = np.where(vals & 0x800000, vals | ~0xFFFFFF, vals)
        arr = (vals >> 8).astype(np.int16)
    elif sample_width == 4:
        arr = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1 and arr.size:
        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return sample_rate, arr.reshape(1, -1)


def _tts_stream_config(voice: str, language: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.config",
        "voice": voice,
        "response_format": "pcm",
        "stream_audio": True,
        "split_granularity": "sentence",
    }
    if config.SELF_TTS_MODEL:
        payload["model"] = config.SELF_TTS_MODEL
    if language:
        payload["language"] = language
    return payload


def _text_chunks(text: str, chunk_size: int = 512) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _decode_stream_audio_chunk(
    data: bytes,
    audio_format: str,
    sample_rate: int,
    pending_pcm: bytes,
) -> tuple[tuple[int, NDArray[np.int16]] | None, bytes]:
    if data[:4] == b"RIFF" or audio_format.lower() == "wav":
        decoded_sample_rate, decoded_audio = _decode_wav_bytes(data)
        return (decoded_sample_rate, decoded_audio), b""

    raw = pending_pcm + data
    usable_size = len(raw) - (len(raw) % 2)
    if usable_size <= 0:
        return None, raw

    pcm = np.frombuffer(raw[:usable_size], dtype="<i2").astype(np.int16, copy=False).reshape(1, -1)
    return (sample_rate, pcm), raw[usable_size:]


async def _stream_tts_audio(
    text: str,
    voice: str,
    language: str | None = None,
) -> AsyncIterator[tuple[int, NDArray[np.int16]]]:
    async def chunks() -> AsyncIterator[str]:
        for chunk in _text_chunks(text):
            yield chunk

    async for sample_rate, audio in _stream_tts_audio_from_chunks(chunks(), voice, language):
        yield sample_rate, audio


async def _stream_tts_audio_from_chunks(
    text_chunks: AsyncIterator[str],
    voice: str,
    language: str | None = None,
) -> AsyncIterator[tuple[int, NDArray[np.int16]]]:
    url = _websocket_endpoint(config.SELF_TTS_BASE_URL, "audio/speech/stream")
    headers = _auth_headers(config.SELF_TTS_API_KEY)
    connect_kwargs: dict[str, Any] = {"open_timeout": 10, "close_timeout": 10, "max_size": None}
    if headers:
        connect_kwargs["additional_headers"] = headers

    current_sample_rate = config.SELF_TTS_SAMPLE_RATE
    current_format = "pcm"
    pending_pcm = b""

    async with websockets.connect(url, **connect_kwargs) as websocket:
        await websocket.send(json.dumps(_tts_stream_config(voice, language)))

        async def send_text() -> None:
            async for chunk in text_chunks:
                if chunk:
                    await websocket.send(json.dumps({"type": "input.text", "text": chunk}, ensure_ascii=False))
            await websocket.send(json.dumps({"type": "input.done"}))

        sender = asyncio.create_task(send_text(), name="tts-stream-text-sender")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    decoded, pending_pcm = _decode_stream_audio_chunk(
                        message,
                        current_format,
                        current_sample_rate,
                        pending_pcm,
                    )
                    if decoded is not None:
                        yield decoded
                    continue

                payload = json.loads(message)
                message_type = payload.get("type")
                if message_type == "audio.start":
                    current_sample_rate = int(payload.get("sample_rate") or config.SELF_TTS_SAMPLE_RATE)
                    current_format = str(payload.get("format") or "pcm")
                    pending_pcm = b""
                elif message_type == "audio.done":
                    if pending_pcm:
                        logger.warning("Dropping incomplete trailing TTS PCM byte")
                        pending_pcm = b""
                elif message_type == "session.done":
                    break
                elif message_type == "error":
                    raise RuntimeError(f"TTS stream error: {payload.get('message') or payload!r}")
        finally:
            if not sender.done():
                sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass


async def _collect_tts_stream(
    text: str,
    voice: str,
    language: str | None = None,
) -> tuple[int, NDArray[np.int16]]:
    async def chunks() -> AsyncIterator[str]:
        for chunk in _text_chunks(text):
            yield chunk

    return await _collect_tts_stream_from_chunks(chunks(), voice, language)


async def _collect_tts_stream_from_chunks(
    text_chunks: AsyncIterator[str],
    voice: str,
    language: str | None = None,
) -> tuple[int, NDArray[np.int16]]:
    chunks: list[NDArray[np.int16]] = []
    sample_rate = config.SELF_TTS_SAMPLE_RATE
    async for chunk_sample_rate, audio in _stream_tts_audio_from_chunks(text_chunks, voice, language):
        sample_rate = chunk_sample_rate
        chunks.append(audio.reshape(-1))

    if not chunks:
        raise RuntimeError("TTS stream did not return any audio")
    return sample_rate, np.concatenate(chunks).astype(np.int16, copy=False).reshape(1, -1)


def _merge_stream_tool_call(accumulator: dict[int, dict[str, Any]], delta_tool_call: dict[str, Any]) -> None:
    index = int(delta_tool_call.get("index") or 0)
    target = accumulator.setdefault(index, {"type": "function", "function": {"name": "", "arguments": ""}})

    call_id = delta_tool_call.get("id")
    if isinstance(call_id, str) and call_id:
        target["id"] = call_id

    call_type = delta_tool_call.get("type")
    if isinstance(call_type, str) and call_type:
        target["type"] = call_type

    function_delta = delta_tool_call.get("function")
    if isinstance(function_delta, dict):
        function = target.setdefault("function", {"name": "", "arguments": ""})
        name = function_delta.get("name")
        if isinstance(name, str):
            function["name"] = str(function.get("name") or "") + name
        arguments = function_delta.get("arguments")
        if isinstance(arguments, str):
            function["arguments"] = str(function.get("arguments") or "") + arguments


def _debug_log_llm_request(payload: dict[str, Any]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    history_count = max(0, len(messages) - 2) if isinstance(messages, list) else 0
    tool_names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_names.append(function["name"])

    logger.debug(
        "LLM request: model=%s stream=%s history_messages=%s tools=%s messages=%s",
        payload.get("model"),
        payload.get("stream"),
        history_count,
        tool_names,
        json.dumps(messages, ensure_ascii=False, default=str),
    )


async def _read_chat_completion_stream(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    on_content: Callable[[str], Awaitable[None]],
) -> dict[str, Any]:
    response_text = ""
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}

    async with client.stream(
        "POST",
        _endpoint(config.SELF_LLM_BASE_URL, "chat/completions"),
        headers={**_auth_headers(config.SELF_LLM_API_KEY), "Content-Type": "application/json"},
        json=payload,
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            response_text = body.decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM stream request failed with {response.status_code}: {response_text[:500]}")

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break

            data = json.loads(line)
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if not isinstance(delta, dict):
                continue

            content = delta.get("content")
            if isinstance(content, str) and content:
                logger.debug("LLM stream content delta: %r", content)
                content_parts.append(content)
                await on_content(content)

            delta_tool_calls = delta.get("tool_calls") or []
            if isinstance(delta_tool_calls, list):
                for delta_tool_call in delta_tool_calls:
                    if isinstance(delta_tool_call, dict):
                        logger.debug("LLM stream tool_call delta: %s", json.dumps(delta_tool_call, ensure_ascii=False))
                        _merge_stream_tool_call(tool_calls, delta_tool_call)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    logger.debug("LLM stream final message: %s", json.dumps(message, ensure_ascii=False, default=str))
    return message


async def _chat_completion_stream_message(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    on_content: Callable[[str], Awaitable[None]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.SELF_LLM_MODEL,
        "messages": messages,
        "temperature": config.SELF_LLM_TEMPERATURE,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    _debug_log_llm_request(payload)
    try:
        return await _read_chat_completion_stream(client, payload, on_content)
    except RuntimeError as e:
        if not tools:
            raise
        logger.warning("LLM rejected streaming tool schema; retrying without tools: %s", str(e)[:500])
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        return await _read_chat_completion_stream(client, payload, on_content)


async def _stream_chat_completion_text(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    message_holder: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()

    async def on_content(content: str) -> None:
        await queue.put(content)

    async def run_stream() -> None:
        try:
            message = await _chat_completion_stream_message(client, messages, tools or [], on_content)
            if message_holder is not None:
                message_holder["message"] = message
            await queue.put(None)
        except BaseException as e:
            await queue.put(e)

    task = asyncio.create_task(run_stream(), name="llm-chat-stream")
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _chat_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        if spec.get("type") != "function":
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        function_spec: dict[str, Any] = {
            "name": name,
            "parameters": spec.get("parameters", {}),
        }
        description = spec.get("description")
        if isinstance(description, str):
            function_spec["description"] = description
        tools.append({"type": "function", "function": function_spec})
    return tools


def _sanitize_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "camera" and "b64_im" in result:
        sanitized = dict(result)
        sanitized.pop("b64_im", None)
        sanitized["image_attached"] = True
        return sanitized
    return result


def _camera_image_message(result: dict[str, Any]) -> dict[str, Any] | None:
    b64_image = result.get("b64_im")
    if not isinstance(b64_image, str) or not b64_image:
        return None

    mime_type = result.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type:
        mime_type = "image/jpeg"

    question = result.get("question")
    text = "Use this camera image to answer the previous camera tool request."
    if isinstance(question, str) and question.strip():
        text = f"{text} Camera question: {question.strip()}"

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ],
    }


class SelfHostedOpenAIHandler(ConversationHandler):
    """Turn-based voice handler backed by self-hosted OpenAI-compatible HTTP services."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=config.SELF_TTS_SAMPLE_RATE,
            input_sample_rate=config.SELF_ASR_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override = (startup_voice or "").strip() or None

        self.output_queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._client: httpx.AsyncClient | None = None
        self._stop_event = asyncio.Event()
        self._turn_lock = asyncio.Lock()
        self._processing_tasks: set[asyncio.Task[None]] = set()
        self.tool_manager = BackgroundToolManager()

        self._history: list[dict[str, Any]] = []
        self._system_instructions: str | None = None

        self._is_speaking = False
        self._speech_frames: list[NDArray[np.int16]] = []
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._pre_roll: deque[NDArray[np.int16]] = deque()
        self._pre_roll_samples = max(1, int(config.SELF_ASR_SAMPLE_RATE * config.SELF_VAD_PRE_ROLL_MS / 1000))
        self._pre_roll_total = 0
        self.last_activity_time = asyncio.get_event_loop().time()

    def copy(self) -> "SelfHostedOpenAIHandler":
        """Create a copy of the handler."""
        return SelfHostedOpenAIHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def start_up(self) -> None:
        """Start the handler lifecycle and keep it alive until shutdown."""
        self._stop_event.clear()
        self._client = _new_http_client()
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])
        logger.info(
            "Self-hosted voice pipeline ready: ASR %s/%s, LLM %s/%s, TTS %s/%s voice=%s",
            config.SELF_ASR_BASE_URL,
            config.SELF_ASR_MODEL,
            config.SELF_LLM_BASE_URL,
            config.SELF_LLM_MODEL,
            config.SELF_TTS_BASE_URL,
            config.SELF_TTS_MODEL,
            self.get_current_voice(),
        )
        await self._stop_event.wait()

    async def shutdown(self) -> None:
        """Shut down pending work and HTTP resources."""
        self._stop_event.set()
        for task in list(self._processing_tasks):
            task.cancel()
        for task in list(self._processing_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._processing_tasks.clear()

        await self.tool_manager.shutdown()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile for future turns."""
        set_custom_profile(profile)
        self._system_instructions = None
        self._history.clear()
        return "Applied personality."

    async def get_available_voices(self) -> list[str]:
        """Return configured TTS voices."""
        return list(config.SELF_TTS_VOICES)

    def get_current_voice(self) -> str:
        """Return the current TTS voice."""
        return self._voice_override or get_session_voice(get_default_voice_for_backend())

    async def change_voice(self, voice: str) -> str:
        """Change the TTS voice for future turns."""
        voice = voice.strip()
        if not voice:
            return "Voice was not changed."
        self._voice_override = voice
        return f"Voice changed to {voice}."

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio, run local VAD, and schedule completed utterances."""
        input_sample_rate, audio_frame = frame
        pcm = _to_mono_int16(audio_frame, input_sample_rate, config.SELF_ASR_SAMPLE_RATE)
        if pcm.size == 0:
            return

        frame_seconds = pcm.size / float(config.SELF_ASR_SAMPLE_RATE)
        db = _rms_dbfs(pcm)

        if not self._is_speaking:
            self._push_pre_roll(pcm)
            if db >= config.SELF_VAD_START_DBFS:
                self._begin_speech()
            return

        self._speech_frames.append(pcm)
        self._speech_seconds += frame_seconds

        if db <= config.SELF_VAD_STOP_DBFS:
            self._silence_seconds += frame_seconds
        else:
            self._silence_seconds = 0.0

        silence_limit = config.SELF_VAD_SILENCE_MS / 1000.0
        if self._silence_seconds >= silence_limit or self._speech_seconds >= config.SELF_VAD_MAX_UTTERANCE_SECONDS:
            self._finish_speech()

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit queued transcripts or audio frames to the stream."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    def _push_pre_roll(self, pcm: NDArray[np.int16]) -> None:
        self._pre_roll.append(pcm)
        self._pre_roll_total += pcm.size
        while self._pre_roll_total > self._pre_roll_samples and self._pre_roll:
            removed = self._pre_roll.popleft()
            self._pre_roll_total -= removed.size

    def _begin_speech(self) -> None:
        self._is_speaking = True
        self._speech_frames = list(self._pre_roll)
        self._speech_seconds = sum(frame.size for frame in self._speech_frames) / float(config.SELF_ASR_SAMPLE_RATE)
        self._silence_seconds = 0.0
        self._pre_roll.clear()
        self._pre_roll_total = 0
        self.last_activity_time = asyncio.get_event_loop().time()

        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
            self._clear_queue()
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        self.deps.movement_manager.set_listening(True)
        logger.debug("Local VAD: speech started")

    def _finish_speech(self) -> None:
        frames = self._speech_frames
        duration = self._speech_seconds
        self._is_speaking = False
        self._speech_frames = []
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self.deps.movement_manager.set_listening(False)
        logger.debug("Local VAD: speech stopped after %.2fs", duration)

        min_duration = config.SELF_VAD_MIN_SPEECH_MS / 1000.0
        if duration < min_duration or not frames:
            logger.debug("Discarding short utterance: %.2fs", duration)
            return

        utterance = np.concatenate(frames).astype(np.int16, copy=False)
        task = asyncio.create_task(self._process_utterance(utterance), name="self-hosted-voice-turn")
        self._processing_tasks.add(task)
        task.add_done_callback(self._processing_tasks.discard)

    async def _process_utterance(self, pcm: NDArray[np.int16]) -> None:
        async with self._turn_lock:
            try:
                transcript = await self._transcribe(pcm)
                transcript = transcript.strip()
                if not transcript:
                    return
                self.last_activity_time = asyncio.get_event_loop().time()
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

                reply_parts: list[str] = []

                async def reply_chunks() -> AsyncIterator[str]:
                    async for chunk in self._stream_generate_reply(transcript):
                        reply_parts.append(chunk)
                        yield chunk

                async for sample_rate, audio in self._stream_synthesize_chunks(reply_chunks()):
                    await self._queue_audio(sample_rate, audio)

                reply = "".join(reply_parts)
                reply = reply.strip()
                if not reply:
                    return
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": reply}))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Self-hosted voice turn failed")
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": f"[voice pipeline error] {type(e).__name__}: {e}"})
                )

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = _new_http_client()
        return self._client

    async def _transcribe(self, pcm: NDArray[np.int16]) -> str:
        client = await self._client_or_create()
        data: dict[str, str] = {}
        if config.SELF_ASR_MODEL:
            data["model"] = config.SELF_ASR_MODEL
        if config.SELF_ASR_LANGUAGE:
            data["language"] = config.SELF_ASR_LANGUAGE
        files = {"file": ("speech.wav", _wav_bytes(pcm, config.SELF_ASR_SAMPLE_RATE), "audio/wav")}
        response = await client.post(
            _endpoint(config.SELF_ASR_BASE_URL, "audio/transcriptions"),
            headers=_auth_headers(config.SELF_ASR_API_KEY),
            data=data,
            files=files,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"ASR response did not contain text: {payload!r}")
        logger.debug("ASR transcript: %s", text)
        return text

    async def _generate_reply(self, user_text: str) -> str:
        parts: list[str] = []
        async for chunk in self._stream_generate_reply(user_text):
            parts.append(chunk)
        return "".join(parts)

    async def _stream_generate_reply(self, user_text: str) -> AsyncIterator[str]:
        instructions = get_session_instructions()
        if self._system_instructions != instructions:
            self._history.clear()
            self._system_instructions = instructions

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            *self._history,
            {"role": "user", "content": user_text},
        ]
        tools = _chat_tools(get_active_tool_specs(self.deps))

        for round_index in range(config.SELF_LLM_MAX_TOOL_ROUNDS + 1):
            client = await self._client_or_create()
            message_holder: dict[str, Any] = {}
            async for chunk in _stream_chat_completion_text(client, messages, tools, message_holder):
                yield chunk

            message = message_holder.get("message")
            if not isinstance(message, dict):
                raise RuntimeError("LLM stream did not return a final message")
            tool_calls = list(message.get("tool_calls") or [])
            content = str(message.get("content") or "").strip()

            if tool_calls and round_index < config.SELF_LLM_MAX_TOOL_ROUNDS:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                for tool_call in tool_calls:
                    call_id, tool_name, args_json = self._tool_call_parts(tool_call)
                    tool_result = await self._run_tool(tool_name, args_json)
                    visible_tool_result = _sanitize_tool_result(tool_name, tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": json.dumps(visible_tool_result, ensure_ascii=False, default=str),
                        }
                    )
                    image_message = _camera_image_message(tool_result) if tool_name == "camera" else None
                    if image_message is not None:
                        messages.append(image_message)
                continue

            if not content and tool_calls:
                content = "I used the requested tool."
            if not content:
                content = "I am ready."
                yield content

            messages.append({"role": "assistant", "content": content})
            self._history = self._trim_history(messages[1:])
            return

        fallback = "I used the available tools, but I could not complete a final answer."
        yield fallback
        messages.append({"role": "assistant", "content": fallback})
        self._history = self._trim_history(messages[1:])
        return

    async def _chat_completion(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        client = await self._client_or_create()
        payload: dict[str, Any] = {
            "model": config.SELF_LLM_MODEL,
            "messages": messages,
            "temperature": config.SELF_LLM_TEMPERATURE,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        _debug_log_llm_request(payload)
        response = await client.post(
            _endpoint(config.SELF_LLM_BASE_URL, "chat/completions"),
            headers={**_auth_headers(config.SELF_LLM_API_KEY), "Content-Type": "application/json"},
            json=payload,
        )
        if response.status_code >= 400 and tools:
            logger.warning("LLM rejected tool schema; retrying without tools: %s", response.text[:500])
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            response = await client.post(
                _endpoint(config.SELF_LLM_BASE_URL, "chat/completions"),
                headers={**_auth_headers(config.SELF_LLM_API_KEY), "Content-Type": "application/json"},
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response did not contain choices: {data!r}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"LLM choice did not contain a message: {data!r}")
        logger.debug("LLM non-stream final message: %s", json.dumps(message, ensure_ascii=False, default=str))
        return message

    def _tool_call_parts(self, tool_call: Any) -> tuple[str, str, str]:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        call_id = str(tool_call.get("id") or uuid.uuid4()) if isinstance(tool_call, dict) else str(uuid.uuid4())
        tool_name = function.get("name") if isinstance(function, dict) else ""
        args = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
        if not isinstance(tool_name, str):
            tool_name = ""
        if not isinstance(args, str):
            args = json.dumps(args)
        return call_id, tool_name, args

    async def _run_tool(self, tool_name: str, args_json: str) -> dict[str, Any]:
        if not tool_name:
            return {"error": "tool call did not include a function name"}

        logger.debug("Executing tool call: name=%s args=%s", tool_name, args_json)
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": f"Used tool {tool_name} with args {args_json}."})
        )

        if tool_name in _SYSTEM_TOOL_NAMES:
            result = await dispatch_tool_call_with_manager(tool_name, args_json, self.deps, self.tool_manager)
        else:
            result = await dispatch_tool_call(tool_name, args_json, self.deps)

        if not isinstance(result, dict):
            result = {"result": result}
        logger.debug("Tool result: name=%s result=%s", tool_name, json.dumps(result, ensure_ascii=False, default=str))

        visible_result = _sanitize_tool_result(tool_name, result)
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": json.dumps(visible_result, ensure_ascii=False, default=str),
                    "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                }
            )
        )

        if tool_name == "camera" and self.deps.camera_worker is not None:
            frame = self.deps.camera_worker.get_latest_frame()
            if frame is not None:
                rgb_frame = frame[:, :, ::-1].copy() if frame.ndim == 3 and frame.shape[-1] == 3 else frame
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": gr.Image(value=rgb_frame)}))

        return result

    async def _stream_synthesize(self, text: str) -> AsyncIterator[tuple[int, NDArray[np.int16]]]:
        async for sample_rate, audio in _stream_tts_audio(
            text,
            self.get_current_voice(),
            config.SELF_TTS_LANGUAGE,
        ):
            yield sample_rate, audio

    async def _stream_synthesize_chunks(
        self,
        text_chunks: AsyncIterator[str],
    ) -> AsyncIterator[tuple[int, NDArray[np.int16]]]:
        async for sample_rate, audio in _stream_tts_audio_from_chunks(
            text_chunks,
            self.get_current_voice(),
            config.SELF_TTS_LANGUAGE,
        ):
            yield sample_rate, audio

    async def _queue_audio(self, sample_rate: int, audio: NDArray[np.int16]) -> None:
        if audio.size == 0:
            return
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        if self.gradio_mode and self.deps.head_wobbler is not None:
            self.deps.head_wobbler.feed_pcm(audio, sample_rate)

        chunk_samples = max(1, int(sample_rate * 0.2))
        for start in range(0, audio.shape[1], chunk_samples):
            await self.output_queue.put((sample_rate, audio[:, start : start + chunk_samples]))

    def _trim_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_messages = max(2, config.SELF_LLM_MAX_HISTORY_MESSAGES)
        cleaned: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message.get("content"), list):
                replacement = dict(message)
                replacement["content"] = "[camera image omitted from history]"
                cleaned.append(replacement)
            else:
                cleaned.append(dict(message))

        if len(cleaned) <= max_messages:
            return cleaned
        return cleaned[-max_messages:]

    async def _handle_tool_notification(self, bg_tool: ToolNotification) -> None:
        content = bg_tool.error if bg_tool.error is not None else json.dumps(bg_tool.result or {}, default=str)
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": content,
                    "metadata": {"title": f"Tool {bg_tool.tool_name}", "status": "done"},
                }
            )
        )
