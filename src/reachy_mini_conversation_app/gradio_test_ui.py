from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import httpx
import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config, refresh_runtime_config_from_env
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.self_hosted_openai import (
    _auth_headers,
    _endpoint,
    _new_http_client,
    _stream_chat_completion_text,
    _collect_tts_stream_from_chunks,
)


logger = logging.getLogger(__name__)


def _ensure_localhost_no_proxy() -> None:
    local_hosts = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.getenv(key, "").split(",") if item.strip()]
        merged = existing[:]
        for host in local_hosts:
            if host not in merged:
                merged.append(host)
        os.environ[key] = ",".join(merged)


_ensure_localhost_no_proxy()


class LocalPipelineTester:
    """Hardware-free tester for the self-hosted ASR, LLM, and TTS services."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = _new_http_client()
        return self._client

    async def text_to_reply_and_speech(
        self,
        user_text: str,
        voice: str,
        language: str,
    ) -> tuple[str, tuple[int, NDArray[np.int16]]]:
        user_text = user_text.strip()
        if not user_text:
            raise gr.Error("Please enter text.")

        reply_parts: list[str] = []

        async def reply_chunks() -> AsyncIterator[str]:
            async for chunk in self._stream_generate_reply(user_text):
                reply_parts.append(chunk)
                yield chunk

        audio = await self._synthesize(reply_chunks(), voice=voice, language=language)
        reply = "".join(reply_parts).strip()
        return reply, audio

    async def audio_to_transcript_reply_and_speech(
        self,
        audio_file: Any,
        voice: str,
        language: str,
    ) -> tuple[str, str, tuple[int, NDArray[np.int16]]]:
        audio_path = _coerce_audio_filepath(audio_file)
        transcript = (await self._transcribe_file(audio_path)).strip()
        if not transcript:
            raise gr.Error("ASR did not return a transcript.")

        reply_parts: list[str] = []

        async def reply_chunks() -> AsyncIterator[str]:
            async for chunk in self._stream_generate_reply(transcript):
                reply_parts.append(chunk)
                yield chunk

        audio = await self._synthesize(reply_chunks(), voice=voice, language=language)
        reply = "".join(reply_parts).strip()
        return transcript, reply, audio

    async def _transcribe_file(self, audio_path: Path) -> str:
        client = self._client_or_create()
        data: dict[str, str] = {}
        if config.SELF_ASR_MODEL:
            data["model"] = config.SELF_ASR_MODEL
        if config.SELF_ASR_LANGUAGE:
            data["language"] = config.SELF_ASR_LANGUAGE

        files = {
            "file": (
                audio_path.name,
                audio_path.read_bytes(),
                _audio_content_type(audio_path),
            )
        }
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
        return text

    async def _generate_reply(self, user_text: str) -> str:
        parts: list[str] = []
        async for chunk in self._stream_generate_reply(user_text):
            parts.append(chunk)
        return "".join(parts).strip()

    async def _stream_generate_reply(self, user_text: str) -> AsyncIterator[str]:
        client = self._client_or_create()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": get_session_instructions()},
            {"role": "user", "content": user_text},
        ]
        yielded = False
        async for chunk in _stream_chat_completion_text(client, messages):
            yielded = True
            yield chunk
        if not yielded:
            raise RuntimeError("LLM stream did not return content")

    async def _synthesize(
        self,
        text_chunks: AsyncIterator[str],
        voice: str,
        language: str,
    ) -> tuple[int, NDArray[np.int16]]:
        sample_rate, audio = await _collect_tts_stream_from_chunks(
            text_chunks,
            (voice or config.SELF_TTS_VOICE).strip() or config.SELF_TTS_VOICE,
            language.strip(),
        )
        return sample_rate, audio.reshape(-1)


def _coerce_audio_filepath(audio_file: Any) -> Path:
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")
    if isinstance(audio_file, (str, os.PathLike)):
        return Path(audio_file)
    if isinstance(audio_file, dict):
        path = audio_file.get("path") or audio_file.get("name")
        if path:
            return Path(path)
    if isinstance(audio_file, (tuple, list)) and audio_file:
        first = audio_file[0]
        if isinstance(first, (str, os.PathLike)):
            return Path(first)
    raise gr.Error("Could not read the uploaded audio file path.")


def _audio_content_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


def create_test_ui() -> gr.Blocks:
    tester = LocalPipelineTester()

    async def run_text_turn(user_text: str, voice: str, language: str) -> tuple[str, tuple[int, NDArray[np.int16]]]:
        return await tester.text_to_reply_and_speech(user_text, voice, language)

    async def run_audio_turn(
        audio_file: Any,
        voice: str,
        language: str,
    ) -> tuple[str, str, tuple[int, NDArray[np.int16]]]:
        return await tester.audio_to_transcript_reply_and_speech(audio_file, voice, language)

    voice_choices = list(config.SELF_TTS_VOICES)
    default_voice = config.SELF_TTS_VOICE

    with gr.Blocks(title="Local Pipeline Test") as demo:
        gr.Markdown("## Local Pipeline Test")
        gr.Markdown(
            f"ASR `{config.SELF_ASR_BASE_URL}` | LLM `{config.SELF_LLM_BASE_URL}` | "
            f"TTS `{config.SELF_TTS_BASE_URL}`"
        )

        with gr.Row():
            voice = gr.Dropdown(
                label="TTS voice",
                choices=voice_choices,
                value=default_voice,
                allow_custom_value=True,
            )
            language = gr.Textbox(label="TTS language", value=config.SELF_TTS_LANGUAGE)

        with gr.Tab("Text -> LLM -> TTS"):
            text_input = gr.Textbox(label="User text", lines=5)
            text_button = gr.Button("Run", variant="primary")
            text_reply = gr.Textbox(label="LLM reply", lines=8)
            text_audio = gr.Audio(label="TTS audio")
            text_button.click(
                run_text_turn,
                inputs=[text_input, voice, language],
                outputs=[text_reply, text_audio],
            )

        with gr.Tab("Audio -> ASR -> LLM -> TTS"):
            audio_input = gr.Audio(label="Audio upload", sources=["upload"], type="filepath")
            audio_button = gr.Button("Run", variant="primary")
            transcript_output = gr.Textbox(label="ASR transcript", lines=4)
            audio_reply = gr.Textbox(label="LLM reply", lines=8)
            audio_output = gr.Audio(label="TTS audio")
            audio_button.click(
                run_audio_turn,
                inputs=[audio_input, voice, language],
                outputs=[transcript_output, audio_reply, audio_output],
            )

        demo.unload(tester.close)

    return demo


def run_test_ui(args: Any, instance_path: str | None = None) -> None:
    if instance_path is not None:
        env_path = Path(instance_path) / ".env"
        if env_path.exists():
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=str(env_path), override=True)
            refresh_runtime_config_from_env()
            logger.info("Loaded instance configuration from %s", env_path)

    demo = create_test_ui()
    launch_kwargs: dict[str, Any] = {}
    if getattr(args, "server_name", None):
        launch_kwargs["server_name"] = args.server_name
    if getattr(args, "server_port", None) is not None:
        launch_kwargs["server_port"] = args.server_port
    demo.launch(**launch_kwargs)
