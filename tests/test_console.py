"""Tests for the headless console stream."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray
from fastapi.testclient import TestClient

from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.console import LOCAL_PLAYER_BACKEND, LocalStream
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_player=MagicMock(), clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=LOCAL_PLAYER_BACKEND))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    audio = SimpleNamespace(clear_player=MagicMock(), clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()
    assert isinstance(handler.output_queue, asyncio.Queue)
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_play_loop_feeds_head_wobbler_with_local_playback_delay() -> None:
    """Local playback should drive speech wobble using the queued player delay."""
    head_wobbler = MagicMock()
    chunk = np.array([1, -2, 3, -4], dtype=np.int16)

    class Handler:
        def __init__(self) -> None:
            self.deps = SimpleNamespace(head_wobbler=head_wobbler)
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._emitted = False

        async def emit(self) -> tuple[int, NDArray[np.int16]] | None:
            if not self._emitted:
                self._emitted = True
                return (24000, chunk.copy())
            return None

    audio = SimpleNamespace(
        _playback_next_pts_ns=1_500_000_000,
        _get_playback_running_time_ns=lambda: 500_000_000,
    )
    media = SimpleNamespace(
        audio=audio,
        backend=LOCAL_PLAYER_BACKEND,
        get_output_audio_samplerate=lambda: 24000,
        push_audio_sample=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    handler = Handler()
    stream = LocalStream(handler, robot)

    async def stop_soon() -> None:
        await asyncio.sleep(0.01)
        stream._stop_event.set()

    stopper = asyncio.create_task(stop_soon())
    try:
        await asyncio.wait_for(stream.play_loop(), timeout=1.0)
    finally:
        await stopper

    head_wobbler.feed_pcm.assert_called_once()
    args, kwargs = head_wobbler.feed_pcm.call_args
    assert np.array_equal(args[0], chunk.reshape(1, -1))
    assert args[1] == 24000
    assert kwargs["start_delay_s"] == pytest.approx(1.0)
    media.push_audio_sample.assert_called_once()


def test_service_config_persists_self_hosted_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings API should persist self-hosted service configuration."""
    monkeypatch.setenv("SELF_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    client = TestClient(app)
    response = client.post(
        "/service_config",
        json={
            "self_openai_base_url": "http://localhost:9000",
            "self_openai_api_key": "token",
            "self_asr_model": "asr-model",
            "self_llm_model": "llm-model",
            "self_tts_model": "tts-model",
            "self_tts_voice": "voice-a",
            "self_tts_voices": "voice-a,voice-b",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["asr_base_url"] == "http://localhost:9000/v1"
    assert data["llm_model"] == "llm-model"
    assert data["tts_voice"] == "voice-a"

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "SELF_OPENAI_BASE_URL=http://localhost:9000" in env_text
    assert "SELF_OPENAI_API_KEY=token" in env_text
    assert "SELF_ASR_MODEL=asr-model" in env_text
    assert "SELF_LLM_MODEL=llm-model" in env_text
    assert "SELF_TTS_MODEL=tts-model" in env_text


def test_headless_personality_routes_load_builtin_default_tools() -> None:
    """Headless personality UI should expose built-in default tools on initial load."""
    app = FastAPI()
    handler = MagicMock()
    mount_personality_routes(app, handler, lambda: None)

    client = TestClient(app)
    response = client.get("/personalities/load", params={"name": "(built-in default)"})

    assert response.status_code == 200
    data = response.json()
    assert data["tools_text"]
    assert "dance" in data["enabled_tools"]
    assert "camera" in data["enabled_tools"]


def test_headless_personality_routes_apply_voice_accepts_query_param() -> None:
    """Headless personality UI should apply a voice change from a POST query param."""
    app = FastAPI()
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to voice-a.")

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    started.wait(timeout=1.0)

    try:
        mount_personality_routes(app, handler, lambda: loop)

        client = TestClient(app)
        response = client.post("/voices/apply?voice=voice-a")

        assert response.status_code == 200
        assert response.json() == {"ok": True, "status": "Voice changed to voice-a."}
        handler.change_voice.assert_awaited_once_with("voice-a")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_local_stream_persist_personality_stores_voice_override(tmp_path: Path) -> None:
    """Persisting startup settings should write both profile and voice override."""
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality("sorry_bro", "voice-a")

    settings_path = tmp_path / "startup_settings.json"
    assert settings_path.exists()
    assert settings_path.read_text(encoding="utf-8") == '{\n  "profile": "sorry_bro",\n  "voice": "voice-a"\n}\n'
    assert stream._read_persisted_personality() == "sorry_bro"
