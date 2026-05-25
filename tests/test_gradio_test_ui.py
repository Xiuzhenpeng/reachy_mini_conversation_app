from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app.gradio_test_ui import _audio_content_type, _coerce_audio_filepath
from reachy_mini_conversation_app.self_hosted_openai import (
    _text_chunks,
    _tts_stream_config,
    _websocket_endpoint,
    _merge_stream_tool_call,
    _decode_stream_audio_chunk,
)


def test_coerce_audio_filepath_accepts_gradio_filepath(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF")

    assert _coerce_audio_filepath(str(audio_path)) == audio_path


def test_coerce_audio_filepath_accepts_gradio_dict(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF")

    assert _coerce_audio_filepath({"path": str(audio_path)}) == audio_path


def test_coerce_audio_filepath_rejects_missing_file() -> None:
    with pytest.raises(Exception, match="Please upload an audio file"):
        _coerce_audio_filepath(None)


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("speech.wav", "audio/wav"),
        ("speech.mp3", "audio/mpeg"),
        ("speech.flac", "audio/flac"),
        ("speech.ogg", "audio/ogg"),
        ("speech.bin", "application/octet-stream"),
    ],
)
def test_audio_content_type(filename: str, content_type: str) -> None:
    assert _audio_content_type(Path(filename)) == content_type


def test_websocket_endpoint_uses_stream_scheme() -> None:
    assert _websocket_endpoint("http://127.0.0.1:8091/v1", "audio/speech/stream") == (
        "ws://127.0.0.1:8091/v1/audio/speech/stream"
    )
    assert _websocket_endpoint("https://tts.example/v1", "audio/speech/stream") == (
        "wss://tts.example/v1/audio/speech/stream"
    )


def test_tts_stream_config_includes_session_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_mini_conversation_app.config import config

    monkeypatch.setattr(config, "SELF_TTS_MODEL", "tts-model")

    assert _tts_stream_config("vivian", "English") == {
        "type": "session.config",
        "voice": "vivian",
        "response_format": "pcm",
        "stream_audio": True,
        "split_granularity": "sentence",
        "model": "tts-model",
        "language": "English",
    }


def test_decode_stream_audio_chunk_handles_split_pcm_sample() -> None:
    decoded, pending = _decode_stream_audio_chunk(b"\x01", "pcm", 24000, b"")
    assert decoded is None
    assert pending == b"\x01"

    decoded, pending = _decode_stream_audio_chunk(b"\x00\x02", "pcm", 24000, pending)
    assert pending == b"\x02"
    assert decoded is not None
    sample_rate, audio = decoded
    assert sample_rate == 24000
    assert np.array_equal(audio, np.array([[1]], dtype=np.int16))


def test_text_chunks_splits_long_text() -> None:
    assert _text_chunks("abcdef", chunk_size=2) == ["ab", "cd", "ef"]


def test_merge_stream_tool_call_accumulates_arguments() -> None:
    tool_calls: dict[int, dict] = {}
    _merge_stream_tool_call(
        tool_calls,
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "move", "arguments": "{\"x\""}},
    )
    _merge_stream_tool_call(tool_calls, {"index": 0, "function": {"arguments": ": 1}"}})

    assert tool_calls[0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "move", "arguments": "{\"x\": 1}"},
    }
