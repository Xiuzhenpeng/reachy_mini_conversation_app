from __future__ import annotations

from pathlib import Path

import pytest

from reachy_mini_conversation_app.gradio_test_ui import _audio_content_type, _coerce_audio_filepath


def test_coerce_audio_filepath_accepts_gradio_filepath(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF")

    assert _coerce_audio_filepath(str(audio_path)) == audio_path


def test_coerce_audio_filepath_accepts_gradio_dict(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF")

    assert _coerce_audio_filepath({"path": str(audio_path)}) == audio_path


def test_coerce_audio_filepath_rejects_missing_file() -> None:
    with pytest.raises(Exception, match="请上传音频文件"):
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
