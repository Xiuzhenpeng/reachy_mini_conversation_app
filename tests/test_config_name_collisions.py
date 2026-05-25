from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod


def test_config_raises_on_external_profile_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in profile names collide."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)
    (external_profiles / "default").mkdir()

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_tool_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail fast when external/built-in tool names collide."""
    external_tools = tmp_path / "external_tools"
    external_tools.mkdir(parents=True)
    (external_tools / "dance.py").write_text("# collision with built-in dance tool\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", external_tools)

    with pytest.raises(RuntimeError, match="Ambiguous tool names"):
        config_mod.Config()


def test_config_raises_when_selected_external_profile_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when selected profile is absent from external root."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", "missing_profile")
    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Selected profile 'missing_profile' was not found"):
        config_mod.Config()


def test_backend_choice_is_always_self_hosted() -> None:
    """The app has a single self-hosted backend now."""
    assert config_mod.get_backend_choice() == config_mod.SELF_HOSTED_BACKEND
    assert config_mod.get_backend_label() == "Self-hosted OpenAI-compatible"


def test_refresh_runtime_config_reloads_self_hosted_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance-local .env reloads should update self-hosted runtime fields."""
    monkeypatch.setenv("SELF_OPENAI_BASE_URL", "http://localhost:9000")
    monkeypatch.setenv("SELF_OPENAI_API_KEY", "shared-token")
    monkeypatch.setenv("SELF_ASR_MODEL", "asr-test")
    monkeypatch.setenv("SELF_LLM_MODEL", "llm-test")
    monkeypatch.setenv("SELF_TTS_MODEL", "tts-test")
    monkeypatch.setenv("SELF_TTS_VOICE", "voice-test")
    monkeypatch.setenv("SELF_TTS_VOICES", "voice-test,voice-alt")

    config_mod.refresh_runtime_config_from_env()

    assert config_mod.config.SELF_OPENAI_BASE_URL == "http://localhost:9000/v1"
    assert config_mod.config.SELF_ASR_BASE_URL == "http://localhost:9000/v1"
    assert config_mod.config.SELF_LLM_BASE_URL == "http://localhost:9000/v1"
    assert config_mod.config.SELF_TTS_BASE_URL == "http://localhost:9000/v1"
    assert config_mod.config.SELF_OPENAI_API_KEY == "shared-token"
    assert config_mod.config.SELF_ASR_MODEL == "asr-test"
    assert config_mod.config.SELF_LLM_MODEL == "llm-test"
    assert config_mod.config.SELF_TTS_MODEL == "tts-test"
    assert config_mod.get_default_voice_for_backend() == "voice-test"
    assert config_mod.get_available_voices_for_backend() == ["voice-test", "voice-alt"]
