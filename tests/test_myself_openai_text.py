from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastrtc import AdditionalOutputs

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.myself_openai_text import (
    MyselfOpenAITextGenerator,
    to_chat_completion_tools,
    normalize_myself_openai_base_url,
)


def test_normalize_myself_openai_base_url_adds_v1_for_host_root() -> None:
    """A host root should become the standard OpenAI-compatible /v1 base URL."""
    assert normalize_myself_openai_base_url("http://118.191.0.226:26045/") == "http://118.191.0.226:26045/v1"


def test_to_chat_completion_tools_converts_realtime_specs() -> None:
    """Realtime function specs should be converted to Chat Completions tool specs."""
    tools = to_chat_completion_tools(
        [
            {
                "type": "function",
                "name": "move_head",
                "description": "Move head",
                "parameters": {"type": "object"},
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "move_head",
                "description": "Move head",
                "parameters": {"type": "object"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_myself_text_generator_calls_openai_compatible_chat() -> None:
    """The local text generator should call chat.completions on the configured endpoint."""
    generator = MyselfOpenAITextGenerator(
        base_url="http://118.191.0.226:26045/",
        api_key="DUMMY",
        model_name="test-model",
    )
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="local reply", tool_calls=[]),
                    )
                ]
            )

    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    reply = await generator.generate_reply(
        user_text="hello",
        instructions="be concise",
        tool_specs=[],
        deps=ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        tool_manager=MagicMock(),
    )

    assert reply == "local reply"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["messages"][-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_openai_realtime_disables_auto_response_when_myself_text_is_configured(monkeypatch: Any) -> None:
    """Realtime VAD should not auto-create model replies when local text generation is enabled."""
    monkeypatch.setattr(config, "MYSELF_OPENAI_API", "http://118.191.0.226:26045/")
    monkeypatch.setattr(config, "MYSELF_OPENAI_API_KEY", "DUMMY")
    monkeypatch.setattr(config, "MYSELF_OPENAI_MODEL", "test-model")
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda default=None: "alloy")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _deps: [])

    handler = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    session_config = handler._get_session_config([])

    turn_detection = session_config["audio"]["input"]["turn_detection"]
    assert turn_detection["create_response"] is False


@pytest.mark.asyncio
async def test_myself_text_turn_queues_assistant_text_and_tts_response(monkeypatch: Any) -> None:
    """A locally generated text turn should update chat output and request realtime speech."""
    monkeypatch.setattr(config, "MYSELF_OPENAI_API", None)
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(rt_mod, "get_active_tool_specs", lambda _deps: [])

    handler = OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    class FakeTextGenerator:
        async def generate_reply(self, **_kwargs: Any) -> str:
            return "hello from local text"

    handler._myself_text_generator = FakeTextGenerator()  # type: ignore[assignment]
    handler.connection = MagicMock()

    await handler._handle_myself_text_turn("hello")

    output = await handler.output_queue.get()
    assert isinstance(output, AdditionalOutputs)
    assert output.args[0]["content"] == "hello from local text"

    queued_response = await handler._pending_responses.get()
    response = queued_response["response"]
    assert response["output_modalities"] == ["audio"]
    assert "hello from local text" in response["input"][0]["content"][0]["text"]
