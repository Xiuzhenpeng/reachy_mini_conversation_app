from __future__ import annotations
import json
import uuid
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from reachy_mini_conversation_app.tools.tool_constants import SystemTool
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)

_FALLBACK_MODEL = "local-model"
_MAX_HISTORY_MESSAGES = 24
_MAX_TOOL_ROUNDS = 4
_SYSTEM_TOOL_NAMES = {tool.value for tool in SystemTool}


def normalize_myself_openai_base_url(base_url: str) -> str:
    """Normalize a root OpenAI-compatible API URL to the SDK base URL."""
    candidate = base_url.strip()
    if not candidate:
        raise ValueError("MYSELF_OPENAI_API must be a non-empty URL")

    parsed = urlsplit(candidate)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    normalized = parsed._replace(path=path, query="", fragment="")
    return urlunsplit(normalized)


def to_chat_completion_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert this app's tool specs to standard Chat Completions tool specs."""
    chat_tools: list[dict[str, Any]] = []
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
        chat_tools.append({"type": "function", "function": function_spec})
    return chat_tools


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _get_value(part, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _sanitize_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Remove transport-only image payloads before returning tool output to the text model."""
    if tool_name == "camera" and "b64_im" in result:
        sanitized = dict(result)
        sanitized.pop("b64_im", None)
        sanitized["image_attached"] = True
        return sanitized
    return result


def _tool_call_parts(tool_call: Any) -> tuple[str, str, str]:
    function_obj = _get_value(tool_call, "function", {})
    call_id = _get_value(tool_call, "id") or str(uuid.uuid4())
    name = _get_value(function_obj, "name")
    arguments = _get_value(function_obj, "arguments", "{}")
    if not isinstance(name, str):
        name = ""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return str(call_id), name, arguments


def _assistant_message_from_model(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": _message_content_to_text(_get_value(message, "content")),
    }
    if tool_calls:
        assistant_message["tool_calls"] = []
        for tool_call in tool_calls:
            call_id, name, arguments = _tool_call_parts(tool_call)
            assistant_message["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
    return assistant_message


class MyselfOpenAITextGenerator:
    """OpenAI-compatible text generator used before realtime TTS."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str | None = None,
        max_history_messages: int = _MAX_HISTORY_MESSAGES,
        max_tool_rounds: int = _MAX_TOOL_ROUNDS,
    ) -> None:
        """Initialize the OpenAI-compatible chat client and local history."""
        self.base_url = normalize_myself_openai_base_url(base_url)
        self.api_key = api_key or "DUMMY"
        self._configured_model_name = (model_name or "").strip() or None
        self._resolved_model_name: str | None = self._configured_model_name
        self._max_history_messages = max_history_messages
        self._max_tool_rounds = max_tool_rounds
        self._history: list[dict[str, Any]] = []
        self._system_instructions: str | None = None
        self._completion_supports_tools = True
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_config(cls) -> "MyselfOpenAITextGenerator":
        """Build a text generator from runtime config."""
        base_url = (config.MYSELF_OPENAI_API or "").strip()
        if not base_url:
            raise ValueError("MYSELF_OPENAI_API is not configured")
        return cls(
            base_url=base_url,
            api_key=(config.MYSELF_OPENAI_API_KEY or "DUMMY").strip(),
            model_name=(config.MYSELF_OPENAI_MODEL or "").strip() or None,
        )

    def reset(self) -> None:
        """Clear the local conversation history."""
        self._history.clear()
        self._system_instructions = None

    async def generate_reply(
        self,
        *,
        user_text: str,
        instructions: str,
        tool_specs: list[dict[str, Any]],
        deps: ToolDependencies,
        tool_manager: BackgroundToolManager,
    ) -> str:
        """Generate the assistant text, executing OpenAI-style tool calls if requested."""
        if self._system_instructions != instructions:
            self._history.clear()
            self._system_instructions = instructions

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            *self._history,
            {"role": "user", "content": user_text},
        ]
        tools = to_chat_completion_tools(tool_specs)

        for round_index in range(self._max_tool_rounds + 1):
            completion = await self._create_completion(messages, tools)
            message = self._first_message(completion)
            tool_calls = list(_get_value(message, "tool_calls", []) or [])
            content = _message_content_to_text(_get_value(message, "content")).strip()

            if tool_calls and round_index < self._max_tool_rounds:
                messages.append(_assistant_message_from_model(message, tool_calls))
                for tool_call in tool_calls:
                    call_id, tool_name, args_json = _tool_call_parts(tool_call)
                    tool_result = await self._run_tool(tool_name, args_json, deps, tool_manager)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                        }
                    )
                continue

            if not content and tool_calls:
                content = "I used the requested tool."
            if not content:
                content = "I am ready."

            messages.append({"role": "assistant", "content": content})
            self._history = self._trim_history(messages[1:])
            return content

        fallback = "I used the available tools, but I could not complete a final answer."
        messages.append({"role": "assistant", "content": fallback})
        self._history = self._trim_history(messages[1:])
        return fallback

    async def _create_completion(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        model_name = await self._get_model_name()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if tools and self._completion_supports_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            return await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if tools and self._completion_supports_tools:
                self._completion_supports_tools = False
                logger.warning(
                    "MYSELF_OPENAI_API chat completion rejected tools; retrying without tool specs: %s",
                    exc,
                )
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                return await self.client.chat.completions.create(**kwargs)
            raise

    async def _get_model_name(self) -> str:
        if self._resolved_model_name:
            return self._resolved_model_name

        try:
            models = await self.client.models.list()
            data = _get_value(models, "data", []) or []
            if data:
                model_id = _get_value(data[0], "id")
                if isinstance(model_id, str) and model_id:
                    self._resolved_model_name = model_id
                    return model_id
        except Exception as exc:
            logger.warning(
                "Could not discover model from MYSELF_OPENAI_API; falling back to %s: %s",
                _FALLBACK_MODEL,
                exc,
            )

        self._resolved_model_name = _FALLBACK_MODEL
        return self._resolved_model_name

    @staticmethod
    def _first_message(completion: Any) -> Any:
        choices = _get_value(completion, "choices", []) or []
        if not choices:
            raise RuntimeError("MYSELF_OPENAI_API returned no choices")
        return _get_value(choices[0], "message", {})

    async def _run_tool(
        self,
        tool_name: str,
        args_json: str,
        deps: ToolDependencies,
        tool_manager: BackgroundToolManager,
    ) -> dict[str, Any]:
        if not tool_name:
            return {"error": "tool call did not include a function name"}

        if tool_name in _SYSTEM_TOOL_NAMES:
            result = await dispatch_tool_call_with_manager(tool_name, args_json, deps, tool_manager)
        else:
            result = await dispatch_tool_call(tool_name, args_json, deps)

        if isinstance(result, dict):
            return _sanitize_tool_result(tool_name, result)
        return {"result": result}

    def _trim_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self._max_history_messages:
            return list(messages)
        return list(messages[-self._max_history_messages :])
