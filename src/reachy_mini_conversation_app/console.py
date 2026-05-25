"""Bidirectional local audio stream for Reachy Mini headless mode."""

from __future__ import annotations
import os
import time
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.config import config, refresh_runtime_config_from_env
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings


try:
    from fastapi import FastAPI, Response, Request
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    FastAPI = object  # type: ignore
    Response = object  # type: ignore
    Request = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore


logger = logging.getLogger(__name__)

LOCAL_PLAYER_BACKEND = (
    getattr(MediaBackend, "LOCAL", None)
    or getattr(MediaBackend, "GSTREAMER", None)
    or getattr(MediaBackend, "DEFAULT", None)
)

LEGACY_STARTUP_ENV_NAMES = (
    "REACHY_MINI_CUSTOM_PROFILE",
    "REACHY_MINI_VOICE_OVERRIDE",
)


def _estimate_pending_playback_seconds(robot: ReachyMini) -> float:
    """Best-effort estimate of audio still queued in the local player."""
    media = getattr(robot, "media", None)
    audio = getattr(media, "audio", None)
    if audio is None:
        return 0.0

    next_pts_ns = getattr(audio, "_playback_next_pts_ns", None)
    get_running_time_ns = getattr(audio, "_get_playback_running_time_ns", None)
    if next_pts_ns is None or not callable(get_running_time_ns):
        return 0.0

    try:
        pending_ns = int(next_pts_ns) - int(get_running_time_ns())
    except Exception:
        return 0.0

    return max(0.0, pending_ns / 1e9)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: ConversationHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialize the stream with a conversation handler and media pipelines."""
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None

    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or the project template as a list of lines."""
        if env_path.exists():
            try:
                return env_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return []

        candidates = [
            env_path.parent / ".env.example",
            Path.cwd() / ".env.example",
            Path(__file__).parents[2] / ".env.example",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate.read_text(encoding="utf-8").splitlines()
            except Exception:
                pass
        return []

    def _persist_env_values(self, updates: dict[str, str]) -> None:
        """Persist non-empty environment values in memory and in the instance `.env`."""
        normalized = {name: (value or "").strip() for name, value in updates.items()}
        normalized = {name: value for name, value in normalized.items() if value}
        if not normalized:
            return

        for env_name, value in normalized.items():
            os.environ[env_name] = value
        refresh_runtime_config_from_env()

        if not self._instance_path:
            return

        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            for env_name, value in normalized.items():
                replaced = False
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{env_name}="):
                        lines[i] = f"{env_name}={value}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"{env_name}={value}")
            env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            logger.info("Persisted %s to %s", ", ".join(sorted(normalized)), env_path)
        except Exception as e:
            logger.warning("Failed to persist service configuration: %s", e)

    def _remove_persisted_env_values(self, env_names: tuple[str, ...]) -> None:
        """Remove keys from the instance `.env` without mutating the current runtime."""
        if not env_names or not self._instance_path:
            return
        env_path = Path(self._instance_path) / ".env"
        if not env_path.exists():
            return
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
            filtered = [line for line in lines if not any(line.strip().startswith(f"{name}=") for name in env_names)]
            env_path.write_text(("\n".join(filtered).rstrip() + "\n") if filtered else "", encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to remove persisted env values: %s", e)

    def _persist_personality(self, profile: Optional[str], voice_override: Optional[str] = None) -> None:
        """Persist startup profile and voice in instance-local UI settings."""
        selection = (profile or "").strip() or None
        normalized_voice = (voice_override or "").strip() or None
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            write_startup_settings(
                self._instance_path,
                profile=selection,
                voice=normalized_voice,
            )
            self._remove_persisted_env_values(LEGACY_STARTUP_ENV_NAMES)
        except Exception as e:
            logger.warning("Failed to persist startup personality settings: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read the saved startup personality from instance-local UI settings."""
        return read_startup_settings(self._instance_path).profile

    def _settings_payload(self) -> dict[str, object]:
        return {
            "backend": "self_hosted",
            "label": "Self-hosted OpenAI-compatible",
            "asr_base_url": config.SELF_ASR_BASE_URL,
            "asr_model": config.SELF_ASR_MODEL,
            "llm_base_url": config.SELF_LLM_BASE_URL,
            "llm_model": config.SELF_LLM_MODEL,
            "tts_base_url": config.SELF_TTS_BASE_URL,
            "tts_model": config.SELF_TTS_MODEL,
            "tts_voice": config.SELF_TTS_VOICE,
            "tts_voices": config.SELF_TTS_VOICES,
            "tts_response_format": config.SELF_TTS_RESPONSE_FORMAT,
        }

    def _init_settings_ui_if_needed(self) -> None:
        """Attach service and personality settings endpoints."""
        if self._settings_initialized or self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            return JSONResponse(self._settings_payload())

        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            return JSONResponse({"ready": True})

        @self._settings_app.post("/service_config")
        async def _service_config(request: Request) -> JSONResponse:
            env_map = {
                "self_openai_base_url": "SELF_OPENAI_BASE_URL",
                "self_openai_api_key": "SELF_OPENAI_API_KEY",
                "self_asr_base_url": "SELF_ASR_BASE_URL",
                "self_asr_api_key": "SELF_ASR_API_KEY",
                "self_asr_model": "SELF_ASR_MODEL",
                "self_llm_base_url": "SELF_LLM_BASE_URL",
                "self_llm_api_key": "SELF_LLM_API_KEY",
                "self_llm_model": "SELF_LLM_MODEL",
                "self_tts_base_url": "SELF_TTS_BASE_URL",
                "self_tts_api_key": "SELF_TTS_API_KEY",
                "self_tts_model": "SELF_TTS_MODEL",
                "self_tts_voice": "SELF_TTS_VOICE",
                "self_tts_voices": "SELF_TTS_VOICES",
                "self_tts_response_format": "SELF_TTS_RESPONSE_FORMAT",
            }
            try:
                raw = await request.json()
            except Exception:
                raw = {}
            if not isinstance(raw, dict):
                raw = {}
            updates = {env_name: str(raw[field]) for field, env_name in env_map.items() if raw.get(field)}
            self._persist_env_values(updates)
            return JSONResponse({"ok": True, **self._settings_payload()})

        self._settings_initialized = True

    def launch(self) -> None:
        """Start recorder/player and run async processing loops."""
        self._stop_event.clear()

        if self._instance_path:
            try:
                from dotenv import load_dotenv

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    refresh_runtime_config_from_env()
            except Exception:
                pass

        self._init_settings_ui_if_needed()

        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)
        apply_audio_startup_config(self._robot, logger=logger)

        async def runner() -> None:
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                logger.exception("Failed to mount personality routes")

            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="conversation-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines."""
        logger.info("Stopping LocalStream...")
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug("Error stopping recording: %s", e)

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug("Error stopping playback: %s", e)

        self._stop_event.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush queued playback audio immediately."""
        logger.info("User intervention: flushing player queue")
        backend = getattr(self._robot.media, "backend", None)
        audio = getattr(self._robot.media, "audio", None)
        if audio is not None:
            if (
                LOCAL_PLAYER_BACKEND is not None
                and backend == LOCAL_PLAYER_BACKEND
                and hasattr(audio, "clear_player")
                and callable(audio.clear_player)
            ):
                audio.clear_player()
            elif (
                backend == MediaBackend.WEBRTC
                and hasattr(audio, "clear_output_buffer")
                and callable(audio.clear_output_buffer)
            ):
                audio.clear_output_buffer()
            elif hasattr(audio, "clear_output_buffer") and callable(audio.clear_output_buffer):
                audio.clear_output_buffer()
            elif hasattr(audio, "clear_player") and callable(audio.clear_player):
                audio.clear_player()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug("Audio recording started at %s Hz", input_sample_rate)

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "...",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                if audio_data.size == 0:
                    continue

                if audio_data.ndim == 2:
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                audio_frame = audio_to_float32(audio_data)

                if input_sample_rate != output_sample_rate:
                    num_samples = int(len(audio_frame) * output_sample_rate / input_sample_rate)
                    if num_samples == 0:
                        continue
                    audio_frame = resample(audio_frame, num_samples)

                head_wobbler = self.handler.deps.head_wobbler
                if head_wobbler is not None:
                    playback_delay_s = _estimate_pending_playback_seconds(self._robot)
                    head_wobbler.feed_pcm(audio_data.reshape(1, -1), input_sample_rate, start_delay_s=playback_delay_s)

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)
