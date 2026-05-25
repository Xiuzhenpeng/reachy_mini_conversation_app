import os
import sys
import logging
from pathlib import Path
from importlib.resources import files

from dotenv import find_dotenv, load_dotenv


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
PROJECT_ROOT = Path(__file__).parents[2].resolve()

SELF_HOSTED_BACKEND = "self_hosted"
SELF_HOSTED_BACKEND_LABEL = "Self-hosted OpenAI-compatible"

logger = logging.getLogger(__name__)


def _is_source_checkout_root(root: Path) -> bool:
    """Return whether the given root looks like this project's source checkout."""
    return (root / "pyproject.toml").is_file() and (root / "src" / "reachy_mini_conversation_app").is_dir()


def _packaged_profiles_directory() -> Path | None:
    """Return the installed wheel's packaged profiles directory when available."""
    try:
        return Path(str(files("reachy_talk_data").joinpath("profiles")))
    except Exception:
        return None


def _resolve_default_profiles_directory() -> Path:
    """Resolve built-in profiles from source checkout or installed package data."""
    source_profiles = PROJECT_ROOT / "profiles"
    if _is_source_checkout_root(PROJECT_ROOT) and source_profiles.is_dir():
        return source_profiles

    packaged_profiles = _packaged_profiles_directory()
    if packaged_profiles is not None and packaged_profiles.is_dir():
        return packaged_profiles

    return source_profiles


DEFAULT_PROFILES_DIRECTORY = _resolve_default_profiles_directory()


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer value for %s=%r, using default=%s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value for %s=%r, using default=%s", name, raw, default)
        return default


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize a root service URL to an OpenAI-compatible /v1 base URL."""
    candidate = (base_url or "").strip().rstrip("/")
    if not candidate:
        raise ValueError("OpenAI-compatible base URL must be non-empty")
    if candidate.endswith("/v1"):
        return candidate
    return f"{candidate}/v1"


def _collect_profile_names(profiles_root: Path) -> set[str]:
    """Return profile folder names from a profiles root directory."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def _collect_tool_module_names(tools_root: Path) -> set[str]:
    """Return tool module names from a tools directory."""
    if not tools_root.exists() or not tools_root.is_dir():
        return set()
    ignored = {"__init__", "core_tools"}
    return {p.stem for p in tools_root.glob("*.py") if p.is_file() and p.stem not in ignored}


def _raise_on_name_collisions(
    *,
    label: str,
    external_root: Path,
    internal_root: Path,
    external_names: set[str],
    internal_names: set[str],
) -> None:
    """Raise with a clear message when external/internal names collide."""
    collisions = sorted(external_names & internal_names)
    if not collisions:
        return

    raise RuntimeError(
        f"Config.__init__(): Ambiguous {label} names found in both external and built-in libraries: {collisions}. "
        f"External {label} root: {external_root}. Built-in {label} root: {internal_root}. "
        f"Please rename the conflicting external {label}(s) to continue."
    )


# Validate LOCKED_PROFILE at startup.
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info("Configuration loaded from %s", dotenv_path)
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration for self-hosted OpenAI-compatible ASR, LLM, and TTS services."""

    SELF_OPENAI_BASE_URL = normalize_openai_base_url(
        os.getenv("SELF_OPENAI_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "http://127.0.0.1:8000"
    )
    SELF_OPENAI_API_KEY = os.getenv("SELF_OPENAI_API_KEY", "DUMMY")

    SELF_ASR_BASE_URL = normalize_openai_base_url(os.getenv("SELF_ASR_BASE_URL") or SELF_OPENAI_BASE_URL)
    SELF_ASR_API_KEY = os.getenv("SELF_ASR_API_KEY") or SELF_OPENAI_API_KEY
    SELF_ASR_MODEL = os.getenv("SELF_ASR_MODEL", "whisper-1").strip()
    SELF_ASR_LANGUAGE = os.getenv("SELF_ASR_LANGUAGE", "").strip()
    SELF_ASR_SAMPLE_RATE = _env_int("SELF_ASR_SAMPLE_RATE", 16000)

    SELF_LLM_BASE_URL = normalize_openai_base_url(os.getenv("SELF_LLM_BASE_URL") or SELF_OPENAI_BASE_URL)
    SELF_LLM_API_KEY = os.getenv("SELF_LLM_API_KEY") or SELF_OPENAI_API_KEY
    SELF_LLM_MODEL = os.getenv("SELF_LLM_MODEL", "local-model")
    SELF_LLM_TEMPERATURE = _env_float("SELF_LLM_TEMPERATURE", 0.7)
    SELF_LLM_MAX_HISTORY_MESSAGES = _env_int("SELF_LLM_MAX_HISTORY_MESSAGES", 24)
    SELF_LLM_MAX_TOOL_ROUNDS = _env_int("SELF_LLM_MAX_TOOL_ROUNDS", 4)

    SELF_TTS_BASE_URL = normalize_openai_base_url(os.getenv("SELF_TTS_BASE_URL") or SELF_OPENAI_BASE_URL)
    SELF_TTS_API_KEY = os.getenv("SELF_TTS_API_KEY") or SELF_OPENAI_API_KEY
    SELF_TTS_MODEL = os.getenv("SELF_TTS_MODEL", "tts-1").strip()
    SELF_TTS_VOICE = os.getenv("SELF_TTS_VOICE", "default")
    SELF_TTS_VOICES = _split_csv(os.getenv("SELF_TTS_VOICES")) or [SELF_TTS_VOICE]
    SELF_TTS_LANGUAGE = os.getenv("SELF_TTS_LANGUAGE", "").strip()
    SELF_TTS_RESPONSE_FORMAT = os.getenv("SELF_TTS_RESPONSE_FORMAT", "wav").strip().lower() or "wav"
    SELF_TTS_SEND_RESPONSE_FORMAT = _env_flag("SELF_TTS_SEND_RESPONSE_FORMAT", default=False)
    SELF_TTS_SAMPLE_RATE = _env_int("SELF_TTS_SAMPLE_RATE", 24000)

    SELF_VAD_START_DBFS = _env_float("SELF_VAD_START_DBFS", -42.0)
    SELF_VAD_STOP_DBFS = _env_float("SELF_VAD_STOP_DBFS", -50.0)
    SELF_VAD_MIN_SPEECH_MS = _env_int("SELF_VAD_MIN_SPEECH_MS", 250)
    SELF_VAD_SILENCE_MS = _env_int("SELF_VAD_SILENCE_MS", 700)
    SELF_VAD_PRE_ROLL_MS = _env_int("SELF_VAD_PRE_ROLL_MS", 300)
    SELF_VAD_MAX_UTTERANCE_SECONDS = _env_float("SELF_VAD_MAX_UTTERANCE_SECONDS", 30.0)

    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY = Path(_profiles_directory_env) if _profiles_directory_env else DEFAULT_PROFILES_DIRECTORY
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    logger.debug(
        "Self-hosted services: asr=%s model=%s, llm=%s model=%s, tts=%s model=%s voice=%s",
        SELF_ASR_BASE_URL,
        SELF_ASR_MODEL,
        SELF_LLM_BASE_URL,
        SELF_LLM_MODEL,
        SELF_TTS_BASE_URL,
        SELF_TTS_MODEL,
        SELF_TTS_VOICE,
    )

    def __init__(self) -> None:
        """Validate profile and tool roots."""
        if self.REACHY_MINI_CUSTOM_PROFILE and self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            selected_profile_path = self.PROFILES_DIRECTORY / self.REACHY_MINI_CUSTOM_PROFILE
            if not selected_profile_path.is_dir():
                available_profiles = sorted(_collect_profile_names(self.PROFILES_DIRECTORY))
                raise RuntimeError(
                    "Config.__init__(): Selected profile "
                    f"'{self.REACHY_MINI_CUSTOM_PROFILE}' was not found in external profiles root "
                    f"{self.PROFILES_DIRECTORY}. "
                    f"Available external profiles: {available_profiles}. "
                    "Either set 'REACHY_MINI_CUSTOM_PROFILE' to one of the available external profiles "
                    "or unset 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' to use built-in profiles."
                )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            external_profiles = _collect_profile_names(self.PROFILES_DIRECTORY)
            internal_profiles = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
            _raise_on_name_collisions(
                label="profile",
                external_root=self.PROFILES_DIRECTORY,
                internal_root=DEFAULT_PROFILES_DIRECTORY,
                external_names=external_profiles,
                internal_names=internal_profiles,
            )

        if self.TOOLS_DIRECTORY is not None:
            builtin_tools_root = Path(__file__).parent / "tools"
            external_tools = _collect_tool_module_names(self.TOOLS_DIRECTORY)
            internal_tools = _collect_tool_module_names(builtin_tools_root)
            _raise_on_name_collisions(
                label="tool",
                external_root=self.TOOLS_DIRECTORY,
                internal_root=builtin_tools_root,
                external_names=external_tools,
                internal_names=internal_tools,
            )

        logger.info("Using %s services for ASR, LLM, and TTS.", SELF_HOSTED_BACKEND_LABEL)


config = Config()


def refresh_runtime_config_from_env() -> None:
    """Refresh mutable runtime config fields from the current environment."""
    config.SELF_OPENAI_BASE_URL = normalize_openai_base_url(
        os.getenv("SELF_OPENAI_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "http://127.0.0.1:8000"
    )
    config.SELF_OPENAI_API_KEY = os.getenv("SELF_OPENAI_API_KEY", "DUMMY")

    config.SELF_ASR_BASE_URL = normalize_openai_base_url(os.getenv("SELF_ASR_BASE_URL") or config.SELF_OPENAI_BASE_URL)
    config.SELF_ASR_API_KEY = os.getenv("SELF_ASR_API_KEY") or config.SELF_OPENAI_API_KEY
    config.SELF_ASR_MODEL = os.getenv("SELF_ASR_MODEL", "whisper-1").strip()
    config.SELF_ASR_LANGUAGE = os.getenv("SELF_ASR_LANGUAGE", "").strip()
    config.SELF_ASR_SAMPLE_RATE = _env_int("SELF_ASR_SAMPLE_RATE", 16000)

    config.SELF_LLM_BASE_URL = normalize_openai_base_url(os.getenv("SELF_LLM_BASE_URL") or config.SELF_OPENAI_BASE_URL)
    config.SELF_LLM_API_KEY = os.getenv("SELF_LLM_API_KEY") or config.SELF_OPENAI_API_KEY
    config.SELF_LLM_MODEL = os.getenv("SELF_LLM_MODEL", "local-model")
    config.SELF_LLM_TEMPERATURE = _env_float("SELF_LLM_TEMPERATURE", 0.7)
    config.SELF_LLM_MAX_HISTORY_MESSAGES = _env_int("SELF_LLM_MAX_HISTORY_MESSAGES", 24)
    config.SELF_LLM_MAX_TOOL_ROUNDS = _env_int("SELF_LLM_MAX_TOOL_ROUNDS", 4)

    config.SELF_TTS_BASE_URL = normalize_openai_base_url(os.getenv("SELF_TTS_BASE_URL") or config.SELF_OPENAI_BASE_URL)
    config.SELF_TTS_API_KEY = os.getenv("SELF_TTS_API_KEY") or config.SELF_OPENAI_API_KEY
    config.SELF_TTS_MODEL = os.getenv("SELF_TTS_MODEL", "tts-1").strip()
    config.SELF_TTS_VOICE = os.getenv("SELF_TTS_VOICE", "default")
    config.SELF_TTS_VOICES = _split_csv(os.getenv("SELF_TTS_VOICES")) or [config.SELF_TTS_VOICE]
    config.SELF_TTS_LANGUAGE = os.getenv("SELF_TTS_LANGUAGE", "").strip()
    config.SELF_TTS_RESPONSE_FORMAT = os.getenv("SELF_TTS_RESPONSE_FORMAT", "wav").strip().lower() or "wav"
    config.SELF_TTS_SEND_RESPONSE_FORMAT = _env_flag("SELF_TTS_SEND_RESPONSE_FORMAT", default=False)
    config.SELF_TTS_SAMPLE_RATE = _env_int("SELF_TTS_SAMPLE_RATE", 24000)

    config.SELF_VAD_START_DBFS = _env_float("SELF_VAD_START_DBFS", -42.0)
    config.SELF_VAD_STOP_DBFS = _env_float("SELF_VAD_STOP_DBFS", -50.0)
    config.SELF_VAD_MIN_SPEECH_MS = _env_int("SELF_VAD_MIN_SPEECH_MS", 250)
    config.SELF_VAD_SILENCE_MS = _env_int("SELF_VAD_SILENCE_MS", 700)
    config.SELF_VAD_PRE_ROLL_MS = _env_int("SELF_VAD_PRE_ROLL_MS", 300)
    config.SELF_VAD_MAX_UTTERANCE_SECONDS = _env_float("SELF_VAD_MAX_UTTERANCE_SECONDS", 30.0)

    config.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")


def get_backend_choice(model_name: str | None = None) -> str:
    """Return the only supported backend family."""
    return SELF_HOSTED_BACKEND


def get_model_name_for_backend(backend: str | None = None) -> str:
    """Return the configured LLM model name."""
    return config.SELF_LLM_MODEL


def get_backend_label(backend: str | None = None) -> str:
    """Return a human-readable label for the active backend."""
    return SELF_HOSTED_BACKEND_LABEL


def get_available_voices_for_backend(backend: str | None = None) -> list[str]:
    """Return the configured TTS voice list."""
    return list(config.SELF_TTS_VOICES)


def get_default_voice_for_backend(backend: str | None = None) -> str:
    """Return the configured default TTS voice."""
    return config.SELF_TTS_VOICE


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env."""
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        if profile:
            os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
