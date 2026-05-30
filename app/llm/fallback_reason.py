import re

from app.core.config import settings
from app.llm.exceptions import LLMOutputParsingError

_REDACTED = "[REDACTED]"
_MAX_REASON_LENGTH = 300


def sanitize_fallback_reason(exc: Exception, prefix: str | None = None) -> str:
    """Return a short, secret-safe fallback reason without a traceback."""
    message = _sanitize_message(exc)
    reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if prefix:
        reason = f"{prefix}: {reason}"
    return reason[:_MAX_REASON_LENGTH]


def _sanitize_message(exc: Exception) -> str:
    raw = " ".join(str(exc).split())
    raw = _redact_secrets(raw)

    if isinstance(exc, LLMOutputParsingError):
        if "not valid JSON" in raw:
            return "invalid JSON"
        missing_field = _extract_missing_field(raw)
        if missing_field:
            return f"missing required field {missing_field}"
        if "does not match" in raw:
            return "schema validation failed"

    return raw[:_MAX_REASON_LENGTH]


def _extract_missing_field(message: str) -> str | None:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s+Field required", message)
    if match:
        return match.group(1)
    return None


def _redact_secrets(message: str) -> str:
    redacted = message
    for secret in [settings.openai_api_key, settings.gemini_api_key]:
        if secret:
            redacted = redacted.replace(secret, _REDACTED)
    redacted = re.sub(r"sk-[A-Za-z0-9_-]+", _REDACTED, redacted)
    redacted = re.sub(r"AIza[0-9A-Za-z_-]+", _REDACTED, redacted)
    return redacted
