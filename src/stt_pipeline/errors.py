"""Typed pipeline errors.

Every failure a *client* can cause maps to a stable machine-readable code and a
4xx status, so callers can tell "your file is bad" (don't retry) apart from
"we broke" (5xx, safe to retry). The job runner uses the same split to decide
whether a failed job is worth retrying.
"""

from __future__ import annotations


class PipelineError(Exception):
    status: int = 500
    code: str = "internal_error"
    retryable: bool = False

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class EmptyFile(PipelineError):
    status, code = 400, "empty_file"


class FileTooLarge(PipelineError):
    status, code = 413, "file_too_large"


class AudioTooLong(PipelineError):
    status, code = 413, "audio_too_long"


class UnsupportedMedia(PipelineError):
    """Content sniffing did not recognise an allowed container (extension and MIME type are ignored)."""

    status, code = 415, "unsupported_media"


class NoAudioStream(PipelineError):
    status, code = 422, "no_audio_stream"


class UndecodableAudio(PipelineError):
    """Looks like an allowed container but FFmpeg cannot open or decode it."""

    status, code = 422, "undecodable_audio"


class InvalidOptions(PipelineError):
    status, code = 422, "invalid_options"


class QueueFull(PipelineError):
    """Backpressure: we would rather shed load with a clear 503 than accept work we cannot finish."""

    status, code, retryable = 503, "queue_full", True


class EngineFailure(PipelineError):
    """The model failed (OOM, driver error...). Transient by assumption, so the runner retries."""

    status, code, retryable = 500, "engine_failure", True


class ProviderError(PipelineError):
    """A hosted engine rejected *our* configuration (bad key, unknown model, revoked access).
    Server-side problem: 5xx, not retryable, provider details logged rather than echoed."""

    status, code, retryable = 502, "provider_error", False
