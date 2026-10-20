"""12-factor settings: everything deploy-specific comes from STT_* environment variables."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"STT_{name}")
    if raw is None:
        return default
    return cast(raw)


# Env vars that feed Settings.engine_key(): the API and every worker must agree on all of them
# (STT_API_BASE only matters for the openai-compatible engine).
ENGINE_KEY_ENV_VARS = (
    "STT_ENGINE",
    "STT_MODEL",
    "STT_MODEL_REVISION",
    "STT_DEVICE",
    "STT_COMPUTE_TYPE",
    "STT_BEAM_SIZE",
    "STT_CHUNK_SECONDS",
    "STT_API_BASE",
)


@dataclass
class Settings:
    engine: str = field(default_factory=lambda: _env("ENGINE", "faster-whisper"))
    # large-v3-turbo is the production default on GPU; `small` keeps the CPU demo snappy.
    # Unset -> engine-appropriate default (see resolved_model), so switching STT_ENGINE alone works.
    model: str | None = field(default_factory=lambda: _env("MODEL", None))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    compute_type: str = field(default_factory=lambda: _env("COMPUTE_TYPE", "default"))
    batch_size: int = field(default_factory=lambda: _env("BATCH_SIZE", 8, int))
    beam_size: int = field(default_factory=lambda: _env("BEAM_SIZE", 5, int))
    cpu_threads: int = field(default_factory=lambda: _env("CPU_THREADS", 0, int))
    # Pin in production: "turbo" etc. resolve to third-party HF repos that can change underneath you.
    model_revision: str | None = field(default_factory=lambda: _env("MODEL_REVISION", None))
    # Hosted engine (OpenAI-compatible /audio/transcriptions: OpenAI whisper-1, Groq, ...)
    api_base: str = field(default_factory=lambda: _env("API_BASE", "https://api.openai.com/v1"))
    api_key: str | None = field(default_factory=lambda: _env("API_KEY", None), repr=False)

    chunk_s: float = field(default_factory=lambda: _env("CHUNK_SECONDS", 600.0, float))
    max_duration_s: float = field(default_factory=lambda: _env("MAX_DURATION_SECONDS", 4 * 3600.0, float))
    max_upload_bytes: int = field(default_factory=lambda: _env("MAX_UPLOAD_MB", 1024, int) * 1024 * 1024)
    sync_max_duration_s: float = field(default_factory=lambda: _env("SYNC_MAX_DURATION_SECONDS", 300.0, float))
    sync_timeout_s: float = field(default_factory=lambda: _env("SYNC_TIMEOUT_SECONDS", 120.0, float))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    max_queue: int = field(default_factory=lambda: _env("MAX_QUEUE", 100, int))
    max_attempts: int = field(default_factory=lambda: _env("MAX_ATTEMPTS", 3, int))
    lease_s: float = field(default_factory=lambda: _env("LEASE_SECONDS", 60.0, float))
    # A worker that completes no chunk for this long is wedged (hung decoder/driver): it stops renewing
    # its lease and exits so the orchestrator restarts it and the job is retried elsewhere.
    stall_timeout_s: float = field(default_factory=lambda: _env("STALL_TIMEOUT_SECONDS", 1800.0, float))
    # Queued jobs whose engine config has had no live worker for this long are failed (retryable).
    stranded_after_s: float = field(default_factory=lambda: _env("STRANDED_AFTER_SECONDS", 3600.0, float))
    max_split_channels: int = field(default_factory=lambda: _env("MAX_SPLIT_CHANNELS", 4, int))
    result_ttl_s: float = field(default_factory=lambda: _env("RESULT_TTL_SECONDS", 7 * 24 * 3600.0, float))

    @property
    def resolved_model(self) -> str:
        return self.model or ("whisper-1" if self.engine == "openai-compatible" else "small")

    def engine_key(self) -> str:
        """Identity of 'what would produce the transcript', computable without loading weights.
        The API uses it for idempotency keys; workers only claim jobs with their own key, so the
        API and the workers MUST share every setting in ENGINE_KEY_ENV_VARS (compose uses one env anchor).
        Throughput-only knobs (batch size, threads, queue sizes) are deliberately excluded.
        Checkpoint reuse is checked separately against the loaded engine's stricter Engine.engine_id."""
        from . import PIPELINE_VERSION

        device, compute = self.device, self.compute_type
        versions = None
        if self.engine == "faster-whisper":
            # Resolve "auto"/"default" with the engine's own rules, so a CPU pool (int8) and a GPU pool (fp16)
            # never share a key, and pin library versions so an upgrade doesn't replay results from the old one.
            from importlib.metadata import version

            from .engines.devices import resolve_compute_type, resolve_device

            device = resolve_device(device)
            compute = resolve_compute_type(compute, device)
            versions = [version("faster-whisper"), version("ctranslate2")]
        blob = json.dumps(
            [
                self.engine,
                self.resolved_model,
                self.model_revision,
                device,
                compute,
                self.beam_size,
                self.chunk_s,
                self.api_base if self.engine == "openai-compatible" else None,
                versions,
                PIPELINE_VERSION,
            ]
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def engine_kwargs(self) -> dict:
        if self.engine == "mock":
            return {}
        if self.engine == "openai-compatible":
            return {"base_url": self.api_base, "api_key": self.api_key, "model": self.resolved_model}
        return {
            "revision": self.model_revision,
            "model": self.resolved_model,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "beam_size": self.beam_size,
            "cpu_threads": self.cpu_threads,
        }

    def pipeline_kwargs(self) -> dict:
        """Keyword arguments for `PipelineConfig`: the one place deploy settings reach the pipeline."""
        return {
            "chunk_s": self.chunk_s,
            "max_duration_s": self.max_duration_s,
            "max_split_channels": self.max_split_channels,
        }
