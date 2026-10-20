"""Engine boundary.

The pipeline owns everything that is not "PCM in, timed text out": ingestion,
chunking, checkpointing, post-processing, output formats. An engine only has
to transcribe <= ~10 min of 16 kHz mono float32 audio and return segments with
timestamps *relative to that array*. That keeps the model swappable (Whisper
today; Parakeet, a hosted API, or a fine-tune tomorrow) without touching the
rest of the system, and lets tests run the whole pipeline against a fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..schemas import ModelMeta, Segment


@dataclass
class EngineResult:
    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None
    extra: dict = field(default_factory=dict)


class Engine(Protocol):
    max_chunk_s: float | None
    """Largest chunk in seconds the engine accepts (e.g. a hosted API's upload cap), or None for no
    limit of its own. The pipeline caps its chunk size to this."""

    @property
    def meta(self) -> ModelMeta: ...

    @property
    def engine_id(self) -> str:
        """Stable identifier of model + decoding params. Part of the per-chunk checkpoint fingerprint,
        so a checkpoint is never reused across models or settings. (Job idempotency and worker
        routing use Settings.engine_key(), which needs no loaded model.)"""
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> EngineResult: ...

    def warmup(self) -> None:
        """Pay one-off start-up cost (e.g. kernel initialisation) before serving. May be a no-op."""
        ...
