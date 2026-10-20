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
    @property
    def meta(self) -> ModelMeta: ...

    @property
    def engine_id(self) -> str:
        """Stable identifier of model + decoding params; part of every cache/idempotency key."""
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> EngineResult: ...
