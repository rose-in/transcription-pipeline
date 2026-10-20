"""Public data model. The JSON transcript is the canonical output; SRT/VTT/TXT are renderings of it."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .languages import LANGUAGES

SCHEMA_VERSION = "1.0"


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class Segment(BaseModel):
    id: int = 0
    start: float = Field(description="Seconds from the start of the original file")
    end: float
    text: str
    speaker: str | None = Field(None, description="Channel label in split-channel mode")
    # Raw model confidence signals, kept so downstream consumers can apply their own policy.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    flags: list[str] = Field(default_factory=list, description="e.g. low_confidence, redacted")
    words: list[Word] | None = None


class DroppedSegment(BaseModel):
    """Segments removed by post-processing are kept for audit instead of silently vanishing."""

    segment: Segment
    reason: str


class AudioMeta(BaseModel):
    filename: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    container: str
    codec: str
    sample_rate: int
    channels: int
    container_duration_s: float | None = Field(None, description="As reported by the container; can be wrong")
    start_offset_s: float = Field(
        0.0, description="Audio stream start time in the container; add it to map segment times onto a video's timeline"
    )
    decoded_duration_s: float = Field(description="Ground truth: samples actually decoded / 16000")


class ModelMeta(BaseModel):
    engine: str
    model: str
    device: str | None = None
    compute_type: str | None = None
    revision: str | None = None
    params: dict = Field(default_factory=dict)
    versions: dict = Field(default_factory=dict)


class ChunkMeta(BaseModel):
    index: int
    channel: str | None = None
    offset_s: float
    duration_s: float
    segments: int
    cached: bool = False


class Timing(BaseModel):
    decode_s: float = 0.0
    inference_s: float = 0.0
    total_s: float = 0.0
    rtf: float = Field(0.0, description="total_s / audio seconds (lower is better)")


class Transcript(BaseModel):
    schema_version: str = SCHEMA_VERSION
    language: str | None = None
    language_probability: float | None = None
    duration_s: float
    text: str = ""
    segments: list[Segment] = Field(default_factory=list)
    audio: AudioMeta
    model: ModelMeta
    options: dict = Field(default_factory=dict)
    chunks: list[ChunkMeta] = Field(default_factory=list)
    dropped: list[DroppedSegment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timing: Timing = Field(default_factory=Timing)
    diagnostics: dict = Field(
        default_factory=dict,
        description="Engine counters summed over chunks "
        "(e.g. ranges re-decoded by the safety net and the time it took)",
    )


class TranscribeOptions(BaseModel):
    """Per-request knobs. Anything that changes the output must live here so it is part of the cache key."""

    language: str | None = Field(None, description="ISO-639-1 code; auto-detected and pinned if omitted")
    word_timestamps: bool = False
    channel_mode: Literal["mix", "split"] = Field(
        "mix", description="'split' transcribes each channel separately (e.g. agent/customer call recordings)"
    )
    redact_pii: bool = False
    prompt: str | None = Field(None, max_length=800, description="Domain vocabulary hint (names, jargon)")

    @field_validator("language")
    @classmethod
    def _known_language(cls, v: str | None) -> str | None:
        # Validate at the edge: an unknown code would otherwise surface as a retried engine failure.
        if v is None:
            return None
        v = v.strip().lower()
        if v not in LANGUAGES:
            raise ValueError(f"unsupported language {v!r}; use an ISO-639-1 code such as 'en'")
        return v

    def cache_key(self, engine_id: str) -> str:
        blob = json.dumps({"opts": self.model_dump(), "engine": engine_id}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class JobError(BaseModel):
    code: str
    message: str
    status: int | None = None


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    filename: str | None = None
    options: TranscribeOptions
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 0
    progress: float = 0.0
    processed_s: float = 0.0
    duration_s: float | None = None
    error: JobError | None = None
