"""Deterministic fake engine for tests and local development without model weights.

It "transcribes" by energy: every contiguous loud region becomes a segment whose
text encodes its own timing. That makes chunking/offset/merge bugs visible as
wrong numbers, with no model download and millisecond runtimes.
"""

from __future__ import annotations

import numpy as np

from ..audio import TARGET_SR
from ..schemas import ModelMeta, Segment, Word
from .base import EngineResult


class MockEngine:
    max_chunk_s: float | None = None

    def __init__(self, threshold: float = 0.02, min_gap_s: float = 0.3, fail_on_call: int | None = None):
        self.threshold = threshold
        self.min_gap_s = min_gap_s
        self.fail_on_call = fail_on_call  # simulate a crash for checkpoint/resume tests
        self.calls = 0

    @property
    def meta(self) -> ModelMeta:
        return ModelMeta(engine="mock", model="energy-vad", params={"threshold": self.threshold})

    @property
    def engine_id(self) -> str:
        return "mock-v1"

    def warmup(self) -> None:
        """No model to load."""

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, word_timestamps: bool = False, prompt: str | None = None
    ) -> EngineResult:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("simulated engine crash")
        frame = TARGET_SR // 100  # 10 ms
        n = len(audio) - len(audio) % frame
        if n == 0:
            return EngineResult([], language or "en", 1.0)
        rms = np.sqrt(np.mean(audio[:n].reshape(-1, frame) ** 2, axis=1))
        loud = rms > self.threshold
        segments, start, last = [], None, None
        gap = int(self.min_gap_s * 100)
        for i, v in enumerate(loud):
            if v:
                start = i if start is None else start
                last = i
            elif start is not None and i - last > gap:
                segments.append((start, last + 1))
                start = None
        if start is not None:
            segments.append((start, last + 1))
        out = []
        for a, b in segments:
            s, e = a / 100, b / 100
            words = [Word(start=s, end=e, word=" speech", probability=0.99)] if word_timestamps else None
            out.append(
                Segment(
                    start=s,
                    end=e,
                    text=f"speech {s:.2f}-{e:.2f}",
                    avg_logprob=-0.1,
                    no_speech_prob=0.01,
                    compression_ratio=1.2,
                    words=words,
                )
            )
        return EngineResult(out, language or "en", 1.0)
