"""Hosted engine: any OpenAI-compatible /audio/transcriptions endpoint (OpenAI whisper-1, Groq, ...).

Exists to make the build-vs-buy decision a config switch rather than a rewrite
(STT_ENGINE=openai-compatible). Everything around the engine - validation,
decoding, silence-aware chunking under the upload limit, checkpoints, post-
processing, formats - is unchanged.

Capability caveat this adapter enforces: on OpenAI, only `whisper-1` returns
segment/word timestamps (response_format=verbose_json + timestamp_granularities).
The newer gpt-4o-transcribe / gpt-4o-mini-transcribe / gpt-transcribe models
return text only, so they are rejected at construction instead of silently
producing a transcript without the timestamps this pipeline promises.
"""

from __future__ import annotations

import email.utils
import hashlib
import io
import logging
import time

import av
import httpx
import numpy as np

from ..audio import TARGET_SR
from ..errors import EngineFailure, InvalidOptions, ProviderError
from ..languages import to_code
from ..schemas import ModelMeta, Segment, Word
from .base import EngineResult

TEXT_ONLY_MODELS = ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-transcribe")
RETRYABLE = (408, 409, 429, 500, 502, 503, 504)
LOCAL_MODEL_ALIASES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
    "distil-medium.en",
    "distil-small.en",
}
MIN_AUDIO_S = 0.1  # providers reject audio shorter than ~100 ms
CONFIG_ERRORS = (401, 403, 404)  # our key/model/endpoint is wrong: not the client's fault, not transient
log = logging.getLogger(__name__)


def encode_flac(audio: np.ndarray) -> bytes:
    """Lossless, ~50% of PCM size: 10 min of 16 kHz mono speech is ~10 MB, under the 25 MB cap."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="flac") as out:
        st = out.add_stream("flac", rate=TARGET_SR)
        st.layout = "mono"
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)[None, :]
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = TARGET_SR
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return buf.getvalue()


class OpenAICompatibleEngine:
    max_chunk_s: float | None = 600.0  # largest chunk we accept: 10 min of FLAC is ~10 MB, under the 25 MB cap

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str = "whisper-1",
        *,
        max_upload_bytes: int = 25 * 2**20,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
    ):
        if any(model.startswith(m) for m in TEXT_ONLY_MODELS):
            raise ValueError(
                f"{model} returns no segment timestamps; use whisper-1 (OpenAI) or a Whisper model on Groq"
            )
        if model in LOCAL_MODEL_ALIASES:  # e.g. STT_MODEL=large-v3-turbo baked into the image
            raise ValueError(
                f"STT_MODEL={model!r} is a local faster-whisper model; set a provider model "
                "such as 'whisper-1' (OpenAI) or 'whisper-large-v3-turbo' (Groq)"
            )
        if not api_key and client is None:
            raise ValueError("STT_API_KEY is required for the openai-compatible engine")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.max_upload_bytes, self.max_retries = max_upload_bytes, max_retries
        self._client = client or httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    @property
    def meta(self) -> ModelMeta:
        return ModelMeta(engine="openai-compatible", model=self.model, params={"base_url": self.base_url})

    @property
    def engine_id(self) -> str:
        return hashlib.sha256(f"oai|{self.base_url}|{self.model}".encode()).hexdigest()[:12]

    def warmup(self) -> None:
        """Nothing to warm up locally; the provider hosts the model."""

    def transcribe(
        self, audio: np.ndarray, *, language: str | None, word_timestamps: bool = False, prompt: str | None = None
    ) -> EngineResult:
        if len(audio) < MIN_AUDIO_S * TARGET_SR:
            return EngineResult([], language, None)
        body = encode_flac(audio)
        if len(body) > self.max_upload_bytes:
            raise InvalidOptions(f"chunk encodes to {len(body) >> 20} MB, above the provider's upload limit")
        data = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": "0",
            "timestamp_granularities[]": ["segment", "word"] if word_timestamps else ["segment"],
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        payload = self._post(data, body)
        return self._parse(payload, word_timestamps)

    def _post(self, data: dict, body: bytes) -> dict:
        """Retries transient failures with exponential backoff, honouring Retry-After (capped).
        Provider error bodies are logged, never echoed to our clients (they can contain key hints)."""
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(
                    f"{self.base_url}/audio/transcriptions",
                    data=data,
                    files={"file": ("chunk.flac", body, "audio/flac")},
                )
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise EngineFailure(f"provider unreachable: {type(e).__name__}") from e
                time.sleep(min(2**attempt, 30))
                continue
            if r.status_code == 200:
                return r.json()
            log.warning("provider returned %s: %s", r.status_code, r.text[:500])
            if r.status_code in RETRYABLE and attempt < self.max_retries:
                time.sleep(_retry_after(r.headers.get("retry-after"), default=min(2**attempt, 30)))
                continue
            if r.status_code in RETRYABLE:
                raise EngineFailure(f"provider unavailable ({r.status_code}) after {attempt + 1} attempts")
            if r.status_code in CONFIG_ERRORS:
                raise ProviderError(f"hosted engine rejected our credentials or model ({r.status_code})")
            raise InvalidOptions(f"hosted engine rejected the request ({r.status_code})")
        raise EngineFailure("retries exhausted")  # pragma: no cover

    @staticmethod
    def _parse(p: dict, word_timestamps: bool) -> EngineResult:
        raw = [s for s in p.get("segments") or [] if (s.get("text") or "").strip()]
        buckets: list[list[Word]] = [[] for _ in raw]
        for w in p.get("words") or []:
            if not raw:
                break
            word = Word(start=w["start"], end=w["end"], word=" " + w["word"].strip())
            # Exactly one segment per word: the one it overlaps most (ties -> nearest midpoint).
            mid = (word.start + word.end) / 2
            best = max(
                range(len(raw)),
                key=lambda i: (
                    min(word.end, raw[i]["end"]) - max(word.start, raw[i]["start"]),
                    -abs(mid - (raw[i]["start"] + raw[i]["end"]) / 2),
                ),
            )
            buckets[best].append(word)
        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"].strip(),
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
                compression_ratio=s.get("compression_ratio"),
                words=ws if word_timestamps and ws else None,
            )
            for s, ws in zip(raw, buckets)
        ]
        # Providers answer "english" or "en"; anything unmappable becomes None so the pipeline
        # never pins it and echoes it back (which the provider would then reject).
        return EngineResult(segments, to_code(p.get("language")), None)


def _retry_after(value: str | None, *, default: float, cap: float = 60.0) -> float:
    if not value:
        return default
    try:
        return min(max(float(value), 0.0), cap)
    except ValueError:
        try:  # HTTP-date form
            when = email.utils.parsedate_to_datetime(value).timestamp()
            return min(max(when - time.time(), 0.0), cap)
        except (TypeError, ValueError):
            return default
