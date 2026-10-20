"""Orchestration: probe -> stream-decode -> macro-chunk -> engine -> offset/merge -> post-process.

The same `Pipeline.run` backs the CLI, the synchronous HTTP endpoint and the
async job worker, so there is exactly one code path to test and reason about.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio import DecodeStats, iter_pcm, probe
from .chunking import silence_aware_chunks
from .engines.base import Engine, EngineResult
from .errors import AudioTooLong, EngineFailure, InvalidOptions, PipelineError
from .postprocess import build_chain, run_chain
from .schemas import AudioMeta, ChunkMeta, Segment, Timing, TranscribeOptions, Transcript

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, float | None], None]  # (processed_audio_s, total_audio_s or None)


class CheckpointStore(Protocol):
    def load(self, key: str) -> dict | None: ...
    def save(self, key: str, data: dict) -> None: ...


class DirCheckpointStore:
    """One JSON file per finished chunk; atomic rename so a crash never leaves a half-written file."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> dict | None:
        p = self.root / f"{key}.json"
        try:
            return json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save(self, key: str, data: dict) -> None:
        # Unique temp name: at-least-once execution means two workers may write the same chunk.
        tmp = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.root / f"{key}.json")


@dataclass
class PipelineConfig:
    chunk_s: float = 600.0  # macro-chunk target: memory bound + checkpoint granularity
    chunk_search_s: float = 60.0  # look this far back from the target for a pause to cut at
    max_duration_s: float = 4 * 3600.0
    max_split_channels: int = 4  # split mode multiplies work by the channel count


class Pipeline:
    def __init__(self, engine: Engine, config: PipelineConfig | None = None):
        self.engine = engine
        self.config = config or PipelineConfig()

    def run(
        self,
        path: Path,
        options: TranscribeOptions | None = None,
        *,
        filename: str | None = None,
        sha256: str | None = None,
        progress: ProgressFn | None = None,
        checkpoints: CheckpointStore | None = None,
        max_duration_s: float | None = None,
    ) -> Transcript:
        """max_duration_s overrides the config limit for this run (e.g. the stricter sync-API limit);
        it is enforced on decoded samples, so a missing or lying container header can't bypass it."""
        cfg = self.config
        limit = min(cfg.max_duration_s, max_duration_s or cfg.max_duration_s)
        try:
            return self._run(path, options or TranscribeOptions(), limit, filename=filename, sha256=sha256,
                             progress=progress, checkpoints=checkpoints)
        except AudioTooLong as e:
            if limit < cfg.max_duration_s:  # the stricter per-request (sync) limit fired
                raise AudioTooLong(f"{e.message} Use POST /v1/jobs for long audio.") from e
            raise

    def _run(self, path, options, limit, *, filename, sha256, progress, checkpoints) -> Transcript:
        cfg = self.config
        try:  # stored jobs are loaded leniently; validate here before doing any work
            options = TranscribeOptions.model_validate(options.model_dump())
        except ValueError as e:
            raise InvalidOptions(str(e).splitlines()[-1]) from e
        t_start = time.perf_counter()
        info = probe(path, max_duration_s=limit)

        if options.channel_mode == "split" and info.channels > 1:
            if info.channels > cfg.max_split_channels:
                raise InvalidOptions(f"channel_mode=split supports up to {cfg.max_split_channels} channels; "
                                     f"this file has {info.channels}.")
            channels: list[int | None] = list(range(info.channels))
        else:
            channels = [None]
        sha256 = sha256 or (_file_sha256(path) if checkpoints else None)
        # Checkpoints are only valid for the same bytes, engine, options and chunking. Without this
        # a reused checkpoint dir silently returned another file's (or another model's) text.
        fingerprint = hashlib.sha256(json.dumps(
            [sha256, self.engine.engine_id, options.cache_key(self.engine.engine_id), cfg.chunk_s, cfg.chunk_search_s]
        ).encode()).hexdigest()[:16]
        warnings: list[str] = []
        if options.channel_mode == "split" and info.channels == 1:
            warnings.append("channel_mode=split requested but the audio is mono; transcribed as a single channel.")

        # Engines may declare a smaller max input (e.g. hosted APIs with upload caps).
        chunk_s = min(cfg.chunk_s, getattr(self.engine, "max_chunk_s", None) or cfg.chunk_s)
        search_s = min(cfg.chunk_search_s, chunk_s / 4)
        language, language_p = options.language, (1.0 if options.language else None)
        segments: list[Segment] = []
        chunk_meta: list[ChunkMeta] = []
        inference_s = decode_s = 0.0
        decoded_s = 0.0
        total_est = info.container_duration_s * len(channels) if info.container_duration_s else None
        done_s = 0.0

        for ch in channels:
            label = None if ch is None else f"channel_{ch}"
            stats = DecodeStats()
            blocks = iter_pcm(path, info, channel=ch, max_duration_s=limit, stats=stats)
            for chunk in silence_aware_chunks(blocks, target_s=chunk_s, search_s=search_s):
                key = f"{label or 'mix'}-{chunk.index:05d}"
                cached = checkpoints.load(key) if checkpoints else None
                # A checkpoint is reused only if it was produced for the same audio position and the
                # same pinned language; otherwise (e.g. detection changed) the chunk is recomputed.
                if (cached and cached.get("fingerprint") == fingerprint and cached.get("n_samples") == len(chunk.audio)
                        and abs(cached["offset_s"] - chunk.offset_s) < 1e-6 and cached["input_language"] == language):
                    result = EngineResult(
                        [Segment(**s) for s in cached["segments"]], cached["language"], cached["language_probability"]
                    )
                    was_cached = True
                elif chunk.duration_s < 0.1:  # a few ms of tail: nothing to transcribe, don't pay for a call
                    result, was_cached = EngineResult([], language, None), False
                else:
                    t0 = time.perf_counter()
                    try:
                        result = self.engine.transcribe(
                            chunk.audio, language=language, word_timestamps=options.word_timestamps, prompt=options.prompt
                        )
                    except PipelineError:
                        raise
                    except Exception as e:  # OOM, driver errors...: transient by assumption
                        raise EngineFailure(f"engine failed on chunk {chunk.index}: {e}") from e
                    inference_s += time.perf_counter() - t0
                    was_cached = False
                    if checkpoints:
                        checkpoints.save(
                            key,
                            {
                                "fingerprint": fingerprint,
                                "n_samples": len(chunk.audio),
                                "offset_s": chunk.offset_s,
                                "input_language": language,
                                "language": result.language,
                                "language_probability": result.language_probability,
                                "segments": [s.model_dump() for s in result.segments],
                            },
                        )
                # Detect language once, on the first chunk that actually contains speech, then pin
                # it: per-chunk detection lets a noisy chunk flip the language mid-file.
                if language is None and result.segments and result.language:
                    language, language_p = result.language, result.language_probability
                for s in result.segments:
                    segments.append(_shift(s, chunk.offset_s, label))
                chunk_meta.append(
                    ChunkMeta(
                        index=chunk.index, channel=label, offset_s=round(chunk.offset_s, 3),
                        duration_s=round(chunk.duration_s, 3), segments=len(result.segments), cached=was_cached,
                    )
                )
                done_s += chunk.duration_s
                if progress:  # in file seconds, not channel-seconds of work
                    progress(done_s / len(channels), total_est / len(channels) if total_est else None)
                log.info("chunk %s %.0fs+%.0fs -> %d segments%s", key, chunk.offset_s, chunk.duration_s,
                         len(result.segments), " (cached)" if was_cached else "")
            decode_s += stats.decode_s
            decoded_s = max(decoded_s, stats.duration_s)
            warnings += [w for w in stats.warnings if w not in warnings]

        segments.sort(key=lambda s: (s.start, s.speaker or ""))
        total_s = time.perf_counter() - t_start
        transcript = Transcript(
            language=language,
            language_probability=language_p,
            duration_s=round(decoded_s, 3),
            segments=segments,
            audio=AudioMeta(
                filename=filename or path.name, sha256=sha256, size_bytes=info.size_bytes,
                container=info.container, codec=info.codec, sample_rate=info.sample_rate, channels=info.channels,
                container_duration_s=info.container_duration_s and round(info.container_duration_s, 3),
                start_offset_s=info.start_offset_s,
                decoded_duration_s=round(decoded_s, 3),
            ),
            model=self.engine.meta,
            options=options.model_dump(),
            chunks=chunk_meta,
            warnings=warnings,
            timing=Timing(
                decode_s=round(decode_s, 3), inference_s=round(inference_s, 3), total_s=round(total_s, 3),
                rtf=round(total_s / decoded_s, 4) if decoded_s else 0.0,
            ),
        )
        transcript = run_chain(transcript, build_chain(redact=options.redact_pii, words=options.word_timestamps))
        if not transcript.segments:
            transcript.warnings.append("No speech detected.")
        return transcript


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _shift(s: Segment, offset: float, speaker: str | None) -> Segment:
    s = s.model_copy(deep=True)
    s.start, s.end = round(s.start + offset, 3), round(s.end + offset, 3)
    s.speaker = speaker
    if s.words:
        for w in s.words:
            w.start, w.end = round(w.start + offset, 3), round(w.end + offset, 3)
    return s
