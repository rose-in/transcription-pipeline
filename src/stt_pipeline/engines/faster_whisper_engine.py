"""faster-whisper (CTranslate2) engine.

Why this runtime: same Whisper weights as the reference implementation, but
CTranslate2 kernels with int8/fp16 quantisation give a multi-x speedup and
lower memory, it runs on CUDA *and* CPU (so the M1 dev box and the Linux GPU
box execute the same code path), and it ships a batched, VAD-driven long-form
mode. See README for the alternatives considered.

Important behaviour of BatchedInferencePipeline (verified in faster-whisper
1.2.1 source): it decodes each VAD chunk independently with a single
temperature and *ignores* compression_ratio_threshold / log_prob_threshold /
no_speech_threshold. The reference model's safety net is therefore gone; we
re-apply it in post-processing (postprocess.QualityGate) using the per-segment
scores it still returns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np

from ..audio import TARGET_SR as SAMPLE_RATE
from ..schemas import ModelMeta, Segment, Word
from .base import EngineResult

log = logging.getLogger(__name__)
COMPRESSION_RATIO_THRESHOLD = 2.4  # openai/whisper default: gzip ratio above this == repetition loop


def _default_threads() -> int:
    """Performance cores on Apple Silicon (efficiency cores are slower and can hold up
    CTranslate2's parallel loops); elsewhere the core count, capped at 8."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=2)
            if int(out.stdout.strip()) > 0:
                return int(out.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return min(8, os.cpu_count() or 4)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


class _SeekAware:
    """Model proxy for BatchedInferencePipeline that records where each clip's decode stopped.

    Whisper often ends a window early with a timestamp pair, meaning "continue from here in the
    next window" (e.g. "...dismay upon<5.92><5.92>" on an 8 s clip). Sequential decoding acts on
    that by starting the next window there; faster-whisper's batched path computes the same seek
    position (transcribe.py, _split_segments_by_timestamps) but never uses it, so with timestamps
    on, the rest of the clip is silently never decoded. (With faster-whisper's default
    without_timestamps=True there are no timestamp tokens, so this can't happen there.) We record
    the unfinished remainders and decode them in further batched passes.
    """

    def __init__(self, model, min_remaining_s: float = 0.5):
        self._m = model
        self.min_remaining_s = min_remaining_s
        self.remainders: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._m, name)

    def _split_segments_by_timestamps(self, tokenizer, tokens, time_offset, segment_size, segment_duration, seek):
        segments, new_seek, single_ending = self._m._split_segments_by_timestamps(
            tokenizer=tokenizer, tokens=tokens, time_offset=time_offset, segment_size=segment_size,
            segment_duration=segment_duration, seek=seek)
        reached = (new_seek - seek) / self._m.frames_per_second  # seconds into the clip decoding got to
        if not single_ending and segment_duration - reached >= self.min_remaining_s:
            self.remainders.append({"start": time_offset + reached, "end": time_offset + segment_duration})
        return segments, new_seek, single_ending


class FasterWhisperEngine:
    # Fallback tuning (see transcribe), kept selectable so each fix can be measured on its own:
    # coverage_tolerance_s=0 + fallback="sequential" is the original safety net, and
    # continuation=False turns off continuing early-stopped clips (_decode).
    coverage_tolerance_s = 0.3
    fallback = "two_stage"
    continuation = True

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
        batch_size: int = 8,
        beam_size: int = 5,
        cpu_threads: int = 0,
        segment_timestamps: bool = True,
        max_gap_s: float = 1.5,
        max_clip_s: float = 30.0,
        revision: str | None = None,
        download_root: str | None = None,
        coverage_tolerance_s: float | None = None,
        fallback: str | None = None,
        continuation: bool | None = None,
    ):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.device = _resolve_device(device)
        if compute_type == "default":
            # int8 on CPU (no fp16 kernels; ~2x faster than fp32, near-identical WER); float16 on GPU.
            # int8_float16 halves VRAM but is *slower* on many cards - benchmark before switching.
            compute_type = "int8" if self.device == "cpu" else "float16"
        self.compute_type = compute_type
        self.model_name = model
        self.max_gap_s, self.max_clip_s = max_gap_s, max_clip_s
        if coverage_tolerance_s is not None:
            self.coverage_tolerance_s = coverage_tolerance_s
        if fallback is not None:
            self.fallback = fallback
        if continuation is not None:
            self.continuation = continuation
        self.params = {
            "batch_size": batch_size,
            "beam_size": beam_size,
            # without_timestamps=False makes Whisper emit timestamp tokens inside each 30 s VAD
            # chunk, so we get sentence-level segments instead of one segment per chunk.
            "without_timestamps": not segment_timestamps,
        }
        threads = cpu_threads or _default_threads()
        self.cpu_threads = threads
        log.info("loading %s on %s (%s)", model, self.device, compute_type)
        # Pin `revision` in production: aliases like "turbo" resolve to third-party repos.
        self.revision = revision
        self._model = WhisperModel(
            model, device=self.device, compute_type=compute_type, cpu_threads=threads,
            download_root=download_root, revision=revision,
        )
        self._seek = _SeekAware(self._model)
        self._pipe = BatchedInferencePipeline(self._seek)

    @property
    def meta(self) -> ModelMeta:
        import ctranslate2
        import faster_whisper

        return ModelMeta(
            engine="faster-whisper",
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            revision=self.revision,
            params={**self.params, "vad_max_gap_s": self.max_gap_s, "vad_max_clip_s": self.max_clip_s,
                    "coverage_tolerance_s": self.coverage_tolerance_s, "fallback": self.fallback,
                    "continuation": self.continuation, "cpu_threads": self.cpu_threads},
            versions={"faster_whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__},
        )

    @property
    def engine_id(self) -> str:
        m = self.meta
        blob = json.dumps([m.engine, m.model, m.compute_type, m.params, m.versions], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def warmup(self) -> None:
        """One tiny decode at startup so the first real request doesn't pay kernel/JIT init."""
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        self._pipe.transcribe((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                              clip_timestamps=[{"start": 0.0, "end": 1.0}], language="en", **self.params)
        self._seek.remainders = []

    def speech_regions(self, audio: np.ndarray) -> list[tuple[float, float]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        regions = get_speech_timestamps(
            audio, VadOptions(min_silence_duration_ms=160, speech_pad_ms=200, max_speech_duration_s=self.max_clip_s)
        )
        return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE) for r in regions]

    def clips(self, regions: list[tuple[float, float]]) -> list[dict]:
        """VAD 'cut & merge' with a max-gap rule.

        faster-whisper's default batched path merges speech regions up to 30 s and *removes*
        the silence between them, so one segment can span a long pause (in split-channel calls,
        the other party's entire turn). We merge regions only across gaps <= max_gap_s and keep
        the in-between audio, so segments stay within one utterance and timestamp tokens map to
        real time.
        """
        clips: list[dict] = []
        for start, end in regions:
            if clips and end - clips[-1]["start"] <= self.max_clip_s and start - clips[-1]["end"] <= self.max_gap_s:
                clips[-1]["end"] = end
            else:
                clips.append({"start": start, "end": end})
        return clips

    def _decode(self, audio, clips, *, language, word_timestamps, prompt, max_rounds: int = 4):
        """Batched decode of clips, continuing any clip whose decode stopped early (see _SeekAware)
        with further batched passes over just the remainders, as sequential Whisper would: up to
        max_rounds passes in total. Remainders still left after the last pass go uncontinued and
        are left to the coverage check."""
        out: list[Segment] = []
        info = None
        if not getattr(self, "continuation", True):
            max_rounds = 1
        for round_no in range(max_rounds):
            if not clips:
                break
            self._seek.remainders = []
            segs, round_info = self._decode_once(audio, clips, language=language, word_timestamps=word_timestamps,
                                                 prompt=prompt)
            info = info or round_info
            language = language or round_info.language
            if round_no:
                # Decoded from a leftover piece with none of the speech before it. Very short pieces
                # are where Whisper makes things up, so the quality gate holds these to a stricter rule.
                for s in segs:
                    s.flags.append("continued")
            out += segs
            clips = self._seek.remainders
            self._continued += len(clips)  # early stops detected (decoded next pass, if any is left)
        self._left_over = getattr(self, "_left_over", 0) + len(clips)
        return sorted(out, key=lambda x: x.start), info

    def _decode_once(self, audio, clips, *, language, word_timestamps, prompt):
        segments, info = self._pipe.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=prompt,
            clip_timestamps=clips,
            **self.params,
        )
        out: list[Segment] = []
        for s in segments:  # generator: decoding happens while we iterate
            text = s.text.strip()
            if not text:
                continue
            out.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=text,
                    avg_logprob=float(s.avg_logprob),
                    no_speech_prob=float(s.no_speech_prob),
                    compression_ratio=float(s.compression_ratio),
                    words=[Word(start=w.start, end=w.end, word=w.word, probability=w.probability) for w in s.words]
                    if s.words
                    else None,
                )
            )
        return out, info

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        word_timestamps: bool = False,
        prompt: str | None = None,
    ) -> EngineResult:
        regions = self.speech_regions(audio)
        if not regions:  # no speech: never feed pure silence/noise to Whisper (hallucination magnet)
            return EngineResult(segments=[], language=language, language_probability=None, extra={"speech_s": 0.0})
        kw = dict(word_timestamps=word_timestamps, prompt=prompt)
        t0 = time.perf_counter()
        self._continued = self._left_over = 0
        segments, info = self._decode(audio, self.clips(regions), language=language, **kw)
        language = info.language
        stats = {"speech_s": sum(e - s for s, e in regions), "batched_s": time.perf_counter() - t0,
                 "continued_clips": self._continued, "uncontinued_clips": self._left_over,
                 "suspect_ranges": 0, "redecoded_s": 0.0, "stage2_ranges": 0, "fallback_s": 0.0}

        # Safety net for what batched decoding still lacks (early stops are already continued in
        # _decode): no temperature fallback, so a repetition loop stays a loop, and nothing checks
        # that every stretch of speech produced text. So (1) every VAD speech region must be
        # covered by some segment and (2) no segment may look like a loop. Failing regions are
        # re-decoded:
        #   stage 1: batched again, with clips cut to just those regions (cheap; fixes lost tails);
        #   stage 2: whatever still fails goes through the reference sequential decoder with its
        #            full temperature-fallback ladder (expensive; reserved for genuine problems).
        # Coverage tolerates ~0.3 s of timestamp jitter: on real calls an intolerant check fired
        # on correct transcripts and re-decoded minutes of audio at batch size 1.
        t1 = time.perf_counter()
        suspects, segments = self._suspects(regions, segments)
        if suspects:
            ranges = _grow(_merge_ranges(suspects), segments)
            stats["suspect_ranges"] = len(ranges)
            stats["redecoded_s"] = sum(b - a for a, b in ranges)
            replaced = [x for x in segments if _touches(x, ranges)]
            segments = [x for x in segments if x not in replaced]
            if self.fallback == "two_stage":
                sub_regions = [(max(a, rs), min(b, re_)) for rs, re_ in regions for a, b in ranges if rs < b and a < re_]
                redo, _ = self._decode(audio, self.clips(sub_regions), language=language, **kw)
                still, redo = self._suspects(sub_regions, redo)
                if still:
                    hard = _grow(_merge_ranges(still), redo)
                    stats["stage2_ranges"] = len(hard)
                    redo = [x for x in redo if not _touches(x, hard)] + self._sequential(audio, hard, language=language, **kw)
            else:
                stats["stage2_ranges"] = len(ranges)
                redo = self._sequential(audio, ranges, language=language, **kw)
            for x in redo:
                x.flags.append("fallback_decode")
            # If a re-decode returns nothing for a range (the reference decoder's no-speech skip is
            # stricter), keep the batched text we had there rather than losing it.
            for a, b in ranges:
                if not any(a <= (x.start + x.end) / 2 <= b for x in redo):
                    redo += [x for x in replaced if a <= (x.start + x.end) / 2 <= b]
            segments = sorted(segments + redo, key=lambda x: x.start)
        stats["fallback_s"] = time.perf_counter() - t1
        return EngineResult(
            segments=segments,
            language=language,
            language_probability=info.language_probability,
            extra=stats,
        )

    def _suspects(self, regions, segments):
        """(ranges needing a re-decode, segments minus loop segments)."""
        loops = [x for x in segments if (x.compression_ratio or 0) > COMPRESSION_RATIO_THRESHOLD]
        kept = [x for x in segments if x not in loops]  # loops must not count as coverage
        tol = self.coverage_tolerance_s
        gaps = uncovered(regions, kept, tolerance_s=tol, shrink_s=0.2 if tol else 0.0)
        return gaps + [(x.start, x.end) for x in loops], kept

    def _sequential(self, audio, ranges, *, language, word_timestamps, prompt) -> list[Segment]:
        out: list[Segment] = []
        for a, b in ranges:
            lo, hi = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
            gen, _ = self._model.transcribe(
                audio[lo:hi],
                language=language,
                beam_size=self.params["beam_size"],
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                vad_filter=False,  # region already comes from VAD
                word_timestamps=word_timestamps,
                initial_prompt=prompt,
            )
            for x in gen:
                if not x.text.strip():
                    continue
                out.append(Segment(
                    start=round(a + x.start, 3), end=round(a + x.end, 3), text=x.text.strip(),
                    avg_logprob=x.avg_logprob, no_speech_prob=x.no_speech_prob, compression_ratio=x.compression_ratio,
                    words=[Word(start=a + w.start, end=a + w.end, word=w.word, probability=w.probability) for w in x.words]
                    if x.words else None,
                ))
        return out


def _touches(seg: Segment, ranges) -> bool:
    return any(seg.start < b and a < seg.end for a, b in ranges)


def _grow(ranges, segments):
    """Whole segments in, whole segments out: grow ranges until they fully contain every segment
    they touch (replacing part of a segment duplicated its words next to the redo's)."""
    while True:
        grown = _merge_ranges(ranges + [(x.start, x.end) for x in segments if _touches(x, ranges)], pad=0.0)
        if grown == ranges:
            return ranges
        ranges = grown


def _merge_ranges(ranges: list[tuple[float, float]], pad: float = 0.2) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        a = max(0.0, a - pad)
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b + pad)
        else:
            out.append([a, b + pad])
    return [(a, b) for a, b in out]


def uncovered(
    regions: list[tuple[float, float]], segments: list[Segment], *, min_missing_s: float = 1.5,
    min_region_s: float = 1.0, min_cover: float = 0.5, tolerance_s: float = 0.0, shrink_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Speech regions the transcript doesn't account for: either >= min_missing_s seconds of the
    region lie outside every segment (a lost tail in a long region), or a region >= min_region_s
    is less than min_cover covered. Coverage uses the *union* of segment intervals.

    tolerance_s widens each segment and shrink_s trims the VAD padding off each region, so
    ordinary timestamp jitter on a correct transcript doesn't count as missing speech."""
    spans = sorted((x.start - tolerance_s, x.end + tolerance_s) for x in segments)
    out = []
    for r0, r1 in regions:
        rs, re_ = r0 + shrink_s, r1 - shrink_s
        dur = re_ - rs
        if dur < min_region_s:
            continue
        covered, cursor = 0.0, rs
        for a, b in spans:
            a, b = max(a, cursor), min(b, re_)
            if b > a:
                covered += b - a
                cursor = b
        missing = dur - covered
        if missing >= min_missing_s or covered / dur < min_cover:
            out.append((r0, r1))
    return out


