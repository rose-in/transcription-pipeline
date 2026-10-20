"""faster-whisper (CTranslate2) engine.

Why this runtime: same Whisper weights as the reference implementation, but
CTranslate2 kernels with int8/fp16 quantisation give a multi-x speedup and
lower memory, it runs on CUDA *and* CPU (so the M1 dev box and the Linux GPU
box execute the same code path), and it ships a batched, VAD-driven long-form
mode. See docs/RESEARCH.md §1 ("Runtimes for Whisper weights") for the
alternatives considered.

How transcribe() decodes a chunk:
1. VAD clips. Silero VAD finds the speech regions, which are merged into clips
   of up to max_clip_s across gaps of at most max_gap_s (clips()).
2. Batched decode with timestamp tokens. BatchedInferencePipeline decodes the
   clips in parallel; timestamp tokens split each clip into sentence-level
   segments.
3. Continuation of early stops. With timestamps on, Whisper sometimes stops a
   clip early, meaning "continue from here", and the batched path drops the
   rest. _SeekAware records where each clip stopped and _decode() decodes the
   remainders in further batched passes.
4. Safety net (_redecode_suspects()). Speech regions no segment covers, and
   segments that look like repetition loops, are re-decoded in two stages:
   batched again over just those regions (cheap), then whatever still fails
   through the reference sequential decoder with its temperature-fallback
   ladder (expensive, rare).

BatchedInferencePipeline (verified in faster-whisper 1.2.1) decodes each clip
with a single temperature and *ignores* compression_ratio_threshold /
log_prob_threshold / no_speech_threshold, so beyond the loop check above,
postprocess.QualityGate re-applies those thresholds to the per-segment scores
it still returns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..audio import TARGET_SR
from ..schemas import ModelMeta, Segment, Word
from ..whisper_thresholds import COMPRESSION_RATIO_THRESHOLD, LOG_PROB_THRESHOLD, NO_SPEECH_THRESHOLD
from .base import EngineResult
from .coverage import Range, grow_to_segments, intersect, merge_ranges, touches, uncovered
from .devices import default_cpu_threads, resolve_compute_type, resolve_device

if TYPE_CHECKING:
    from faster_whisper.transcribe import TranscriptionInfo

log = logging.getLogger(__name__)

# Silero VAD settings for speech_regions(), which drive both the clips and the coverage check.
VAD_MIN_SILENCE_MS = 160  # BatchedInferencePipeline's own default (Silero's class default is 2000)
VAD_SPEECH_PAD_MS = 200  # padding added to each side of a region; the coverage check trims it back off
# Timestamp jitter the coverage check allows on each side of a segment: boundaries of correct
# segments often sit ~0.3 s inside the speech, and an exact check would re-decode them for nothing.
COVERAGE_TOLERANCE_S = 0.3
REDECODE_PAD_S = 0.2  # context added to each side of a suspect range before it is re-decoded


class _SeekAware:
    """Model proxy for BatchedInferencePipeline that records where each clip's decode stopped.

    Whisper often ends a window early with a timestamp pair, meaning "continue from here in the
    next window" (e.g. "...dismay upon<5.92><5.92>" on an 8 s clip). Sequential decoding acts on
    that by starting the next window there; faster-whisper's batched path computes the same seek
    position (transcribe.py, _split_segments_by_timestamps) but never uses it, so with timestamps
    on, the rest of the clip is silently never decoded. (With faster-whisper's default
    without_timestamps=True there are no timestamp tokens, so this can't happen there.) We record
    the unfinished remainders and decode them in further batched passes.

    This hooks a private faster-whisper method that BatchedInferencePipeline reaches through
    `self.model` (verified against 1.2.1; re-verify before upgrading). If the method is missing,
    construction fails rather than continuation silently turning off.
    """

    def __init__(self, model, min_remaining_s: float = 0.5):
        if not callable(getattr(model, "_split_segments_by_timestamps", None)):
            raise RuntimeError(
                "faster-whisper's WhisperModel has no _split_segments_by_timestamps, which _SeekAware hooks to "
                "continue early-stopped clips; install a faster-whisper version this engine is verified with"
            )
        self._m = model
        self.min_remaining_s = min_remaining_s
        self.remainders: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._m, name)

    def _split_segments_by_timestamps(self, tokenizer, tokens, time_offset, segment_size, segment_duration, seek):
        segments, new_seek, single_ending = self._m._split_segments_by_timestamps(
            tokenizer=tokenizer,
            tokens=tokens,
            time_offset=time_offset,
            segment_size=segment_size,
            segment_duration=segment_duration,
            seek=seek,
        )
        reached = (new_seek - seek) / self._m.frames_per_second  # seconds into the clip decoding got to
        if not single_ending and segment_duration - reached >= self.min_remaining_s:
            self.remainders.append({"start": time_offset + reached, "end": time_offset + segment_duration})
        return segments, new_seek, single_ending


class _DecodePass(NamedTuple):
    segments: list[Segment]
    info: TranscriptionInfo | None  # from the first round; None only if there were no clips
    early_stops: int  # early stops detected over all rounds (each decoded in the next round, if one is left)
    left_over: int  # early stops still undecoded after the last round


class FasterWhisperEngine:
    max_chunk_s: float | None = None  # no input cap of its own: the pipeline's chunk_s applies

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
        batch_size: int = 8,
        beam_size: int = 5,
        cpu_threads: int = 0,
        max_gap_s: float = 1.5,
        max_clip_s: float = 30.0,
        revision: str | None = None,
        download_root: str | None = None,
        continuation: bool = True,
    ):
        """`continuation=False` skips decoding the rest of clips Whisper stopped early on (see
        _SeekAware); it exists to measure what continuation buys (scripts/eval_real_data.py)."""
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.model_name = model
        self.max_gap_s, self.max_clip_s = max_gap_s, max_clip_s
        self.continuation = continuation
        self.params = {
            "batch_size": batch_size,
            "beam_size": beam_size,
            # without_timestamps=False makes Whisper emit timestamp tokens inside each clip, so we get
            # sentence-level segments instead of one segment per clip; _SeekAware relies on them too.
            "without_timestamps": False,
        }
        self.cpu_threads = cpu_threads or default_cpu_threads()
        log.info("loading %s on %s (%s)", model, self.device, self.compute_type)
        # Pin `revision` in production: aliases like "turbo" resolve to third-party repos.
        self.revision = revision
        self._model = WhisperModel(
            model,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=download_root,
            revision=revision,
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
            # Recorded in every transcript and hashed into engine_id; "fallback" names the
            # safety-net strategy (see _redecode_suspects).
            params={
                **self.params,
                "vad_max_gap_s": self.max_gap_s,
                "vad_max_clip_s": self.max_clip_s,
                "coverage_tolerance_s": COVERAGE_TOLERANCE_S,
                "fallback": "two_stage",
                "continuation": self.continuation,
                "cpu_threads": self.cpu_threads,
            },
            versions={"faster_whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__},
        )

    @property
    def engine_id(self) -> str:
        m = self.meta
        blob = json.dumps([m.engine, m.model, m.compute_type, m.params, m.versions], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def warmup(self) -> None:
        """One tiny decode at startup so the first real request doesn't pay kernel/JIT init."""
        t = np.arange(TARGET_SR) / TARGET_SR
        self._pipe.transcribe(
            (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
            clip_timestamps=[{"start": 0.0, "end": 1.0}],
            language="en",
            **self.params,
        )
        self._seek.remainders = []

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
        first = self._decode(audio, self.clips(regions), language=language, **kw)
        language = first.info.language
        # The early-stop counts describe this first pass only, not the safety net's re-decode.
        stats = {
            "speech_s": sum(e - s for s, e in regions),
            "batched_s": time.perf_counter() - t0,
            "continued_clips": first.early_stops,
            "uncontinued_clips": first.left_over,
        }
        t1 = time.perf_counter()
        segments, net_stats = self._redecode_suspects(audio, regions, first.segments, language=language, **kw)
        stats.update(net_stats)
        stats["fallback_s"] = time.perf_counter() - t1
        return EngineResult(
            segments=segments,
            language=language,
            language_probability=first.info.language_probability,
            extra=stats,
        )

    def speech_regions(self, audio: np.ndarray) -> list[Range]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        opts = VadOptions(
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
            max_speech_duration_s=self.max_clip_s,
        )
        return [(r["start"] / TARGET_SR, r["end"] / TARGET_SR) for r in get_speech_timestamps(audio, opts)]

    def clips(self, regions: list[Range]) -> list[dict]:
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

    def _decode(
        self,
        audio: np.ndarray,
        clips: list[dict],
        *,
        language: str | None,
        word_timestamps: bool,
        prompt: str | None,
        max_rounds: int = 4,
    ) -> _DecodePass:
        """Batched decode of clips, continuing any clip whose decode stopped early (see _SeekAware)
        with further batched passes over just the remainders, as sequential Whisper would: up to
        max_rounds passes in total. Remainders still left after the last pass go uncontinued and
        are left to the coverage check."""
        out: list[Segment] = []
        info = None
        early_stops = 0
        if not self.continuation:
            max_rounds = 1
        for round_no in range(max_rounds):
            if not clips:
                break
            self._seek.remainders = []
            segs, round_info = self._decode_once(
                audio, clips, language=language, word_timestamps=word_timestamps, prompt=prompt
            )
            info = info or round_info
            language = language or round_info.language
            if round_no:
                # Decoded from a leftover piece with none of the speech before it. Very short pieces
                # are where Whisper makes things up, so the quality gate holds these to a stricter rule.
                for s in segs:
                    s.flags.append("continued")
            out += segs
            clips = self._seek.remainders
            early_stops += len(clips)  # decoded in the next round, if one is left
        return _DecodePass(sorted(out, key=lambda x: x.start), info, early_stops, left_over=len(clips))

    def _decode_once(
        self, audio: np.ndarray, clips: list[dict], *, language: str | None, word_timestamps: bool, prompt: str | None
    ) -> tuple[list[Segment], TranscriptionInfo]:
        segments, info = self._pipe.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=prompt,
            clip_timestamps=clips,
            **self.params,
        )
        return _to_segments(segments), info  # `segments` is a generator: the decode runs as it is read

    def _redecode_suspects(
        self,
        audio: np.ndarray,
        regions: list[Range],
        segments: list[Segment],
        *,
        language: str | None,
        word_timestamps: bool,
        prompt: str | None,
    ) -> tuple[list[Segment], dict]:
        """Safety net for what batched decoding still lacks (early stops are already continued in
        _decode): no temperature fallback, so a repetition loop stays a loop, and nothing checks
        that every stretch of speech produced text. So (1) every VAD speech region must be covered
        by some segment and (2) no segment may look like a loop. Failing regions are re-decoded:
          stage 1: batched again, with clips cut to just those regions (cheap; fixes lost tails);
          stage 2: whatever still fails goes through the reference sequential decoder with its
                   full temperature-fallback ladder (expensive; reserved for genuine problems).
        Returns the repaired segments and the suspect_ranges / redecoded_s / stage2_ranges stats."""
        stats = {"suspect_ranges": 0, "redecoded_s": 0.0, "stage2_ranges": 0}
        kw = dict(word_timestamps=word_timestamps, prompt=prompt)
        suspects, segments = self._suspects(regions, segments)
        if not suspects:
            return segments, stats
        ranges = grow_to_segments(merge_ranges(suspects, REDECODE_PAD_S), segments)
        stats["suspect_ranges"] = len(ranges)
        stats["redecoded_s"] = sum(b - a for a, b in ranges)
        replaced = [x for x in segments if touches(x, ranges)]
        segments = [x for x in segments if not touches(x, ranges)]
        # Stage 1. Its early-stop counts are not reported: continued_clips describes the first pass.
        sub_regions = intersect(regions, ranges)
        redo = self._decode(audio, self.clips(sub_regions), language=language, **kw).segments
        still, redo = self._suspects(sub_regions, redo)
        if still:  # stage 2
            hard = grow_to_segments(merge_ranges(still, REDECODE_PAD_S), redo)
            stats["stage2_ranges"] = len(hard)
            redo = [x for x in redo if not touches(x, hard)] + self._sequential(audio, hard, language=language, **kw)
        for x in redo:
            x.flags.append("fallback_decode")
        # If a re-decode returns nothing for a range (the reference decoder's no-speech skip is
        # stricter), keep the batched text we had there rather than losing it.
        for a, b in ranges:
            if not any(a <= (x.start + x.end) / 2 <= b for x in redo):
                redo += [x for x in replaced if a <= (x.start + x.end) / 2 <= b]
        return sorted(segments + redo, key=lambda x: x.start), stats

    def _suspects(self, regions: list[Range], segments: list[Segment]) -> tuple[list[Range], list[Segment]]:
        """(ranges needing a re-decode, segments minus loop segments)."""
        loops = [x for x in segments if _is_loop(x)]
        kept = [x for x in segments if not _is_loop(x)]  # loops must not count as coverage
        gaps = uncovered(regions, kept, tolerance_s=COVERAGE_TOLERANCE_S, shrink_s=VAD_SPEECH_PAD_MS / 1000)
        return gaps + [(x.start, x.end) for x in loops], kept

    def _sequential(
        self,
        audio: np.ndarray,
        ranges: list[Range],
        *,
        language: str | None,
        word_timestamps: bool,
        prompt: str | None,
    ) -> list[Segment]:
        """Reference sequential decode of each range, with Whisper's temperature-fallback ladder."""
        out: list[Segment] = []
        for a, b in ranges:
            lo, hi = int(a * TARGET_SR), int(b * TARGET_SR)
            gen, _ = self._model.transcribe(
                audio[lo:hi],
                language=language,
                beam_size=self.params["beam_size"],
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=LOG_PROB_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                condition_on_previous_text=False,
                vad_filter=False,  # region already comes from VAD
                word_timestamps=word_timestamps,
                initial_prompt=prompt,
            )
            out += _to_segments(gen, offset=a, round_bounds=True)
        return out


def _is_loop(seg: Segment) -> bool:
    """Whisper's repetition-loop signal: gzip compression ratio above the reference threshold."""
    return (seg.compression_ratio or 0) > COMPRESSION_RATIO_THRESHOLD


def _to_segments(raw: Iterable, offset: float = 0.0, *, round_bounds: bool = False) -> list[Segment]:
    """Convert faster-whisper segments to ours, dropping any whose text is empty.

    `offset` moves times decoded from an audio slice onto the chunk's timeline. `round_bounds`
    rounds the shifted start/end to the millisecond, as the batched pipeline already returns them;
    word times are kept as faster-whisper gives them.
    """
    out: list[Segment] = []
    for s in raw:
        text = s.text.strip()
        if not text:
            continue
        start, end = offset + s.start, offset + s.end
        if round_bounds:
            start, end = round(start, 3), round(end, 3)
        out.append(
            Segment(
                start=float(start),
                end=float(end),
                text=text,
                avg_logprob=float(s.avg_logprob),
                no_speech_prob=float(s.no_speech_prob),
                compression_ratio=float(s.compression_ratio),
                words=[
                    Word(start=offset + w.start, end=offset + w.end, word=w.word, probability=w.probability)
                    for w in s.words
                ]
                if s.words
                else None,
            )
        )
    return out
