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

import numpy as np

from ..audio import TARGET_SR as SAMPLE_RATE
from ..schemas import ModelMeta, Segment, Word
from .base import EngineResult

log = logging.getLogger(__name__)
COMPRESSION_RATIO_THRESHOLD = 2.4  # openai/whisper default: gzip ratio above this == repetition loop


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


class FasterWhisperEngine:
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
        self.params = {
            "batch_size": batch_size,
            "beam_size": beam_size,
            # without_timestamps=False makes Whisper emit timestamp tokens inside each 30 s VAD
            # chunk, so we get sentence-level segments instead of one segment per chunk.
            "without_timestamps": not segment_timestamps,
        }
        threads = cpu_threads or min(8, os.cpu_count() or 4)
        log.info("loading %s on %s (%s)", model, self.device, compute_type)
        # Pin `revision` in production: aliases like "turbo" resolve to third-party repos.
        self.revision = revision
        self._model = WhisperModel(
            model, device=self.device, compute_type=compute_type, cpu_threads=threads,
            download_root=download_root, revision=revision,
        )
        self._pipe = BatchedInferencePipeline(self._model)

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
            params={**self.params, "vad_max_gap_s": self.max_gap_s, "vad_max_clip_s": self.max_clip_s},
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

    def _decode(self, audio, clips, *, language, word_timestamps, prompt):
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
        segments, info = self._decode(audio, self.clips(regions), language=language, **kw)
        language = info.language

        # Narrow sequential fallback. Batched decoding is fast but has no temperature fallback
        # and no "seek and continue": a repetition loop stays a loop, and if the model emits
        # end-of-text early in a ~30 s clip the rest of the clip is silently lost (observed on
        # our fixtures with small models). So: (1) every VAD speech region must be mostly
        # covered by some segment, and (2) no segment may look like a loop. Regions failing
        # either check are re-decoded with the reference algorithm's full safety net -- only
        # there, so the batched speed is kept everywhere else.
        loops = [x for x in segments if (x.compression_ratio or 0) > COMPRESSION_RATIO_THRESHOLD]
        segments = [x for x in segments if x not in loops]  # loops must not count as coverage
        suspects = uncovered(regions, segments) + [(x.start, x.end) for x in loops]
        if suspects:
            # Whole segments in, whole segments out: grow each range to fully contain any segment
            # it touches, then replace those segments with the redo. (Keeping a partially
            # overlapping batched segment duplicated its words next to the redo's.)
            ranges = _merge_ranges(suspects)
            while True:
                grown = _merge_ranges(ranges + [(x.start, x.end) for x in segments
                                                if any(x.start < b and a < x.end for a, b in ranges)], pad=0.0)
                if grown == ranges:
                    break
                ranges = grown
            replaced = [x for x in segments if any(x.start < b and a < x.end for a, b in ranges)]
            segments = [x for x in segments if x not in replaced]
            log.info("fallback: re-decoding %d range(s) sequentially (%d loop segment(s))", len(ranges), len(loops))
            redo = self._sequential(audio, ranges, language=language, **kw)
            for x in redo:
                x.flags.append("fallback_decode")
            # If the reference decoder returns nothing for a range (its no-speech skip is stricter),
            # keep the batched text we had there rather than losing it.
            for a, b in ranges:
                if not any(a <= (x.start + x.end) / 2 <= b for x in redo):
                    redo += [x for x in replaced if a <= (x.start + x.end) / 2 <= b]
            segments = sorted(segments + redo, key=lambda x: x.start)
        return EngineResult(
            segments=segments,
            language=language,
            language_probability=info.language_probability,
            extra={"speech_s": sum(e - s for s, e in regions), "fallback_regions": len(suspects)},
        )

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
    min_region_s: float = 1.0, min_cover: float = 0.5,
) -> list[tuple[float, float]]:
    """Speech regions the transcript doesn't account for: either >= min_missing_s seconds of the
    region lie outside every segment (a lost tail in a long region), or a region >= min_region_s
    is less than min_cover covered. Coverage uses the *union* of segment intervals."""
    spans = sorted((x.start, x.end) for x in segments)
    out = []
    for rs, re_ in regions:
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
            out.append((rs, re_))
    return out


