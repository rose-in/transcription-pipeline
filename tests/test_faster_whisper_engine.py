"""FasterWhisperEngine orchestration (VAD clips, coverage check, re-decode safety net, continuation of
early-stopped clips) with the model calls replaced by fakes: no weights are loaded."""

from types import SimpleNamespace

import numpy as np
import pytest

from stt_pipeline.engines.coverage import uncovered
from stt_pipeline.engines.faster_whisper_engine import (
    COVERAGE_TOLERANCE_S,
    VAD_SPEECH_PAD_MS,
    FasterWhisperEngine,
    _DecodePass,
    _SeekAware,
)

from .conftest import S

_INFO = SimpleNamespace(language="en", language_probability=1.0)  # stands in for faster-whisper's TranscriptionInfo


class _ScriptedWhisper(FasterWhisperEngine):
    """FasterWhisperEngine with VAD, the batched decode and the sequential decoder scripted."""

    def __init__(self, regions, batched, redo):
        self.max_gap_s, self.max_clip_s = 1.5, 30.0
        self.params = {"beam_size": 5}
        self._regions, self._batched, self._redo, self.redo_ranges = regions, batched, redo, None

    def speech_regions(self, audio):
        return self._regions

    def _decode(self, audio, clips, **kw):
        return _DecodePass([s.model_copy() for s in self._batched], _INFO, 0, 0)

    def _sequential(self, audio, ranges, **kw):
        self.redo_ranges = ranges
        return [s.model_copy() for s in self._redo]


class _TwoPass(_ScriptedWhisper):
    """First batched pass loses the tail; the targeted batched re-decode recovers it."""

    def __init__(self):
        super().__init__([(10.0, 30.0)], batched=[S(10, 14, "Good start.")], redo=[S(10, 30, "should not be needed")])
        self.decode_calls = 0

    def _decode(self, audio, clips, **kw):
        self.decode_calls += 1
        if self.decode_calls == 1:
            return _DecodePass([S(10, 14, "Good start.")], _INFO, 0, 0)
        return _DecodePass([S(10, 14, "Good start."), S(14.2, 29.8, "And the rest of the sentence.")], _INFO, 0, 0)


class _Continuing(FasterWhisperEngine):
    """FasterWhisperEngine with _decode_once scripted: the first pass stops early at 6.07 s, the second
    decodes the remainder."""

    def __init__(self):
        self._seek = SimpleNamespace(remainders=[])
        self.continuation, self.calls = True, []

    def _decode_once(self, audio, clips, **kw):
        self.calls.append(clips)
        if len(self.calls) == 1:
            self._seek.remainders = [{"start": 6.07, "end": 7.94}]  # first pass stopped early
            return [S(0.15, 6.07, "dismay upon")], _INFO
        self._seek.remainders = []
        return [S(6.07, 7.9, "banister in the farther corner")], _INFO


def _transcribe(eng):
    return eng.transcribe(np.zeros(16000 * 31, np.float32), language=None)


def _decode_8s_clip(eng):
    return eng._decode(
        np.zeros(16000 * 8, np.float32),
        [{"start": 0.15, "end": 7.94}],
        language="en",
        word_timestamps=False,
        prompt=None,
    )


# ------------------------------------------------------------------ VAD clips


def test_clips_merge_across_short_gaps_only():
    eng = _ScriptedWhisper([], batched=[], redo=[])  # max_gap_s=1.5, max_clip_s=30
    assert eng.clips([(0, 1), (2, 3), (5, 6)]) == [{"start": 0, "end": 3}, {"start": 5, "end": 6}]
    assert eng.clips([(0, 20), (21, 35)]) == [
        {"start": 0, "end": 20},
        {"start": 21, "end": 35},
    ]  # max_clip_s caps the merge


# ------------------------------------------------------------------ coverage check and re-decode


def test_timestamp_jitter_alone_does_not_trigger_a_redecode():
    regions = [(20.0, 21.6)]  # a 1.2 s utterance (20.2-21.4) plus 0.2 s VAD padding each side
    segs = [S(20.55, 21.15, "yes that's right")]  # correct text, boundaries ~0.3 s inside the speech
    tolerant = dict(tolerance_s=COVERAGE_TOLERANCE_S, shrink_s=VAD_SPEECH_PAD_MS / 1000)  # as the engine checks
    assert uncovered(regions, segs)  # with zero tolerance, jitter alone counts as missing speech
    assert uncovered(regions, segs, **tolerant) == []
    assert uncovered([(10.0, 30.0)], [S(10.2, 14.0, "a")], **tolerant)  # a real lost tail still fires


def test_fallback_replaces_partial_batched_text_instead_of_duplicating_it():
    eng = _ScriptedWhisper(
        [(10.0, 30.0)],
        batched=[S(10, 14, "Hello there, how are you")],
        redo=[S(10, 15.5, "Hello there, how are you doing today?"), S(16, 29, "Rest of it.")],
    )
    texts = [s.text for s in _transcribe(eng).segments]
    assert texts == ["Hello there, how are you doing today?", "Rest of it."]


def test_fallback_replaces_partially_overlapping_segments_whole():
    # batched segment ends mid-sentence inside the re-decoded range
    eng = _ScriptedWhisper(
        [(19.0, 30.2)],
        batched=[S(19.75, 26.73, "the main risks? We need two more weeks of")],
        redo=[S(19.75, 29.9, "the main risks? We need two more weeks of testing.")],
    )
    texts = [s.text for s in _transcribe(eng).segments]
    assert texts == ["the main risks? We need two more weeks of testing."]


def test_loop_segments_do_not_count_as_coverage():
    loop = S(14, 22, "la la la la la la la la", cr=3.0)
    eng = _ScriptedWhisper([(10.0, 30.0)], batched=[S(10, 14, "Good start."), loop], redo=[])
    _transcribe(eng)
    assert eng.redo_ranges and eng.redo_ranges[-1][1] >= 30.0  # the untranscribed 22-30 s is re-decoded


def test_fallback_keeps_batched_text_when_the_redo_returns_nothing():
    loop = S(14, 22, "la la la la la la la la", cr=3.0)
    eng = _ScriptedWhisper([(10.0, 30.0)], batched=[S(10, 14, "Good start."), loop], redo=[])
    texts = [s.text for s in _transcribe(eng).segments]
    assert texts == ["Good start."]


def test_lost_tail_is_fixed_by_the_cheap_batched_stage():
    eng = _TwoPass()
    r = _transcribe(eng)
    assert [s.text for s in r.segments] == ["Good start.", "And the rest of the sentence."]
    assert eng.redo_ranges is None  # the expensive sequential decoder never ran
    assert r.extra["suspect_ranges"] == 1 and r.extra["stage2_ranges"] == 0


# ------------------------------------------------------------------ continuation of early-stopped clips


def test_seek_aware_records_where_a_clip_stopped_early():
    # Whisper output "...upon<5.92><5.92>" on an 8 s clip: faster-whisper returns seek=5.92 s
    # (592 frames) and single_timestamp_ending=False, i.e. "continue from 5.92 s".
    fake = SimpleNamespace(
        frames_per_second=100, _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 592, False)
    )
    proxy = _SeekAware(fake)
    proxy._split_segments_by_timestamps(None, [1], time_offset=0.15, segment_size=800, segment_duration=7.79, seek=0)
    assert proxy.remainders == [{"start": pytest.approx(6.07), "end": pytest.approx(7.94)}]
    finished = SimpleNamespace(
        frames_per_second=100,  # ended with a single timestamp: nothing left
        _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 800, True),
    )
    done = _SeekAware(finished)
    done._split_segments_by_timestamps(None, [1], time_offset=0, segment_size=800, segment_duration=8.0, seek=0)
    assert done.remainders == []
    almost = SimpleNamespace(
        frames_per_second=100,  # stopped 0.3 s before the end: too short to continue
        _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 770, False),
    )
    short = _SeekAware(almost)
    short._split_segments_by_timestamps(None, [1], time_offset=0, segment_size=800, segment_duration=8.0, seek=0)
    assert short.remainders == []


def test_seek_aware_refuses_a_model_without_the_private_hook():
    with pytest.raises(RuntimeError, match="_split_segments_by_timestamps"):
        _SeekAware(SimpleNamespace(frames_per_second=100))


def test_early_stopped_clips_are_continued_in_another_batched_pass():
    eng = _Continuing()
    result = _decode_8s_clip(eng)
    assert [s.text for s in result.segments] == ["dismay upon", "banister in the farther corner"]
    assert eng.calls[1] == [{"start": 6.07, "end": 7.94}]
    assert (result.early_stops, result.left_over) == (1, 0)
    assert [s.flags for s in result.segments] == [[], ["continued"]]


def test_continuation_can_be_switched_off_and_leftovers_are_counted():
    eng = _Continuing()
    eng.continuation = False
    result = _decode_8s_clip(eng)
    assert [s.text for s in result.segments] == ["dismay upon"] and len(eng.calls) == 1
    assert (result.early_stops, result.left_over) == (1, 1)  # detected, not continued
