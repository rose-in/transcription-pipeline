import numpy as np
import pytest

from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.errors import EngineFailure, InvalidOptions
from stt_pipeline.pipeline import DirCheckpointStore, Pipeline, PipelineConfig
from stt_pipeline.schemas import TranscribeOptions

from .conftest import bursts, write_wav

SPEC = [(t, t + 2.0) for t in np.arange(1.0, 175.0, 3.5)]  # 50 utterances over 3 minutes


@pytest.fixture
def long_wav(tmp_path):
    return write_wav(tmp_path / "long.wav", bursts(SPEC, 180.0))


def small_chunks():
    return PipelineConfig(chunk_s=40.0, chunk_search_s=10.0)


class _DetectsGerman(MockEngine):
    """Without a forced language, "detects" 'de' on speech and 'en' on silence (Whisper still guesses on
    silence); records the language the pipeline passed in on each call."""

    def __init__(self):
        super().__init__()
        self.languages_in: list[str | None] = []

    def transcribe(self, audio, *, language, **kwargs):
        self.languages_in.append(language)
        result = super().transcribe(audio, language=language, **kwargs)
        result.language = language or ("de" if result.segments else "en")
        return result


def test_timestamps_survive_chunking(long_wav):
    t = Pipeline(MockEngine(), small_chunks()).run(long_wav)
    assert len(t.chunks) >= 4  # really exercised multi-chunk path
    assert len(t.segments) == len(SPEC)
    for seg, (s, e) in zip(t.segments, SPEC):
        assert abs(seg.start - s) < 0.03 and abs(seg.end - e) < 0.03
    assert [s.id for s in t.segments] == list(range(len(SPEC)))
    assert t.duration_s == pytest.approx(180.0, abs=0.01)


def test_crash_then_resume_from_checkpoints(long_wav, tmp_path):
    store = DirCheckpointStore(tmp_path / "ckpt")
    crashing = MockEngine(fail_on_call=3)
    with pytest.raises(EngineFailure):
        Pipeline(crashing, small_chunks()).run(long_wav, checkpoints=store)
    resumed_engine = MockEngine()
    resumed = Pipeline(resumed_engine, small_chunks()).run(long_wav, checkpoints=store)
    clean = Pipeline(MockEngine(), small_chunks()).run(long_wav)
    n_chunks = len(clean.chunks)
    assert resumed_engine.calls == n_chunks - 2  # the two finished chunks were not recomputed
    assert sum(c.cached for c in resumed.chunks) == 2
    assert [s.model_dump() for s in resumed.segments] == [s.model_dump() for s in clean.segments]


def test_checkpoint_dir_never_returns_another_files_text(tmp_path):
    store = DirCheckpointStore(tmp_path / "ckpt")
    a = write_wav(tmp_path / "a.wav", bursts([(1, 3)], 10.0))
    b = write_wav(tmp_path / "b.wav", bursts([(5, 7), (8, 9)], 10.0, seed=3))
    Pipeline(MockEngine()).run(a, checkpoints=store)
    t = Pipeline(MockEngine()).run(b, checkpoints=store)
    assert [round(s.start) for s in t.segments] == [5, 8]


def test_checkpoints_from_a_different_chunk_size_are_not_reused(tmp_path):
    store = DirCheckpointStore(tmp_path / "ckpt")
    p = write_wav(tmp_path / "a.wav", bursts([(t, t + 2) for t in range(1, 119, 4)], 120.0))
    Pipeline(MockEngine(), small_chunks()).run(p, checkpoints=store)
    t = Pipeline(MockEngine(), PipelineConfig(chunk_s=80, chunk_search_s=10)).run(p, checkpoints=store)
    assert len(t.segments) == 30  # no audio silently lost or duplicated


def test_language_is_detected_on_the_first_speech_chunk_then_pinned(tmp_path):
    spec = [(t, t + 2.0) for t in np.arange(45.0, 175.0, 3.5)]  # first ~35 s chunk is pure silence
    p = write_wav(tmp_path / "late_start.wav", bursts(spec, 180.0))
    engine = _DetectsGerman()
    t = Pipeline(engine, small_chunks()).run(p)
    assert len(engine.languages_in) >= 4
    assert engine.languages_in[:2] == [None, None]  # silent chunk does not pin; first speech chunk detects
    assert set(engine.languages_in[2:]) == {"de"}  # every later chunk is forced to the detected language
    assert t.language == "de"


def test_split_channels_labels_and_interleaves(tmp_path):
    left = bursts([(1, 3), (8, 10)], 12.0, seed=1)
    right = bursts([(4, 6.5)], 12.0, seed=2)
    p = write_wav(tmp_path / "call.wav", np.stack([left, right]))
    t = Pipeline(MockEngine()).run(p, TranscribeOptions(channel_mode="split"))
    assert [(round(s.start), s.speaker) for s in t.segments] == [(1, "channel_0"), (4, "channel_1"), (8, "channel_0")]


def test_split_mode_rejects_more_channels_than_the_limit(tmp_path):
    p = write_wav(tmp_path / "call.wav", np.stack([bursts([(1, 2)], 3.0, seed=1), bursts([(1, 2)], 3.0, seed=2)]))
    with pytest.raises(InvalidOptions, match="channel_mode=split"):
        Pipeline(MockEngine(), PipelineConfig(max_split_channels=1)).run(p, TranscribeOptions(channel_mode="split"))


def test_split_mode_on_mono_warns_and_transcribes_one_channel(tmp_path):
    p = write_wav(tmp_path / "mono.wav", bursts([(1, 2), (4, 5)], 6.0))
    t = Pipeline(MockEngine()).run(p, TranscribeOptions(channel_mode="split"))
    assert any("audio is mono" in w for w in t.warnings)
    assert len(t.segments) == 2 and all(s.speaker is None for s in t.segments)


def test_silence_returns_empty_transcript_with_warning(tmp_path):
    p = write_wav(tmp_path / "quiet.wav", np.zeros(16000 * 5, dtype=np.float32))
    t = Pipeline(MockEngine()).run(p)
    assert t.segments == [] and t.text == ""
    assert "No speech detected." in t.warnings


def test_output_records_provenance(long_wav):
    t = Pipeline(MockEngine(), small_chunks()).run(long_wav, TranscribeOptions(language="en"))
    assert t.model.engine == "mock" and t.options["language"] == "en"
    assert t.audio.container == "wav" and t.audio.decoded_duration_s == pytest.approx(180.0, abs=0.01)
    assert t.timing.rtf > 0
