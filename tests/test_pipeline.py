import numpy as np
import pytest

from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.errors import EngineFailure
from stt_pipeline.pipeline import DirCheckpointStore, Pipeline, PipelineConfig
from stt_pipeline.schemas import TranscribeOptions

from .conftest import bursts, write_wav

SPEC = [(t, t + 2.0) for t in np.arange(1.0, 175.0, 3.5)]  # 50 utterances over 3 minutes


@pytest.fixture
def long_wav(tmp_path):
    return write_wav(tmp_path / "long.wav", bursts(SPEC, 180.0))


def small_chunks():
    return PipelineConfig(chunk_s=40.0, chunk_search_s=10.0)


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


def test_split_channels_labels_and_interleaves(tmp_path):
    left = bursts([(1, 3), (8, 10)], 12.0, seed=1)
    right = bursts([(4, 6.5)], 12.0, seed=2)
    p = write_wav(tmp_path / "call.wav", np.stack([left, right]))
    t = Pipeline(MockEngine()).run(p, TranscribeOptions(channel_mode="split"))
    assert [(round(s.start), s.speaker) for s in t.segments] == [(1, "channel_0"), (4, "channel_1"), (8, "channel_0")]


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
