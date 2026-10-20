"""One test per defect found in the adversarially-verified code review, so none can come back."""

import json
import struct
import time
from fractions import Fraction

import av
import httpx
import numpy as np
import pytest

from stt_pipeline.audio import TARGET_SR, load_pcm, probe
from stt_pipeline.config import Settings
from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.engines.openai_compat import OpenAICompatibleEngine, _retry_after
from stt_pipeline.errors import InvalidOptions, ProviderError, UndecodableAudio
from stt_pipeline.formats import render, to_vtt
from stt_pipeline.jobs import JobStore, Worker
from stt_pipeline.pipeline import DirCheckpointStore, Pipeline, PipelineConfig
from stt_pipeline.postprocess import QualityGate, redact_pii, run_chain
from stt_pipeline.schemas import AudioMeta, JobError, ModelMeta, Segment, TranscribeOptions, Transcript, Word

from .conftest import bursts, write_wav


def T(*segments, dropped=()):
    t = Transcript(duration_s=100.0, segments=list(segments),
                   audio=AudioMeta(container="wav", codec="pcm", sample_rate=16000, channels=1, decoded_duration_s=100.0),
                   model=ModelMeta(engine="t", model="t"))
    t.dropped = list(dropped)
    return t


def S(start, end, text, **kw):
    return Segment(start=start, end=end, text=text, avg_logprob=kw.pop("lp", -0.2), no_speech_prob=0.01,
                   compression_ratio=1.2, **kw)


# ------------------------------------------------------------------ checkpoints

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
    Pipeline(MockEngine(), PipelineConfig(chunk_s=40, chunk_search_s=10)).run(p, checkpoints=store)
    t = Pipeline(MockEngine(), PipelineConfig(chunk_s=80, chunk_search_s=10)).run(p, checkpoints=store)
    assert len(t.segments) == 30  # no audio silently lost or duplicated


# ------------------------------------------------------------------ ingestion

def _adts(path, rate, seconds):
    with av.open(str(path), "w", format="adts") as out:
        st = out.add_stream("aac", rate=rate)
        st.layout = "mono"
        cc = st.codec_context
        t = np.arange(int(rate * seconds)) / rate
        x = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)[None, :]
        fifo = av.AudioFifo()
        f = av.AudioFrame.from_ndarray(x, format="fltp", layout="mono")
        f.sample_rate = rate
        fifo.write(f)
        pts = 0
        while fifo.samples:
            fr = fifo.read(min(cc.frame_size, fifo.samples))
            fr.pts, fr.time_base = pts, Fraction(1, rate)
            pts += fr.samples
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)


def test_mid_stream_sample_rate_change_is_resampled(tmp_path):
    a, b = tmp_path / "a.aac", tmp_path / "b.aac"
    _adts(a, 16000, 3.0)
    _adts(b, 44100, 3.0)
    joined = tmp_path / "joined.aac"
    joined.write_bytes(a.read_bytes() + b.read_bytes())  # 16 kHz first: the passthrough trap
    seconds = len(load_pcm(joined)) / TARGET_SR
    assert 5.8 < seconds < 6.3, seconds  # was 11.5 s: the 44.1 kHz half got time-stretched


def test_unknown_codec_is_a_422_not_a_crash(tmp_path):
    data = b"\x00\x01" * 16000
    fmt = struct.pack("<HHIIHH", 0x1234, 1, 16000, 32000, 2, 16)
    wav = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt
    (tmp_path / "x.wav").write_bytes(wav + b"data" + struct.pack("<I", len(data)) + data)
    with pytest.raises(UndecodableAudio):
        probe(tmp_path / "x.wav")


def test_bitrate_estimated_duration_does_not_reject_up_front(tmp_path):
    p = tmp_path / "vbr.mp3"
    with av.open(str(p), "w", format="mp3") as out:  # low-bitrate start skews FFmpeg's estimate
        st = out.add_stream("libmp3lame", rate=16000)
        st.layout, st.bit_rate = "mono", 8000
        x = np.zeros((1, 16000 * 20), dtype=np.float32)
        f = av.AudioFrame.from_ndarray(x, format="fltp", layout="mono")
        f.sample_rate = 16000
        for pkt in st.encode(f):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    info = probe(p, max_duration_s=10.0)  # 20 s estimate vs 10 s limit: advisory, decode decides
    assert info.duration_is_estimate
    from stt_pipeline.errors import AudioTooLong

    with pytest.raises(AudioTooLong):  # ...but an estimate far over (>3x) is rejected up front
        probe(p, max_duration_s=5.0)


# ------------------------------------------------------------------ post-processing

def test_pii_is_redacted_in_dropped_segments_too():
    t = T(*[S(i * 2.5, i * 2.5 + 2, "My number is 415-555-0132 ok.") for i in range(3)])  # 3rd copy is dropped
    t = redact_pii(QualityGate()(t))
    assert "415-555-0132" not in t.model_dump_json()


def test_card_split_across_a_pause_is_redacted():
    t = redact_pii(T(S(0, 2, "It is 4111 1111"), S(3, 5, "1111 1111 thanks")))
    assert not any(ch.isdigit() for s in t.segments for ch in s.text)


def test_word_timings_never_keep_digits_the_text_redacted():
    words = [Word(start=i, end=i + 0.5, word=w) for i, w in enumerate([" My", " SSN", " is", " 123", " 45", " 6789"])]
    t = redact_pii(T(S(0, 6, "My SSN is 123-45-6789.", words=words)))
    assert t.segments[0].text == "My SSN is [SSN]."
    assert not any(c.isdigit() for w in t.segments[0].words for c in w.word)


def test_cross_segment_loops_are_removed():
    t = QualityGate()(T(*[S(i * 1.5, i * 1.5 + 1, "I'm sorry.") for i in range(12)]))
    assert [s.text for s in t.segments] == ["I'm sorry."] * 3  # "No. No. No." is speech; x12 is a loop


def test_duplicate_rule_is_unicode_aware():
    t = QualityGate()(T(S(0, 1, "iPhone 15 Pro を買いました"), S(1.5, 2.5, "iPhone 15 Pro が欲しいです")))
    assert len(t.segments) == 2


def test_overlapping_split_channel_cues_keep_their_own_time():
    long = " ".join(["word"] * 40)
    t = run_chain(T(S(10, 25, long, speaker="channel_0"), S(12, 13, "Mm-hmm, right.", speaker="channel_1")), [])
    vtt = to_vtt(t)
    assert "00:00:12.000 --> 00:00:13.000\n<v channel_1>Mm-hmm, right." in vtt


def test_chunk_citation_range_covers_every_segment():
    t = run_chain(T(S(10, 40, "long turn " * 10, speaker="channel_0"), S(12, 13, "ok", speaker="channel_1")), [])
    c = json.loads(render(t, "chunks"))["chunks"][0]
    assert (c["start"], c["end"]) == (10, 40)


def test_invalid_language_is_rejected_at_the_edge():
    with pytest.raises(ValueError):
        TranscribeOptions(language="en-US")
    assert TranscribeOptions(language="EN").language == "en"


# ------------------------------------------------------------------ engine orchestration

class _ScriptedWhisper(FasterWhisperEngine):
    """FasterWhisperEngine with the model calls replaced, to test the fallback orchestration."""

    def __init__(self, regions, batched, redo):
        self.max_gap_s, self.max_clip_s = 1.5, 30.0
        self.params = {"beam_size": 5}
        self._regions, self._batched, self._redo, self.redo_ranges = regions, batched, redo, None

    def speech_regions(self, audio):
        return self._regions

    def _decode(self, audio, clips, **kw):
        return [s.model_copy() for s in self._batched], type("Info", (), {"language": "en", "language_probability": 1.0})

    def _sequential(self, audio, ranges, **kw):
        self.redo_ranges = ranges
        return [s.model_copy() for s in self._redo]


def test_fallback_replaces_partial_batched_text_instead_of_duplicating_it():
    eng = _ScriptedWhisper([(10.0, 30.0)], batched=[S(10, 14, "Hello there, how are you")],
                           redo=[S(10, 15.5, "Hello there, how are you doing today?"), S(16, 29, "Rest of it.")])
    texts = [s.text for s in eng.transcribe(np.zeros(16000 * 31, np.float32), language=None).segments]
    assert texts == ["Hello there, how are you doing today?", "Rest of it."]


def test_loop_segments_do_not_count_as_coverage():
    loop = S(14, 22, "la la la la la la la la")
    loop.compression_ratio = 3.0
    eng = _ScriptedWhisper([(10.0, 30.0)], batched=[S(10, 14, "Good start."), loop], redo=[])
    eng.transcribe(np.zeros(16000 * 31, np.float32), language=None)
    assert eng.redo_ranges and eng.redo_ranges[-1][1] >= 30.0  # the untranscribed 22-30 s is re-decoded


# ------------------------------------------------------------------ hosted adapter

def _client(responses):
    return httpx.Client(transport=httpx.MockTransport(lambda r: responses.pop(0)))


def test_provider_language_names_map_to_iso_and_unknowns_are_not_pinned():
    for name, code in (("polish", "pl"), ("Swedish", "sv"), ("en", "en"), ("klingon", None)):
        eng = OpenAICompatibleEngine("https://x/v1", "k", client=_client([httpx.Response(200, json={"language": name, "segments": []})]))
        assert eng.transcribe(np.zeros(16000, np.float32), language=None).language == code


def test_each_word_lands_in_exactly_one_segment():
    payload = {"language": "en", "segments": [{"start": 0, "end": 5, "text": "a b"}, {"start": 5, "end": 9, "text": "c"}],
               "words": [{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": 4.98, "end": 5.02},
                         {"word": "c", "start": 6, "end": 7}]}
    eng = OpenAICompatibleEngine("https://x/v1", "k", client=_client([httpx.Response(200, json=payload)]))
    segs = eng.transcribe(np.zeros(16000, np.float32), language=None, word_timestamps=True).segments
    assert sum(len(s.words) for s in segs) == 3


def test_auth_errors_are_server_side_not_client_errors():
    eng = OpenAICompatibleEngine("https://x/v1", "k", client=_client([httpx.Response(401, text="Incorrect API key sk-abc")]))
    with pytest.raises(ProviderError) as e:
        eng.transcribe(np.zeros(16000, np.float32), language=None)
    assert e.value.status == 502 and "sk-" not in e.value.message


def test_retry_after_accepts_http_dates_and_is_capped():
    assert _retry_after("Wed, 21 Oct 2015 07:28:00 GMT", default=1.0) == 0.0
    assert _retry_after("99999", default=1.0) == 60.0
    assert _retry_after("garbage", default=2.0) == 2.0


def test_hosted_engine_defaults_to_a_timestamp_capable_model():
    s = Settings()
    s.engine, s.model = "openai-compatible", None
    assert s.engine_kwargs()["model"] == "whisper-1"


# ------------------------------------------------------------------ queue

KEY = "k"


def _submit(store, tmp_path, name):
    p = write_wav(tmp_path / f"{name}.wav", bursts([(1, 2)], 3.0))
    return store.submit(name, KEY, p, "x.wav", TranscribeOptions())[0]


def test_poison_jobs_cannot_be_resurrected_by_resubmitting(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "p")
    for _ in range(3):
        store.claim("w", KEY, lease_s=-1, max_attempts=3)
    store.claim("w", KEY, lease_s=60, max_attempts=3)
    job, created = store.submit("p", KEY, write_wav(tmp_path / "again.wav", bursts([(1, 2)], 3.0)), "x", TranscribeOptions())
    assert not created and job.status == "failed" and job.error.code == "poison_job"


def test_result_persistence_failure_does_not_kill_the_worker(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "j")
    w = Worker(store, Pipeline(MockEngine()), KEY)
    monkeypatch.setattr(store, "complete", lambda *a: (_ for _ in ()).throw(OSError("No space left on device")))
    assert w.run_once()  # returned normally
    job = store.get("j")
    assert job.status == "queued" and job.error.code == "storage_error"


def test_stalled_job_stops_renewing_its_lease(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "slow")

    class Hang(MockEngine):
        def transcribe(self, *a, **k):
            time.sleep(1.0)
            return super().transcribe(*a, **k)

    w = Worker(store, Pipeline(Hang()), KEY, lease_s=0.2, stall_timeout_s=0.3)
    w.run_once()
    assert not w.healthy  # worker marked itself wedged (the CLI worker would exit here)


def test_sync_caller_upgrades_a_queued_async_duplicate(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "d")
    store.submit("d", KEY, write_wav(tmp_path / "d2.wav", bursts([(1, 2)], 3.0)), "x", TranscribeOptions(), priority=0)
    assert store._one("SELECT priority FROM jobs WHERE id='d'")["priority"] == 0


def test_orphaned_running_job_can_be_erased(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "o")
    store.claim("dead-worker", KEY, lease_s=-1, max_attempts=3)
    assert store.delete("o") and store.get("o") is None


def test_stranded_jobs_fail_instead_of_blocking_backpressure(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "s")
    assert store.fail_stranded(older_than_s=-1) == 1
    assert store.get("s").error.code == "engine_unavailable"


def test_oversized_body_is_rejected_before_parsing(tmp_path):
    from fastapi.testclient import TestClient

    from stt_pipeline.api import create_app

    s = Settings()
    s.data_dir, s.engine, s.max_upload_bytes = tmp_path / "d", "mock", 1024
    with TestClient(create_app(s)) as c:
        r = c.post("/v1/jobs", files={"file": ("big.wav", b"\0" * (3 << 20))})
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    assert not list((tmp_path / "d" / "uploads").glob("*")) if (tmp_path / "d" / "uploads").exists() else True


def test_fallback_replaces_partially_overlapping_segments_whole():
    # turbo case: batched segment ends mid-sentence inside the region that gets re-decoded
    eng = _ScriptedWhisper([(19.0, 30.2)], batched=[S(19.75, 26.73, "the main risks? We need two more weeks of")],
                           redo=[S(19.75, 29.9, "the main risks? We need two more weeks of testing.")])
    texts = [s.text for s in eng.transcribe(np.zeros(16000 * 31, np.float32), language=None).segments]
    assert texts == ["the main risks? We need two more weeks of testing."]


# ------------------------------------------------------------------ second review pass (fix verification)

def test_sync_too_long_does_not_block_the_async_path(tmp_path):
    from fastapi.testclient import TestClient

    from stt_pipeline.api import create_app

    s = Settings()
    s.data_dir, s.engine, s.sync_max_duration_s = tmp_path / "d", "mock", 5
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2), (8, 9)], 12.0))
    with TestClient(create_app(s, worker_engine=MockEngine())) as c:
        with open(p, "rb") as f:
            r = c.post("/v1/transcriptions", files={"file": ("a.wav", f)})
        assert r.status_code == 413 and "/v1/jobs" in r.json()["error"]["message"]
        with open(p, "rb") as f:
            r = c.post("/v1/jobs", files={"file": ("a.wav", f)})
        assert r.status_code == 202  # a new async job, not the failed sync one replayed


def test_worker_restart_does_not_fail_the_backlog(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "old")
    with store._tx() as c:
        c.execute("UPDATE jobs SET created_at = created_at - 7200")  # waited 2 h: normal for long files
    store.register_worker("w1", KEY, "m")
    store.remove_worker("w1")  # clean shutdown during a deploy; replacement still loading
    assert store.fail_stranded(older_than_s=3600) == 0
    assert store.get("old").status == "queued"


def test_rows_with_legacy_options_do_not_crash_the_worker(tmp_path):
    store = JobStore(tmp_path)
    _submit(store, tmp_path, "legacy")
    with store._tx() as c:
        c.execute("""UPDATE jobs SET options='{"language": "en-US", "word_timestamps": false, "channel_mode": "mix",
                     "redact_pii": false, "prompt": null}'""")
    Worker(store, Pipeline(MockEngine()), KEY).run_once()  # must not raise
    job = store.get("legacy")
    assert job.status == "failed" and job.error.code == "invalid_options" and job.error.status == 422


def test_phone_context_rules():
    from stt_pipeline.postprocess import _apply_spans, _pii_spans

    redact = lambda x: _apply_spans(x, _pii_spans(x))
    assert redact("Call me on 415, 555, 0132.") == "Call me on [PHONE]."
    assert redact("tiers are 250, 500, 1000 dollars") == "tiers are 250, 500, 1000 dollars"
    assert redact("order 1234567890") == "order 1234567890"
    assert redact("option 2, 4111 1111 1111 1111") == "option 2, [CARD]"
    assert redact("tracking 9400 1000 0000 0000 0000 00") == "tracking 9400 1000 0000 0000 0000 00"


def test_card_split_around_the_other_partys_backchannel():
    t = redact_pii(T(S(0, 2, "It is 4111 1111", speaker="channel_1"), S(2.2, 2.6, "Mm-hmm.", speaker="channel_0"),
                     S(3, 4, "1111", speaker="channel_1"), S(4.5, 6, "1111 thanks", speaker="channel_1")))
    assert not any(c.isdigit() for s in t.segments for c in s.text)


def test_words_follow_text_redaction_for_spoken_email_across_a_pause():
    def ws(start, text):
        return [Word(start=start + i * 0.4, end=start + i * 0.4 + 0.3, word=" " + w) for i, w in enumerate(text.split())]
    a = S(0, 3, "write to john dot smith", words=ws(0, "write to john dot smith"))
    b = S(3.5, 6, "at example dot com", words=ws(3.5, "at example dot com"))
    t = redact_pii(T(a, b))
    assert [w.word.strip() for w in t.segments[0].words] == ["write", "to", "[EMAIL]"]
    assert [w.word.strip() for w in t.segments[1].words] == ["[EMAIL]"]


def test_fallback_keeps_batched_text_when_the_redo_returns_nothing():
    loop = S(14, 22, "la la la la la la la la")
    loop.compression_ratio = 3.0
    eng = _ScriptedWhisper([(10.0, 30.0)], batched=[S(10, 14, "Good start."), loop], redo=[])
    texts = [s.text for s in eng.transcribe(np.zeros(16000 * 31, np.float32), language=None).segments]
    assert texts == ["Good start."]


def test_srt_fallback_cues_all_carry_the_speaker():
    from stt_pipeline.formats import to_srt

    t = run_chain(T(S(0, 20, "Okay. Sounds good to me, thanks.", speaker="channel_1")), [])
    cues = [c.split("\n")[2:] for c in to_srt(t).strip().split("\n\n")]
    assert len(cues) >= 2 and all(c[0].startswith("channel_1: ") for c in cues)
    assert all(len(line) <= 42 for c in cues for line in c)


def test_captions_apply_the_audio_start_offset():
    t = run_chain(T(S(1.0, 2.0, "Hello.")), [])
    t.audio.start_offset_s = 1.5
    assert "00:00:02,500 --> 00:00:03,500" in render(t, "srt")


def test_hosted_engine_refuses_local_model_names():
    with pytest.raises(ValueError, match="local faster-whisper model"):
        OpenAICompatibleEngine("https://x/v1", "k", "large-v3-turbo")


# ------------------------------------------------------------------ real-data tuning (Earnings-22 audit)

def test_timestamp_jitter_alone_does_not_trigger_a_redecode():
    from stt_pipeline.engines.faster_whisper_engine import uncovered

    regions = [(20.0, 21.6)]  # a 1.2 s utterance (20.2-21.4) plus 0.2 s VAD padding each side
    segs = [S(20.55, 21.15, "yes that's right")]  # correct text, boundaries ~0.3 s inside the speech
    assert uncovered(regions, segs)  # the original, intolerant check fires on jitter alone
    assert uncovered(regions, segs, tolerance_s=0.3, shrink_s=0.2) == []
    long_region = [(10.0, 30.0)]
    assert uncovered(long_region, [S(10.2, 14.0, "a")], tolerance_s=0.3, shrink_s=0.2)  # a real lost tail still fires


class _TwoPass(_ScriptedWhisper):
    """First batched pass loses the tail; the targeted batched re-decode recovers it."""

    def __init__(self):
        super().__init__([(10.0, 30.0)], batched=[S(10, 14, "Good start.")],
                         redo=[S(10, 30, "should not be needed")])
        self.decode_calls = 0

    def _decode(self, audio, clips, **kw):
        self.decode_calls += 1
        info = type("Info", (), {"language": "en", "language_probability": 1.0})
        if self.decode_calls == 1:
            return [S(10, 14, "Good start.")], info
        return [S(10, 14, "Good start."), S(14.2, 29.8, "And the rest of the sentence.")], info


def test_lost_tail_is_fixed_by_the_cheap_batched_stage():
    eng = _TwoPass()
    r = eng.transcribe(np.zeros(16000 * 31, np.float32), language=None)
    assert [s.text for s in r.segments] == ["Good start.", "And the rest of the sentence."]
    assert eng.redo_ranges is None  # the expensive sequential decoder never ran
    assert r.extra["suspect_ranges"] == 1 and r.extra["stage2_ranges"] == 0


# ------------------------------------------------------------------ batched continuation (root cause of lost tails)

def test_seek_aware_records_where_a_clip_stopped_early():
    from types import SimpleNamespace

    from stt_pipeline.engines.faster_whisper_engine import _SeekAware

    # Whisper output "...upon<5.92><5.92>" on an 8 s clip: faster-whisper returns seek=5.92 s
    # (592 frames) and single_timestamp_ending=False, i.e. "continue from 5.92 s".
    fake = SimpleNamespace(frames_per_second=100,
                           _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 592, False))
    proxy = _SeekAware(fake)
    proxy._split_segments_by_timestamps(None, [1], time_offset=0.15, segment_size=800, segment_duration=7.79, seek=0)
    assert proxy.remainders == [{"start": pytest.approx(6.07), "end": pytest.approx(7.94)}]
    finished = SimpleNamespace(frames_per_second=100,  # ended with a single timestamp: nothing left
                               _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 800, True))
    done = _SeekAware(finished)
    done._split_segments_by_timestamps(None, [1], time_offset=0, segment_size=800, segment_duration=8.0, seek=0)
    assert done.remainders == []
    almost = SimpleNamespace(frames_per_second=100,  # stopped 0.3 s before the end: too short to continue
                             _split_segments_by_timestamps=lambda **kw: ([{"tokens": [1]}], 770, False))
    short = _SeekAware(almost)
    short._split_segments_by_timestamps(None, [1], time_offset=0, segment_size=800, segment_duration=8.0, seek=0)
    assert short.remainders == []


def test_continuation_can_be_switched_off_and_leftovers_are_counted():
    eng = _Continuing()
    eng.continuation = False
    segs, _ = eng._decode(np.zeros(16000 * 8, np.float32), [{"start": 0.15, "end": 7.94}],
                          language="en", word_timestamps=False, prompt=None)
    assert [s.text for s in segs] == ["dismay upon"] and len(eng.calls) == 1
    assert eng._continued == 1 and eng._left_over == 1  # detected, not continued


class _Continuing(FasterWhisperEngine):
    def __init__(self):
        from types import SimpleNamespace

        self._seek = SimpleNamespace(remainders=[])
        self._continued, self._left_over, self.calls = 0, 0, []

    def _decode_once(self, audio, clips, **kw):
        self.calls.append(clips)
        info = type("Info", (), {"language": "en", "language_probability": 1.0})
        if len(self.calls) == 1:
            self._seek.remainders = [{"start": 6.07, "end": 7.94}]  # first pass stopped early
            return [S(0.15, 6.07, "dismay upon")], info
        self._seek.remainders = []
        return [S(6.07, 7.9, "banister in the farther corner")], info


def test_early_stopped_clips_are_continued_in_another_batched_pass():
    eng = _Continuing()
    segs, _ = eng._decode(np.zeros(16000 * 8, np.float32), [{"start": 0.15, "end": 7.94}],
                          language="en", word_timestamps=False, prompt=None)
    assert [s.text for s in segs] == ["dismay upon", "banister in the farther corner"]
    assert eng.calls[1] == [{"start": 6.07, "end": 7.94}] and eng._continued == 1
    assert [s.flags for s in segs] == [[], ["continued"]]


def test_weak_continuations_are_dropped_but_confident_ones_and_normal_segments_are_kept():
    # Real cases from Earnings-22: a 1.6 s leftover piece decoded with no context turned "and fully
    # utilize" into "I'm free to die." (avg log-prob -1.12); a 0.8 s one came back correctly as
    # "Volumes" (-0.94).
    t = QualityGate()(T(
        S(0, 4, "Our strategy is unchanged.", lp=-1.12),                          # normal, weak: flagged only
        S(4, 5.6, "I'm free to die.", lp=-1.12, flags=["continued"]),             # dropped
        S(6, 6.8, "Volumes", lp=-0.94, flags=["continued"]),                      # kept
    ))
    assert [s.text for s in t.segments] == ["Our strategy is unchanged.", "Volumes"]
    assert t.segments[0].flags == ["low_confidence"]
    assert [(d.segment.text, d.reason) for d in t.dropped] == [("I'm free to die.", "weak_continuation")]
