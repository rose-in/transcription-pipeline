"""Hosted adapter, tested against a fake provider (no network, no API key)."""

import json

import httpx
import numpy as np
import pytest

from stt_pipeline.engines.openai_compat import OpenAICompatibleEngine
from stt_pipeline.errors import EngineFailure, InvalidOptions
from stt_pipeline.pipeline import Pipeline, PipelineConfig

from .conftest import bursts, write_wav

VERBOSE = {
    "language": "english", "duration": 10.0, "text": "Hello there. General Kenobi.",
    "segments": [
        {"id": 0, "start": 1.0, "end": 2.0, "text": " Hello there.", "avg_logprob": -0.2, "no_speech_prob": 0.01,
         "compression_ratio": 1.1},
        {"id": 1, "start": 5.0, "end": 6.0, "text": " General Kenobi.", "avg_logprob": -0.3, "no_speech_prob": 0.02,
         "compression_ratio": 1.2},
    ],
    "words": [{"word": "Hello", "start": 1.0, "end": 1.4}, {"word": "there.", "start": 1.5, "end": 2.0},
              {"word": "General", "start": 5.0, "end": 5.5}, {"word": "Kenobi.", "start": 5.5, "end": 6.0}],
}


def fake_provider(responses, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_request_shape_and_parsing():
    seen = []
    eng = OpenAICompatibleEngine("https://api.example/v1", "k", "whisper-1",
                                 client=fake_provider([httpx.Response(200, json=VERBOSE)], seen))
    r = eng.transcribe(np.zeros(16000 * 10, dtype=np.float32), language="en", word_timestamps=True, prompt="Kenobi")
    req = seen[0]
    body = req.read()
    assert req.url.path == "/v1/audio/transcriptions"
    assert b"fLaC" in body  # lossless FLAC upload, not raw PCM
    for field in (b'name="response_format"\r\n\r\nverbose_json', b'name="timestamp_granularities[]"\r\n\r\nword',
                  b'name="language"\r\n\r\nen', b'name="prompt"\r\n\r\nKenobi'):
        assert field in body
    assert r.language == "en"  # provider's "english" mapped to ISO
    assert [(s.start, s.text) for s in r.segments] == [(1.0, "Hello there."), (5.0, "General Kenobi.")]
    assert [w.word for w in r.segments[1].words] == [" General", " Kenobi."]


def test_retries_rate_limits_then_succeeds():
    seen = []
    client = fake_provider([httpx.Response(429, headers={"retry-after": "0"}),
                            httpx.Response(503, headers={"retry-after": "0"}),
                            httpx.Response(200, json=VERBOSE)], seen)
    r = OpenAICompatibleEngine("https://x/v1", "k", client=client).transcribe(np.zeros(16000, np.float32), language=None)
    assert len(seen) == 3 and len(r.segments) == 2


def test_permanent_vs_transient_errors():
    bad = fake_provider([httpx.Response(400, json={"error": "bad"})], [])
    with pytest.raises(InvalidOptions):
        OpenAICompatibleEngine("https://x/v1", "k", client=bad).transcribe(np.zeros(16000, np.float32), language=None)
    down = fake_provider([httpx.Response(500, headers={"retry-after": "0"})] * 2, [])
    with pytest.raises(EngineFailure):  # retryable -> job runner backs off and retries
        OpenAICompatibleEngine("https://x/v1", "k", client=down, max_retries=1).transcribe(
            np.zeros(16000, np.float32), language=None)


def test_text_only_models_are_rejected_up_front():
    with pytest.raises(ValueError, match="no segment timestamps"):
        OpenAICompatibleEngine("https://api.openai.com/v1", "k", "gpt-4o-transcribe")


def test_pipeline_chunks_to_fit_provider_limits(tmp_path):
    seen = []
    eng = OpenAICompatibleEngine("https://x/v1", "k", client=fake_provider(
        [httpx.Response(200, json={"language": "en", "segments": []}) for _ in range(10)], seen))
    p = write_wav(tmp_path / "a.wav", bursts([(t, t + 2) for t in range(1, 1499, 5)], 1500.0))
    t = Pipeline(eng, PipelineConfig(chunk_s=3600)).run(p)  # 25 min file, engine caps chunks at 10 min
    assert len(seen) == 3 and all(c.duration_s <= 600 for c in t.chunks)
