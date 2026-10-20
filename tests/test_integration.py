"""End-to-end with a real Whisper model. Slow-ish; run with `pytest -m slow`.

Uses STT_TEST_MODEL (default tiny.en) so CI stays fast; WER thresholds are loose on purpose:
these tests catch pipeline regressions (a format that silently decodes to garbage, a chunking
bug that drops speech), not model-quality changes - that's what scripts/eval_synthetic.py is for.
"""

import json
import os
import re

import jiwer
import pytest
from whisper_normalizer.english import EnglishTextNormalizer

from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from stt_pipeline.pipeline import Pipeline
from stt_pipeline.schemas import TranscribeOptions

pytestmark = pytest.mark.slow
norm = EnglishTextNormalizer()


@pytest.fixture(scope="module")
def pipeline(fixtures_dir):
    # Requests fixtures_dir so every test here skips, before a model is downloaded, when fixtures/ is missing.
    return Pipeline(FasterWhisperEngine(os.environ.get("STT_TEST_MODEL", "tiny.en")))


def wer(ref: str, hyp: str) -> float:
    return jiwer.wer(norm(ref), norm(hyp))


def test_format_matrix_transcribes_consistently(pipeline, manifest, fixtures_dir):
    ref = (fixtures_dir / "meeting_short.reference.txt").read_text()
    for m in manifest:
        if m["kind"] == "valid" and m["file"].startswith("meeting_short"):
            t = pipeline.run(fixtures_dir / m["file"], TranscribeOptions(language="en"))
            assert wer(ref, t.text) < 0.15, (m["file"], t.text)


def test_silence_does_not_hallucinate(pipeline, fixtures_dir):
    t = pipeline.run(fixtures_dir / "silence_noise.wav")
    assert t.segments == []


def test_dual_channel_call_turns(pipeline, fixtures_dir):
    truth = json.loads((fixtures_dir / "support_call_dual_channel.truth.json").read_text())
    t = pipeline.run(
        fixtures_dir / "support_call_dual_channel.wav", TranscribeOptions(channel_mode="split", redact_pii=True)
    )
    assert [s.speaker for s in t.segments] == [x["speaker"] for x in truth]
    for seg, x in zip(t.segments, truth):
        assert abs(seg.start - x["start"]) < 1.0
    # Small models mangle number formatting (seen: "415-55-0132"), so assert on the outcome -
    # no phone-like digit run survives - rather than on which pattern caught it.
    assert not re.search(r"\d{3}\D?\d{2,3}\D?\d{4}", t.text), t.text
    assert any("redacted" in s.flags for s in t.segments)
