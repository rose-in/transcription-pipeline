from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
import pytest

from stt_pipeline.audio import TARGET_SR
from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.jobs import JobStore
from stt_pipeline.pipeline import Pipeline, PipelineConfig
from stt_pipeline.schemas import AudioMeta, Job, ModelMeta, Segment, TranscribeOptions, Transcript
from stt_pipeline.worker import Worker

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_stt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings() reads STT_* from the environment; drop the developer's shell values so tests see config.py defaults.
    STT_TEST_MODEL is kept: it deliberately selects the model for the slow integration tests."""
    for name in list(os.environ):
        if name.startswith("STT_") and name != "STT_TEST_MODEL":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Generated audio fixtures (gitignored). Skips the requesting test when they have not been generated."""
    if not (FIXTURES / "manifest.json").exists():
        pytest.skip("fixtures not generated: run `uv run python scripts/make_fixtures.py` (macOS)")
    return FIXTURES


@pytest.fixture(scope="session")
def manifest(fixtures_dir: Path) -> list[dict]:
    return json.loads((fixtures_dir / "manifest.json").read_text())


# ------------------------------------------------------------------ audio


def write_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> Path:
    """audio: (n,) mono or (channels, n) float32."""
    planar = audio[None, :] if audio.ndim == 1 else audio
    layout = "mono" if planar.shape[0] == 1 else "stereo"
    with av.open(str(path), "w", format="wav") as out:
        st = out.add_stream("pcm_s16le", rate=sr)
        st.layout = layout
        frame = av.AudioFrame.from_ndarray(
            (np.clip(planar, -1, 1) * 32767).astype(np.int16).T.reshape(1, -1).copy(), format="s16", layout=layout
        )
        frame.sample_rate = sr
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return path


def bursts(spec: list[tuple[float, float]], total_s: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """Noise bursts at [(start, end), ...] seconds over silence: stand-in 'speech' with known timing."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(total_s * TARGET_SR), dtype=np.float32)
    for s, e in spec:
        a, b = int(s * TARGET_SR), int(e * TARGET_SR)
        x[a:b] = rng.uniform(-amp, amp, b - a)
    return x


# ------------------------------------------------------------------ transcripts


def T(*segments: Segment) -> Transcript:
    """A 100 s mono-WAV transcript holding exactly these segments."""
    return Transcript(
        duration_s=100.0,
        segments=list(segments),
        audio=AudioMeta(container="wav", codec="pcm", sample_rate=16000, channels=1, decoded_duration_s=100.0),
        model=ModelMeta(engine="test", model="test"),
    )


def S(start: float, end: float, text: str, *, lp: float = -0.2, ns: float = 0.05, cr: float = 1.3, **kw) -> Segment:
    """A segment with confident, non-repetitive Whisper scores unless lp/ns/cr override them."""
    return Segment(start=start, end=end, text=text, avg_logprob=lp, no_speech_prob=ns, compression_ratio=cr, **kw)


# ------------------------------------------------------------------ job queue

ENGINE_KEY = "test-engine"


def submit_job(store: JobStore, tmp_path: Path, name: str, seconds: float = 3.0) -> Job:
    """Queue a fresh job `name` (one 1 s burst of 'speech') for ENGINE_KEY."""
    p = write_wav(tmp_path / f"{name}.wav", bursts([(1, 2)], seconds))
    job, created = store.submit(name, ENGINE_KEY, p, f"{name}.wav", TranscribeOptions())
    assert created
    return job


def job_worker(store: JobStore, engine=None, **kw) -> Worker:
    """A Worker for ENGINE_KEY over a MockEngine pipeline with 20 s chunks."""
    return Worker(
        store, Pipeline(engine or MockEngine(), PipelineConfig(chunk_s=20, chunk_search_s=5)), ENGINE_KEY, **kw
    )
