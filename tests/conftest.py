from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from stt_pipeline.audio import TARGET_SR

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


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


@pytest.fixture(scope="session")
def manifest() -> list[dict]:
    p = FIXTURES / "manifest.json"
    if not p.exists():
        pytest.skip("fixtures not generated: run `uv run python scripts/make_fixtures.py` (macOS)")
    return json.loads(p.read_text())
