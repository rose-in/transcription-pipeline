import json

import numpy as np

from stt_pipeline.audio import TARGET_SR, load_pcm
from stt_pipeline.chunking import find_cut, silence_aware_chunks

from .conftest import FIXTURES, bursts


def energy_regions(window, sr):
    """Test double for Silero: frames above an energy threshold are 'speech'."""
    frame = sr // 100
    n = len(window) - len(window) % frame
    loud = np.sqrt(np.mean(window[:n].reshape(-1, frame) ** 2, axis=1)) > 0.02
    regions, start = [], None
    for i, v in enumerate(loud):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start * frame, i * frame))
            start = None
    if start is not None:
        regions.append((start * frame, n))
    return regions


def blocks_of(x, block_s=7.0):
    b = int(block_s * TARGET_SR)
    return (x[i : i + b] for i in range(0, len(x), b))


def test_chunks_are_lossless_and_cut_in_silence():
    # 'speech' every 4 s with 1.5 s pauses, 5 minutes total
    spec = [(t, t + 2.5) for t in np.arange(0.5, 299, 4.0)]
    x = bursts(spec, 300.0)
    cut = lambda w, sr: find_cut(w, sr, speech_fn=energy_regions)
    chunks = list(silence_aware_chunks(blocks_of(x), target_s=60, search_s=20, cut_fn=cut))
    assert np.array_equal(np.concatenate([c.audio for c in chunks]), x)  # no lost/duplicated samples
    for prev, c in zip(chunks, chunks[1:]):
        assert c.offset_s == prev.offset_s + prev.duration_s
        boundary = c.offset_s
        assert not any(s < boundary < e for s, e in spec), f"cut at {boundary:.2f}s splits a word"
    assert all(c.duration_s <= 60 for c in chunks)


def test_wall_to_wall_speech_falls_back_to_quietest_point():
    x = bursts([(0, 100)], 100.0)
    x[int(45.3 * TARGET_SR) : int(45.35 * TARGET_SR)] *= 0.01  # one tiny dip
    chunks = list(silence_aware_chunks(blocks_of(x), target_s=50, search_s=10,
                                       cut_fn=lambda w, sr: find_cut(w, sr, speech_fn=lambda *_: [(0, len(w))])))
    assert abs(chunks[1].offset_s - 45.325) < 0.05


def test_real_vad_cuts_between_tts_turns(manifest):
    x = load_pcm(FIXTURES / "meeting_short.wav")
    truth = json.loads((FIXTURES / "meeting_short.truth.json").read_text())
    chunks = list(silence_aware_chunks(blocks_of(x, 5.0), target_s=12, search_s=6))
    assert len(chunks) >= 3
    for c in chunks[1:]:
        assert not any(t["start"] + 0.15 < c.offset_s < t["end"] - 0.15 for t in truth), c.offset_s
