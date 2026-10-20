"""Silence-aware macro-chunking for long audio.

Two levels of chunking, each solving a different problem:

* Macro-chunks (this module, ~10 min, cut at the longest pause near the target):
  bound memory regardless of file length, give checkpoint/resume and progress
  granularity, and are the unit you would shard across workers. Cutting in a
  pause means no word straddles a boundary, so there is nothing to stitch or
  de-duplicate (unlike fixed windows with overlap).

* Micro-chunks (inside the engine: VAD speech regions merged up to Whisper's
  30 s window, decoded in batches): this is what makes long-form Whisper fast
  and stops it hallucinating through silence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR

# Shortest chunk worth transcribing: a few ms of tail after the last cut. It is also the minimum
# hosted providers accept (they reject sub-100 ms audio).
MIN_CHUNK_S = 0.1


@dataclass
class Chunk:
    index: int
    offset_s: float  # position of audio[0] in the original file
    audio: np.ndarray

    @property
    def duration_s(self) -> float:
        return len(self.audio) / TARGET_SR


def _speech_regions(window: np.ndarray, sr: int) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # Tight padding so gap lengths reflect real pauses: this profile only locates a cut point.
    # The decode-time profile is FasterWhisperEngine.speech_regions.
    opts = VadOptions(threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=30, min_speech_duration_ms=100)
    return [(t["start"], t["end"]) for t in get_speech_timestamps(window, opts, sampling_rate=sr)]


def find_cut(window: np.ndarray, sr: int = TARGET_SR, speech_fn=_speech_regions) -> int:
    """Return the sample index (within `window`) of the best place to cut: the middle of the
    longest non-speech gap per Silero VAD. Falls back to the quietest 50 ms frame if the
    window is wall-to-wall speech."""
    n = len(window)
    regions = speech_fn(window, sr)
    edges = [0] + [x for r in regions for x in r] + [n]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]  # (silence_start, silence_end)
    best = max(gaps, key=lambda g: g[1] - g[0])
    if best[1] - best[0] >= int(0.1 * sr):
        return (best[0] + best[1]) // 2
    frame = int(0.05 * sr)
    usable = n - n % frame
    if usable == 0:
        return n
    rms = np.sqrt(np.mean(window[:usable].reshape(-1, frame) ** 2, axis=1))
    return int(np.argmin(rms)) * frame + frame // 2


def silence_aware_chunks(
    blocks: Iterable[np.ndarray],
    *,
    target_s: float = 600.0,
    search_s: float = 60.0,
    sr: int = TARGET_SR,
    cut_fn: Callable[[np.ndarray, int], int] = find_cut,
) -> Iterator[Chunk]:
    """Re-cut a stream of PCM blocks into chunks of ~target_s, each ending in a pause found
    within the last `search_s` seconds before the target. Holds at most ~target_s + one block
    in memory. Offsets are exact sample counts, so concatenating chunks reproduces the input."""
    if search_s >= target_s:
        raise ValueError("search_s must be smaller than target_s")
    target = int(target_s * sr)
    search = int(search_s * sr)
    pending: list[np.ndarray] = []
    pending_n = 0
    consumed = 0  # samples already emitted
    index = 0
    for block in blocks:
        pending.append(block)
        pending_n += len(block)
        if pending_n < target:
            continue
        buf = np.concatenate(pending)
        pending.clear()  # release the blocks now: peak stays ~one chunk, not blocks + concat + copies
        while len(buf) >= target:
            lo = target - search
            cut = lo + cut_fn(buf[lo:target], sr)
            cut = max(1, min(cut, len(buf)))
            yield Chunk(index, consumed / sr, buf[:cut])  # a view: consumers must not keep the audio
            consumed += cut
            index += 1
            buf = buf[cut:]
        buf = buf.copy()  # drop the reference to the large concatenated base array
        pending, pending_n = [buf], len(buf)
    if pending_n:
        yield Chunk(index, consumed / sr, np.concatenate(pending))
