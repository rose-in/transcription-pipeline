"""Helpers shared by the evaluation scripts (eval_synthetic.py, eval_real_data.py).

Module-level imports are standard library only, so importing this in a measurement subprocess
doesn't move its RSS baseline.
"""

from __future__ import annotations

import functools
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


@functools.cache
def _normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Whisper's English text normaliser (the Open ASR Leaderboard convention), built once per process."""
    return _normalizer()(text)


def percentile(xs: list[float], q: float) -> float | None:
    """Nearest-rank q-quantile: element min(n-1, int(q*n)) of the sorted values, rounded to 2 dp; None if empty."""
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None
