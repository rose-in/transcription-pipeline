"""Interval maths for the faster-whisper engine's safety net: which VAD speech regions a transcript
fails to account for, and how failing spans become whole-segment re-decode ranges.

Ranges are (start_s, end_s) tuples on the same timeline as the segments.
"""

from __future__ import annotations

from ..schemas import Segment

Range = tuple[float, float]


def touches(seg: Segment, ranges: list[Range]) -> bool:
    return any(seg.start < b and a < seg.end for a, b in ranges)


def merge_ranges(ranges: list[Range], pad: float) -> list[Range]:
    """Sort ranges, widen each by `pad` seconds per side (clamped at 0) and merge overlaps."""
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        a = max(0.0, a - pad)
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b + pad)
        else:
            out.append([a, b + pad])
    return [(a, b) for a, b in out]


def grow_to_segments(ranges: list[Range], segments: list[Segment]) -> list[Range]:
    """Whole segments in, whole segments out: grow ranges until they fully contain every segment
    they touch (replacing part of a segment duplicated its words next to the redo's)."""
    while True:
        grown = merge_ranges(ranges + [(x.start, x.end) for x in segments if touches(x, ranges)], pad=0.0)
        if grown == ranges:
            return ranges
        ranges = grown


def intersect(regions: list[Range], ranges: list[Range]) -> list[Range]:
    """The parts of each region that fall inside some range, in region order (sorted in, sorted out)."""
    return [(max(a, rs), min(b, re_)) for rs, re_ in regions for a, b in ranges if rs < b and a < re_]


def uncovered(
    regions: list[Range],
    segments: list[Segment],
    *,
    min_missing_s: float = 1.5,
    min_region_s: float = 1.0,
    min_cover: float = 0.5,
    tolerance_s: float = 0.0,
    shrink_s: float = 0.0,
) -> list[Range]:
    """Speech regions the transcript doesn't account for: either >= min_missing_s seconds of the
    region lie outside every segment (a lost tail in a long region), or a region >= min_region_s
    is less than min_cover covered. Coverage uses the *union* of segment intervals.

    tolerance_s widens each segment and shrink_s trims the VAD padding off each region, so
    ordinary timestamp jitter on a correct transcript doesn't count as missing speech."""
    spans = sorted((x.start - tolerance_s, x.end + tolerance_s) for x in segments)
    out = []
    for r0, r1 in regions:
        rs, re_ = r0 + shrink_s, r1 - shrink_s
        dur = re_ - rs
        if dur < min_region_s:
            continue
        covered, cursor = 0.0, rs
        for a, b in spans:
            a, b = max(a, cursor), min(b, re_)
            if b > a:
                covered += b - a
                cursor = b
        missing = dur - covered
        if missing >= min_missing_s or covered / dur < min_cover:
            out.append((r0, r1))
    return out
