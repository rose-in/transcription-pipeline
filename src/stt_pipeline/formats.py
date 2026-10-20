"""Renderings of the canonical JSON transcript for downstream consumers.

json   - full fidelity (scores, flags, words, audit trail); the system of record
jsonl  - one segment per line; streams into log pipelines / search indexers
txt    - readable paragraphs (new paragraph on speaker change or long pause)
srt/vtt- captions: <=42 chars/line, <=2 lines, <=7 s per cue (common broadcast guidelines)
chunks - ~1-2k char passages with start/end, ready for embedding / LLM summarisation,
         so every generated claim can cite a timestamp back into the audio
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Callable, Iterator

from .schemas import Segment, Transcript

MEDIA_TYPES = {
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip",
    "vtt": "text/vtt; charset=utf-8",
    "chunks": "application/json",
}
Cue = tuple[float, float, list[str], str | None]  # (start_s, end_s, lines, speaker)


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _cues(
    segments: list[Segment], *, prefix_speaker: bool, max_chars: int = 42, max_lines: int = 2, max_dur: float = 7.0
) -> Iterator[Cue]:
    """Split segments into caption cues of <= max_lines lines of <= max_chars, <= max_dur seconds.
    Yields (start, end, lines, speaker). Uses word timings when present; otherwise allocates time
    proportionally to characters (approximate, but monotonic and never overlapping)."""

    def wrap(text: str, spk: str | None, width: int = max_chars) -> list[str]:
        return textwrap.wrap(f"{spk}: {text}" if (prefix_speaker and spk) else text, width) or [""]

    for seg in segments:
        lines = wrap(seg.text, seg.speaker)
        if len(lines) <= max_lines and seg.end - seg.start <= max_dur:
            yield seg.start, seg.end, lines, seg.speaker
            continue
        if seg.words:
            # Balanced split: if the text needs n cues, aim for ~len/n characters each instead of
            # filling greedily, which strands a one-word orphan cue ("tier.") at the end.
            n = max(
                math.ceil(len(seg.text) / (max_chars * max_lines * 0.9)), math.ceil((seg.end - seg.start) / max_dur)
            )
            target = len(seg.text) / n
            cur, c_start, prev_end = "", None, seg.start
            for w in seg.words:
                cand = cur + w.word
                too_big = len(wrap(cand.strip(), seg.speaker)) > max_lines or (
                    c_start is not None and w.end - c_start > max_dur
                )
                at_clause = cur.rstrip().endswith((",", ";", ":", "—")) and len(cur.strip()) >= 0.6 * target
                if cur and (too_big or at_clause or len(cur.strip()) >= target):
                    yield c_start, prev_end, wrap(cur.strip(), seg.speaker), seg.speaker
                    cur, c_start, cand = "", None, w.word
                c_start = w.start if c_start is None else c_start
                cur, prev_end = cand, w.end
            if cur.strip():
                yield c_start, prev_end, wrap(cur.strip(), seg.speaker), seg.speaker
            continue
        n_by_time = math.ceil((seg.end - seg.start) / max_dur)
        groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
        if len(groups) < n_by_time:  # long in time, short in text: one shorter line per cue
            width = min(max_chars, max(12, len(seg.text) // n_by_time))
            # Wrap the bare text, then label every cue: a speaker prefix must never become a
            # cue of its own, and continuation cues still need to say who is speaking.
            groups = [wrap(piece, seg.speaker) for piece in textwrap.wrap(seg.text, width)]
        total = sum(len(" ".join(g)) for g in groups) or 1
        t = seg.start
        for g in groups:
            d = (seg.end - seg.start) * len(" ".join(g)) / total
            yield t, min(t + d, t + max_dur), g, seg.speaker  # a lone long word can't stretch a cue
            t += d


def _valid_cues(t: Transcript, *, prefix_speaker: bool) -> Iterator[Cue]:
    """Caption players reject zero-length or backwards cues (WebVTT requires end > start and
    non-decreasing starts). Sort rather than clamp: in split-channel transcripts cues from two
    speakers legitimately overlap, and clamping would move a reply to the wrong time.
    Cue times include the audio stream's start offset, so captions line up with a video's
    timeline when the audio track doesn't start at 0."""
    off = t.audio.start_offset_s or 0.0
    for s, e, lines, spk in sorted(_cues(t.segments, prefix_speaker=prefix_speaker), key=lambda c: (c[0], c[1])):
        yield s + off, max(e, s + 0.02) + off, lines, spk


def _vtt_escape(text: str) -> str:
    # Cue payloads are parsed for tags/entities: escape them, and "-->" is forbidden outright.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_srt(t: Transcript) -> str:
    out = []
    for i, (s, e, lines, _) in enumerate(_valid_cues(t, prefix_speaker=True), 1):
        out.append(f"{i}\n{_ts(s, ',')} --> {_ts(e, ',')}\n" + "\n".join(lines).replace("-->", "->") + "\n")
    return "\n".join(out)


def to_vtt(t: Transcript) -> str:
    out = ["WEBVTT", ""]
    for s, e, lines, spk in _valid_cues(t, prefix_speaker=False):
        body = _vtt_escape("\n".join(lines))
        voice = f"<v {_vtt_escape(spk)}>" if spk else ""
        out.append(f"{_ts(s, '.')} --> {_ts(e, '.')}\n{voice}{body}\n")
    return "\n".join(out)


def to_txt(t: Transcript, paragraph_gap_s: float = 2.0) -> str:
    paras, cur, prev = [], [], None
    for s in t.segments:
        if prev and (s.speaker != prev.speaker or s.start - prev.end > paragraph_gap_s) and cur:
            paras.append(cur)
            cur = []
        cur.append(s)
        prev = s
    if cur:
        paras.append(cur)
    lines = []
    for p in paras:
        head = f"[{_ts(p[0].start, '.')[:8]}]" + (f" {p[0].speaker}:" if p[0].speaker else "")
        lines.append(f"{head} " + " ".join(s.text for s in p))
    return "\n\n".join(lines) + "\n"


def to_json(t: Transcript) -> str:
    """Full-fidelity canonical transcript: the system of record."""
    return t.model_dump_json(indent=1)


def to_jsonl(t: Transcript) -> str:
    return "".join(json.dumps(s.model_dump(exclude_none=True)) + "\n" for s in t.segments)


def to_chunks(t: Transcript, max_chars: int = 1500) -> str:
    """Group whole segments into passages for RAG/LLM use; never splits a segment."""

    def size(x: Segment) -> int:
        return len(x.text) + 1 + (len(x.speaker) + 2 if x.speaker else 0)

    chunks, cur = [], []
    for s in t.segments:
        if cur and sum(size(x) for x in cur) + size(s) > max_chars:
            chunks.append(cur)
            cur = []
        cur.append(s)
    if cur:
        chunks.append(cur)
    payload = [
        {
            "chunk": i,
            "start": min(s.start for s in c),
            "end": max(s.end for s in c),
            "speakers": sorted({s.speaker for s in c if s.speaker}),
            "segment_ids": [s.id for s in c],
            "text": " ".join((f"{s.speaker}: " if s.speaker else "") + s.text for s in c),
        }
        for i, c in enumerate(chunks)
    ]
    return json.dumps({"language": t.language, "duration_s": t.duration_s, "chunks": payload}, indent=1)


_RENDERERS: dict[str, Callable[[Transcript], str]] = {
    "json": to_json,
    "jsonl": to_jsonl,
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "chunks": to_chunks,
}
FORMATS = tuple(_RENDERERS)


def render(t: Transcript, fmt: str) -> str:
    return _RENDERERS[fmt](t)
