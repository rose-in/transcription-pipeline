"""Caption, text and chunk renderers in formats.py."""

import json

from stt_pipeline.formats import FORMATS, MEDIA_TYPES, render, to_srt, to_txt, to_vtt
from stt_pipeline.postprocess import run_chain
from stt_pipeline.schemas import Word

from .conftest import S, T


def srt_cues(srt: str) -> list[tuple[float, float, list[str]]]:
    """Parse SRT text into (start_s, end_s, text_lines) per cue."""

    def seconds(ts: str) -> float:
        h, m, s = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    cues = []
    for block in srt.strip().split("\n\n"):
        _index, timing, *lines = block.split("\n")
        start, end = timing.split(" --> ")
        cues.append((seconds(start), seconds(end), lines))
    return cues


def test_every_format_renders_and_has_a_media_type():
    # The API accepts exactly FORMATS and serves each with MEDIA_TYPES[fmt].
    assert set(MEDIA_TYPES) == set(FORMATS)
    t = run_chain(T(S(0, 1, "Hello.", speaker="channel_0")), [])
    for fmt in FORMATS:
        assert "Hello." in render(t, fmt), fmt


# ------------------------------------------------------------------ captions


def test_srt_and_vtt_formatting():
    t = run_chain(T(S(0.0, 2.5, "Hello world."), S(3661.5, 3663.004, "One hour in.", speaker="channel_1")), [])
    srt = to_srt(t)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nHello world.\n")
    assert "01:01:01,500 --> 01:01:03,004\nchannel_1: One hour in." in srt
    vtt = to_vtt(t)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world.")
    assert "<v channel_1>One hour in." in vtt


def test_long_segments_become_readable_cues():
    text = " ".join(["word"] * 60)  # ~300 chars over 20 s
    cues = srt_cues(to_srt(run_chain(T(S(0, 20, text)), [])))
    assert len(cues) >= 3
    for start, end, lines in cues:
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines)
        assert end - start <= 7.0 + 1e-6


def test_long_sentences_split_into_balanced_cues_not_orphans():
    text = "Revenue grew 12% compared to last quarter, mostly driven by the new subscription tier."
    words = [Word(start=4.3 + i * 0.3, end=4.3 + i * 0.3 + 0.25, word=" " + w) for i, w in enumerate(text.split())]
    srt = to_srt(run_chain(T(S(4.3, 4.3 + 0.3 * len(words), text, words=words)), []))
    cues = [" ".join(lines) for _, _, lines in srt_cues(srt)]
    assert len(cues) == 2 and min(len(c) for c in cues) > 25  # no "tier." orphan


def test_srt_fallback_cues_all_carry_the_speaker():
    t = run_chain(T(S(0, 20, "Okay. Sounds good to me, thanks.", speaker="channel_1")), [])
    cues = [lines for _, _, lines in srt_cues(to_srt(t))]
    assert len(cues) >= 2 and all(c[0].startswith("channel_1: ") for c in cues)
    assert all(len(line) <= 42 for c in cues for line in c)


def test_vtt_is_spec_valid_for_hostile_text_and_zero_length_segments():
    t = run_chain(T(S(1.0, 1.0, "x < y && a --> b"), S(0.5, 2.0, "earlier start")), [])
    vtt = to_vtt(t)
    assert "x &lt; y &amp;&amp; a --&gt; b" in vtt  # no raw '<', '&' or '-->' in payloads
    times = [line.split(" --> ") for line in vtt.splitlines() if " --> " in line]
    for start, end in times:
        assert end > start  # strictly positive duration
    assert [s for s, _ in times] == sorted(s for s, _ in times)  # non-decreasing starts


def test_overlapping_split_channel_cues_keep_their_own_time():
    long = " ".join(["word"] * 40)
    t = run_chain(T(S(10, 25, long, speaker="channel_0"), S(12, 13, "Mm-hmm, right.", speaker="channel_1")), [])
    vtt = to_vtt(t)
    assert "00:00:12.000 --> 00:00:13.000\n<v channel_1>Mm-hmm, right." in vtt


def test_captions_apply_the_audio_start_offset():
    t = run_chain(T(S(1.0, 2.0, "Hello.")), [])
    t.audio.start_offset_s = 1.5
    assert "00:00:02,500 --> 00:00:03,500" in render(t, "srt")


# ------------------------------------------------------------------ text and chunks


def test_txt_starts_a_paragraph_on_speaker_change_or_long_pause():
    t = T(
        S(0, 1, "Hi.", speaker="channel_0"),
        S(1.5, 2.5, "How are you?", speaker="channel_0"),
        S(3, 4, "Fine.", speaker="channel_1"),
        S(6.5, 7, "Anyway.", speaker="channel_1"),
    )  # 2.5 s pause > 2 s
    assert to_txt(t) == (
        "[00:00:00] channel_0: Hi. How are you?\n\n[00:00:03] channel_1: Fine.\n\n[00:00:06] channel_1: Anyway.\n"
    )


def test_chunks_format_keeps_timestamps_for_citation():
    segs = [S(i * 10, i * 10 + 9, "x" * 400) for i in range(10)]
    payload = json.loads(render(run_chain(T(*segs), []), "chunks"))
    assert len(payload["chunks"]) > 1
    c0 = payload["chunks"][0]
    assert c0["start"] == 0 and c0["segment_ids"][0] == 0 and len(c0["text"]) <= 1500


def test_chunk_citation_range_covers_every_segment():
    t = run_chain(T(S(10, 40, "long turn " * 10, speaker="channel_0"), S(12, 13, "ok", speaker="channel_1")), [])
    c = json.loads(render(t, "chunks"))["chunks"][0]
    assert (c["start"], c["end"]) == (10, 40)
