import json

import pytest

from stt_pipeline.formats import render, to_srt, to_vtt
from stt_pipeline.postprocess import (
    QualityGate, normalize_text, redact_pii, resegment_sentences, run_chain, sanitize_timestamps, tighten_to_words,
)
from stt_pipeline.schemas import AudioMeta, ModelMeta, Segment, Transcript, Word


def T(*segments: Segment) -> Transcript:
    return Transcript(
        duration_s=100.0,
        segments=list(segments),
        audio=AudioMeta(container="wav", codec="pcm", sample_rate=16000, channels=1, decoded_duration_s=100.0),
        model=ModelMeta(engine="test", model="test"),
    )


def S(start, end, text, lp=-0.2, ns=0.05, cr=1.3, **kw) -> Segment:
    return Segment(start=start, end=end, text=text, avg_logprob=lp, no_speech_prob=ns, compression_ratio=cr, **kw)


def test_quality_gate_reapplies_whisper_thresholds():
    t = QualityGate()(T(
        S(0, 2, "Hello there."),
        S(3, 30, "Thank you for watching!", ns=0.7, lp=-1.2),        # classic silence hallucination
        S(31, 40, "the the the the the the the the the", cr=3.1),     # repetition loop
        S(41, 43, "Thank you so much.", ns=0.02, lp=-0.1),            # real 'thank you' must survive
        S(44, 46, "Thank you.", ns=0.5, lp=-0.9),                     # ...but not when the model doubts it
        S(47, 49, "Mumble mumble mumble.", lp=-1.3, ns=0.1),          # low confidence: keep + flag
        S(50, 52, "Mumble mumble mumble.", lp=-1.3, ns=0.1),          # one repeat: plausibly real
        S(53, 55, "Mumble mumble mumble.", lp=-1.3, ns=0.1),          # third copy: a loop
        S(56, 57, "Yes.", lp=-0.1), S(57.5, 58, "Yes."),               # short real repeats survive
    ))
    assert [s.text for s in t.segments] == ["Hello there.", "Thank you so much.", "Mumble mumble mumble.",
                                           "Mumble mumble mumble.", "Yes.", "Yes."]
    assert t.segments[2].flags == ["low_confidence"]
    assert [d.reason for d in t.dropped] == ["no_speech", "repetition_loop", "hallucination_phrase", "duplicate"]


def test_normalize_text():
    t = normalize_text(T(S(0, 1, "  Café   au\n lait  ")))
    assert t.segments[0].text == "Café au lait"


def test_tighten_to_words_fixes_snapped_starts():
    seg = S(9.9, 18.4, "Thanks.", words=[Word(start=10.7, end=11.0, word=" Thanks."), Word(start=11.1, end=18.3, word=" ok")])
    t = tighten_to_words(T(seg))
    assert (t.segments[0].start, t.segments[0].end) == (10.7, 18.3)


def test_redact_pii_text_and_words():
    words = [Word(start=i, end=i + 0.5, word=w) for i, w in enumerate([" Call", " 415-555-0132", " or", " jane@example.com", " please."])]
    t = redact_pii(T(S(0, 5, "Call 415-555-0132 or jane@example.com please.", words=words)))
    s = t.segments[0]
    assert s.text == "Call [PHONE] or [EMAIL] please."
    assert "redacted" in s.flags
    assert [w.word for w in s.words] == [" Call", " [PHONE]", " or", " [EMAIL]", " please."]
    assert s.words[1].start == 1 and s.words[1].end == 1.5  # timing kept -> can bleep the audio


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
    srt = to_srt(run_chain(T(S(0, 20, text)), []))
    cues = [c for c in srt.strip().split("\n\n")]
    assert len(cues) >= 3
    for c in cues:
        lines = c.split("\n")[2:]
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines)
        start, end = c.split("\n")[1].split(" --> ")
        to_s = lambda ts: int(ts[:2]) * 3600 + int(ts[3:5]) * 60 + float(ts[6:].replace(",", "."))
        assert to_s(end) - to_s(start) <= 7.0 + 1e-6


def test_chunks_format_keeps_timestamps_for_citation():
    segs = [S(i * 10, i * 10 + 9, "x" * 400) for i in range(10)]
    payload = json.loads(render(run_chain(T(*segs), []), "chunks"))
    assert len(payload["chunks"]) > 1
    c0 = payload["chunks"][0]
    assert c0["start"] == 0 and c0["segment_ids"][0] == 0 and len(c0["text"]) <= 1500


def test_sanitize_timestamps_enforces_invariants():
    words = [Word(start=5.0, end=5.4, word=" a"), Word(start=4.9, end=4.8, word=" b"), Word(start=6.5, end=9.0, word=" c")]
    t = sanitize_timestamps(T(
        S(-0.5, 2.0, "starts before zero"),
        S(98.0, 104.0, "runs past the end"),
        S(5.0, 6.0, "abc", words=words),
        S(10.0, 12.0, "x" * 90),  # 45 chars/s
    ))
    a, b, c, d = t.segments
    assert (a.start, b.end) == (0.0, 100.0)
    assert [w.start for w in c.words] == sorted(w.start for w in c.words)
    assert all(c.start - 0.05 <= w.start <= w.end <= c.end + 0.05 for w in c.words)
    assert "timestamp_anomaly" in d.flags and "timestamp_anomaly" not in a.flags


def test_vtt_is_spec_valid_for_hostile_text_and_zero_length_segments():
    t = run_chain(T(S(1.0, 1.0, "x < y && a --> b"), S(0.5, 2.0, "earlier start")), [])
    vtt = to_vtt(t)
    assert "x &lt; y &amp;&amp; a --&gt; b" in vtt  # no raw '<', '&' or '-->' in payloads
    times = [line.split(" --> ") for line in vtt.splitlines() if " --> " in line]
    for start, end in times:
        assert end > start  # strictly positive duration
    assert [s for s, _ in times] == sorted(s for s, _ in times)  # non-decreasing starts


def test_resegment_sentences_fixes_mid_sentence_boundaries():
    def ws(start, text):
        return [Word(start=start + i * 0.4, end=start + i * 0.4 + 0.3, word=" " + w) for i, w in enumerate(text.split())]
    a = S(0.0, 2.0, "Revenue grew. Thanks, we", words=ws(0.0, "Revenue grew. Thanks, we"))
    b = S(2.0, 4.0, "shipped it. Great.", words=ws(2.0, "shipped it. Great."))
    t = resegment_sentences(T(a, b))
    assert [s.text for s in t.segments] == ["Revenue grew.", "Thanks, we shipped it.", "Great."]
    assert (t.segments[1].start, t.segments[1].end) == pytest.approx((0.8, 2.7))


def test_long_sentences_split_into_balanced_cues_not_orphans():
    text = "Revenue grew 12% compared to last quarter, mostly driven by the new subscription tier."
    words = [Word(start=4.3 + i * 0.3, end=4.3 + i * 0.3 + 0.25, word=" " + w) for i, w in enumerate(text.split())]
    srt = to_srt(run_chain(T(S(4.3, 4.3 + 0.3 * len(words), text, words=words)), []))
    cues = [" ".join(c.split("\n")[2:]) for c in srt.strip().split("\n\n")]
    assert len(cues) == 2 and min(len(c) for c in cues) > 25  # no "tier." orphan
