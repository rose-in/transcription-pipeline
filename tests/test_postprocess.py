import pytest

from stt_pipeline.postprocess import (
    QualityGate,
    normalize_text,
    redact_pii,
    resegment_sentences,
    sanitize_timestamps,
    tighten_to_words,
)
from stt_pipeline.schemas import Word

from .conftest import S, T


def words_at(start: float, text: str) -> list[Word]:
    """One Word per whitespace token of `text`, 0.4 s apart from `start`."""
    return [Word(start=start + i * 0.4, end=start + i * 0.4 + 0.3, word=" " + w) for i, w in enumerate(text.split())]


# ------------------------------------------------------------------ QualityGate


def test_quality_gate_reapplies_whisper_thresholds():
    t = QualityGate()(
        T(
            S(0, 2, "Hello there."),
            S(3, 30, "Thank you for watching!", ns=0.7, lp=-1.2),  # classic silence hallucination
            S(31, 40, "the the the the the the the the the", cr=3.1),  # repetition loop
            S(41, 43, "Thank you so much.", ns=0.02, lp=-0.1),  # real 'thank you' must survive
            S(44, 46, "Thank you.", ns=0.5, lp=-0.9),  # ...but not when the model doubts it
            S(47, 49, "Mumble mumble mumble.", lp=-1.3, ns=0.1),  # low confidence: keep + flag
            S(50, 52, "Mumble mumble mumble.", lp=-1.3, ns=0.1),  # one repeat: plausibly real
            S(53, 55, "Mumble mumble mumble.", lp=-1.3, ns=0.1),  # third copy: a loop
            S(56, 57, "Yes.", lp=-0.1),
            S(57.5, 58, "Yes."),  # short real repeats survive
        )
    )
    assert [s.text for s in t.segments] == [
        "Hello there.",
        "Thank you so much.",
        "Mumble mumble mumble.",
        "Mumble mumble mumble.",
        "Yes.",
        "Yes.",
    ]
    assert t.segments[2].flags == ["low_confidence"]
    assert [d.reason for d in t.dropped] == ["no_speech", "repetition_loop", "hallucination_phrase", "duplicate"]


def test_cross_segment_loops_are_removed():
    t = QualityGate()(T(*[S(i * 1.5, i * 1.5 + 1, "I'm sorry.") for i in range(12)]))
    assert [s.text for s in t.segments] == ["I'm sorry."] * 3  # "No. No. No." is speech; x12 is a loop


def test_duplicate_rule_is_unicode_aware():
    t = QualityGate()(T(S(0, 1, "iPhone 15 Pro を買いました"), S(1.5, 2.5, "iPhone 15 Pro が欲しいです")))
    assert len(t.segments) == 2


def test_possible_loop_is_flagged_across_a_window_of_segments():
    a, b = "And then we went back to the office.", "And then we went back to the office again."
    t = QualityGate()(T(*[S(i * 2.0, i * 2.0 + 1.5, a if i % 2 == 0 else b) for i in range(5)]))
    assert len(t.segments) == 5 and not t.dropped  # each segment passes alone; alternating text never trips 'duplicate'
    assert all(s.flags == ["possible_loop"] for s in t.segments)


def test_weak_continuations_are_dropped_but_confident_ones_and_normal_segments_are_kept():
    # Real cases from Earnings-22: a 1.6 s leftover piece decoded with no context turned "and fully
    # utilize" into "I'm free to die." (avg log-prob -1.12); a 0.8 s one came back correctly as
    # "Volumes" (-0.94).
    t = QualityGate()(
        T(
            S(0, 4, "Our strategy is unchanged.", lp=-1.12),  # normal, weak: flagged only
            S(4, 5.6, "I'm free to die.", lp=-1.12, flags=["continued"]),  # dropped
            S(6, 6.8, "Volumes", lp=-0.94, flags=["continued"]),  # kept
        )
    )
    assert [s.text for s in t.segments] == ["Our strategy is unchanged.", "Volumes"]
    assert t.segments[0].flags == ["low_confidence"]
    assert [(d.segment.text, d.reason) for d in t.dropped] == [("I'm free to die.", "weak_continuation")]


# ------------------------------------------------------------------ text and timestamps


def test_normalize_text_composes_unicode_and_collapses_whitespace():
    t = normalize_text(T(S(0, 1, "  Café   au\n lait  ")))  # NFD: "e" + combining acute accent
    assert t.segments[0].text == "Café au lait"  # NFC: one precomposed code point


def test_tighten_to_words_fixes_snapped_starts():
    seg = S(
        9.9,
        18.4,
        "Thanks.",
        words=[Word(start=10.7, end=11.0, word=" Thanks."), Word(start=11.1, end=18.3, word=" ok")],
    )
    t = tighten_to_words(T(seg))
    assert (t.segments[0].start, t.segments[0].end) == (10.7, 18.3)


def test_sanitize_timestamps_enforces_invariants():
    words = [
        Word(start=5.0, end=5.4, word=" a"),
        Word(start=4.9, end=4.8, word=" b"),
        Word(start=6.5, end=9.0, word=" c"),
    ]
    t = sanitize_timestamps(
        T(
            S(-0.5, 2.0, "starts before zero"),
            S(98.0, 104.0, "runs past the end"),
            S(5.0, 6.0, "abc", words=words),
            S(10.0, 12.0, "x" * 90),  # 45 chars/s
        )
    )
    a, b, c, d = t.segments
    assert (a.start, b.end) == (0.0, 100.0)
    assert [w.start for w in c.words] == sorted(w.start for w in c.words)
    assert all(c.start - 0.05 <= w.start <= w.end <= c.end + 0.05 for w in c.words)
    assert "timestamp_anomaly" in d.flags and "timestamp_anomaly" not in a.flags


def test_resegment_sentences_fixes_mid_sentence_boundaries():
    a = S(0.0, 2.0, "Revenue grew. Thanks, we", words=words_at(0.0, "Revenue grew. Thanks, we"))
    b = S(2.0, 4.0, "shipped it. Great.", words=words_at(2.0, "shipped it. Great."))
    t = resegment_sentences(T(a, b))
    assert [s.text for s in t.segments] == ["Revenue grew.", "Thanks, we shipped it.", "Great."]
    assert (t.segments[1].start, t.segments[1].end) == pytest.approx((0.8, 2.7))


# ------------------------------------------------------------------ PII redaction


def test_redact_pii_text_and_words():
    words = [
        Word(start=i, end=i + 0.5, word=w)
        for i, w in enumerate([" Call", " 415-555-0132", " or", " jane@example.com", " please."])
    ]
    t = redact_pii(T(S(0, 5, "Call 415-555-0132 or jane@example.com please.", words=words)))
    s = t.segments[0]
    assert s.text == "Call [PHONE] or [EMAIL] please."
    assert "redacted" in s.flags
    assert [w.word for w in s.words] == [" Call", " [PHONE]", " or", " [EMAIL]", " please."]
    assert s.words[1].start == 1 and s.words[1].end == 1.5  # timing kept -> can bleep the audio


def test_words_that_do_not_align_with_text_redact_any_digit_or_at_sign():
    # ' later' is not in the text, so word spans can't be mapped; nothing matches a PII pattern
    words = [
        Word(start=i * 0.5, end=i * 0.5 + 0.4, word=w)
        for i, w in enumerate([" Option", " 2,", " write", " me@x", " later"])
    ]
    s = redact_pii(T(S(0, 2.5, "Option 2, write me@x", words=words))).segments[0]
    assert s.text == "Option 2, write me@x"
    assert [w.word for w in s.words] == [" Option", " [REDACTED]", " write", " [REDACTED]", " later"]


def test_pii_is_redacted_in_dropped_segments_too():
    t = T(*[S(i * 2.5, i * 2.5 + 2, "My number is 415-555-0132 ok.") for i in range(3)])  # 3rd copy is dropped
    t = redact_pii(QualityGate()(t))
    assert "415-555-0132" not in t.model_dump_json()


def test_phone_context_rules():
    def redact(text):
        return redact_pii(T(S(0, 5, text))).segments[0].text

    assert redact("Call me on 415, 555, 0132.") == "Call me on [PHONE]."
    assert redact("tiers are 250, 500, 1000 dollars") == "tiers are 250, 500, 1000 dollars"
    assert redact("order 1234567890") == "order 1234567890"
    assert redact("option 2, 4111 1111 1111 1111") == "option 2, [CARD]"
    assert redact("tracking 9400 1000 0000 0000 0000 00") == "tracking 9400 1000 0000 0000 0000 00"


def test_card_split_across_a_pause_is_redacted():
    t = redact_pii(T(S(0, 2, "It is 4111 1111"), S(3, 5, "1111 1111 thanks")))
    assert not any(ch.isdigit() for s in t.segments for ch in s.text)


def test_card_split_around_the_other_partys_backchannel():
    t = redact_pii(
        T(
            S(0, 2, "It is 4111 1111", speaker="channel_1"),
            S(2.2, 2.6, "Mm-hmm.", speaker="channel_0"),
            S(3, 4, "1111", speaker="channel_1"),
            S(4.5, 6, "1111 thanks", speaker="channel_1"),
        )
    )
    assert not any(c.isdigit() for s in t.segments for c in s.text)


def test_word_timings_never_keep_digits_the_text_redacted():
    words = [Word(start=i, end=i + 0.5, word=w) for i, w in enumerate([" My", " SSN", " is", " 123", " 45", " 6789"])]
    t = redact_pii(T(S(0, 6, "My SSN is 123-45-6789.", words=words)))
    assert t.segments[0].text == "My SSN is [SSN]."
    assert not any(c.isdigit() for w in t.segments[0].words for c in w.word)


def test_words_follow_text_redaction_for_spoken_email_across_a_pause():
    a = S(0, 3, "write to john dot smith", words=words_at(0, "write to john dot smith"))
    b = S(3.5, 6, "at example dot com", words=words_at(3.5, "at example dot com"))
    t = redact_pii(T(a, b))
    assert [w.word.strip() for w in t.segments[0].words] == ["write", "to", "[EMAIL]"]
    assert [w.word.strip() for w in t.segments[1].words] == ["[EMAIL]"]
