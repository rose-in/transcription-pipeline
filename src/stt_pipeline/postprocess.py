"""Post-processing chain: raw model output -> transcript that is safe to hand downstream.

Each processor is a callable Transcript -> Transcript (it may modify the transcript in
place and returns it), so the chain is configurable per request and each step is
unit-testable in isolation.
Removed content is never silently discarded: it goes to `transcript.dropped`
with a reason, so a reviewer can audit what the filters did. (PII redaction
covers the audit trail too: a dropped segment is still a place PII can leak.)
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections.abc import Callable, Sequence

from .schemas import DroppedSegment, Segment, Transcript, Word
from .whisper_thresholds import COMPRESSION_RATIO_THRESHOLD, LOG_PROB_THRESHOLD, NO_SPEECH_THRESHOLD

Processor = Callable[[Transcript], Transcript]

# Phrases Whisper is known to emit on silence/music, learned from subtitle credits in its
# training data. Only dropped when the model's own scores also say "probably not speech".
HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you for watching and please subscribe",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amaraorg community",
    "subtitles by the amara org community",
    "transcription by castingwords",
    "thank you",
    "thank you very much",
    "you",
    "bye",
    "music",
    "applause",
}


def _norm(text: str) -> str:
    """Script-agnostic comparison key: NFKC + casefold, punctuation/symbols removed, letters and
    digits of every script kept (an ASCII-only key made different Japanese sentences 'equal')."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c for c in text if unicodedata.category(c)[0] not in "PS").split())


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


class QualityGate:
    """Re-applies the reference Whisper fallback thresholds that batched decoding skips, plus
    loop detection across segments (batched mode scores each short subsegment on its own, so a
    loop of "I'm sorry." x12 never trips a per-segment compression ratio).

    - no_speech_prob > 0.6 and avg_logprob < -1.0          -> drop  (model thinks it was silence)
    - compression_ratio > 2.4                              -> drop  (repetition within a segment)
    - 3rd+ consecutive copy of a sentence (4th+ of a <3-word phrase) from the same speaker,
      each within 2 s of the previous                      -> drop  (loop across segments; a
                                                                     single repeat is kept)
    - known hallucination phrase + weak evidence of speech -> drop
    - segment flagged `continued` and avg_logprob < -1.0   -> drop  (decoded from a short
      leftover with no preceding context; below Whisper's own failure threshold these are
      mostly garbled real phrases, so dropping trades a few real words for keeping
      confident-looking wrong text out of search and summaries; see docs/eval_real_data.md and README)
    - a window of 5 consecutive segments (within 30 s) compressing > 2.4 -> flag `possible_loop`
    - avg_logprob < -1.0                                   -> keep, flag `low_confidence`
    """

    REPEAT_MAX_GAP_S = 2.0  # a repeat continues a loop only if it starts within this of the previous copy's end
    PHRASE_NO_SPEECH_THRESHOLD = 0.3  # known hallucination phrase + (no_speech_prob above this
    PHRASE_LOG_PROB_THRESHOLD = -0.7  #   or avg_logprob below this) = weak evidence of speech -> drop
    LOOP_WINDOW = 5  # consecutive same-speaker segments checked together for a cross-segment loop
    LOOP_WINDOW_MAX_SPAN_S = 30.0  # windows with a longer first-end-to-last-start gap are skipped

    def __init__(
        self,
        compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold: float = LOG_PROB_THRESHOLD,
        no_speech_threshold: float = NO_SPEECH_THRESHOLD,
    ) -> None:
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold

    def __call__(self, t: Transcript) -> Transcript:
        kept: list[Segment] = []
        last: dict[str | None, tuple[str, float, int]] = {}  # speaker -> (norm text, end, run length)
        for s in t.segments:
            lp = s.avg_logprob if s.avg_logprob is not None else 0.0
            ns = s.no_speech_prob or 0.0
            cr = s.compression_ratio or 0.0
            key = _norm(s.text)
            prev_key, prev_end, run = last.get(s.speaker, ("", -1e9, 0))
            run = run + 1 if (key and key == prev_key and s.start - prev_end < self.REPEAT_MAX_GAP_S) else 1
            last[s.speaker] = (key, s.end, run)  # updated for kept AND dropped segments
            reason = None
            if ns > self.no_speech_threshold and lp < self.log_prob_threshold:
                reason = "no_speech"
            elif cr > self.compression_ratio_threshold:
                reason = "repetition_loop"
            elif run >= 3 and (len(key.split()) >= 3 or run >= 4):
                # Two identical sentences in a row are plausibly real (agreement, an echo);
                # decoding loops repeat many times, so only later copies are dropped.
                reason = "duplicate"
            elif key in HALLUCINATION_PHRASES and (
                ns > self.PHRASE_NO_SPEECH_THRESHOLD or lp < self.PHRASE_LOG_PROB_THRESHOLD
            ):
                reason = "hallucination_phrase"
            elif "continued" in s.flags and lp < self.log_prob_threshold:
                reason = "weak_continuation"
            if reason:
                t.dropped.append(DroppedSegment(segment=s, reason=reason))
                continue
            if lp < self.log_prob_threshold and "low_confidence" not in s.flags:
                s.flags.append("low_confidence")
            kept.append(s)
        self._flag_windows(kept)
        t.segments = kept
        return t

    def _flag_windows(self, segs: list[Segment]) -> None:
        size = self.LOOP_WINDOW
        by_speaker: dict[str | None, list[Segment]] = {}
        for s in segs:
            by_speaker.setdefault(s.speaker, []).append(s)
        for run in by_speaker.values():
            for i in range(len(run) - size + 1):
                window = run[i : i + size]
                if window[-1].start - window[0].end > self.LOOP_WINDOW_MAX_SPAN_S:
                    continue
                if _compression_ratio(" ".join(x.text for x in window)) > self.compression_ratio_threshold:
                    for x in window:
                        if "possible_loop" not in x.flags:
                            x.flags.append("possible_loop")


def normalize_text(t: Transcript) -> Transcript:
    """Unicode NFC (so 'é' has one byte representation for search/diff) and whitespace cleanup."""
    for s in t.segments:
        s.text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", s.text)).strip()
        if s.words:
            for w in s.words:
                w.word = unicodedata.normalize("NFC", w.word)
    return t


def sanitize_timestamps(t: Transcript) -> Transcript:
    """Enforce timeline invariants every consumer assumes, whatever the engine did:
    0 <= start <= end <= duration, words inside their segment (+-50 ms) and non-decreasing.
    Implausible speaking rates are flagged (DTW/VAD remapping glitches do happen)."""
    dur = t.duration_s
    for s in t.segments:
        s.start = round(min(max(0.0, s.start), dur), 3)
        s.end = round(min(max(s.start, s.end), dur), 3)
        if s.words:
            prev = s.start
            hi = min(s.end + 0.05, dur)
            for w in s.words:
                w.start = round(min(max(w.start, prev, s.start - 0.05), s.end), 3)
                w.end = round(min(max(w.end, w.start), hi), 3)
                prev = w.start
        span = s.end - s.start
        if span > 1.0 and len(s.text) / span > 30 and "timestamp_anomaly" not in s.flags:
            s.flags.append("timestamp_anomaly")  # > 30 chars/s is faster than humans speak
    return t


def tighten_to_words(t: Transcript) -> Transcript:
    """Whisper's segment start is often the previous segment's end (it snaps across pauses).
    When word timings exist, clamp the segment to its first/last word instead."""
    for s in t.segments:
        if s.words:
            s.start = round(max(s.start, s.words[0].start), 3)
            s.end = round(min(s.end, s.words[-1].end), 3) if s.words[-1].end > s.start else s.end
    return t


_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")


def resegment_sentences(t: Transcript, *, max_gap_s: float = 1.0, max_len_s: float = 15.0) -> Transcript:
    """Rebuild segments as sentences from word timings. Some models (large-v3-turbo especially)
    place segment boundaries mid-sentence, which makes captions and RAG passages read badly and
    pushes segment starts away from where utterances begin. Runs only when every segment has
    words; scores are inherited conservatively (worst of the source segments)."""
    if not t.segments or any(not s.words for s in t.segments):
        return t
    out: list[Segment] = []
    cur: list[tuple[Word, Segment]] = []

    def flush() -> None:
        if not cur:
            return
        srcs = list({id(src): src for _, src in cur}.values())
        words = [w for w, _ in cur]
        lps = [x.avg_logprob for x in srcs if x.avg_logprob is not None]
        out.append(
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text="".join(w.word for w in words).strip(),
                speaker=srcs[0].speaker,
                words=words,
                avg_logprob=min(lps) if lps else None,
                no_speech_prob=max((x.no_speech_prob or 0.0) for x in srcs),
                compression_ratio=max((x.compression_ratio or 0.0) for x in srcs),
                flags=sorted({f for x in srcs for f in x.flags}),
            )
        )
        cur.clear()

    for seg in sorted(t.segments, key=lambda s: (s.speaker or "", s.start)):
        for w in seg.words:
            if cur and (
                seg.speaker != cur[-1][1].speaker
                or w.start - cur[-1][0].end > max_gap_s
                or w.end - cur[0][0].start > max_len_s
            ):
                flush()
            cur.append((w, seg))
            if w.word.strip().endswith(_SENTENCE_END):
                flush()
    flush()
    t.segments = sorted(out, key=lambda s: (s.start, s.speaker or ""))
    return t


# --------------------------------------------------------------------------------------------- PII


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


_GROUP_SEP = r"(?:[ .-]|, ?)?"  # card digit groups: permissive, because Luhn validates the result
_TLD = r"(?:com|org|net|edu|gov|io|co|uk|de|fr|info|biz|us|ca|au|in)"
# ASR renders phone numbers as "415, 555, 0132" or "4155550132" - but so are prices and order
# numbers ("250, 500, 1000 dollars"). Those loose forms count only near a phone-ish word.
_PHONE_CONTEXT = re.compile(
    r"\b(?:phone|number|call|calling|reach|dial|text|cell|mobile|contact|ext|extension)\b", re.I
)
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    # ASR writes what was *said*: "john.smith at example dot com", "John.Smith.at.example.com".
    ("EMAIL", re.compile(rf"\b[\w+-]+(?:(?:\.| dot )[\w+-]+)*(?: at |\.at\.)[\w-]+(?:(?:\.| dot ){_TLD})+\b", re.I)),
    ("CARD", re.compile(rf"\b\d(?:{_GROUP_SEP}\d){{12,30}}\b")),  # narrowed to a Luhn-valid group window below
    ("SSN", re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b")),
    ("PHONE", re.compile(r"\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}\b")),  # international
    ("PHONE", re.compile(r"(?<![\w+])(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}\b")),  # separated 3-3-4
    ("PHONE?", re.compile(r"(?<![\w+])\d{3}(?:, ?)?\d{3}(?:, ?)?\d{4}\b")),  # loose 3-3-4: needs context
    ("PHONE", re.compile(r"(?<![\w-])\d{3}-\d{4}\b")),  # local 555-0132
]
# Recorded in transcript.options["pii_redaction"]; bump when PII_PATTERNS or the span logic changes.
PII_REDACTOR_VERSION = "regex-v3"


def _card_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Cards are read in digit groups, possibly next to other numbers ("option 2, 4111 ...") or a
    trailing CVV. Accept a Luhn-valid window of 13-19 digits that starts AND ends on group
    boundaries. Arbitrary sub-windows are not tried: ~1 in 10 random digit strings passes Luhn."""
    pos = [i for i in range(start, end) if text[i].isdigit()]
    breaks = [k + 1 for k in range(len(pos) - 1) if pos[k + 1] - pos[k] > 1]
    starts, ends = [0] + breaks, breaks + [len(pos)]
    for i in starts:
        for j in sorted(ends, reverse=True):
            digits = "".join(text[k] for k in pos[i:j])
            # Issuer digit 2-6 (Mastercard/Amex/Visa/Discover): zero-padded IDs pass Luhn trivially.
            if 13 <= j - i <= 19 and digits[0] in "23456" and _luhn_ok(digits):
                return pos[i], pos[j - 1] + 1
    return None


def _pii_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for label, rx in PII_PATTERNS:
        for m in rx.finditer(text):
            a, b = m.start(), m.end()
            if label == "CARD":
                hit = _card_span(text, a, b)
                if hit is None:
                    continue
                a, b = hit
            elif label == "PHONE?":
                if not _PHONE_CONTEXT.search(text[max(0, a - 40) : a]):
                    continue
                label = "PHONE"
            if any(a < e and s < b for s, e, _ in spans):
                continue  # earlier (more specific) pattern already covers it
            spans.append((a, b, label))
    return sorted(spans)


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    out: list[list] = []
    for a, b, label in sorted(spans):
        if out and a < out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b, label])
    return [tuple(x) for x in out]


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    for s, e, label in reversed(spans):
        text = text[:s] + f"[{label}]" + text[e:]
    return text


def redact_pii(t: Transcript) -> Transcript:
    """Regex + checksum redaction for structured PII (email, phone, card, SSN).

    Deliberately simple for v1: deterministic, explainable, zero model cost. Names and
    addresses need NER (e.g. Microsoft Presidio) - noted as the v2 upgrade. Covers:
      * kept AND dropped segments (the audit trail ships in the JSON too);
      * PII read across pauses - split over 2-3 consecutive segments of the same speaker, even
        with the other party's backchannel in between (split-channel calls);
      * word timings, mapped from the *same* character spans as the text, so the words can
        never keep what the text removed (and the spans could later be bleeped in the audio).
    """
    for segs in (t.segments, [d.segment for d in t.dropped]):
        spans: dict[int, list] = {id(s): _pii_spans(s.text) for s in segs}
        _cross_segment_spans(segs, spans)
        for s in segs:
            sp = _merge_spans(spans[id(s)])
            original = s.text
            if sp:
                s.text = _apply_spans(original, sp)
                if "redacted" not in s.flags:
                    s.flags.append("redacted")
            if s.words and (sp or _align(s.words, original) is None):
                s.words = _redact_words(s.words, original, sp)
    t.options["pii_redaction"] = PII_REDACTOR_VERSION
    return t


def _cross_segment_spans(segs: list[Segment], spans: dict[int, list], max_gap_s: float = 5.0) -> None:
    """'4111 1111' <pause> '1111' <pause> '1111 thanks': no part matches alone. Detect on the
    joined text of windows of 2-3 consecutive segments *of the same speaker* and map any span
    that crosses a join back onto each segment's own characters."""
    by_speaker: dict[str | None, list[Segment]] = {}
    for s in sorted(segs, key=lambda x: x.start):
        by_speaker.setdefault(s.speaker, []).append(s)
    for run in by_speaker.values():
        for size in (2, 3):
            for i in range(len(run) - size + 1):
                win = run[i : i + size]
                if any(b.start - a.end > max_gap_s for a, b in zip(win, win[1:])):
                    continue
                offsets, pos = [], 0
                for x in win:
                    offsets.append(pos)
                    pos += len(x.text) + 1
                joined = " ".join(x.text for x in win)
                for a, b, label in _pii_spans(joined):
                    touched = [(x, off) for x, off in zip(win, offsets) if a < off + len(x.text) and off < b]
                    if len(touched) < 2:
                        continue  # single-segment spans are found per segment already
                    for x, off in touched:
                        spans[id(x)].append((max(a - off, 0), min(b - off, len(x.text)), label))


def _align(words: list[Word], text: str) -> list[tuple[int, int]] | None:
    """Character range of each word inside the segment text, or None if they don't line up."""
    out, pos = [], 0
    for w in words:
        token = w.word.strip()
        if not token:
            out.append((pos, pos))
            continue
        i = text.find(token, pos)
        if i < 0:
            return None
        out.append((i, i + len(token)))
        pos = i + len(token)
    return out


def _redact_words(words: list[Word], text: str, spans: list[tuple[int, int, str]]) -> list[Word]:
    bounds = _align(words, text)
    if bounds is None:
        # Words don't line up with the text (e.g. a hosted engine assigned a word to the
        # neighbouring segment): detect on the words themselves and, as a backstop, redact any
        # word carrying digits or '@'. Over-redaction is the safe failure here.
        joined = ""
        bounds = []
        for w in words:
            bounds.append((len(joined), len(joined) + len(w.word)))
            joined += w.word
        spans = _pii_spans(joined)
        backstop = True
    else:
        backstop = False
    out: list[Word] = []
    prev_hit = None
    for (ws, we), w in zip(bounds, words):
        hit = next((sp for sp in spans if ws < sp[1] and sp[0] < we), None)
        if hit is None and backstop and (any(c.isdigit() for c in w.word) or "@" in w.word):
            hit = (ws, we, "REDACTED")
        if hit is None:
            out.append(w)
            prev_hit = None
        elif prev_hit is not None and prev_hit == hit:
            out[-1].end = w.end  # extend the placeholder over the rest of the same span
        else:
            out.append(Word(start=w.start, end=w.end, word=f" [{hit[2]}]", probability=None))
            prev_hit = hit
    return out


def build_chain(*, redact: bool, words: bool) -> Sequence[Processor]:
    chain: list[Processor] = [sanitize_timestamps, normalize_text]
    if words:
        chain.append(tighten_to_words)  # before the gate, so duration-based rules see true spans
    chain.append(QualityGate())
    if words:
        chain.append(resegment_sentences)  # after the gate: junk is judged on the model's own segments
    if redact:
        chain.append(redact_pii)
    return chain


def run_chain(t: Transcript, chain: Sequence[Processor]) -> Transcript:
    for p in chain:
        t = p(t)
    t.segments.sort(key=lambda s: (s.start, s.speaker or ""))  # processors may move starts
    for i, s in enumerate(t.segments):
        s.id = i
    t.text = " ".join(s.text for s in t.segments)
    return t
