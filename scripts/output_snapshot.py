"""Snapshot transcription output on a fixed set of real and synthetic inputs, and diff two snapshots.

Used to check that a code change is behaviour-preserving: capture before, change, capture after,
diff. Compares segments (text, timestamps, scores, flags, word timings), dropped segments, language
and diagnostic counts; wall-clock timings are excluded. Decoding is deterministic on CPU, so two
captures of the same code are identical.

    uv run python scripts/eval_real_data.py download        # datasets/ (LibriSpeech + Earnings-22)
    uv run python scripts/make_fixtures.py                  # fixtures/ (macOS)
    uv run python scripts/output_snapshot.py capture before.json
    uv run python scripts/output_snapshot.py capture after.json
    uv run python scripts/output_snapshot.py diff before.json after.json

Inputs (34): one Earnings-22 call and 20 LibriSpeech clips with `small`; a split-channel call with
PII redaction and word timings, and a meeting with word timings, with `small`; the 11 format
variants of the same meeting with `tiny.en`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_common import ROOT

TIMING_KEYS = {"batched_s", "fallback_s"}  # wall-clock: differs run to run


def _dump(t) -> dict:
    return {
        "language": t.language,
        "segments": [s.model_dump(exclude={"id"}) for s in t.segments],
        "dropped": [{"reason": d.reason, **d.segment.model_dump(exclude={"id"})} for d in t.dropped],
        "diagnostics": {k: v for k, v in t.diagnostics.items() if k not in TIMING_KEYS},
    }


def capture(out: Path) -> None:
    from stt_pipeline.engines.devices import default_cpu_threads
    from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
    from stt_pipeline.pipeline import Pipeline
    from stt_pipeline.schemas import TranscribeOptions

    fx, data, threads = ROOT / "fixtures", ROOT / "datasets", default_cpu_threads()
    english = TranscribeOptions(language="en")
    res: dict[str, dict] = {}

    small = Pipeline(FasterWhisperEngine("small", cpu_threads=threads))
    res["e22_china"] = _dump(small.run(data / "earnings22/4472895.mp3", english))
    clips = [json.loads(line) for line in (data / "librispeech/manifest.jsonl").read_text().splitlines()]
    for c in clips[::15][:20]:
        res[f"ls_{c['id']}"] = _dump(small.run(ROOT / c["file"], english))
    res["call_split_pii_words"] = _dump(
        small.run(
            fx / "support_call_dual_channel.wav",
            TranscribeOptions(channel_mode="split", redact_pii=True, word_timestamps=True),
        )
    )
    res["meeting_words"] = _dump(small.run(fx / "meeting_short.wav", TranscribeOptions(word_timestamps=True)))

    tiny = Pipeline(FasterWhisperEngine("tiny.en", cpu_threads=threads))
    for m in json.loads((fx / "manifest.json").read_text()):
        if m["kind"] == "valid" and m["file"].startswith("meeting_short"):
            res[f"fmt_{m['file']}"] = _dump(tiny.run(fx / m["file"], english))

    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"captured {len(res)} inputs -> {out}")


def diff(a: Path, b: Path) -> int:
    before, after = json.loads(a.read_text()), json.loads(b.read_text())
    differ = 0
    for key in sorted(set(before) | set(after)):
        x, y = before.get(key), after.get(key)
        if x != y:
            differ += 1
            fields = [
                f for f in ("language", "segments", "dropped", "diagnostics") if (x or {}).get(f) != (y or {}).get(f)
            ]
            print(f"DIFF {key}: {', '.join(fields) or 'missing'}")
    print(f"{len(before)} vs {len(after)} inputs; {differ} differ")
    return 1 if differ else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture").add_argument("out", type=Path)
    d = sub.add_parser("diff")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.out)
    else:
        sys.exit(diff(args.a, args.b))


if __name__ == "__main__":
    main()
