"""Run every valid `meeting_short` format variant through the pipeline and print a markdown table
of WER, clips continued after an early stop, and ranges flagged for re-decode.

    uv run python scripts/make_fixtures.py          # once, macOS (TTS)
    uv run python scripts/check_format_matrix.py --model tiny.en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
from stt_pipeline.pipeline import Pipeline
from stt_pipeline.schemas import TranscribeOptions

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny.en")
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = ap.parse_args()
    norm = EnglishTextNormalizer()
    ref = norm((args.fixtures / "meeting_short.reference.txt").read_text())
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    pipe = Pipeline(FasterWhisperEngine(args.model))
    print("| file | WER | clips continued | ranges flagged for re-decode |\n|---|---|---|---|")
    for m in manifest:
        if m["kind"] == "valid" and m["file"].startswith("meeting_short"):
            t = pipe.run(args.fixtures / m["file"], TranscribeOptions(language="en"))
            d = t.diagnostics
            print(f"| {m['file']} | {100 * jiwer.wer(ref, norm(t.text)):.1f}% | {d.get('continued_clips', 0):.0f} | "
                  f"{d.get('suspect_ranges', 0):.0f} |")


if __name__ == "__main__":
    main()
