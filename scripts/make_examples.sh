#!/usr/bin/env bash
# Regenerate examples/ from the fixtures with the default CPU model, so readers can see real
# outputs without running anything. Requires fixtures (scripts/make_fixtures.py).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p examples
STT="uv run stt transcribe"

# Two-party call: per-channel transcription (speaker labels for free) + PII redaction
$STT fixtures/support_call_dual_channel.wav --split-channels --redact-pii --format txt -o examples/support_call.txt
$STT fixtures/support_call_dual_channel.wav --split-channels --redact-pii --format vtt -o examples/support_call.vtt

# Short meeting with word timings -> sentence-level segments, captions, canonical JSON
$STT fixtures/meeting_short.mp3 --words --format srt  -o examples/meeting_short.srt
$STT fixtures/meeting_short.mp3 --words --format json -o examples/meeting_short.json

# Long meeting (14 min, silence + hold music mid-file): readable text + RAG chunks
$STT fixtures/long_meeting.mp3 --format txt    -o examples/long_meeting.txt
$STT fixtures/long_meeting.mp3 --format chunks -o examples/long_meeting.chunks.json

# Hostile input: precise error, no crash
$STT fixtures/playlist_ssrf.mp3 --format json 2> examples/playlist_ssrf.stderr.txt || true
