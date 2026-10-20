# Transcription pipeline

The pipeline takes any audio file (or the audio track of a video), transcribes it, and returns a
transcript with per-segment timestamps. The transcript is post-processed for downstream use: a
quality gate, PII redaction, captions, and RAG-ready chunks. It runs as a CLI and as an HTTP
service with a durable job queue for long files.

The brief asked for engineering decisions rather than a model. So this README is mostly the
*why*: what each choice solves that the alternatives don't, and what measurements back it.
Research notes with sources are in [docs/RESEARCH.md](docs/RESEARCH.md). Raw benchmark output
is in [docs/evaluation_raw.md](docs/evaluation_raw.md).

```bash
uv sync                                                   # Python 3.12, no system ffmpeg needed
uv run stt transcribe meeting.mp3 --format srt            # CLI: any format, any length
uv run stt serve --with-worker                            # API on :8000 + one worker process
curl -F file=@meeting.mp3 localhost:8000/v1/transcriptions
```

---

## 1. What it does

```
                ┌─────────────── API process (holds no model) ────────────────┐
  upload ─────▶ │ size cap before parsing ─▶ stream to disk + sha256          │──▶ 4xx now, never a failed job later
                │ ─▶ sniff + FFmpeg probe in a sandbox subprocess (allowlists)│    (empty / not audio / no audio stream / too long)
                │ ─▶ job row (idempotent id)                                   │
                └──────────────────────────────┬──────────────────────────────┘
                                               │  SQLite job table (Postgres in prod): lease, heartbeat, attempts
                ┌──────────────────────────────▼──── worker process(es), model loaded once ───────────────────┐
                │ stream-decode ─▶ 16 kHz mono float32 (or one stream per channel for call recordings)       │
                │   ─▶ silence-aware ~10 min chunks (cut mid-pause; bounded memory; checkpoint unit)          │
                │        per chunk: Silero VAD ─▶ clips ≤30 s, never spanning a long pause                    │
                │                   ─▶ batched Whisper with timestamp tokens (fast path)                      │
                │                   ─▶ coverage + loop check ─▶ sequential fallback only where needed         │
                │                   ─▶ (word timings on) sentence re-segmentation                             │
                │                   ─▶ checkpoint                                                             │
                │   ─▶ shift by chunk offsets ─▶ post-processing chain ─▶ result.json                         │
                └─────────────────────────────────────────────────────────────────────────────────────────────┘
  outputs: json (canonical) · jsonl · txt · srt · vtt · chunks (RAG)
```

| Requirement | Where |
|---|---|
| Accept an audio file (WAV, MP3, …) | `stt transcribe <file>` · `POST /v1/transcriptions` (sync) · `POST /v1/jobs` (async) |
| Transcribe speech to text | [engines/faster_whisper_engine.py](src/stt_pipeline/engines/faster_whisper_engine.py), behind a 1-method engine protocol |
| Timestamps per segment | every segment has `start`/`end` in seconds on the original file's timeline; optional per-word timings |
| Different audio formats | [audio.py](src/stt_pipeline/audio.py): §2 |
| Long audio files | [chunking.py](src/stt_pipeline/chunking.py), [pipeline.py](src/stt_pipeline/pipeline.py), [jobs.py](src/stt_pipeline/jobs.py): §3 |
| Processing for downstream use | [postprocess.py](src/stt_pipeline/postprocess.py), [formats.py](src/stt_pipeline/formats.py): §4 |

A segment from the canonical JSON looks like this:

```json
{"id": 3, "start": 15.43, "end": 17.21, "text": "Sure, it's [PHONE].", "speaker": "channel_1",
 "avg_logprob": -0.18, "no_speech_prob": 0.01, "compression_ratio": 0.9, "flags": ["redacted"]}
```

Each transcript also carries provenance: model, revision, compute type, decoding parameters and
library versions. It records audio facts (sha256, container, codec, rate, channels, and both the
container-reported and the actually-decoded duration). It includes a per-chunk breakdown,
timings and RTF, warnings, and a `dropped` list that keeps every segment the filters removed,
with the reason.

---

## 2. How different audio formats are handled

**Principle: one decode path.** Every input goes through the same path, and comes out as the
model's native input: **16 kHz mono float32 PCM**. Nothing downstream ever branches on format.
That path is FFmpeg's libraries via PyAV.

**Why PyAV:**
- Its wheels bundle FFmpeg 9, so there's no system ffmpeg to install, patch, or let drift
  between a Mac and a Linux GPU box.
- It decodes everything users actually upload: M4A/AAC from phones, WebM/Opus from browsers,
  AMR from old phones, the audio track of MP4/MOV, and 8 kHz µ-law telephony.
- It streams frame by frame, so we can count samples and enforce limits while decoding.

**Alternatives and why not:**
- soundfile can't open AAC/M4A, WebM or AMR.
- pydub is unmaintained and breaks on Python 3.13.
- torchaudio now needs system FFmpeg libraries via TorchCodec.
- The ffmpeg CLI is a fine equivalent (documented fallback), but it costs a system dependency
  and stderr parsing.

### Validation runs before anything is queued

| Check | Why | Result |
|---|---|---|
| Body-size cap *before* multipart parsing (ASGI middleware) | Starlette spools whole bodies to disk with no total limit | `413 file_too_large` |
| Magic-byte sniff (fast path) | Extension and MIME type are attacker-controlled and often wrong | known container → FFmpeg |
| FFmpeg `format_whitelist` + `protocol_whitelist=file`; >16 streams rejected | The real security boundary. A crafted HLS/concat playlist makes FFmpeg fetch URLs (SSRF) or read local files; only allowlisted demuxers can ever run. Files the sniff doesn't know get one try with these demuxers | `415` if nothing allowlisted opens it |
| Probe runs in a **disposable sandbox process per upload** (CPU/memory rlimits, timeout) | FFmpeg ships 23–25 CVEs a year; a crash or hang on one upload must not take the API down, or fail anyone else's upload (a shared process pool would) | crash or timeout → `422` |
| Choose a *decodable* audio stream (`streams.best`) | Video files: audio isn't stream 0; cover art is a "video" stream; unknown codecs have no decoder | `422 no_audio_stream` / `undecodable_audio` |
| Max duration enforced **while decoding**, by sample count | Container durations lie (VBR MP3, MediaRecorder WebM), and a few MB of FLAC silence can decode to hours. Up front, a bitrate-*estimated* duration only rejects a file that is >3× over the limit | `413 audio_too_long` |
| Decoded vs container duration; demuxer errors mid-file | A truncated file decodes partially. That isn't fatal, but the caller must know | warning in result |
| Clipping and very-low-level detection | Cheap quality signals | warning in result |

### Choices inside the decode step (each one fixed a measured problem)

- **Float32 end to end.** openai-whisper and faster-whisper go through int16, which re-quantizes
  24-bit audio and clips the >0 dBFS peaks that lossy decoders produce (measured: 1.23 peak on
  an MP3).
- **Stereo is averaged in numpy, not downmixed by FFmpeg.** FFmpeg's default stereo→mono uses a
  constant-power law, so identical L/R channels came out **+3 dB louder** than the mono encodes
  of the same audio (RMS 0.178 vs 0.126) and peaked over full scale. Now every format decodes to
  the same level (test-enforced, ±10%). For more than 2 channels FFmpeg's dialog-aware matrix is
  kept.
- **Split-channel mode** (`channel_mode=split`) for call recordings: each channel is transcribed
  separately and labelled. That's free, perfect speaker attribution for agent/customer audio.
  Downmixing it would throw that away.
- **Timestamps come from sample counts**, not container PTS. Broken or missing PTS can't skew
  them. The stream's start offset is recorded (`audio.start_offset_s`) so captions can be mapped
  back onto a video's timeline.
- **The resampler is rebuilt on any mid-stream format change.** PyAV's resampler goes into
  passthrough when the first frame already matches 16 kHz, and then never checks again. A stream
  that switched to 44.1 kHz was decoded as **11.5 s instead of 6 s** (time-stretched audio, wrong
  timestamps). Found in review; regression-tested.
- **No denoising or loudness normalization.** Recent studies found speech enhancement *raises*
  Whisper/Parakeet WER, and Whisper's log-mel is gain-invariant up to an offset
  (see [RESEARCH §3](docs/RESEARCH.md#3-audio-ingestion)).

### Tested format matrix

Every one of these transcribes to the same text (0.0% WER with `tiny.en`, see §6):

- WAV: s16 44.1 kHz stereo, s24 96 kHz, f32 48 kHz, 8 kHz µ-law
- MP3, M4A/AAC, Ogg/Opus, WebM/Opus, FLAC, AIFF
- MP4 video (audio on stream 1)

And each of these hostile or broken inputs gets its documented status, never a 500:

- `empty.wav`: 400
- `not_audio.mp3`: 415
- HLS playlist renamed `.mp3`: 415
- `video_only.mp4`: 422
- `corrupt.wav`: 422
- `truncated.mp3`: 200 with a warning

---

## 3. How long audio files are handled

Whisper sees 30 s at a time, so every long-form strategy is a chunking strategy. Here's how
the options compare (ablation numbers in §6):

| Strategy | Problem |
|---|---|
| Reference sequential decoding (openai-whisper style) | Slow, and can't batch. Errors propagate through the text prompt. **Hallucinated 64 words** in our 90 s silence + 20 s hold-music stretch. Loads the whole file. |
| faster-whisper `BatchedInferencePipeline` defaults | Fast, but returns **one "segment" per 30 s VAD chunk** (median start error 6.9 s). Batching also silently disables all of Whisper's anti-hallucination fallbacks, and it loads the whole file. |
| HF chunk-with-overlap + stitch | "Experimental" for seq2seq; worse WER; loads the whole file. |
| Single-pass long-context model (Parakeet, ≤3 h) | Can't checkpoint; needs a large GPU; English/EU only. Still better to shard. |

### What this pipeline does: two levels of chunking, each solving a different problem

1. **Macro-chunks (~10 min, cut mid-pause).**
   - Audio is stream-decoded and re-cut into ~10 min chunks. Each cut lands in the middle of
     the longest pause Silero VAD finds in the last 60 s before the target (energy fallback for
     wall-to-wall speech).
   - Because cuts fall in silence, no word straddles a boundary. So there's nothing to overlap,
     stitch or de-duplicate. Offsets are exact sample counts, and concatenating the chunks
     reproduces the input bit-for-bit (test-enforced).
   - **Memory is bounded** by the chunk, not the file: measured flat at ~124 MB of audio and
     feature buffers from 1 h to 3.2 h, versus ~715 MB per hour for whole-file processing (§6).
   - The chunk is also the **checkpoint unit**: a crash at 2h50m of a 3 h file resumes at 2h50m.
     It's the unit of progress reporting, and the natural unit to fan out across GPUs in v2.

2. **Clips inside each chunk (≤30 s).**
   - Silero VAD regions are merged into clips of at most 30 s, but **never across a pause
     longer than `max_gap_s`**. The audio between merged regions is kept.
   - These clips are passed to faster-whisper as `clip_timestamps`, decoded in batches **with
     timestamp tokens on**. That gives sentence-level segments on the real timeline.
   - It also fixes something the default path gets wrong: faster-whisper's own merge *removes*
     the silence between regions. In a split-channel call, that glued three separate customer
     turns into one segment and broke the conversation order.
   - No speech in a chunk means no Whisper call at all. That's the strongest hallucination
     defence.

3. **Narrow sequential fallback.** Batched decoding has no temperature fallback and no
   "seek and continue". After the batched pass, two checks run:
   - **coverage:** no VAD speech region may have ≥1.5 s outside every segment (measured on the
     union of segment intervals), or be under half covered;
   - **loops:** no segment may have a gzip compression ratio > 2.4. Loop segments don't count
     as coverage.

   Regions that fail either check are re-decoded with the *reference* algorithm's full safety
   net: temperature fallback, compression, log-prob and no-speech thresholds. That happens only
   there, so batched speed is kept everywhere else.

   Each redo range is grown to fully contain every segment it touches, and those segments are
   replaced ("whole segments in, whole segments out"). Keeping a partially overlapping segment
   duplicated words in practice.

   Both failure modes are real:
   - With `tiny.en`, the model stopped early inside a ~29 s clip on **5 of 11** format-matrix
     files, silently dropping a whole sentence (23% WER). The coverage check brought all 11
     back to 0.0%.
   - With `large-v3-turbo`, a batched segment ended mid-sentence inside a re-decoded region.
     Before the whole-segment rule this produced duplicated words (6.4% WER); after it, 0.0%.

4. **Language** is detected on the first chunk that actually contains speech, then pinned for
   the rest of the file. Per-chunk detection lets one noisy chunk flip the language mid-file.

### Service side

Long files go through `POST /v1/jobs`, which returns `202` plus a job URL. The client polls, or
the sync endpoint degrades to that. The queue is a SQLite table (Postgres in production) with
these properties:

- **Idempotent.** The job id is a hash of the audio bytes, options and engine config. A client
  retry or double-submit returns the existing job instead of transcribing twice. Sync and async
  submissions of the same file are separate jobs, because they carry different duration limits:
  a sync-only `audio_too_long` must not stick to the async path it recommends.
- **Leases + heartbeats.** A worker that dies stops heartbeating. Another worker claims the job
  after the lease expires and **resumes from the chunk checkpoints**.
- **Poison-job quarantine.** Attempts are counted at claim time, so a file that crashes workers
  (decoder segfault, OOM-kill) is failed after N tries instead of crash-looping the fleet.
- **Retries.** Transient failures (engine/provider errors) retry with exponential backoff.
  Client errors (bad file) fail immediately with their 4xx code.
- **Graceful drain.** SIGTERM (deploy, spot interruption) hands the job back *immediately*,
  without counting it as an attempt, and another worker resumes from the last checkpoint.
  - It doesn't depend on the current chunk finishing within the grace period.
  - The work loop runs off the main thread, because Python only runs signal handlers between
    bytecodes. A main thread stuck in one long CTranslate2 call would otherwise miss the window
    before SIGKILL.
- **Stall detection.** A worker that finishes no chunk for `STALL_TIMEOUT_SECONDS` (a wedged
  decoder or driver call) stops renewing its lease and exits, so the orchestrator restarts it.
  Otherwise a hung native call would hold its job forever with a perfectly fresh heartbeat.
- **Per-job limits.** Sync jobs carry the 5-minute limit into the worker, enforced on decoded
  samples. A browser WebM with no duration header can't use the sync path to run for hours at
  sync priority.
- **Stranded-job sweep and retention.** Queued jobs for an engine config that *no worker has
  served for an hour* (a retired config) are failed (retryable) instead of blocking backpressure
  forever. A routine restart or model load never trips it. Expired audio and transcripts, and
  long-orphaned runs, are purged periodically, not just at startup.
- **Engine routing.** Workers only claim jobs for their own model config, so a `small` pool and a
  `large-v3-turbo` pool can share one queue.

Why not Celery/RQ for v1: their default timeouts silently kill or re-run long tasks (RQ 180 s,
Arq 300 s, Dramatiq 600 s; Celery+Redis with `acks_late` re-delivers after 1 h). They'd also add
a second source of truth for job state. A DB table with leases is one transactional source of
truth, and it maps 1:1 to Postgres `SELECT … FOR UPDATE SKIP LOCKED`.

---

## 4. Processing for downstream use

The canonical JSON is the system of record. Processors are small pure functions
`Transcript → Transcript`, run in order, each unit-tested:

| Processor | What it does and why |
|---|---|
| `sanitize_timestamps` | Enforces `0 ≤ start ≤ end ≤ duration`, words inside their segment and monotonic, and flags >30 chars/s as `timestamp_anomaly`. Consumers assume these invariants; engines don't guarantee them. |
| `normalize_text` | Unicode NFC and whitespace cleanup. Lossless; scoring normalization stays out of stored text. |
| `tighten_to_words` | With word timings, clamps segment bounds to the first and last word. |
| `QualityGate` | Re-applies Whisper's reference thresholds that batched decoding skips. **Drops:**<br>• `no_speech > 0.6 ∧ logprob < −1` (silence)<br>• `compression_ratio > 2.4` (loop inside a segment)<br>• **loops across segments**: the same ≥3-word sentence again within 2 s, or a fourth copy of a short phrase ("No. No. No." survives; "I'm sorry." ×12 doesn't). Batched decoding scores each short segment alone, so a per-segment ratio never catches these.<br>• known hallucination phrases ("thanks for watching"…), **only when the model's own scores also doubt them** (a real "Thank you so much." survives, test-enforced).<br>**Flags:** `low_confidence` (`logprob < −1`) and `possible_loop` (a 5-segment window compressing > 2.4) for human review.<br>Text comparison is Unicode-aware. Dropped segments go to `transcript.dropped` with a reason; nothing vanishes silently. |
| `resegment_sentences` | With word timings, rebuilds segments as sentences. `large-v3-turbo` puts segment boundaries mid-sentence. Re-segmenting gave sentence-level segments with starts within ~0.05 s of ground truth on the test clip. |
| `redact_pii` (opt-in) | Email / phone / card / SSN → `[EMAIL]` etc. **What it covers:**<br>• text *and* word timings (so the same spans could be bleeped in audio)<br>• the `dropped` audit trail too (it ships in the JSON)<br>• PII read across pauses: split over 2–3 segments of the same speaker, even around the other party's "mm-hmm" in split-channel calls<br>• **spoken forms**: ASR writes what was said, e.g. `john.smith at example.com`, `John.Smith.at.example.com`, `415, 555, 0132`<br>**Rules that keep it precise:**<br>• word timings are redacted from the *same character spans* as the text, so words can never keep what the text removed<br>• card numbers need an issuer digit and a Luhn-valid window on digit-group boundaries: a trailing spoken CVV is handled, order and tracking numbers aren't touched<br>• loose phone forms ("250, 500, 1000") need a phone word nearby<br>Regex is deterministic and explainable for v1; names and addresses need NER (Presidio) in v2. |

**Renderers:**
- `srt` / `vtt`: captions, ≤42 chars × 2 lines, ≤7 s per cue (Netflix guidelines).
  - WebVTT is spec-valid: `end > start`, non-decreasing starts, `&`/`<` escaped, speaker `<v>`
    tags.
  - Cue times include the audio stream's start offset, so captions line up with a video whose
    audio track doesn't start at 0.
- `txt`: readable paragraphs, broken on speaker change or a long pause.
- `jsonl`: one segment per line, for log pipelines and search indexers.
- `chunks`: ~1.5k-char passages that never split a segment, each with `start`/`end`/speakers.
  Ready for embedding or LLM summarisation, so every generated claim can cite a timestamp back
  into the audio.

---

## 5. Key decisions and alternatives

| Decision | Chosen | Alternatives considered | What the choice buys |
|---|---|---|---|
| Model family | Whisper via **faster-whisper**: `large-v3-turbo` on GPU, `small` for the CPU demo | Parakeet-TDT (faster and better English WER, 25 EU languages, NeMo stack); Qwen3-ASR / Cohere / Canary (better WER, but no timestamps or ≤40 s windows); whisper.cpp / mlx (fast on Mac, but no dev/prod parity); WhisperX (alignment + diarization, torch stack) | 99 languages, segment *and* word timestamps, MIT, no torch. **The same code path on the M1 and a CUDA box.** Parakeet is the documented upgrade for EN/EU at high volume, and plugs in behind the same engine interface. |
| Engine boundary | `transcribe(pcm ≤ 10 min) → segments`; the pipeline owns everything else | Engine owns files/chunking | Swapping models or vendors is config (`STT_ENGINE`). Tests run the whole system against a fake engine in ms. |
| Build vs buy | Self-host by default; **hosted adapter included** (`openai-compatible`: OpenAI whisper-1, Groq) | Deepgram / AssemblyAI / ElevenLabs etc. | Hosted is $0.04–0.36/audio-h, self-host roughly $0.01–0.03 marginal. Crossover is ~5–20k audio-h/month; privacy can force self-hosting at any volume. The adapter refuses OpenAI's text-only models (`gpt-4o-transcribe`…), which would silently drop timestamps. |
| Decoder | PyAV (bundled FFmpeg), float32 | ffmpeg CLI, soundfile, pydub, torchaudio | See §2. Also: faster-whisper's own `decode_audio` is **broken by PyAV 19** (verified), and we don't depend on it. |
| Long-form | Silence-cut macro-chunks + VAD clips + batched + narrow fallback | Sequential; batched defaults; overlap-stitch; single-pass Parakeet | See §3 and the ablation in §6. |
| Segment timestamps | Whisper timestamp tokens inside VAD clips; optional DTW word timings and sentence re-segmentation | Batched defaults (30 s blobs); wav2vec2 forced alignment (WhisperX) | p90 start error 0.20–0.22 s vs 12.7–13.8 s, with no extra model. Forced alignment is the v2 step when word precision matters (karaoke, audio redaction). |
| Queue | SQLite table with leases; separate worker processes | Celery/RQ/Arq/Dramatiq; Temporal | See §3. Crash isolation: an FFmpeg or CUDA crash kills a disposable worker, not the API. |
| Sync API | Enqueue at high priority + bounded wait; `202` on timeout | Run the model in the request handler | One execution path; no model contention; no 504s. |

---

## 6. Evaluation

**How it was measured:**
- All numbers are measured on this machine (Apple M1 Pro, CPU only, int8; CTranslate2 has no
  Metal backend).
- Audio is synthetic speech from the macOS TTS (6 voices/accents), with exact ground-truth text
  and utterance timings.
- **Caveat:** TTS speech is far cleaner than real calls, so absolute WER is optimistic. These
  fixtures are for regression-testing the *pipeline*, not for choosing a model.
- Reproduce with `scripts/make_fixtures.py` and `scripts/evaluate.py`.

### Ablation: three ways to run the same model on long audio

The test file is `small` on a 14.3-min, 6-voice meeting with a 90 s silence + 20 s hold-music
stretch mid-file.

| approach | WER | segments | start error median / p90 | longest segment | words placed in the silence | RTF |
|---|---|---|---|---|---|---|
| A. sequential, reference-style | 3.13% | 222 | 0.16 s / 0.67 s | 29.0 s | **64** | 0.101 |
| B. faster-whisper batched defaults | 0.20% | 25 | 6.86 s / 12.67 s | **143.9 s** | 84 | **0.055** |
| C. **this pipeline** | 0.30% | 147 | **0.16 s / 0.22 s** | 7.6 s | **0** | 0.081 |

What is actually in the non-speech stretch:
- **A** invents text. It repeats an earlier sentence ("Let's make sure that we freeze new
  features…") three times across 510–590 s, over pure silence and tone: the classic
  conditioning loop. With `tiny.en` it was "Thank you very much." ×6.
- **B** puts real speech in the wrong place. One "segment" spans 482–625 s, gluing speech from
  both sides of the silence. Its words are right; its timestamps are meaningless.
- **C** puts nothing there.

C costs about 1.5× B's compute. That buys two things: timestamps usable for captions, search
and citations (p90 0.22 s vs 12.7 s), and the restored safety net. Compute was measured on a
shared, busy laptop, so treat RTF as ±30%; the relative ordering was stable across runs.

### Model comparison (this pipeline, CPU int8)

| model | clean 38 s (WER) | 8 kHz µ-law telephony (WER) | 14-min meeting (WER / start err p50 / RTF) | noise-only file | sequential fallbacks (14 min) | peak RSS |
|---|---|---|---|---|---|---|
| tiny.en | 0.0% | 0.0% | 1.39% / 0.17 s / **0.028** | 0 segments | 17 | 1.5 GB |
| small | 0.0% | 1.06% | **0.30%** / 0.16 s / 0.115 | 0 segments | 1 | 2.9 GB |
| large-v3-turbo | 0.0% | 0.0% | 0.40% / 0.18 s / 0.131 | 0 segments | 0 | 3.9 GB |

- Every model produced **zero hallucinated words**, both in the silence stretch and on the
  noise-only file.
- The fallback count tracks model strength. `tiny.en` stops early often (17 re-decoded ranges in
  14 min), and the fallback is what keeps its WER at 1.4% instead of losing whole sentences.
- On CPU, turbo costs only ~15% more than `small` for long-form audio (4 decoder layers).
  - With a GPU, turbo is the production default.
  - On a CPU-only box, `small` is the default here because it is 3–4× cheaper on short clips at
    equal quality on this data.
  - Real, noisy, accented audio would likely favour turbo; measure on your own domain audio.

### Long file end to end (64 min, 6 voices, 360 turns; `small`)

| approach | WER | start error p50 / p90 | longest segment | RTF | peak RSS |
|---|---|---|---|---|---|
| faster-whisper batched defaults (whole file) | 1.48% | 7.40 s / 13.77 s | 143.2 s | **0.059** | 3.1 GB |
| **this pipeline** (streamed, 10-min chunks, 8 fallback ranges) | **0.62%** | **0.17 s / 0.20 s** | 12.6 s | 0.104 | 3.1 GB |

This table also records a mistake caught by measurement. The first run of this benchmark had
the pipeline *worse* (1.87% WER). The cause: the duplicate-loop rule deleted 9 real sentences
where two speakers said the same thing back to back. The rule now keeps a single repeat; loops
repeat many times. Re-measured: 0.62%, 0 drops.

### Memory that grows with duration (audio + log-mel features, model excluded)

| audio length | whole file (faster-whisper batched path) | this pipeline (streamed) |
|---|---|---|
| 1.06 h | 757 MB | 123 MB |
| 2.12 h | 1,513 MB | 124 MB |
| 3.18 h | 2,270 MB | 124 MB |

- Measured with `tracemalloc`, which counts the numpy buffers exactly. RSS on macOS is
  unreliable here: compressed pages aren't counted, and the RSS figures were non-monotonic.
- The whole-file path grows ~715 MB per hour of audio. The streamed path is flat, bounded by one
  10-minute chunk.
- End to end, peak RSS (~3 GB) is dominated by the model runtime, so at ≤1 h the two approaches
  look the same in RSS. The difference shows in the multi-hour files the brief asks about. One
  user reported 26–27 GB on an 11 h file through faster-whisper's non-batched path.

---

## 7. Running it

```bash
uv sync
uv run python scripts/make_fixtures.py          # macOS: TTS fixtures + format matrix + hostile files
uv run pytest -m "not slow"                     # 105 fast tests (mock engine), ~25 s
uv run pytest -m slow                           # real model (tiny.en): format matrix, silence, call
uv run python scripts/smoke_e2e.py              # real API + worker processes, SIGTERM mid-job, resume
./scripts/make_examples.sh                      # regenerate examples/
```

**Example outputs**, produced by `scripts/make_examples.sh` with the default `small` model:
- [support_call.txt](examples/support_call.txt): split channels plus PII redaction.
- [meeting_short.srt](examples/meeting_short.srt): sentence-level captions.
- [meeting_short.json](examples/meeting_short.json): the canonical transcript.
- [long_meeting.chunks.json](examples/long_meeting.chunks.json): RAG passages with timestamps.
- [playlist_ssrf.stderr.txt](examples/playlist_ssrf.stderr.txt): the error for a hostile upload.

**Smoke test** with real processes ([scripts/smoke_e2e.py](scripts/smoke_e2e.py)). Observed:
- readiness is 503 until a worker is up;
- a sync SRT returns 200; the disguised HLS playlist gets 415;
- an async job is sent SIGTERM at 38%, is released immediately (attempts back to 0), and is
  finished by a second worker that **reused 3 of 8 chunks** from the first worker's checkpoints;
- resubmitting the same file returns 200 `Idempotent-Replay` with the same job id.

**Reproducing §6:**

```bash
uv run python scripts/make_fixtures.py --long-minutes 60 --out fixtures_60min --seed 11
mkdir -p fixtures_scaling && for k in 1 2 3; do for i in $(seq $k); do cat fixtures_60min/long_meeting.mp3; done > fixtures_scaling/meeting_x$k.mp3; done
uv run python scripts/evaluate.py --out docs/evaluation_raw.md ablation --model small
uv run python scripts/evaluate.py --out docs/evaluation_raw.md models --models tiny.en small large-v3-turbo
uv run python scripts/evaluate.py --out docs/evaluation_raw.md memory --file fixtures_60min/long_meeting.mp3
uv run python scripts/evaluate.py --out docs/evaluation_raw.md scaling fixtures_scaling/meeting_x{1,2,3}.mp3
```

**CLI:**

```bash
uv run stt transcribe call.wav --split-channels --redact-pii --format txt
uv run stt transcribe lecture.mp3 --format vtt -o lecture.vtt --checkpoint-dir .ckpt   # resumable
uv run stt transcribe interview.m4a --model large-v3-turbo --words --format json
```

**Service (two processes; scale workers independently):**

```bash
uv run stt serve --port 8000          # API: validates, enqueues, serves results. No model.
uv run stt worker                     # one per GPU / CPU box; loads + warms the model once
```

| Endpoint | |
|---|---|
| `POST /v1/transcriptions` | multipart `file` + `language`, `word_timestamps`, `channel_mode=mix\|split`, `redact_pii`, `prompt`, `response_format=json\|jsonl\|txt\|srt\|vtt\|chunks`. ≤5 min of audio; `202` + job link if it runs past the wait budget. |
| `POST /v1/jobs` | same options; `202` + `Location`; same file and options again → `200 Idempotent-Replay` |
| `GET /v1/jobs/{id}` | status, progress, attempts, error |
| `GET /v1/jobs/{id}/result?format=…` | `409 not_ready` until done |
| `DELETE /v1/jobs/{id}` | hard-delete audio + transcript (retention/erasure); results also expire after `STT_RESULT_TTL_SECONDS` |
| `GET /healthz` · `/readyz` · `/metrics` | liveness · a worker for this engine config is heartbeating · Prometheus text |

Errors are typed JSON (`{"error": {"code": "unsupported_media", "message": …}}`) with stable
codes, so clients can distinguish "your file is bad" (4xx, don't retry) from "we failed" (5xx /
`Retry-After`).

**Config** is via `STT_*` environment variables ([config.py](src/stt_pipeline/config.py)):
- engine selection: `ENGINE`, `MODEL`, `MODEL_REVISION`, `DEVICE`, `COMPUTE_TYPE`
- decoding: `BATCH_SIZE`, `BEAM_SIZE`
- long audio: `CHUNK_SECONDS`
- limits: `MAX_DURATION_SECONDS`, `MAX_UPLOAD_MB`, `SYNC_MAX_DURATION_SECONDS`
- queue: `MAX_QUEUE`, `MAX_ATTEMPTS`, `LEASE_SECONDS`, `STALL_TIMEOUT_SECONDS`, `STRANDED_AFTER_SECONDS`
- split mode: `MAX_SPLIT_CHANNELS`
- retention: `RESULT_TTL_SECONDS`
- hosted engine: `API_BASE`, `API_KEY`

**Deploy:**
- [Dockerfile](Dockerfile) builds one image for both roles. It uses CUDA 12 cuBLAS via wheel
  (CTranslate2 ≥4.6.3 needs no cuDNN), bakes **pinned** weights in for offline, reproducible
  deploys, and runs as a non-root user.
- [docker-compose.yml](docker-compose.yml) runs the API plus a GPU worker on a shared volume.
- *Not built or run here: the Docker daemon wasn't running on the dev machine, and there's no
  NVIDIA GPU.*

```
src/stt_pipeline/
  audio.py          sniff · probe · stream-decode (format handling, security limits)
  chunking.py       silence-aware macro-chunks
  pipeline.py       orchestration, checkpoints, language pinning, offsets
  engines/          base protocol · faster_whisper_engine · openai_compat (hosted) · mock
  postprocess.py    sanitizer · normalizer · quality gate · PII
  formats.py        json · jsonl · txt · srt · vtt · chunks
  jobs.py           SQLite job queue + Worker (leases, retries, poison, drain)
  api.py · cli.py · config.py · schemas.py · errors.py
scripts/make_fixtures.py   TTS fixtures with ground truth, format matrix, hostile inputs
scripts/evaluate.py        ablation · model comparison · long-file · memory scaling, each in a fresh process
scripts/smoke_e2e.py       end to end with real API + worker processes and signals
examples/                  real outputs (txt, srt, json, chunks, error)
docs/RESEARCH.md           sourced research notes (fact-checked)
```

---

## 8. How this was checked

1. **Research, fact-checked.**
   - Six parallel research passes covered engines, hosted APIs, ingestion, long-form, timestamps
     and post-processing, and serving.
   - Each was followed by an independent pass that tried to *refute* its load-bearing claims
     against primary sources: 52 confirmed, 8 corrected, 0 refuted.
   - Key library claims were then re-verified by reading the installed source.
   - Details: [docs/RESEARCH.md](docs/RESEARCH.md).
2. **Measured, not assumed.** Every design choice in §2–§3 traces to a measured problem on real
   fixtures. Several came from surprises: the PyAV/faster-whisper break, the +3 dB downmix,
   merged turns, dropped tails, turbo's duplicated words.
3. **Adversarial code review.**
   - Five reviewers each took one dimension (ingestion/security, long-audio correctness,
     queue/concurrency, post-processing/formats, hosted adapter/config/deploy). Each reviewer's
     findings went to a verifier that tried to refute them.
   - Result: **75 reported, 69 confirmed (~60 unique), 6 refuted.**
   - Notable confirmed defects:
     - The compose file gave the API and worker different engine keys, so no job would ever run.
     - A reused CLI checkpoint dir could return another file's text.
     - PII leaked through the `dropped` audit list.
     - The resampler passthrough time-stretched audio.
     - A hung decoder could hold a job forever.
     - The API process ran FFmpeg on untrusted input.
   - All were fixed, each with a regression test ([tests/test_review_regressions.py](tests/test_review_regressions.py)).
4. **Fix verification.**
   - A second adversarial pass re-checked every finding against the fixed code: **51 fixed,
     17 partially fixed, 1 not fixed**.
   - It also found **16 new defects introduced by the fixes** themselves. The notable ones:
     - The new sync duration limit was stored on a job id shared with `/v1/jobs`, so a sync
       `audio_too_long` blocked the async path its own message recommends.
     - The stranded-job sweep would fail the whole backlog during a routine worker restart.
     - One crashing upload broke the probe process pool for every concurrent upload.
     - A new phone pattern redacted "250, 500, 1000 dollars".
   - A second round fixed these with more regression tests. Remaining trade-offs are listed
     in §10.
5. **Tests.**
   - 105 fast tests on the mock engine: format matrix, hostile inputs, chunking invariants,
     checkpoint/resume, queue semantics (atomic claims, lease expiry, poison, drain), API
     contracts, formats, PII, and every regression from both review rounds.
   - 3 real-model integration tests.

## 9. Production roadmap (v2)

These are described rather than built. Each one is triggered by a measured need, not added up
front.

**Scale-out**
- Uploads go directly to object storage via presigned URLs; the API receives a key.
- Postgres replaces SQLite, with the same lease semantics.
- Macro-chunks fan out as separate tasks across GPUs, with a merge step.
- KEDA autoscaling on queue depth; spot instances (the chunk checkpoints already make
  interruption cheap).

**Throughput**
- Cross-request batching of 30 s clips.
- A Parakeet-TDT tier for English/EU traffic (~7–13× throughput on GPU).
- Benchmark int8_float16 vs fp16 per GPU type before choosing; it's card-dependent.

**Quality**
- Speaker diarization (pyannote community-1) on the *full* file, with each word assigned to the
  speaker it overlaps most.
- Forced alignment per segment for word-precise uses.
- Domain vocabulary via `prompt`/hotwords per tenant.
- A WER regression gate in CI on real, consented domain audio.

**Safety**
- Worker-side decode in a network-less, seccomp-restricted container. (The API's probe is
  already per-process sandboxed. The worker decodes in-process, and a crash there is bounded by
  poison quarantine.)
- NER-based PII (Presidio) after ITN.
- Per-tenant dedup scope: cross-tenant hashing leaks whether content exists.

**Delivery**
- Signed webhooks (Standard Webhooks); SSE progress.

**Ops**
- OpenTelemetry traces api → queue → worker; an RTF histogram; GPU metrics (DCGM).

## 10. Known limitations

- **Absolute WER is not representative.** Accuracy was measured on TTS speech. The pipeline
  measurements (timestamps, hallucination, memory, robustness) transfer; absolute WER doesn't.
- **GPU numbers are not measured here.** GPU throughput and cost figures come from published
  benchmarks.
- **One language per file**, no code-switching.
- **Regex PII misses names, addresses, and badly garbled spoken forms.** We observed `tiny.en`
  write "John, Don't Smith, at example.com".
- **The worker decodes in-process.** The API's probe is sandboxed per upload; in the worker, a
  decoder crash kills that worker and poison quarantine stops it repeating.
- **The duplicate rule drops genuine immediate repeats.** A ≥3-word sentence said twice within
  2 s by the same speaker is treated as a decoding loop. The 60-min synthetic script contains 2
  such repeats. They are kept in `dropped` for audit.
- **A drained worker abandons its in-flight chunk.** That work is recomputed elsewhere from the
  last checkpoint.
- **Changing the engine-key settings strands queued jobs.** Those settings are the model,
  device, compute type, beam, chunking and library versions. Stranded jobs are failed
  (retryable) after an hour, and there's no in-place re-keying.
- **No forced alignment.** Word timings are Whisper's DTW; per-segment alignment is the v2 path.
- **The body-size cap counts streamed bytes; it doesn't replace an edge limit.** Keep a proxy
  limit (or presigned uploads) in production.
