# Transcription pipeline

The pipeline takes any audio file (or the audio track of a video), transcribes it, and returns a
transcript with per-segment timestamps. The transcript is post-processed for downstream use: a
quality gate, PII redaction, captions, and RAG-ready chunks. It runs as a CLI and as an HTTP
service with a durable job queue for long files.

The brief asked for engineering decisions rather than a model. So this README is mostly the
*why*: what each choice solves that the alternatives don't, and what measurements back it.
Research notes with sources are in [docs/RESEARCH.md](docs/RESEARCH.md). Results on real speech
(LibriSpeech and three Earnings-22 earnings calls) are in
[docs/eval_real_data.md](docs/eval_real_data.md); the synthetic-audio benchmarks used for
regression testing are in [docs/evaluation_raw.md](docs/evaluation_raw.md).

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
                │                   ─▶ continue clips Whisper stopped early, in more batched passes           │
                │                   ─▶ coverage + loop check ─▶ re-decode only where needed                   │
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

Every one of these transcribes to the same text (0.0% WER with `tiny.en`, see §3; the slow test
enforces < 15%):

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
| Reference sequential decoding (openai-whisper style) | Slow, and can't batch. Errors propagate through the text prompt. In our synthetic 90 s silence + 20 s hold-music stretch it **invented 42 words** (one sentence looped three times) and stretched real sentences into the gap. Loads the whole file. |
| faster-whisper `BatchedInferencePipeline` defaults | Fast, but returns **one "segment" per 30 s VAD chunk**: on real earnings calls the median segment was about 30 s and a sentence start was typically ~8 s from the nearest segment start. Batching also silently disables Whisper's anti-hallucination fallbacks, and it loads the whole file. |
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

3. **Continue clips that Whisper stopped early.** This is the bug real calls exposed.
   - Whisper sometimes ends a window with two timestamps and no more text
     (`…dismay upon<|5.92|><|5.92|>`). That means "I stopped here; start the next window at
     5.92 s". Reference Whisper does exactly that.
   - With timestamps turned on, which segment times need, faster-whisper's batched pipeline
     computes that seek position and then throws it away. The rest of the clip is never decoded,
     and whole sentences go missing without any error. (Its defaults turn timestamps off, so
     the defaults don't hit this; they get 30 s segments instead.)
   - The engine wraps the model so it records where each clip stopped, then decodes just those
     leftover pieces in further batched passes (up to 3 after the first). Anything still left
     after that goes to the coverage check below.
   - Measured by switching it off on the three Earnings-22 calls, so that the coverage fallback
     below has to catch early stops instead: WER 14.92% → 14.62% with `small` and
     12.77% → 12.25% with turbo, for about the same CPU. The gain comes from fewer missing
     words: deletions fell on all six call/model pairs (8.29 → 7.12% and 7.24 → 6.57% pooled).
     With `small` most of that is offset by garbled extra words from short leftovers
     (substitutions 5.37 → 5.84%, insertions 1.26 → 1.66%); with turbo, hardly at all (+0.08 and
     +0.06 pt). WER was lower on all three calls with turbo and on two of three with `small`.
     Without continuation, the fallback re-decodes 15% (`small`) and 8% (turbo) of the speech
     and still leaves more words missing.
   - Leftover pieces are decoded without the speech before them, and very short ones are where
     Whisper garbles: "and fully utilize" came out as "I'm free to die.", "12 months" as
     "I'll see you in 12 minutes." Their segments are flagged `continued`, and the quality gate
     (§4) drops any whose average log-prob is below Whisper's own failure threshold of −1.0.

4. **Narrow fallback for what's left.** Batched decoding still has no temperature fallback, and
   nothing checks that every stretch of speech produced text. So two checks run:
   - **coverage:** no VAD speech region of about 1.4 s or more may have ≥1.5 s outside every
     segment, or be under half covered. It's measured on the union of segment intervals, after
     trimming the VAD's 0.2 s padding and allowing 0.3 s of timestamp jitter per segment edge.
     Shorter regions aren't checked;
   - **loops:** no segment may have a gzip compression ratio > 2.4. Loop segments don't count
     as coverage.

   Regions that fail are re-decoded in two stages. First they're batched again with clips cut to
   just those regions, which is cheap. Whatever still fails goes through the *reference*
   algorithm's full safety net: temperature fallback, plus compression, log-prob and
   no-speech thresholds. Batched speed is kept everywhere else.

   Each redo range is grown to fully contain every segment it touches, and those segments are
   replaced ("whole segments in, whole segments out"). Keeping a partially overlapping segment
   duplicated words in practice: with `large-v3-turbo`, a batched segment ended mid-sentence
   inside a re-decoded region and produced 6.4% WER on a synthetic clip, 0.0% after the rule.

   Before continuation existed, this fallback did most of the work. Early stops showed up as
   coverage gaps: with `tiny.en`, on 5 of 11 synthetic format-matrix files. Re-checked with
   continuation, the same 5 files each have one continued clip, all 11 are at 0.0% WER, and
   nothing is re-decoded (`scripts/check_format_matrix.py`; output in
   [docs/evaluation_raw.md](docs/evaluation_raw.md)). On real
   calls, an earlier development run (not in the published results) had the fallback, plus a
   coverage check that didn't tolerate timestamp jitter, re-decoding so much audio that the
   pipeline cost ~2.8× the default batched path. Today's two-stage fallback is much cheaper:
   with continuation switched off, it catches the early stops at about the same total cost as
   continuation (§6), just less completely. With continuation on, the checks flagged **0
   ranges** (nothing re-decoded) on the three real calls, for both models, and the pipeline
   costs 1.5–1.7× the defaults.

5. **Language** is detected on the first chunk that actually contains speech, then pinned for
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
| `QualityGate` | Re-applies Whisper's reference thresholds that batched decoding skips. **Drops:**<br>• `no_speech > 0.6 ∧ logprob < −1` (silence)<br>• `compression_ratio > 2.4` (loop inside a segment)<br>• **loops across segments**: a third copy of the same ≥3-word sentence, each within 2 s of the last, or a fourth copy of a short phrase ("No. No. No." survives; "I'm sorry." ×12 doesn't). Batched decoding scores each short segment alone, so a per-segment ratio never catches these.<br>• known hallucination phrases ("thanks for watching"…), **only when the model's own scores also doubt them** (a real "Thank you so much." survives, test-enforced).<br>• `continued` segments (decoded from a leftover piece, §3) with `logprob < −1`. On three real calls with `small`, 5 of 89 continued segments fell below the line, and all 5 garbled a short real phrase: only 4 of their 23 words were right ("and fully utilize" → "I'm free to die."). The 84 above it had 79% of their words right. Dropping loses a few real words but keeps confident-looking wrong text out of search and summaries. It doesn't catch everything (§10).<br>**Flags:** `low_confidence` (`logprob < −1`) and `possible_loop` (a 5-segment window compressing > 2.4) for human review.<br>Text comparison is Unicode-aware. Dropped segments go to `transcript.dropped` with a reason; nothing vanishes silently. |
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
| Model family | Whisper via **faster-whisper**: `large-v3-turbo` in production, `small` only for a cheap CPU demo (turbo was more accurate on every real-speech set, at ~2× the CPU) | Parakeet-TDT (faster, better short-form English WER, but worse than turbo on long-form Earnings-21/22; 25 EU languages; NeMo stack); Qwen3-ASR / Cohere / canary-qwen (better short-form WER, but timestamps only through a second aligner model (Qwen3), or none at all (Cohere, canary-qwen; canary-qwen also ≤40 s clips)); whisper.cpp / mlx (fast on Mac, but no dev/prod parity); WhisperX (alignment + diarization, torch stack) | 99 languages, segment *and* word timestamps, MIT, no torch. **The same code path on the M1 and a CUDA box.** Parakeet is the documented upgrade for EN/EU at high volume, and plugs in behind the same engine interface. |
| Engine boundary | `transcribe(pcm ≤ 10 min) → segments`; the pipeline owns everything else | Engine owns files/chunking | Swapping models or vendors is config (`STT_ENGINE`). Tests run the whole system against a fake engine in ms. |
| Build vs buy | Self-host by default; **hosted adapter included** (`openai-compatible`: OpenAI whisper-1, Groq) | Deepgram / AssemblyAI / ElevenLabs etc. | Hosted is $0.04–0.36/audio-h, self-host roughly $0.01–0.03 marginal. Crossover is ~5–20k audio-h/month; privacy can force self-hosting at any volume. The adapter refuses OpenAI's text-only models (`gpt-4o-transcribe`…), which would silently drop timestamps. |
| Decoder | PyAV (bundled FFmpeg), float32 | ffmpeg CLI, soundfile, pydub, torchaudio | See §2. Also: faster-whisper's own `decode_audio` is **broken by PyAV 19** (verified), and we don't depend on it. |
| Long-form | Silence-cut macro-chunks + VAD clips + batched + continuation of early stops + narrow fallback | Sequential; batched defaults; overlap-stitch; single-pass Parakeet | See §3 and the ablation in §6. |
| Segment timestamps | Whisper timestamp tokens inside VAD clips; optional DTW word timings and sentence re-segmentation | Batched defaults (30 s blobs); wav2vec2 forced alignment (WhisperX) | On real earnings calls (the two with usable word times): median sentence-start error 0.3–0.8 s vs ~8 s, and 54–71% of sentence starts within 1 s vs 9–10%, with no extra model. Forced alignment is the v2 step when word precision matters (karaoke, audio redaction). |
| Queue | SQLite table with leases; separate worker processes | Celery/RQ/Arq/Dramatiq; Temporal | See §3. Crash isolation: an FFmpeg or CUDA crash kills a disposable worker, not the API. |
| Sync API | Enqueue at high priority + bounded wait; `202` on timeout | Run the model in the request handler | One execution path; no model contention; no 504s. |

---

## 6. Evaluation

All numbers are measured on this machine: Apple M1 Pro, CPU only, int8 (CTranslate2 has no
Metal backend).

### On real speech: LibriSpeech and three earnings calls

**How it was measured** ([scripts/eval_datasets.py](scripts/eval_datasets.py), full output in
[docs/eval_real_data.md](docs/eval_real_data.md)):
- **LibriSpeech:** a random sample of 200 test-clean clips (21 speakers) and 100 test-other
  clips (17 speakers). Each clip is run through the model alone and through the pipeline, and
  the difference gets a paired bootstrap 95% confidence interval.
- **Earnings-22:** three real earnings calls with accented speakers (South Africa, China,
  Chile; 80 min). Timing is scored against the dataset's forced-aligned word times, on China
  and Chile; the South Africa alignment in the dataset is unusable.
- WER uses Whisper's English text normaliser, the Open ASR Leaderboard convention.
- Every approach gets the same 6 CPU threads; cost is CPU seconds per audio second.

**Does the pipeline change accuracy?** Not detectably.

| model | LibriSpeech split | model alone | through the pipeline | difference (95% CI) | Open ASR Leaderboard (full set, HF transformers) |
|---|---|---|---|---|---|
| small | test-clean | 3.66% | 3.58% | −0.08 pt (−0.49 to +0.30) | 3.05% (small.en) |
| small | test-other | 8.74% | 8.93% | +0.19 pt (−0.61 to +1.07) | 7.25% (small.en) |
| large-v3-turbo | test-clean | 2.19% | 2.19% | 0.00 pt (−0.24 to +0.25) | 2.10% |
| large-v3-turbo | test-other | 4.43% | 4.63% | +0.19 pt (−0.15 to +0.57) | 4.24% |

**Earnings calls: about the same accuracy as the defaults, much better timestamps, 1.5–1.7×
the compute.**

| model | approach | WER (3 calls) | sub / del / ins | segments > 30 s (3 calls) | sentence-start error, median | sentence starts within 1 s | CPU s per audio s |
|---|---|---|---|---|---|---|---|
| small | faster-whisper batched defaults | 14.06% | 5.15 / 7.76 / 1.15 | 63 | 7.9 s | 9–10% | 0.247 |
| small | pipeline, continuation off | 14.92% | 5.37 / 8.29 / 1.26 | 0 | 0.29–0.76 s | 58–72% | 0.396 (1.60×) |
| small | **this pipeline** | 14.62% | 5.84 / 7.12 / 1.66 | 0 | 0.29–0.77 s | 57–71% | 0.381 (1.54×) |
| large-v3-turbo | faster-whisper batched defaults | 12.75% | 4.06 / 7.66 / 1.03 | 63 | 7.9 s | 9–10% | 0.453 |
| large-v3-turbo | pipeline, continuation off | 12.77% | 4.29 / 7.24 / 1.25 | 0 | 0.38–0.75 s | 56–65% | 0.749 (1.65×) |
| large-v3-turbo | **this pipeline** | 12.25% | 4.37 / 6.57 / 1.31 | 0 | 0.40–0.80 s | 54–65% | 0.757 (1.67×) |

The two sentence-start columns are for China and Chile, the calls with usable word times; the
ranges are those two calls.

What this shows:
- **No detectable accuracy change.** Every LibriSpeech 95% confidence interval includes zero
  (the widest, `small` test-other, runs from −0.61 to +1.07 pt), and turbo lands close to its
  leaderboard numbers. On the calls the pipeline is 0.56 pt worse pooled with `small` and
  0.50 pt better with turbo. Per call, the difference goes both ways: South Africa / China /
  Chile are +1.5 / −0.1 / −0.1 pt with `small` and +0.5 / −0.8 / −1.5 pt with turbo (worse on
  South Africa with both). Three calls can't separate the pooled differences.
- **The error mix shifts.** The pipeline misses fewer words (deletions 7.76 → 7.12% with
  `small`, 7.66 → 6.57% with turbo) but writes more wrong or extra ones (substitutions
  5.15 → 5.84% and 4.06 → 4.37%; insertions 1.15 → 1.66% and 1.03 → 1.31%). Extra words are
  output words not in the reference. Some are speech the reference skipped, and some are
  garbled short pieces (§3, §10).
- **The timestamps are the gain.** The defaults return ~30 s blocks (median 29.3–29.9 s per
  call, up to 55 s), so a sentence typically starts ~8 s away from the nearest segment start.
  The pipeline's segments have a median of 3.7–6.1 s per call, with none over 30 s. Half of the
  segments start within ~0.1 s of the aligned time of their first word, and 9 in 10 within
  0.5–0.8 s. The long tail of the sentence-start error (p90 2.8–3.7 s) is sentences that begin
  in the middle of a segment rather than at one. That is looser than on synthetic audio (p90
  0.22 s), so the synthetic timing numbers below are best-case.
  - "Sentence start" here means a forced-aligned word after sentence-ending punctuation, a
    speaker change, or a pause of ≥1 s.
- **The cost is 1.5–1.7× the defaults' CPU.** The checks flagged nothing for re-decoding on
  any call: continuation (§3) handles early stops in batch, and the coverage check tolerates
  timestamp jitter. Switching continuation off costs about the same (1.60× and 1.65×). The
  coverage fallback then catches most early stops itself, re-decoding 15% and 8% of the speech
  (62 and 39 ranges), but leaves more words missing (deletions +1.2 and +0.7 pt). WER ends up
  0.3 and 0.5 pt higher, because with `small` continuation also adds some garbled words.
  Timing is essentially the same either way.
- **The quality gate dropped little.** With `small`: 5 weak continuations and 2 known-phrase
  segments. One was a duplicated "Thank you very much."; the other was a real, low-scoring
  "Thank you." an analyst said (§10). With turbo: 1 known-phrase segment.
- **Turbo beats `small` on every set** (2.19 vs 3.58%, 4.63 vs 8.93%, 12.25 vs 14.62%) for
  ~2× the CPU (2.0× on the calls, 2.2× on LibriSpeech). On the synthetic audio below they looked
  equally accurate, so that data couldn't have chosen the model.
- For reference, turbo's Open ASR Leaderboard long-form Earnings-22 WER is 13.10% (whole
  dataset, HF transformers). Three calls aren't comparable to that, but it's the same range.

### On synthetic audio (regression fixtures)

**How it was measured:**
- Audio is synthetic speech from the macOS TTS (6 voices/accents), with exact ground-truth text
  and utterance timings. That makes timing and hallucination easy to score exactly.
- **Caveat:** TTS speech is far cleaner than real calls, and the real-speech results above show
  where it misled: absolute WER, the WER gap to the defaults, model choice, and the turbo/`small`
  cost ratio (1.14× in the RTFs below, 2.0–2.2× CPU on real speech with equal threads). These
  fixtures are for regression-testing the *pipeline*.
- These runs predate the continuation fix and the current fallback (§3). Early stops then
  showed up as re-decodes by the sequential decoder, under a coverage check with no jitter
  tolerance, which is why most fallback counts below are non-zero.
- Reproduce with `scripts/make_fixtures.py` and `scripts/evaluate.py`.

#### Ablation: three ways to run the same model on long audio

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

C costs about 1.5× B's compute in this run. A separate run of the same configuration (the
model comparison below) measured RTF 0.115 rather than 0.081, which would make it 2.1×.
Compute was measured on a shared, busy laptop without fixed thread counts, so the controlled
real-speech numbers above (1.5–1.7×) are the ones to use. C buys two things: timestamps usable
for captions, search and citations (p90 0.22 s vs 12.7 s), and the restored safety net.

#### Model comparison (this pipeline, CPU int8)

| model | clean 38 s (WER) | 8 kHz µ-law telephony (WER) | 14-min meeting (WER / start err p50 / RTF) | noise-only file | segments from fallback re-decodes (14 min) | peak RSS |
|---|---|---|---|---|---|---|
| tiny.en | 0.0% | 0.0% | 1.39% / 0.17 s / **0.028** | 0 segments | 17 | 1.5 GB |
| small | 0.0% | 1.06% | **0.30%** / 0.16 s / 0.115 | 0 segments | 1 | 2.9 GB |
| large-v3-turbo | 0.0% | 0.0% | 0.40% / 0.18 s / 0.131 | 0 segments | 0 | 3.9 GB |

- Every model produced **zero hallucinated words**, both in the silence stretch and on the
  noise-only file.
- The fallback count tracks model strength. `tiny.en` stops early often (17 output segments came
  from fallback re-decodes in 14 min), and recovering those is what keeps its WER at 1.4%
  instead of losing whole sentences. Those early stops are now continued in a batched pass instead (§3).
- `small` and turbo look equally accurate on this clean audio. On real speech turbo was better
  on every set, for ~2× `small`'s CPU. So turbo is the production default, and `small` is only
  the cheap option for a CPU-only demo.

#### Long file end to end (64 min, 6 voices, 360 turns; `small`)

| approach | WER | start error p50 / p90 | longest segment | RTF | peak RSS |
|---|---|---|---|---|---|
| faster-whisper batched defaults (whole file) | 1.48% | 7.40 s / 13.77 s | 143.2 s | **0.059** | 3.1 GB |
| **this pipeline** (streamed, 10-min chunks, 8 segments from fallback re-decodes) | **0.62%** | **0.17 s / 0.20 s** | 12.6 s | 0.104 | 3.1 GB |

This table also records a mistake caught by measurement. The first run of this benchmark had
the pipeline *worse* (1.87% WER). The cause: the duplicate-loop rule deleted 9 real sentences
where two speakers said the same thing back to back. The rule now keeps a single repeat; loops
repeat many times. Re-measured: 0.62%, 0 drops.

#### Memory that grows with duration (audio + log-mel features, model excluded)

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
uv run pytest -m "not slow"                     # 111 fast tests (mock engine), ~25 s
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

**Reproducing the real-speech results** (downloads ~55 MB: a random LibriSpeech sample from the
Hugging Face datasets server, and three Earnings-22 calls with their references):

```bash
uv run python scripts/eval_datasets.py download
uv run python scripts/eval_datasets.py run --out docs/eval_real_data.md     # all approaches, ~1.5 h on an M1 Pro CPU
# re-run only some approaches, taking the other rows from an earlier run (write to a new file, then copy it over)
uv run python scripts/eval_datasets.py run --modes pipeline --reuse docs/eval_real_data.json --out /tmp/eval_rerun.md
uv run python scripts/eval_datasets.py report docs/eval_real_data.json     # rebuild the .md from the saved .json
```

**Reproducing the synthetic benchmarks:**

```bash
uv run python scripts/make_fixtures.py --long-minutes 60 --out fixtures_60min --seed 11
mkdir -p fixtures_scaling && for k in 1 2 3; do for i in $(seq $k); do cat fixtures_60min/long_meeting.mp3; done > fixtures_scaling/meeting_x$k.mp3; done
uv run python scripts/evaluate.py --out docs/evaluation_raw.md ablation --model small
uv run python scripts/evaluate.py --out docs/evaluation_raw.md models --models tiny.en small large-v3-turbo
uv run python scripts/evaluate.py --out docs/evaluation_raw.md memory --file fixtures_60min/long_meeting.mp3
uv run python scripts/evaluate.py --out docs/evaluation_raw.md scaling fixtures_scaling/meeting_x{1,2,3}.mp3
uv run python scripts/check_format_matrix.py --model tiny.en   # format matrix: WER, continued clips, re-decodes
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
2. **Measured, not assumed.** Every design choice in §2–§3 traces to a measured problem.
   Several came from surprises: the PyAV/faster-whisper break, the +3 dB downmix, merged
   turns, dropped tails, turbo's duplicated words.
3. **Tested on real speech, which changed things.** Synthetic TTS audio was enough to build
   the pipeline but flattered it. On LibriSpeech and real earnings calls:
   - the pipeline's WER advantage on synthetic audio didn't hold: on real speech there's no
     detectable difference from the model alone or the default batched path;
   - the timestamp advantage did hold, though it's smaller than on synthetic audio;
   - `large-v3-turbo` beat `small` on every set, which overturned "`small` at equal quality";
   - the "dropped tails" turned out to be faster-whisper's batched mode, with timestamps on,
     ignoring Whisper's "continue from here" signal. Fixing that at the source (§3) left fewer
     words missing than catching the gaps afterwards: deletions were lower on all six
     call/model pairs, measured with it switched off. Pooled WER was 0.3–0.5 pt lower, which
     is within call-to-call noise for `small`;
   - separately, a coverage check that tolerates timestamp jitter and re-decodes in batch
     first brought the cost from ~2.8× the default path (an earlier development run) to about
     1.6×. Continuation costs about the same as letting that fallback catch the early stops.

   The evaluation itself was audited before trusting it, and three flaws were fixed: a timing
   metric that only looked from the segment side, unequal CPU threads across approaches, and a
   LibriSpeech sample drawn from too few speakers. Then independent checkers re-derived the
   real-speech numbers in this README from the raw results (and the continued-segment counts
   from a separate diagnostic run), in two rounds. They caught a bug in my own diagnostic
   script, which had tagged only half the continued segments, and several overstatements.
   Those are corrected here. The continuation effect in §3 is now a measured on/off run. An
   earlier estimate simply deleted the continued text (~4 pt with `small`); the measured effect
   is much smaller (0.3–0.5 pt), because the fallback catches most early stops on its own.
4. **Adversarial code review.**
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
5. **Fix verification.**
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
6. **Tests.**
   - 111 fast tests on the mock engine: format matrix, hostile inputs, chunking invariants,
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

- **The real-speech test is small.** 300 LibriSpeech clips and three earnings calls (80 min),
  English only. Enough to show no detectable accuracy cost and that the timestamp gain holds on
  real speech; not enough to rank models for a specific domain. The pooled WER differences
  between the pipeline and the defaults (+0.56 pt with `small`, −0.50 pt with turbo) are well
  inside the spread of per-call differences (−1.5 to +1.5 pt), so three calls can't tell them
  apart.
- **Hallucination over real non-speech wasn't measured.** The "0 words in the silence" result
  is from a synthetic silence + hold-music stretch. The earnings calls have little non-speech.
- **A made-up continued segment can survive.** On one call `small` decoded a 0.7 s leftover
  piece as "Thank you very much for watching, and I'll see you in the next video." Its
  log-prob (−0.85) is above the weak-continuation line, and the phrase filter only drops exact
  matches of listed short phrases, so neither rule caught it. Its timestamps run 6 s past the
  leftover piece it came from, which a future check could use. More generally, the −1.0 line
  doesn't catch all garbling: in the diagnostic run, 15 of the 84 continued segments above it
  had at most half their words right.
- **The phrase filter can drop a real short "Thank you."** It drops listed phrases when the
  model's scores are weak (no-speech > 0.3 or log-prob < −0.7). On one call that removed an
  analyst's real, quietly scored "Thank you." It's kept in `dropped` for audit.
- **Continued segments with low confidence are dropped, not repaired.** That removes garbled
  text but also the few real words it stood for.
- **GPU numbers are not measured here.** GPU throughput and cost figures come from published
  benchmarks.
- **One language per file**, no code-switching.
- **Regex PII misses names, addresses, and badly garbled spoken forms.** We observed `tiny.en`
  write "John, Don't Smith, at example.com".
- **The worker decodes in-process.** The API's probe is sandboxed per upload; in the worker, a
  decoder crash kills that worker and poison quarantine stops it repeating.
- **The duplicate rule drops genuine repeats said three or more times.** A ≥3-word sentence
  repeated a third time within 2 s by the same speaker (a fourth time for short phrases) is
  treated as a decoding loop. A single repeat is kept. Dropped copies stay in `dropped` for
  audit.
- **A drained worker abandons its in-flight chunk.** That work is recomputed elsewhere from the
  last checkpoint.
- **Changing the engine-key settings strands queued jobs.** Those settings are the model,
  device, compute type, beam, chunking and library versions. Stranded jobs are failed
  (retryable) after an hour, and there's no in-place re-keying.
- **No forced alignment.** Word timings are Whisper's DTW; per-segment alignment is the v2 path.
- **The body-size cap counts streamed bytes; it doesn't replace an edge limit.** Keep a proxy
  limit (or presigned uploads) in production.
