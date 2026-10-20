# Transcription pipeline

Audio or video file in, timestamped transcript out, as a CLI or as an HTTP service with a durable
job queue. It reads every common format, handles files from seconds to hours long (memory stays
flat; the default cap is 4 hours, `STT_MAX_DURATION_SECONDS`), and post-processes the transcript
for downstream use: hallucination filtering, optional PII redaction, captions, and LLM-ready
chunks. Every segment carries start and end times measured from the start of the audio.

It runs Whisper (`large-v3-turbo` in production, `small` for a cheap CPU demo) through
faster-whisper, and was tested on real speech: LibriSpeech and real earnings calls.

```bash
uv sync                                                  # Python 3.12; no system ffmpeg needed
uv run stt transcribe meeting.mp3 --format srt           # CLI
uv run stt serve --with-worker                           # API on :8000 plus one worker process
curl -F file=@meeting.mp3 localhost:8000/v1/jobs         # long files: 202 + a job to poll
```

---

## What's in the repo

```
src/stt_pipeline/
  audio.py              format sniffing, streaming decode to 16 kHz mono float32, duration limits
  sandbox.py            runs the upload probe in a disposable child process (CPU/memory limits)
  chunking.py           silence-aware ~10 min chunks (bounded memory, checkpoint unit)
  engines/
    base.py             the engine interface, built around transcribe(pcm) -> timestamped segments
    faster_whisper_engine.py   VAD clips, batched Whisper, continuation, safety net
    coverage.py         range maths for the coverage check
    devices.py          device / compute-type / CPU-thread defaults
    openai_compat.py    hosted engine (OpenAI whisper-1, Groq) behind the same interface
    mock.py             deterministic fake engine for tests
  pipeline.py           decode -> chunk -> transcribe -> post-process, with checkpoints
  postprocess.py        timestamp sanity, quality gate, sentence re-segmentation, PII redaction
  formats.py            json, jsonl, txt, srt, vtt, chunks (RAG)
  whisper_thresholds.py Whisper's reference failure thresholds, shared by engine and gate
  jobs.py               SQLite job queue: leases, idempotency, retention
  worker.py             queue consumer: heartbeat, retries, drain, stall detection
  api.py                FastAPI service
  cli.py · config.py · schemas.py · errors.py · languages.py
tests/                  150 tests, mostly one file per module (147 fast on a fake engine, 3 on a real model)
scripts/
  eval_real_data.py     LibriSpeech + Earnings-22 evaluation (results in docs/)
  eval_synthetic.py     synthetic-audio checks: silence, formats, memory
  eval_common.py        shared helpers for the evaluation scripts
  output_snapshot.py    snapshot and diff transcription output (behaviour-preservation check)
  make_fixtures.py      synthetic test audio with exact ground truth (macOS TTS)
  make_examples.sh      regenerates examples/ from the fixtures with the default `small` model
  smoke_e2e.py          real API + worker processes: SIGTERM mid-job, resume, idempotency
docs/
  RESEARCH.md           sourced research behind the decisions below
  eval_real_data.md     real-speech results (+ .json raw records)
  eval_synthetic.md     synthetic-audio results
examples/               real outputs: SRT, VTT, JSON, RAG chunks, a rejected hostile upload
Dockerfile · docker-compose.yml · .github/workflows/ci.yml (lint, format check, fast tests)
```

## Architecture

```
               ┌──────────── API process (never loads the model) ─────────────┐
  upload ────▶ │ size cap ─▶ stream to disk + sha256 ─▶ sniff + probe (child)  │──▶ most bad input: 4xx at upload
               │ ─▶ job row (id = hash of audio + options + engine config)     │
               │ GET /v1/jobs/{id}/result?format= renders json/jsonl/txt/srt/  │
               │ vtt/chunks from the stored transcript                         │
               └───────────────────────────────┬───────────────────────────────┘
                                               │ job table: lease, heartbeat, attempts
               ┌───────────────────────────────▼──── worker process (model loaded once) ───────┐
               │ stream-decode ─▶ 16 kHz mono float32 (or one stream per channel for calls)    │
               │ ─▶ ~10 min chunks cut at pauses (checkpoint after each)                        │
               │     per chunk: VAD ─▶ clips ≤30 s ─▶ batched Whisper with timestamps           │
               │                ─▶ continue clips Whisper stopped early ─▶ coverage/loop check   │
               │ ─▶ post-processing chain ─▶ transcript JSON (stored)                            │
               └───────────────────────────────────────────────────────────────────────────────┘
```

The API and the workers are separate processes. Receiving an upload is mostly waiting on the
network; transcribing it keeps a CPU or GPU busy for minutes and needs gigabytes of memory for
the model. Keeping them apart means a decoder crash or a slow job never takes the API down, and
workers scale on their own (one per GPU). Input problems that only show up while decoding (the
true duration of a VBR file, the channel count in split mode, a stream that won't decode) fail
the job with the same 4xx code, without retries.

The model sits behind a small interface (`engines/base.py`) built around
`transcribe(pcm ≤ 10 min) -> segments`, plus metadata, an id used in checkpoint fingerprints, a
warm-up hook and an optional maximum chunk size. The pipeline owns everything else (decoding,
chunking, checkpoints, post-processing), so moving to a hosted API is a config change, another
model is one more adapter, and the whole system runs against a fake engine in tests.

---

## Design decisions

### Model: Whisper large-v3-turbo on faster-whisper

On the Open ASR Leaderboard several open models beat turbo on English word error rate, and each
has a catch for this pipeline. Cohere Transcribe and canary-qwen return no timestamps (canary-qwen
also only takes ≤40 s clips), Qwen3-ASR needs a second aligner model for them, Parakeet-TDT covers
25 European languages and is worse than turbo on long-form earnings calls, distil-large-v3.5 is
English-only, and full large-v3 is slightly better on short clips (7.16 vs 7.64) but worse on
long-form audio (13.92 vs 13.60) and much slower. Turbo has segment and word timestamps, about 99
languages, an MIT licence, and faster-whisper runs it with the same code on a laptop CPU or a
CUDA GPU, without PyTorch.

Self-hosting has an estimated marginal cost of $0.01–0.03 per audio hour at good GPU utilisation
(from published benchmarks; not measured here), against $0.04 (Groq) to $0.36 for hosted APIs.
Including always-on GPUs and ops time, it breaks even at roughly 5–20k audio hours a month
against Deepgram/AssemblyAI pricing (higher against Groq), and it is required at any volume when
audio must stay on the operator's own infrastructure. A hosted adapter (`openai-compatible`:
OpenAI whisper-1, Groq) replaces the local model with one setting (`STT_ENGINE`). It refuses
OpenAI's newer `gpt-4o-transcribe` models, which return no timestamps. Sources:
[docs/RESEARCH.md](docs/RESEARCH.md).

### One decode path for every format

Whisper works on raw 16 kHz mono samples, so every input is decoded the same way: through
FFmpeg's libraries via PyAV, which bundles FFmpeg in its wheels (no system install, same build on
macOS and Linux), streaming frame by frame rather than loading the file into memory. It reads what
people actually upload: WAV, FLAC, MP3, iPhone M4A, browser WebM/Opus, 8 kHz telephony, and the
soundtrack of MP4/MOV (AMR is allowlisted too). soundfile can't open M4A or WebM, pydub is
unmaintained and shells out to ffmpeg anyway, and torchaudio now needs system FFmpeg.

- **Checked before anything is queued.** File contents are sniffed (extensions and MIME types are
  often wrong), then probed in a disposable child process with CPU and memory limits, under a
  demuxer and protocol allowlist that also applies when the worker decodes. FFmpeg ships 20+ CVEs
  a year, and a crafted playlist can make it fetch internal URLs. A crash or hang while probing
  costs one child process; during the full decode it costs one worker, and a file that does it
  three times is quarantined.
- **Limits are enforced by counting decoded samples.** Header durations are often wrong (VBR MP3,
  browser WebM), and a few MB of FLAC silence can decode to hours.
- **Float32 end to end, stereo averaged in numpy.** FFmpeg's default float downmix made stereo
  files 3 dB louder than the same audio in mono; averaging keeps every format at the same level.
- **Split-channel mode** for call recordings transcribes each channel separately and labels the
  segments `channel_0`, `channel_1`: exact speaker labels at no extra cost when each party is
  recorded on their own channel.
- **No denoising.** Published studies found speech enhancement before Whisper makes accuracy
  worse; Whisper was trained on noisy audio and copes better with noise than with denoiser
  artefacts.

### Long audio: two levels of chunking

Whisper sees 30 seconds at a time, so long-form transcription is a question of how to cut.

1. **~10-minute chunks, cut at a pause.** The file is stream-decoded and cut in the middle of the
   longest pause a voice detector finds in the minute before each 10-minute mark (if there is no
   pause of at least 0.1 s, at the quietest 50 ms). Normally no word straddles a cut, so there is
   nothing to overlap or stitch. Memory stays flat however long the file is, and each chunk is a
   checkpoint: a crashed job resumes from the last finished chunk.
2. **Speech clips of up to 30 seconds inside each chunk.** Silero VAD finds the speech; regions
   are merged into clips but never across a pause longer than 1.5 s, and the audio between merged
   regions is kept so timestamps stay on the real timeline. Long silences never reach the model,
   which is the main defence against Whisper inventing text over silence and music.

Clips are decoded in batches with Whisper's **timestamp tokens turned on**. faster-whisper's
batched mode turns them off by default and returns one "segment" per VAD chunk of up to 30 s of
speech; on real earnings calls those segments spanned 30–55 s of the recording, and a typical
sentence started about 8 seconds from the nearest segment start.

### Continuing clips Whisper stopped early

Whisper sometimes ends a window with two timestamps and no more text (`…dismay upon<|5.92|><|5.92|>`),
meaning "start the next window at 5.92 s". Sequential Whisper does exactly that. With timestamps
on, faster-whisper's batched pipeline computes that position and then discards it (its own source
shows the value is never used), so the rest of the clip is silently never decoded. A thin proxy
around the model (`_SeekAware`) records where each clip stopped, and the leftovers are decoded in
up to three further batched passes. Leftovers are decoded without the speech before them, so
their segments are flagged `continued` and held to a stricter confidence rule by the quality
gate. Because the proxy hooks a private faster-whisper method, the dependency is capped at the
verified minor version and the engine checks the hook exists at startup.

### A narrow safety net instead of slow decoding everywhere

Batched decoding skips Whisper's temperature fallback, and nothing checks that every stretch of
speech produced text. So after decoding, two checks run: every VAD speech region must be covered
by segments (allowing for timestamp jitter), and no segment may look like a repetition loop
(gzip compression ratio > 2.4). Failing regions are re-decoded, first batched again, then, if
they still fail, with faster-whisper's sequential decoder using Whisper's reference
temperature-fallback ladder and thresholds. Batched speed is kept everywhere else.

### Post-processing for downstream use

The canonical JSON transcript is the system of record; every other format is rendered from it.
Processors are small functions run in order:

- **Timestamp sanity:** `0 ≤ start ≤ end ≤ duration`, words inside their segment, impossible
  speaking rates flagged.
- **Quality gate:** re-applies Whisper's reference thresholds that batched decoding skips
  (silence, repetition), drops repetition loops that span several segments, drops known
  hallucination phrases ("thanks for watching") only when the model's own scores also doubt them,
  and drops `continued` segments below Whisper's −1.0 log-prob line. Nothing the gate removes
  vanishes silently: those segments are kept in `transcript.dropped` with a reason. (Text the
  engine's safety net re-decodes is replaced, and counted in `transcript.diagnostics`.)
- **Sentence re-segmentation** from word timings, whenever word timestamps are requested.
- **PII redaction** (opt-in): email, phone, card and SSN in both text and word timings, including
  spoken forms like "john dot smith at example dot com".
- **Formats:** JSON, JSONL, plain text, SRT and WebVTT captions (≤42 characters × 2 lines, ≤7 s
  per cue; captions also add the audio stream's start offset so they line up with a video), and
  ~1.5k-character chunks for embeddings or an LLM, each with start and end times so anything
  generated from them can cite the moment in the recording.

### Running as a service

Short clips (≤5 min) can use `POST /v1/transcriptions`, which waits for the result and returns
`202` plus a job link if it overruns, never a 504. Long files go through `POST /v1/jobs` and are
polled. The queue is a database table with leases. It is SQLite here, so the API and workers share
one host; the same design maps one-to-one onto Postgres `SELECT … FOR UPDATE SKIP LOCKED` for
multiple hosts, which is not implemented yet.

- **Idempotent:** the job id is a hash of the audio, options and engine config, so a retry or
  double submit returns the existing job.
- **Leases and heartbeats:** a dead worker's lease expires and another worker resumes the job from
  its chunk checkpoints.
- **Retries and poison jobs:** transient failures retry with backoff; client errors fail at once
  with their 4xx; a file that crashes every worker is quarantined after three attempts.
- **Graceful drain and stall detection:** on SIGTERM the worker hands its job back immediately;
  a worker stuck in a native call stops renewing its lease and exits so it gets restarted.
- **Engine routing:** each job carries the engine key of the API that accepted it, and workers
  only claim jobs with their own key, so two deployments with different model configs can share
  one job table without taking each other's work.

Celery, RQ and similar task queues aren't used. RQ (180 s), Arq (300 s) and Dramatiq (600 s) kill
long tasks by default, Celery with `acks_late` on Redis re-delivers them after the 1 h visibility
timeout, and any of them would add a second source of truth for job state.

### Research behind these decisions

The decisions above rest on research across six areas (models, hosted APIs, audio ingestion,
long-form strategies, timestamps and post-processing, serving), with each load-bearing claim
checked against primary sources (60 checked: 52 confirmed, 8 corrected, none refuted) and key
library behaviour verified by reading the installed source, such as the PyAV 19 break in
faster-whisper's own audio loader. Notes and sources: [docs/RESEARCH.md](docs/RESEARCH.md).

---

## Testing and results

All runs: Apple M1 Pro CPU, int8, with the same number of CPU threads for every approach compared.
The synthetic and end-to-end results below come from the code as published. The real-speech
evaluation ran on the same decoding logic just before a final code reorganisation, which was
checked to give identical output (see "Automated tests"). Real-speech testing is also what found
the discarded seek position, a coverage check that needed to tolerate timestamp jitter, and the
garbled short leftovers that the weak-continuation rule now drops.

### Automated tests

150 tests, mostly one file per module. 147 run in about 12 seconds against a fake engine; 3 run a
real Whisper model (`tiny.en`) on the synthetic fixtures (`pytest -m slow`). All 150 pass. CI runs
lint, a format check and the fast suite on every push; 6 fast tests need the macOS-generated audio
fixtures and skip there.

| area | tests | what they pin down |
|---|---|---|
| audio | 28 | every supported container and codec decodes to the same canonical PCM; hostile uploads (empty file, text renamed `.mp3`, HLS and ffconcat playlists that would fetch URLs or local files) get precise 4xx codes; duration limits enforced while decoding; resampler handles a mid-stream rate change |
| chunking | 3 | chunks are lossless and cut in silence; wall-to-wall speech falls back to the quietest point |
| engine | 11 | clip merging; continuation of early stops (and the switch to turn it off); coverage and loop checks; re-decodes replace whole segments; jitter alone never triggers a re-decode |
| pipeline | 10 | timestamps survive chunking; crash then resume from checkpoints; language pinned after the first speech chunk; split-channel mode; provenance recorded |
| post-processing | 17 | quality-gate rules (silence, loops within and across segments, phrase filter, weak continuations); PII redaction in text, word timings and the audit list, including spoken forms and numbers split across pauses |
| formats | 11 | SRT/WebVTT cue limits and spec validity, speaker labels, start offsets, citation ranges in chunks |
| job queue and worker | 14 + 9 | no double claims under concurrency; lease expiry, takeover and lost leases; poison-job quarantine; retries with backoff; drain releases without counting an attempt; stall detection; retention |
| API | 19 | every endpoint and error code; sync timeout becomes a job; idempotent replay; body-size cap including chunked uploads; sync limit at intake and in the worker; uploaded filenames never exposed |
| config, CLI, schemas | 12 + 2 + 1 | which settings change the job-routing key and which don't; CLI output and exit codes; language validation |
| hosted engine | 10 | refuses text-only and local model names; retries rate limits honouring `Retry-After`; permanent vs transient errors |
| real model (slow) | 3 | each of the 11 format variants transcribes within 15% WER of the reference; pure noise gives no segments; a dual-channel call keeps its turn order |

`scripts/output_snapshot.py` captures transcription output for 34 inputs (an earnings call, 20
LibriSpeech clips, a split-channel call with PII redaction and word timings, a meeting with word
timings, and the 11 format variants) and diffs two captures: segments, text, timestamps, scores,
flags, dropped segments and diagnostic counts. Decoding is deterministic on CPU, so two captures
of the same code are identical; the final reorganisation produced no differences.

### Real speech: LibriSpeech and three earnings calls

Word error rate uses Whisper's English text normaliser (the Open ASR Leaderboard convention).
LibriSpeech is a random sample of 200 test-clean clips (21 speakers) and 100 test-other clips (17
speakers), each run through the model alone and through the pipeline, with a paired bootstrap
95% confidence interval on the difference. Earnings-22 is three real earnings calls with
accented speakers (companies in South Africa, China and Chile; 80 minutes), with timing scored
against the dataset's forced-aligned word times on China and Chile (the South Africa alignment
is unusable).

**The pipeline doesn't change accuracy measurably** (every confidence interval includes zero):

| model | LibriSpeech | model alone | through the pipeline | difference (95% CI) | leaderboard (full set, HF transformers; context only) |
|---|---|---|---|---|---|
| large-v3-turbo | test-clean | 2.19% | 2.19% | 0.00 pt (−0.24 to +0.25) | 2.10% |
| large-v3-turbo | test-other | 4.43% | 4.63% | +0.19 pt (−0.15 to +0.57) | 4.24% |
| small | test-clean | 3.66% | 3.58% | −0.08 pt (−0.49 to +0.30) | 3.05% (small.en) |
| small | test-other | 8.74% | 8.93% | +0.19 pt (−0.61 to +1.07) | 7.25% (small.en) |

**On the earnings calls its accuracy is within ±0.6 pt of faster-whisper's defaults (better with
turbo, worse with `small`), and it fixes the timestamps,** for 1.5–1.7× the compute:

| model | approach | WER (3 calls) | segments > 30 s (3 calls) | sentence-start error, median (China / Chile) | sentence starts within 1 s (China / Chile) | CPU s per audio s |
|---|---|---|---|---|---|---|
| large-v3-turbo | faster-whisper batched defaults | 12.75% | 63 | 7.97 / 7.86 s | 10.2 / 8.7% | 0.453 |
| large-v3-turbo | **this pipeline** | **12.25%** | **0** | **0.40 / 0.80 s** | **64.5 / 54.4%** | 0.757 (1.67×) |
| small | faster-whisper batched defaults | 14.06% | 63 | 7.97 / 7.86 s | 10.2 / 8.7% | 0.247 |
| small | **this pipeline** | 14.62% | **0** | **0.29 / 0.77 s** | **71.1 / 56.7%** | 0.381 (1.54×) |

- The defaults return blocks of ~30 s (up to 55 s), so a sentence typically starts ~8 s from the
  nearest segment start. The pipeline's segments have a median of 4–6 s; half of them start within
  ~0.1 s of when their first word was spoken, and 9 in 10 within 0.5–0.8 s.
- Per call, the WER difference against the defaults goes both ways (−1.5 to +1.5 pt), so three
  calls can't separate the pooled −0.50 pt (turbo) and +0.56 pt (`small`). The pipeline misses
  fewer words (deletions 7.66 → 6.57% with turbo) but writes slightly more wrong or extra ones.
- Turbo is more accurate than `small` on every set (and on every call through the pipeline), for
  about 2× the CPU.
- **Continuation, measured on and off:** without it the safety net still catches most early
  stops, but re-decodes 8% (turbo) to 15% (`small`) of the speech and leaves more words missing.
  With it, WER is 12.77% → 12.25% (turbo) and 14.92% → 14.62% (`small`), deletions are lower on
  all six call/model pairs, and nothing needs re-decoding, for about the same compute.

Full tables, per-call numbers and the raw records: [docs/eval_real_data.md](docs/eval_real_data.md).

### Synthetic audio: silence, formats, memory

Synthetic speech (macOS TTS, six voices) has exact ground truth for timing and silence, which
real recordings don't.

- **Silence and hold music.** A 14-minute, six-voice meeting has 90 s of silence and 20 s of tone
  in the middle. Sequential, reference-style decoding (faster-whisper without VAD) filled the gap:
  it looped a reworded earlier sentence three times (42 invented words) and stretched real
  sentences into it. faster-whisper's defaults put real words there inside one 144-second
  segment. This pipeline put nothing there, and its segment starts were within 0.16 s of the true
  utterance starts at the median and 0.22 s at p90 (defaults: 6.86 s and 12.67 s). It ran at 1.3×
  the defaults' time.
- **Formats.** Eleven encodings of the same recording (WAV in four sample formats including 8 kHz
  µ-law telephony, MP3, M4A, Ogg, WebM, FLAC, AIFF, and the soundtrack of an MP4) all give the
  same text with `tiny.en`, at 0.0% WER. Five of them hit an early stop, and continuation
  recovered all five without any re-decoding.
- **Memory.** Streaming keeps audio and feature buffers flat at ~124 MB from 1 to 3.2 hours of
  audio; loading the whole file, as faster-whisper's batched path does, grows from 757 to
  2,270 MB (~715 MB per hour).

Full output: [docs/eval_synthetic.md](docs/eval_synthetic.md).

### End to end with real processes

`scripts/smoke_e2e.py` starts the real API and worker processes and drives them over HTTP, with
`tiny.en` and 2-minute chunks. Readiness reports 503 until a worker is up; a sync request returns
captions; an HLS playlist disguised as `.mp3` is refused with 415; an async job is sent SIGTERM at
38% progress and released without counting an attempt; a second worker finishes it, reusing the 3
chunks (of 8) the first one had checkpointed; and submitting the same file again returns the same
job with `Idempotent-Replay: true`. The script asserts completion and checkpoint reuse and prints
the rest.

---

## Running it

```bash
uv sync
uv run stt transcribe call.wav --split-channels --redact-pii --format txt
uv run stt transcribe lecture.mp3 --format vtt -o lecture.vtt --checkpoint-dir .ckpt   # resumable
uv run stt transcribe interview.m4a --model large-v3-turbo --words --format json

uv run stt serve --port 8000          # API: validates, enqueues, serves results; no model
uv run stt worker                     # one per GPU / CPU box; loads and warms the model once
```

| Endpoint | |
|---|---|
| `POST /v1/transcriptions` | multipart `file` plus `language`, `word_timestamps`, `channel_mode=mix\|split`, `redact_pii`, `prompt`, `response_format=json\|jsonl\|txt\|srt\|vtt\|chunks`. ≤5 min of audio; `202` and a job link if it runs past the wait budget |
| `POST /v1/jobs` | the same options except `response_format` (chosen when fetching the result); `202` and `Location`; the same file and options again return the existing job with `200` and `Idempotent-Replay: true` |
| `GET /v1/jobs/{id}` · `GET /v1/jobs/{id}/result?format=…` | status and progress · the transcript (`409 not_ready` until done) |
| `DELETE /v1/jobs/{id}` | deletes audio and transcript; results also expire after `STT_RESULT_TTL_SECONDS` |
| `GET /healthz` · `/readyz` · `/metrics` | liveness · a worker for this engine config is up · Prometheus text |

Errors are typed JSON (`{"error": {"code", "message"}}`) with stable codes, so clients can tell
input problems (4xx, e.g. `unsupported_media`, `audio_too_long`, `invalid_options`) from
server-side failures (5xx). Backpressure (`queue_full`, 503) carries `Retry-After`. Configuration
is through `STT_*` environment variables ([config.py](src/stt_pipeline/config.py)). The
[Dockerfile](Dockerfile) builds one image for both roles with pinned model weights baked in;
[docker-compose.yml](docker-compose.yml) runs the API and a GPU worker.

```bash
uv run pytest                          # fast tests (6 skip until fixtures are generated)
uv run python scripts/make_fixtures.py # synthetic test audio (macOS: uses `say`)
uv run pytest -m slow                  # real-model tests (tiny.en; needs the fixtures)
uv run python scripts/smoke_e2e.py     # real API + worker processes (needs the fixtures)
uv run python scripts/eval_real_data.py download && uv run python scripts/eval_real_data.py run --out docs/eval_real_data.md
uv run python scripts/eval_synthetic.py ablation --model small --out docs/eval_synthetic.md   # also: formats, scaling (needs the fixtures)
```

## Limitations and next steps

- **Small real-speech test:** 300 LibriSpeech clips and three earnings calls, English only.
  Enough to show no detectable accuracy cost and a clear timestamp gain; not enough to pick a
  model for a specific domain. Hallucination over real silence or hold music wasn't measured.
- **Some garbled text gets through:** a 0.7 s leftover piece once became "Thank you very much for
  watching, and I'll see you in the next video." with a confidence above the drop line. The phrase
  filter only drops exact matches, and it once dropped a real, quietly scored "Thank you.".
- **One language per file**, no code-switching. **Regex PII** misses names and addresses.
- **Single host:** the SQLite job table means the API and workers share one machine.
- **Not measured here:** GPU throughput and cost (figures above come from published benchmarks).
  The Docker image and compose file haven't been built or run, so the CUDA path is unexercised.
- **Next:** speaker diarization over the whole file, forced alignment for word-precise timing,
  presigned uploads and Postgres for scale-out, a Parakeet tier for English/EU volume, signed
  webhooks, and a WER regression gate in CI on consented domain audio.
