# Research notes: what is "best for production" in speech-to-text (October 2026)

How this was produced: six parallel research passes (open-source engines, hosted APIs and
build-vs-buy, audio ingestion, long-form strategy, timestamps and post-processing, serving
architecture), each followed by an independent fact-check against primary sources (GitHub
source and release notes, PyPI, vendor pricing pages, model cards, papers). The fact-check
confirmed 52 of 60 load-bearing claims, corrected 8 (mostly precision: upload caps, CVE fix
versions, leaderboard revisions) and refuted none. The corrected versions are used below.
Several claims were also re-verified by hand against the installed library code while building
(marked ✔ — *verified in code here*).

---

## 1. Engines

### What the leaderboards say, and how much to trust them

The HF Open ASR Leaderboard was **re-based in 2026**. It now uses cleaned datasets and private
sets, so current averages can't be compared with the 2025 numbers printed on model cards. Its
RTFx figures for Whisper come from HF transformers at batch 64 on a datacenter GPU, not from
faster-whisper. Use them for ranking, not for capacity planning.

| Model (open weights) | Short-form avg WER | Long-form avg WER | Timestamps | Languages | Licence |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | 5.40 | – | only via separate aligner (≤5 min) | multi | Apache-2.0 |
| Cohere Transcribe 03-2026 | 5.77 | 12.23 (best) | **none** | EN | Apache-2.0 |
| canary-qwen-2.5b | 5.94 | 14.08 | none; ≤40 s clips | EN | CC-BY-4.0 |
| parakeet-tdt-0.6b-v3 | 6.18 | 13.37 | char/word/segment (opt-in) | 25 EU | CC-BY-4.0 |
| distil-large-v3.5 | 6.76 | – | yes | EN | MIT |
| whisper-large-v3 | 7.16 | 13.92 | segment + word (DTW) | 99 | Apache-2.0/MIT |
| whisper-large-v3-turbo | 7.64 | 13.60 | segment + word (DTW) | 99 | MIT |

Per long-form dataset, Parakeet's better average comes from CORAAL. On earnings calls turbo is
better: Earnings-21 9.84 vs 9.96, Earnings-22 13.10 vs 14.58 (leaderboard long-form CSV).

LibriSpeech per split on the same leaderboard (HF transformers; used only as context for this
repo's own runs in docs/eval_real_data.md): large-v3-turbo 2.10 test-clean / 4.24 test-other;
small.en 3.05 / 7.25. Fetched 2026-10-20.

Short-form figures are from the 17-10-2026 leaderboard Space. Long-form figures are from the
same Space's long-form tab (Earnings-21/22 + CORAAL); the repo CSV and the paper (v4) report
different averages, so cite one column consistently.

**Conclusions:**
- Whisper is no longer the English accuracy leader.
- Parakeet-TDT is roughly 7–13× faster on GPU at similar or better English WER. But it is
  worse than Whisper on some EU languages (e.g. PT, NL on the multilingual tab) and needs a
  NeMo/torch stack.
- Several "better WER" models have **no timestamps** (Cohere, canary-qwen) or ≤40 s windows.
  That disqualifies them for this brief no matter how good their WER is.

### Runtimes for Whisper weights

| Runtime | Status (Oct 2026) | Why / why not |
|---|---|---|
| **faster-whisper** (CTranslate2) | 1.2.1 (2025-10-31); master active; CT2 4.8.3 | ≈4× openai-whisper (README benchmark: 2m23s vs 1m03s, large-v2 fp16). int8/fp16. Same code on CUDA and CPU. Batched VAD pipeline. No torch. **Chosen.** |
| openai-whisper | 20250625 | Reference; needs system ffmpeg; no batching; slowest. |
| whisper.cpp | v1.9.5 | Metal/CoreML/CUDA/Vulkan, single binary. CLI reads WAV/MP3/FLAC/OGG; community Python bindings. Good edge/Mac option. |
| mlx-whisper / mlx-audio | 0.4.3 / 0.6.0 | Fastest on Apple Silicon, but Mac-only, so no dev/prod parity. |
| WhisperX | 3.8.6 | faster-whisper + wav2vec2 alignment + pyannote. torch pins, HF token for diarization. Add only when word precision or speakers are required. |
| HF transformers pipeline | 5.x | Chunk+stride stitching is "experimental" for seq2seq (distil-whisper card: 10.0 → 11.0 WER). Loads the whole file. |

**faster-whisper gotchas** (✔ all verified in the 1.2.1 source here):
- `BatchedInferencePipeline` defaults to `without_timestamps=True`, so each "segment" is a whole
  VAD chunk of up to 30 s.
- Batched mode uses only `temperature[0]` and hard-codes `condition_on_previous_text=False`.
  `compression_ratio_threshold`, `log_prob_threshold` and `no_speech_threshold` are accepted
  but drive nothing.
- `decode_audio()` calls `av.open(..., metadata_errors=...)`, which **PyAV 19 removed**: every
  file raises `TypeError`. The fix is on master, unreleased.
- CTranslate2 has **no Metal backend**: on a Mac it is CPU-only (int8 via Ruy).
- The `large-v3-turbo`/`turbo` alias resolves to a third-party repo that now redirects to
  `dropbox-dash/faster-whisper-large-v3-turbo` (sha `0a363e91…`, verified via the HF API). Pin the
  revision.

## 2. Hosted APIs and build vs buy

| Provider | Batch $/audio-hour | Timestamps | Upload limit | Notes |
|---|---|---|---|---|
| Groq whisper-large-v3-turbo | 0.04 | segment+word (verbose_json) | 25 MB direct (100 MB via URL on dev tier) | same model as self-host |
| Google STT v2 dynamic batch | 0.18 | word (Chirp 3: ≤20 min with word ts) | GCS input | fulfilled within 24 h |
| gpt-4o-mini-transcribe | 0.18 | **none** | 25 MB | text only |
| AssemblyAI Universal-3.5 Pro | 0.21 (+0.02 diarization) | word | 2.2 GB upload / 5 GB URL, 10 h | EU endpoint, self-serve BAA |
| ElevenLabs Scribe v2 | 0.22 | word | ~3 GB, 10 h | ZDR enterprise-only |
| Deepgram Nova-3 | 0.258 | word | 2 GB; 10-min processing timeout | diarization incl., self-host option |
| OpenAI whisper-1 | 0.36 | segment+word | 25 MB | only timestamped OpenAI path (besides -diarize) |
| AWS Transcribe | ~0.36 | word | 2 GB, 8 h | HIPAA-eligible |
| Speechmatics | 0.24–0.75 | word | 1 GB | on-prem containers |

**Critical capability gap:** OpenAI's newest transcription models (`gpt-4o-transcribe`,
`gpt-4o-mini-transcribe`, `gpt-transcribe`) return **text only**. An adapter that "just calls
OpenAI" would silently drop the timestamps this brief requires. This repo's hosted adapter
refuses those models at construction.

**Self-host economics** (estimates; GPU throughput is not measured here):
- Instance prices: L4 (AWS g6.xlarge) $0.805/h on-demand; T4 $0.526/h; A10G $1.006/h.
- Published single-stream turbo throughput: ~31× real time on an L4 (RunPod, int8). Batching
  adds about 3.7× (faster-whisper README, large-v2).
- Marginal cost is therefore about **$0.01–0.03 per audio-hour** at good utilization.
- Break-even:
  - With two always-on L4s plus 0.25 FTE of ops (~$5.4k/month fixed): about 20k audio-h/month
    against Deepgram/AssemblyAI pricing.
  - If an existing platform team absorbs ops: about 5k audio-h/month.

**Decision rule:**
- **Buy** below ~5–20k audio-h/month, or when accuracy on hard audio matters more than cost.
  The best commercial engines are ~20–30% relatively better on WER than self-hosted turbo.
- **Self-host** above that volume, or when data residency or privacy forbids sending audio out.
  The latter can force it at any volume.
- **Either way, keep the engine behind an adapter.** Then the decision is a config switch, and a
  hosted engine can serve as overflow or fallback.

## 3. Audio ingestion

**Decoder options:**
- **PyAV 19** bundles FFmpeg 9.0.2 in its wheels. It covers MP3, AAC, Vorbis, Opus, FLAC,
  AMR-NB/WB and µ/A-law decoders, and libmp3lame/libopus encoders (✔ checked here). It resamples
  via libswresample (no soxr).
- **soundfile/libsndfile** can't open AAC/M4A, WebM or AMR, which are exactly what phones and
  browsers upload.
- **pydub** last shipped in 2021 and imports `audioop`, which Python 3.13 removed.
- **torchaudio** I/O is now a thin wrapper over TorchCodec, which needs system FFmpeg libraries.

**Decode to float32 directly.** Going through s16 (as openai-whisper and faster-whisper do)
re-quantizes 24-bit input and clips float WAVs above 0 dBFS.

**Validation:**
- Header durations lie: VBR MP3 without a Xing header gets a bitrate estimate, and
  MediaRecorder WebM often has no Duration. So enforce limits on **decoded sample counts**.
- A few MB of FLAC or Opus silence can decode to hours, so a byte cap alone isn't enough.

**Security:**
- FFmpeg logs 23–25 CVEs per year.
- Playlist demuxers are the classic SSRF and local-file-read surface (CVE-2023-6601, fixed in
  6.1; CVE-2023-6602 and -6605, fixed in 8.0 with backports).
- The primary control is `format_whitelist` (demuxer allowlist). `protocol_whitelist` alone is
  insufficient. Belt-and-braces: decode in a disposable, resource-limited process.

**Pre-processing:**
- **Don't denoise.** Two 2025–26 studies found speech enhancement *raised* WER for Whisper and
  Parakeet in every configuration tested.
- No published evidence for loudness normalization either way. Whisper's log-mel clamps
  relative to the clip maximum, so gain is mostly a constant offset. Leave it off; A/B test
  before adding.

## 4. Long audio

- **Reference algorithm** (openai-whisper): a 30 s window that seeks to the last timestamp
  token, conditioned on previous text, with temperature fallback. It can't batch, and errors
  propagate.
  - Whisper paper, Table 7, long-form WER: 11.0 greedy → 10.6 beam → 10.2 +VAD → 10.0
    +conditioning. Conditioning hurt one dataset.
  - Whisper large-v3 hallucinated text on **40.3%** of 301k non-speech inputs. "thank you" was
    24.8% of those outputs and "thanks for watching" 10.3%.
- **VAD cut & merge + batching** (WhisperX, Interspeech 2023): 11.8× faster on TED-LIUM,
  WER 9.70 vs 10.52, fewer repetitions. That baseline is large-v2 greedy without conditioning,
  not the full reference. faster-whisper's `BatchedInferencePipeline` is the same idea
  (README: 3.7× on GPU, 2× on CPU).
- **Memory:** both reference implementations decode the whole file into RAM (3 h ≈ 690 MB float32
  plus an s16 buffer). One user reported 26–27 GB on an 11 h MP3 through non-batched `transcribe`.
  → Stream-decode and shard.
- **Queues:**
  - Defaults kill or re-run long tasks: RQ 180 s, Arq 300 s, Dramatiq 600 s. Celery with
    `acks_late` on Redis re-delivers after the 1 h visibility timeout. SQS caps visibility at 12 h.
  - That argues for shard-sized tasks with idempotent, deterministic outputs whatever the engine.
- **Parakeet** can do 24 min in one pass (full attention, A100-80GB) or about 3 h with local
  attention. That's still no substitute for sharding in a service: you need checkpoints, bounded
  memory and parallelism.

## 5. Timestamps and post-processing

- **Whisper timestamps:**
  - Timestamp tokens are 20 ms resolution but phrase-level accuracy.
  - Word timestamps come from cross-attention DTW, which stretches words across pauses.
  - Published word-boundary accuracy, AMI at a 200 ms collar (P/R): Whisper 78.9/52.1,
    WhisperX 84.1/60.3.
  - Whole-file forced alignment can fail catastrophically on long inputs, so align per segment.
- **Captions:**
  - Netflix rules: ≤42 chars/line, ≤2 lines, ≤7 s, ~20 cps.
  - WebVTT: cue end strictly > start, non-decreasing starts, escape `&`/`<`, never `-->` in
    payloads.
  - SRT: no formal spec; write UTF-8 and 2-digit hours.
- **Whisper scores are not calibrated confidences.** `avg_logprob`/`no_speech_prob` are per
  decode window, so treat them as review triggers, not probabilities.
- **Library notes:**
  - Diarization: pyannote `community-1` (CC-BY-4.0, gated) on the *full* file, then assign each
    word to the speaker it overlaps most.
  - PII: Presidio ("no guarantee"; struggles on unpunctuated or spoken-form ASR text).
  - Inverse text normalization: NeMo ITN (pynini isn't pip-installable on macOS).
  - Avoid stable-ts (archived), whisper-timestamped (AGPL per repo licence) and MMS_FA default
    weights (CC-BY-NC) as production dependencies.
- **Evaluation:** jiwer 4 with Whisper's `EnglishTextNormalizer` (standalone in
  `whisper-normalizer`) is the leaderboard convention.

## 6. Serving architecture

- **Keep the model out of the API process.** One worker process per GPU loads the model once.
  A decoder segfault or CUDA OOM then kills a disposable worker, not the API.
- **CTranslate2:**
  - It releases the GIL, but `BatchedInferencePipeline` keeps mutable state, so don't share it
    across threads.
  - CT2 ≥ 4.6.3 Linux wheels are built for CUDA 12 without cuDNN and dlopen `libcublas.so.12`.
    So use a CUDA 12 base, or a slim base plus the `nvidia-cublas-cu12` wheel with
    `LD_LIBRARY_PATH`. Not CUDA 13 NGC images.
  - int8 vs fp16 speed on GPU is card-dependent (RunPod: int8 slower for large-v3 on 8 of 10
    cards). Benchmark before choosing.
- **Job state:** a DB table works as the queue (Postgres `FOR UPDATE SKIP LOCKED`) with
  lease/heartbeat/attempts. It gives one transactional source of truth, and dead-letter is just
  a status.
- **Health checks:** readiness = "a worker with the model loaded is heartbeating"; liveness =
  "process up".
- **v2:**
  - Uploads go straight to object storage via presigned URLs.
  - Autoscale on queue depth (KEDA).
  - Spot instances with per-chunk checkpoints (AWS gives a 2-minute notice).
  - Standard Webhooks (HMAC) for delivery.
  - Cheap GPUs plus batching beat big GPUs: single-stream large-v3 goes only 15.5× → 24.9× from
    L4 to H100.

## Key sources

Engines and runtimes:
- faster-whisper [repo](https://github.com/SYSTRAN/faster-whisper), [v1.2.1 transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py), [PyAV fix PR #1495](https://github.com/SYSTRAN/faster-whisper/pull/1495)
- [CTranslate2 hardware support](https://opennmt.net/CTranslate2/hardware_support.html) · [parallelism](https://opennmt.net/CTranslate2/parallel.html)
- Open ASR Leaderboard [Space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), [paper](https://arxiv.org/abs/2510.06961), [long-form CSV](https://github.com/huggingface/open_asr_leaderboard/blob/main/scripts/data/en_longform.csv)
- [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) · [canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)

Long-form and quality:
- [WhisperX paper](https://arxiv.org/abs/2303.00747) · [Whisper paper](https://arxiv.org/abs/2212.04356) · [Whisper hallucinations on non-speech](https://arxiv.org/html/2501.11378)
- Denoising hurts ASR: [2512.17562](https://arxiv.org/abs/2512.17562), [2603.04710](https://arxiv.org/abs/2603.04710)

Ingestion and security:
- PyAV [changelog](https://github.com/PyAV-Org/PyAV/blob/v19.0.1/CHANGELOG.rst) · [FFmpeg security](https://ffmpeg.org/security.html) · [FFmpeg format options](https://ffmpeg.org/ffmpeg-formats.html)

Hosted APIs:
- [OpenAI STT guide](https://developers.openai.com/api/docs/guides/speech-to-text) and [SDK params](https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/audio/transcription_create_params.py)
- [Groq STT](https://console.groq.com/docs/speech-to-text) · [Deepgram pricing](https://deepgram.com/pricing) · [AssemblyAI pricing](https://www.assemblyai.com/pricing)
- [Artificial Analysis STT](https://artificialanalysis.ai/speech-to-text) · [RunPod Whisper GPU benchmark](https://www.runpod.io/articles/guides/best-gpu-for-whisper)

Captions and diarization:
- [WebVTT spec](https://www.w3.org/TR/webvtt1/) · [Netflix timed text guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide) · [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

Queues:
- [Celery Redis broker caveats](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html) · [SQS visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
