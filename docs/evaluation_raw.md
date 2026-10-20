### Ablation: small, `long_meeting.mp3` (14.3 min, 6 voices, 90 s silence + 20 s tone mid-file)

| approach | wer | segments | start_err_median | start_err_p90 | max_seg_s | words_in_silence | rtf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|
| A. sequential (reference-style, no VAD) | 0.0313 | 222 | 0.16 | 0.67 | 29.0 | 64 | 0.1007 | 1782 |
| B. faster-whisper batched defaults | 0.002 | 25 | 6.86 | 12.67 | 143.9 | 84 | 0.0551 | 2716 |
| C. this pipeline | 0.003 | 147 | 0.16 | 0.22 | 7.6 | 0 | 0.0807 | 3043 |

Text placed inside the non-speech stretch:

- **A. sequential (reference-style, no VAD)**: [485-510] I have a conflict this week. | [510-536] Let's make sure that we freeze new features until the end of the month. | [536-565] Let's make sure that we freeze new features until the end of the month. | [565-590] Let's make sure that we freeze new features until the end of the month. | [590-606] I think the billing service is in good shape, but we still need to finish the
- **B. faster-whisper batched defaults**: [482-625] Can we move the review of the billing service to Tuesday? I have a conflict this week. I think the billing service is in
- **C. this pipeline**: (none)

### Model comparison (this pipeline, CPU int8, Apple M1 Pro)

| model | clip | wer | start_err_median | words_in_silence | fallback_decode | segments | rtf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|
| tiny.en | meeting_short.wav | 0.0 | 0.17 | 0 | 0 | 4 | 0.0355 | 612 |
| tiny.en | meeting_short_8k_mulaw.wav | 0.0 | 0.17 | 0 | 1 | 5 | 0.0322 | 618 |
| tiny.en | long_meeting.mp3 | 0.0139 | 0.17 | 0 | 17 | 125 | 0.028 | 1482 |
| tiny.en | silence_noise.wav |  |  |  |  | 0 | 0.004 | 338 |
| small | meeting_short.wav | 0.0 | 0.15 | 0 | 0 | 8 | 0.2063 | 1506 |
| small | meeting_short_8k_mulaw.wav | 0.0106 | 0.15 | 0 | 1 | 7 | 0.299 | 1586 |
| small | long_meeting.mp3 | 0.003 | 0.16 | 0 | 1 | 147 | 0.1148 | 2896 |
| small | silence_noise.wav |  |  |  |  | 0 | 0.0044 | 894 |
| large-v3-turbo | meeting_short.wav | 0.0 | 0.17 | 0 | 3 | 7 | 0.3694 | 2233 |
| large-v3-turbo | meeting_short_8k_mulaw.wav | 0.0 | 0.17 | 0 | 0 | 5 | 0.2438 | 2508 |
| large-v3-turbo | long_meeting.mp3 | 0.004 | 0.18 | 0 | 0 | 148 | 0.1307 | 3890 |
| large-v3-turbo | silence_noise.wav |  |  |  |  | 0 | 0.0055 | 2181 |

### Long file end to end: `fixtures_60min/long_meeting.mp3` (64 min), small

| approach | wer | start_err_median | start_err_p90 | max_seg_s | dropped | fallback_decode | rtf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|
| this pipeline (streamed, 10 min chunks) | 0.0062 | 0.17 | 0.2 | 12.6 | 0 | 8 | 0.1036 | 3120 |
| faster-whisper batched defaults (whole file) | 0.0148 | 7.4 | 13.77 | 143.2 |  |  | 0.0593 | 3145 |

### Memory scaling with duration (audio + features only, model excluded)

`array_peak_mb` = peak bytes held in Python/numpy buffers (tracemalloc, exact). RSS is shown for
completeness but is unreliable on macOS (compressed pages are not counted).

| approach | hours | windows | array_peak_mb | peak_rss_mb | wall_s |
|---|---|---|---|---|---|
| whole file (faster-whisper batched) | 1.06 | 124 | 757 | 1761 | 22.0 |
| this pipeline (streamed) | 1.06 | 127 | 123 | 682 | 23.2 |
| whole file (faster-whisper batched) | 2.12 | 246 | 1513 | 3012 | 43.4 |
| this pipeline (streamed) | 2.12 | 253 | 124 | 762 | 45.3 |
| whole file (faster-whisper batched) | 3.18 | 369 | 2270 | 1984 | 67.8 |
| this pipeline (streamed) | 3.18 | 381 | 124 | 693 | 73.0 |

