# Synthetic-audio results

Generated with `scripts/eval_synthetic.py` on the TTS fixtures from `scripts/make_fixtures.py`
(macOS voices with exact ground-truth text and timings), on the final code. CPU int8, Apple M1
Pro, the same number of CPU threads for every approach. Results on real speech are in
[eval_real_data.md](eval_real_data.md).

### Ablation: small, `long_meeting.mp3` (14.3 min, 6 voices, 90 s silence + 20 s tone mid-file)

| approach | wer | segments | start_err_median | start_err_p90 | max_seg_s | words_in_silence | rtf | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|
| A. sequential (reference-style, no VAD) | 0.0313 | 222 | 0.16 | 0.67 | 29.0 | 64 | 0.109 | 1994 |
| B. faster-whisper batched defaults | 0.002 | 25 | 6.86 | 12.67 | 143.9 | 84 | 0.0451 | 3185 |
| C. this pipeline | 0.003 | 148 | 0.16 | 0.22 | 7.6 | 0 | 0.0584 | 3210 |

Text placed inside the non-speech stretch:

- **A. sequential (reference-style, no VAD)**: [485-510] I have a conflict this week. | [510-536] Let's make sure that we freeze new features until the end of the month. | [536-565] Let's make sure that we freeze new features until the end of the month. | [565-590] Let's make sure that we freeze new features until the end of the month. | [590-606] I think the billing service is in good shape, but we still need to finish the
- **B. faster-whisper batched defaults**: [482-625] Can we move the review of the billing service to Tuesday? I have a conflict this week. I think the billing service is in
- **C. this pipeline**: (none)

### Format matrix: tiny.en

`clips continued` = clips where Whisper stopped early and the engine decoded the rest. `ranges flagged for re-decode` = coverage gaps sent to the fallback.

| file | WER | clips continued | ranges flagged for re-decode |
|---|---|---|---|
| meeting_short.wav | 0.0% | 0 | 0 |
| meeting_short_f32_48k.wav | 0.0% | 1 | 0 |
| meeting_short_s24.wav | 0.0% | 0 | 0 |
| meeting_short_8k_mulaw.wav | 0.0% | 1 | 0 |
| meeting_short.mp3 | 0.0% | 0 | 0 |
| meeting_short.m4a | 0.0% | 0 | 0 |
| meeting_short.ogg | 0.0% | 0 | 0 |
| meeting_short.webm | 0.0% | 1 | 0 |
| meeting_short.flac | 0.0% | 1 | 0 |
| meeting_short.aiff | 0.0% | 1 | 0 |
| meeting_short_video.mp4 | 0.0% | 0 | 0 |

### Memory scaling with duration (audio + features only, model excluded)

`array_peak_mb` = peak bytes held in Python/numpy buffers (tracemalloc, exact). RSS is shown for
completeness but is unreliable on macOS (compressed pages are not counted).

| approach | hours | windows | array_peak_mb | peak_rss_mb | wall_s |
|---|---|---|---|---|---|
| whole file (faster-whisper batched) | 1.06 | 124 | 757 | 1695 | 20.6 |
| this pipeline (streamed) | 1.06 | 127 | 123 | 754 | 21.8 |
| whole file (faster-whisper batched) | 2.12 | 246 | 1513 | 2864 | 41.5 |
| this pipeline (streamed) | 2.12 | 253 | 124 | 777 | 44.2 |
| whole file (faster-whisper batched) | 3.18 | 369 | 2270 | 2413 | 63.0 |
| this pipeline (streamed) | 3.18 | 381 | 124 | 754 | 66.6 |

