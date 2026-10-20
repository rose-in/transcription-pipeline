## Real-data evaluation

CPU int8, Apple M1 Pro, every approach with the same number of threads. WER after Whisper's English normaliser (the leaderboard convention). `del` = reference words missing from the output (on Earnings-22 this includes verbatim repeats and false starts the references keep); `ins` = output words not in the reference (not necessarily made up).

### LibriSpeech: does the pipeline change accuracy? (random sample, paired on the same clips)

| model | split | clips / speakers | model alone (plain) | through the pipeline | difference (95% CI) | published, full set, other implementation |
|---|---|---|---|---|---|---|
| small | test-clean | 200 / 21 | 3.66% | 3.58% | -0.08 pt (-0.49 to +0.30) | 3.05 (small.en) |
| small | test-other | 100 / 17 | 8.74% | 8.93% | +0.19 pt (-0.61 to +1.07) | 7.25 (small.en) |
| large-v3-turbo | test-clean | 200 / 21 | 2.19% | 2.19% | +0.00 pt (-0.24 to +0.25) | 2.10 |
| large-v3-turbo | test-other | 100 / 17 | 4.43% | 4.63% | +0.19 pt (-0.15 to +0.57) | 4.24 |

### Earnings-22: three real calls (South Africa 27 min, China 25 min, Chile 28 min)

| model | approach | WER | sub / del / ins | per call (SA / China / Chile) | wall RTF | CPU s per audio s | peak RSS |
|---|---|---|---|---|---|---|---|
| small | batched_default | 14.06% | 5.15 / 7.76 / 1.15 | 11.43 / 14.11 / 17.07 | 0.0426 | 0.247 | 3725 MB |
| small | no_continuation | 14.92% | 5.37 / 8.29 / 1.26 | 12.6 / 14.11 / 18.27 | 0.0748 | 0.396 | 3151 MB |
| small | pipeline | 14.62% | 5.84 / 7.12 / 1.66 | 12.95 / 14.04 / 17.01 | 0.0659 | 0.381 | 3809 MB |
| large-v3-turbo | batched_default | 12.75% | 4.06 / 7.66 / 1.03 | 9.46 / 12.01 / 17.15 | 0.0804 | 0.453 | 4902 MB |
| large-v3-turbo | no_continuation | 12.77% | 4.29 / 7.24 / 1.25 | 10.14 / 12.01 / 16.44 | 0.1327 | 0.749 | 4762 MB |
| large-v3-turbo | pipeline | 12.25% | 4.37 / 6.57 / 1.31 | 10.0 / 11.26 / 15.65 | 0.1366 | 0.757 | 5017 MB |

Segment timing against the forced-aligned reference (China and Chile; the South Africa alignment in the dataset is unusable):

| model | approach | call | segments | median / max segment | segments > 30 s | sentence-start error p50 / p90 | sentence starts within 1 s | boundary precision p50 / p90 |
|---|---|---|---|---|---|---|---|---|
| small | batched_default | China | 48 | 29.5 / 49.6 s | 19 | 7.97 / 14.64 s | 10.2% | 0.24 / 2.45 s |
| small | batched_default | Chile | 53 | 29.3 / 55.1 s | 18 | 7.86 / 15.65 s | 8.7% | 0.19 / 0.63 s |
| small | no_continuation | China | 252 | 5.0 / 14.0 s | 0 | 0.29 / 2.77 s | 71.7% | 0.11 / 0.58 s |
| small | no_continuation | Chile | 292 | 4.8 / 26.0 s | 0 | 0.76 / 3.79 s | 57.8% | 0.09 / 0.77 s |
| small | pipeline | China | 256 | 5.0 / 13.0 s | 0 | 0.29 / 2.77 s | 71.1% | 0.11 / 0.62 s |
| small | pipeline | Chile | 308 | 4.7 / 26.0 s | 0 | 0.77 / 3.56 s | 56.7% | 0.09 / 0.68 s |
| large-v3-turbo | batched_default | China | 48 | 29.5 / 49.6 s | 19 | 7.97 / 14.64 s | 10.2% | 0.26 / 3.0 s |
| large-v3-turbo | batched_default | Chile | 53 | 29.3 / 55.1 s | 18 | 7.86 / 15.65 s | 8.7% | 0.2 / 0.67 s |
| large-v3-turbo | no_continuation | China | 210 | 6.2 / 27.2 s | 0 | 0.38 / 3.43 s | 65.1% | 0.12 / 0.52 s |
| large-v3-turbo | no_continuation | Chile | 270 | 5.0 / 28.1 s | 0 | 0.75 / 3.64 s | 55.9% | 0.13 / 0.87 s |
| large-v3-turbo | pipeline | China | 215 | 6.1 / 27.2 s | 0 | 0.4 / 3.34 s | 64.5% | 0.12 / 0.54 s |
| large-v3-turbo | pipeline | Chile | 281 | 5.0 / 28.1 s | 0 | 0.8 / 3.73 s | 54.4% | 0.13 / 0.83 s |

Safety-net activity (all three calls):

| model | approach | early stops detected | ranges flagged for re-decode | sent to the slow sequential decoder | audio re-decoded | time in safety net | segments dropped by the quality gate |
|---|---|---|---|---|---|---|---|
| small | no_continuation | 88 | 62 | 2 | 637 s | 59 s of 358 s | {'hallucination_phrase': 1} |
| small | pipeline | 88 | 0 | 0 | 0 s | 0 s of 316 s | {'weak_continuation': 5, 'hallucination_phrase': 2} |
| large-v3-turbo | no_continuation | 46 | 39 | 3 | 340 s | 84 s of 636 s | 0 |
| large-v3-turbo | pipeline | 46 | 0 | 0 | 0 s | 0 s of 655 s | {'hallucination_phrase': 1} |
