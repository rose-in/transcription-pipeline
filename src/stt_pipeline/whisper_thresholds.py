"""openai/whisper's reference decode-failure thresholds.

Sequential Whisper re-decodes a window at a higher temperature when its output fails one of these
checks. faster-whisper's batched path skips them, so the engine passes them to its sequential
fallback decoder and postprocess.QualityGate re-applies them to batched output.
"""

COMPRESSION_RATIO_THRESHOLD = 2.4  # gzip ratio above this == repetition loop
LOG_PROB_THRESHOLD = -1.0  # mean token log-prob below this == failed / low-confidence decode
NO_SPEECH_THRESHOLD = 0.6  # no-speech prob above this (with log-prob below LOG_PROB_THRESHOLD) == silence
