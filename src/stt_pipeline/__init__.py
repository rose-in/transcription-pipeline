"""Audio file -> timestamped, post-processed transcript."""

# Bump when post-processing, chunking or engine decoding logic changes the output for the same
# audio + model: it is part of Settings.engine_key() and of every checkpoint fingerprint, so
# idempotent results and chunk checkpoints from older logic aren't reused.
PIPELINE_VERSION = "1.0"
