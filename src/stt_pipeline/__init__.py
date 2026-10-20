"""Audio file -> timestamped, post-processed transcript."""

# Bump when post-processing or chunking changes the output for the same audio + model:
# it is part of the engine key, so cached/idempotent results from older logic aren't reused.
PIPELINE_VERSION = "1.0"
