# syntax=docker/dockerfile:1.7
# One image, two roles: `stt serve` (API, no model) and `stt worker` (model, one per GPU).
#
# GPU notes (verified against CTranslate2 4.8.x): Linux wheels are built for CUDA 12 *without*
# cuDNN and dlopen libcublas.so.12 by soname, so a slim Python base + the nvidia-cublas-cu12 wheel
# + LD_LIBRARY_PATH is enough. Do not base this on CUDA 13 NGC images. Host needs an R525+ driver.
FROM python:3.12-slim-bookworm

ARG WITH_GPU=1
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project $([ "$WITH_GPU" = 1 ] && echo --extra gpu)
ENV PATH=/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib

# Bake pinned weights into the image: immutable, reproducible deploys with no Hugging Face
# dependency (or rate limit) at startup. "large-v3-turbo" resolves to a third-party repo, so pin
# the commit; bump deliberately.
ARG STT_MODEL=large-v3-turbo
ARG STT_MODEL_REVISION=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf
ENV STT_MODEL=$STT_MODEL STT_MODEL_REVISION=$STT_MODEL_REVISION HF_HOME=/models
RUN python -c "import os; from faster_whisper.utils import download_model as d; \
d(os.environ['STT_MODEL'], revision=os.environ['STT_MODEL_REVISION'])"
ENV HF_HUB_OFFLINE=1 STT_DATA_DIR=/data

# Project code last, so source/README edits rebuild only this layer (not deps or weights).
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev $([ "$WITH_GPU" = 1 ] && echo --extra gpu)

RUN useradd --system --uid 10001 stt && mkdir -p /data && chown stt /data
USER stt
EXPOSE 8000
CMD ["stt", "worker"]
