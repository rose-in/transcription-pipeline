"""HTTP service. Holds no model: it validates, enqueues and serves results.

POST   /v1/transcriptions     sync: short audio (<= SYNC_MAX_DURATION); result in the response
POST   /v1/jobs               async: up to MAX_DURATION -> 202 + Location (idempotent)
GET    /v1/jobs/{id}          status + progress
GET    /v1/jobs/{id}/result   ?format=json|jsonl|txt|srt|vtt|chunks
DELETE /v1/jobs/{id}          delete audio + transcript (retention / erasure requests)
GET    /healthz  /readyz  /metrics

Isolation: inference runs in separate worker processes (`stt worker`), and the
API's own look at untrusted media (the FFmpeg probe) runs in a sandbox
subprocess, so a decoder crash or hang costs one disposable child, never the
API. The sync endpoint is "enqueue at high priority and wait (bounded)": one
execution path, no model contention between sync and async traffic, and on
timeout the client gets a 202 + job URL instead of a 504.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import ENGINE_KEY_ENV_VARS, Settings
from .engines import Engine
from .errors import LONG_AUDIO_HINT, AudioTooLong, FileTooLarge, InvalidOptions, PipelineError, QueueFull
from .formats import MEDIA_TYPES, render
from .jobs import PRIORITY_ASYNC, PRIORITY_SYNC, JobStore, job_id_for
from .pipeline import Pipeline, PipelineConfig
from .sandbox import ProbeSandbox
from .schemas import Job, TranscribeOptions, Transcript
from .worker import Worker

log = logging.getLogger(__name__)

# Must equal formats.FORMATS (tests/test_api.py checks it).
OutputFormat = Literal["json", "jsonl", "txt", "srt", "vtt", "chunks"]

_MAINTENANCE_INTERVAL_S = 300  # how often retention / stranded-job / orphaned-upload cleanup runs
_ORPHAN_UPLOAD_MAX_AGE_S = 3600  # partial uploads left behind by a vanished client or a crash
_FORM_OVERHEAD_BYTES = 1 << 20  # multipart boundaries + form fields on top of the file itself
_RETRY_AFTER_S = 5  # Retry-After hint on retryable errors (e.g. queue_full)


def _error_response(code: str, message: str, status: int, **extra: object) -> JSONResponse:
    """The API's error envelope: {"error": {"code", "message"}, **extra}."""
    return JSONResponse({"error": {"code": code, "message": message}, **extra}, status_code=status)


# ------------------------------------------------------------------------ upload form


@dataclass
class UploadForm:
    """The multipart fields both upload endpoints share."""

    file: UploadFile
    option_fields: dict

    def options(self) -> TranscribeOptions:
        """Validated per-request options; raises InvalidOptions (422) otherwise."""
        try:
            return TranscribeOptions(**self.option_fields)
        except ValueError as e:
            raise InvalidOptions.from_validation_error(e) from e


def upload_form(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    word_timestamps: bool = Form(False),
    channel_mode: Literal["mix", "split"] = Form("mix"),
    redact_pii: bool = Form(False),
    prompt: str | None = Form(None),
) -> UploadForm:
    """FastAPI dependency for the shared fields. Options are validated by the handler (UploadForm.options),
    so FastAPI reports malformed fields first, including endpoint-only ones such as response_format."""
    return UploadForm(
        file,
        {
            "language": language or None,
            "word_timestamps": word_timestamps,
            "channel_mode": channel_mode,
            "redact_pii": redact_pii,
            "prompt": prompt or None,
        },
    )


# ------------------------------------------------------------------------ body limit


class BodyLimit:
    """Reject oversized request bodies *before* multipart parsing spools them to disk (Starlette
    has no total cap). Content-Length is checked up front; chunked bodies are counted as they
    stream. In production the edge proxy should enforce this too (defence in depth)."""

    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            return await self._reject(send)
        state = {"received": 0, "exceeded": False, "started": False}

        async def guarded_receive() -> Message:
            if state["exceeded"]:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                state["received"] += len(msg.get("body", b""))
                if state["received"] > self.limit:
                    state["exceeded"] = True
                    return {"type": "http.disconnect"}  # stop the app from reading further
            return msg

        async def guarded_send(msg: Message) -> None:
            if state["exceeded"] and not state["started"]:
                return  # our 413 replaces whatever the app would answer to a "disconnect"
            if msg["type"] == "http.response.start":
                state["started"] = True
            await send(msg)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except Exception:
            if not state["exceeded"]:
                raise
        if state["exceeded"] and not state["started"]:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = (
            f'{{"error": {{"code": "file_too_large", "message": "Request body exceeds {self.limit // 2**20} MB."}}}}'
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"connection", b"close")],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ------------------------------------------------------------------------ app


def create_app(settings: Settings | None = None, *, worker_engine: Engine | None = None) -> FastAPI:
    """worker_engine: run one worker thread in-process with this engine (tests / single-process dev).
    In production leave it None and run `stt worker` processes."""
    settings = settings or Settings()
    engine_key = settings.engine_key()
    store = JobStore(settings.data_dir)
    sandbox = ProbeSandbox()
    uploads = settings.data_dir / "uploads"

    def maintenance() -> None:
        """Retention and hygiene; runs every _MAINTENANCE_INTERVAL_S."""
        purged = store.purge_expired(settings.result_ttl_s)
        stranded = store.fail_stranded(settings.stranded_after_s)
        cutoff = time.time() - _ORPHAN_UPLOAD_MAX_AGE_S
        for f in uploads.glob("*") if uploads.exists() else []:
            with suppress(FileNotFoundError):
                if f.stat().st_mtime < cutoff:  # orphaned partial uploads (client vanished, crash)
                    f.unlink()
        if purged or stranded:
            log.info("maintenance: purged %d expired job(s), failed %d stranded job(s)", purged, stranded)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async def maintenance_loop() -> None:
            while True:
                try:
                    await run_in_threadpool(maintenance)
                except Exception:  # e.g. DB locked / disk error: log it, the next pass retries
                    log.exception("maintenance failed; retrying in %ss", _MAINTENANCE_INTERVAL_S)
                await asyncio.sleep(_MAINTENANCE_INTERVAL_S)

        task = asyncio.create_task(maintenance_loop())
        worker, thread = None, None
        if worker_engine is not None:
            pipeline = Pipeline(worker_engine, PipelineConfig(**settings.pipeline_kwargs()))
            worker = Worker(
                store,
                pipeline,
                engine_key,
                lease_s=settings.lease_s,
                poll_s=0.02,
                max_attempts=settings.max_attempts,
                stall_timeout_s=settings.stall_timeout_s,
            )
            thread = threading.Thread(target=worker.run, name="embedded-worker", daemon=True)
            thread.start()
        yield
        task.cancel()
        if worker is not None:
            worker.drain()
            thread.join(timeout=10)

    app = FastAPI(title="Transcription pipeline", version="1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_upload_bytes + _FORM_OVERHEAD_BYTES)
    app.state.store = store

    @app.exception_handler(PipelineError)
    async def pipeline_error(_: Request, e: PipelineError) -> JSONResponse:
        headers = {"Retry-After": str(_RETRY_AFTER_S)} if e.retryable else None
        return JSONResponse({"error": e.to_dict()}, status_code=e.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def request_invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        # Malformed fields get the same typed envelope as every other error, not FastAPI's {"detail": [...]}.
        fields = [
            "{}: {}".format(".".join(str(x) for x in err["loc"] if x not in ("body", "query")), err["msg"])
            for err in e.errors()
        ]
        return _error_response(InvalidOptions.code, "; ".join(fields) or "invalid request", InvalidOptions.status)

    async def save_upload(file: UploadFile) -> tuple[Path, str]:
        uploads.mkdir(parents=True, exist_ok=True)
        dest, h, n = uploads / uuid.uuid4().hex, hashlib.sha256(), 0  # never the client's filename
        try:
            with dest.open("wb") as f:
                while chunk := await file.read(1 << 20):
                    n += len(chunk)
                    if n > settings.max_upload_bytes:
                        raise FileTooLarge(f"Upload exceeds {settings.max_upload_bytes // 2**20} MB.")
                    await run_in_threadpool(lambda c=chunk: (h.update(c), f.write(c)))  # keep the loop free
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return dest, h.hexdigest()

    async def intake(file: UploadFile, options: TranscribeOptions, *, sync: bool) -> tuple[Job, bool]:
        """Persist + validate *before* queueing, so a bad file gets an immediate precise 4xx
        rather than a failed job the client has to poll to discover."""
        path, sha = await save_upload(file)
        try:
            limit = settings.sync_max_duration_s if sync else settings.max_duration_s
            info = await run_in_threadpool(sandbox.probe, path, settings.max_duration_s)
            if (
                sync
                and info.container_duration_s
                and not info.duration_is_estimate
                and info.container_duration_s > settings.sync_max_duration_s
            ):
                raise AudioTooLong(
                    f"Audio is {info.container_duration_s:.0f}s; synchronous requests are limited to "
                    f"{settings.sync_max_duration_s:.0f}s. {LONG_AUDIO_HINT}"
                )
            # Sync and async submissions are distinct jobs: they carry different duration limits,
            # and a sync-only failure (audio_too_long) must not stick to the async path.
            job_id = job_id_for(sha, options, engine_key + (":sync" if sync else ""))
            existing = await run_in_threadpool(store.get, job_id)
            if existing is None or existing.status == "failed":
                if await run_in_threadpool(store.queue_depth, engine_key) >= settings.max_queue:
                    raise QueueFull("Transcription queue is full; retry later.")  # backpressure, not unbounded backlog
            # The sync limit is stored on the job and enforced on *decoded samples* by the worker,
            # so a file with a missing or lying duration header can't use the sync path as a bypass.
            return await run_in_threadpool(
                lambda: store.submit(
                    job_id,
                    engine_key,
                    path,
                    file.filename,
                    options,
                    priority=PRIORITY_SYNC if sync else PRIORITY_ASYNC,
                    duration_s=info.container_duration_s,
                    max_duration_s=limit,
                    sha256=sha,
                )
            )
        finally:
            path.unlink(missing_ok=True)  # moved into the job dir on success; removed on any error

    def job_view(job: Job) -> dict:
        # filename is deliberately not echoed: with content-hash dedup, a second submitter would
        # otherwise see the first submitter's metadata.
        d = job.model_dump(mode="json", exclude={"filename"})
        if d.get("error"):
            d["error"].pop("status", None)
        d["links"] = {"self": f"/v1/jobs/{job.id}", "result": f"/v1/jobs/{job.id}/result"}
        return d

    def render_response(transcript: Transcript | None, fmt: str) -> Response:
        if transcript is None:
            return not_found()
        body = render(transcript, fmt)
        if fmt in ("json", "chunks"):
            return Response(body, media_type=MEDIA_TYPES[fmt])
        return PlainTextResponse(body, media_type=MEDIA_TYPES[fmt])

    def not_found() -> JSONResponse:
        return _error_response("not_found", "No such job.", 404)

    # ------------------------------------------------------------------ routes
    @app.post("/v1/transcriptions")
    async def transcribe_sync(form: UploadForm = Depends(upload_form), response_format: OutputFormat = Form("json")):
        job, _ = await intake(form.file, form.options(), sync=True)
        deadline = time.monotonic() + settings.sync_timeout_s
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            await asyncio.sleep(0.05 if job.status == "running" else 0.02)
            job = await run_in_threadpool(store.get, job.id)
            if job is None:  # deleted while we waited
                return not_found()
        if job.status == "succeeded":
            return render_response(await run_in_threadpool(store.result, job.id), response_format)
        if job.status == "failed":
            err = job.error.model_dump(exclude={"status"}) if job.error else None
            return JSONResponse({"error": err}, status_code=(job.error and job.error.status) or 500)
        # Still running: the work continues; give the client a job to poll instead of a 504.
        return JSONResponse(job_view(job), status_code=202, headers={"Location": f"/v1/jobs/{job.id}"})

    @app.post("/v1/jobs", status_code=202)
    async def create_job(form: UploadForm = Depends(upload_form)):
        job, created = await intake(form.file, form.options(), sync=False)
        if created:
            return JSONResponse(job_view(job), status_code=202, headers={"Location": f"/v1/jobs/{job.id}"})
        # Same audio + same options + same engine config => same job. Safe retries, no double billing.
        return JSONResponse(job_view(job), status_code=200, headers={"Idempotent-Replay": "true"})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await run_in_threadpool(store.get, job_id)
        return job_view(job) if job else not_found()

    @app.get("/v1/jobs/{job_id}/result")
    async def get_result(job_id: str, format: OutputFormat = Query("json")):
        job = await run_in_threadpool(store.get, job_id)
        if job is None:
            return not_found()
        if job.status != "succeeded":
            return _error_response("not_ready", f"Job is {job.status}.", 409, job=job_view(job))
        return render_response(await run_in_threadpool(store.result, job_id), format)

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str):
        if not await run_in_threadpool(store.delete, job_id):
            return _error_response("conflict", "Job is running.", 409)
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz():  # liveness: the process is up. Never depends on the model or workers.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():  # readiness: someone can actually run our jobs (fresh worker heartbeat)
        workers = await run_in_threadpool(store.live_workers, engine_key, settings.lease_s)
        body = {
            "ready": bool(workers),
            "engine_key": engine_key,
            "workers": [w["id"] for w in workers],
            "queue_depth": await run_in_threadpool(store.queue_depth, engine_key),
        }
        if not workers:
            body["hint"] = (
                "No live worker for this engine configuration. API and workers must share these "
                "settings: " + ", ".join(ENGINE_KEY_ENV_VARS) + "."
            )
        return JSONResponse(body, status_code=200 if workers else 503)

    @app.get("/metrics")
    async def metrics():
        s = await run_in_threadpool(store.stats)
        lines = ["# TYPE stt_jobs gauge"]
        for status in ("queued", "running", "succeeded", "failed"):
            lines.append(f'stt_jobs{{status="{status}"}} {s["by_status"].get(status, 0)}')
        lines += [
            "# TYPE stt_audio_seconds_retained gauge",
            f"stt_audio_seconds_retained {s['audio_seconds_succeeded']:.1f}",
            "# TYPE stt_queue_wait_seconds_avg gauge",
            f"stt_queue_wait_seconds_avg {s['avg_queue_wait_s']:.3f}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app
