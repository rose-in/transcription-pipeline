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
import multiprocessing
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .audio import probe
from .config import Settings
from .engines import Engine
from .errors import AudioTooLong, FileTooLarge, InvalidOptions, PipelineError, QueueFull, UndecodableAudio
from .formats import FORMATS, MEDIA_TYPES, render
from .jobs import PRIORITY_ASYNC, PRIORITY_SYNC, JobStore, Worker, job_id_for
from .pipeline import Pipeline, PipelineConfig
from .schemas import Job, TranscribeOptions

log = logging.getLogger(__name__)
Fmt = Literal["json", "jsonl", "txt", "srt", "vtt", "chunks"]
assert set(FORMATS) == set(Fmt.__args__)  # keep the API enum and the renderers in sync


# ------------------------------------------------------------------------ sandboxed probe

def _probe_child(conn, path: str, max_duration_s: float) -> None:  # pragma: no cover - runs in the child
    import resource

    for lim, val in ((resource.RLIMIT_CPU, 30), (resource.RLIMIT_AS, 2 << 30)):
        with suppress(ValueError, OSError):  # RLIMIT_AS is not enforced on macOS; it is on Linux
            resource.setrlimit(lim, (val, val))
    try:
        conn.send(("ok", probe(Path(path), max_duration_s=max_duration_s)))
    except PipelineError as e:
        conn.send(("err", e))
    except Exception as e:
        conn.send(("err", UndecodableAudio(f"Could not inspect the file: {type(e).__name__}")))
    finally:
        conn.close()


class ProbeSandbox:
    """FFmpeg probing of untrusted uploads, one disposable process per file: a crash (FFmpeg
    ships ~23-25 CVEs a year), a hang or a resource-limit kill becomes a 422 for *that* upload
    only. (A shared process pool would fail every concurrent probe when one child dies.)
    forkserver keeps per-probe cost to a fork of a warm, single-threaded server process."""

    def __init__(self, timeout_s: float = 30.0, max_concurrent: int = 4):
        self.timeout_s = timeout_s
        methods = multiprocessing.get_all_start_methods()
        self._ctx = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
        if self._ctx.get_start_method() == "forkserver":
            self._ctx.set_forkserver_preload(["stt_pipeline.audio"])
        self._slots = threading.BoundedSemaphore(max_concurrent)

    def probe(self, path: Path, max_duration_s: float):
        with self._slots:
            recv, send = self._ctx.Pipe(duplex=False)
            proc = self._ctx.Process(target=_probe_child, args=(send, str(path), max_duration_s), daemon=True)
            proc.start()
            send.close()
            try:
                if not recv.poll(self.timeout_s):
                    raise UndecodableAudio("Timed out inspecting the file.")
                try:
                    status, value = recv.recv()
                except EOFError:  # child died without answering: crash or rlimit kill
                    raise UndecodableAudio("The decoder crashed on this file.") from None
            finally:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=5)
                recv.close()
        if status == "err":
            raise value
        return value

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------ body limit

class BodyLimit:
    """Reject oversized request bodies *before* multipart parsing spools them to disk (Starlette
    has no total cap). Content-Length is checked up front; chunked bodies are counted as they
    stream. In production the edge proxy should enforce this too (defence in depth)."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            return await self._reject(send)
        state = {"received": 0, "exceeded": False, "started": False}

        async def guarded_receive():
            if state["exceeded"]:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                state["received"] += len(msg.get("body", b""))
                if state["received"] > self.limit:
                    state["exceeded"] = True
                    return {"type": "http.disconnect"}  # stop the app from reading further
            return msg

        async def guarded_send(msg):
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

    async def _reject(self, send):
        body = (f'{{"error": {{"code": "file_too_large", "message": "Request body exceeds '
                f'{self.limit // 2**20} MB."}}}}').encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"connection", b"close")]})
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
        """Retention and hygiene, every few minutes (not just at startup)."""
        purged = store.purge_expired(settings.result_ttl_s)
        stranded = store.fail_stranded(settings.stranded_after_s)
        cutoff = time.time() - 3600
        for f in uploads.glob("*") if uploads.exists() else []:
            with suppress(FileNotFoundError):
                if f.stat().st_mtime < cutoff:  # orphaned partial uploads (client vanished, crash)
                    f.unlink()
        if purged or stranded:
            log.info("maintenance: purged %d expired job(s), failed %d stranded job(s)", purged, stranded)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def maintenance_loop():
            while True:
                with suppress(Exception):
                    await run_in_threadpool(maintenance)
                await asyncio.sleep(300)

        task = asyncio.create_task(maintenance_loop())
        worker, thread = None, None
        if worker_engine is not None:
            pipeline = Pipeline(worker_engine, PipelineConfig(chunk_s=settings.chunk_s, max_duration_s=settings.max_duration_s,
                                                              max_split_channels=settings.max_split_channels))
            worker = Worker(store, pipeline, engine_key, lease_s=settings.lease_s, poll_s=0.02,
                            max_attempts=settings.max_attempts, stall_timeout_s=settings.stall_timeout_s)
            thread = threading.Thread(target=worker.run, name="embedded-worker", daemon=True)
            thread.start()
        yield
        task.cancel()
        sandbox.close()
        if worker is not None:
            worker.drain()
            thread.join(timeout=10)

    app = FastAPI(title="Transcription pipeline", version="1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_upload_bytes + (1 << 20))  # + form-field overhead
    app.state.store = store

    @app.exception_handler(PipelineError)
    async def pipeline_error(_: Request, e: PipelineError):
        headers = {"Retry-After": "5"} if e.retryable else None
        return JSONResponse({"error": e.to_dict()}, status_code=e.status, headers=headers)

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

    def parse_options(language, word_timestamps, channel_mode, redact_pii, prompt) -> TranscribeOptions:
        try:
            return TranscribeOptions(language=language or None, word_timestamps=word_timestamps,
                                     channel_mode=channel_mode, redact_pii=redact_pii, prompt=prompt or None)
        except ValueError as e:
            raise InvalidOptions(str(e).splitlines()[-1] if str(e) else "invalid options") from e

    async def intake(file: UploadFile, options: TranscribeOptions, *, sync: bool) -> tuple[Job, bool]:
        """Persist + validate *before* queueing, so a bad file gets an immediate precise 4xx
        rather than a failed job the client has to poll to discover."""
        path, sha = await save_upload(file)
        try:
            limit = settings.sync_max_duration_s if sync else settings.max_duration_s
            info = await run_in_threadpool(sandbox.probe, path, settings.max_duration_s)
            if sync and info.container_duration_s and not info.duration_is_estimate \
                    and info.container_duration_s > settings.sync_max_duration_s:
                raise AudioTooLong(
                    f"Audio is {info.container_duration_s:.0f}s; synchronous requests are limited to "
                    f"{settings.sync_max_duration_s:.0f}s. Use POST /v1/jobs for long audio."
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
                lambda: store.submit(job_id, engine_key, path, file.filename, options,
                                     priority=PRIORITY_SYNC if sync else PRIORITY_ASYNC,
                                     duration_s=info.container_duration_s, max_duration_s=limit, sha256=sha))
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

    def render_response(transcript, fmt: str) -> Response:
        if transcript is None:
            return not_found()
        body = render(transcript, fmt)
        if fmt in ("json", "chunks"):
            return Response(body, media_type=MEDIA_TYPES[fmt])
        return PlainTextResponse(body, media_type=MEDIA_TYPES[fmt])

    def not_found() -> JSONResponse:
        return JSONResponse({"error": {"code": "not_found", "message": "No such job."}}, status_code=404)

    # ------------------------------------------------------------------ routes
    @app.post("/v1/transcriptions")
    async def transcribe_sync(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        word_timestamps: bool = Form(False),
        channel_mode: Literal["mix", "split"] = Form("mix"),
        redact_pii: bool = Form(False),
        prompt: str | None = Form(None),
        response_format: Fmt = Form("json"),
    ):
        options = parse_options(language, word_timestamps, channel_mode, redact_pii, prompt)
        job, _ = await intake(file, options, sync=True)
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
    async def create_job(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        word_timestamps: bool = Form(False),
        channel_mode: Literal["mix", "split"] = Form("mix"),
        redact_pii: bool = Form(False),
        prompt: str | None = Form(None),
    ):
        options = parse_options(language, word_timestamps, channel_mode, redact_pii, prompt)
        job, created = await intake(file, options, sync=False)
        if created:
            return JSONResponse(job_view(job), status_code=202, headers={"Location": f"/v1/jobs/{job.id}"})
        # Same audio + same options + same engine config => same job. Safe retries, no double billing.
        return JSONResponse(job_view(job), status_code=200, headers={"Idempotent-Replay": "true"})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await run_in_threadpool(store.get, job_id)
        return job_view(job) if job else not_found()

    @app.get("/v1/jobs/{job_id}/result")
    async def get_result(job_id: str, format: Fmt = Query("json")):
        job = await run_in_threadpool(store.get, job_id)
        if job is None:
            return not_found()
        if job.status != "succeeded":
            return JSONResponse({"error": {"code": "not_ready", "message": f"Job is {job.status}."},
                                 "job": job_view(job)}, status_code=409)
        return render_response(await run_in_threadpool(store.result, job_id), format)

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str):
        if not await run_in_threadpool(store.delete, job_id):
            return JSONResponse({"error": {"code": "conflict", "message": "Job is running."}}, status_code=409)
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz():  # liveness: the process is up. Never depends on the model or workers.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():  # readiness: someone can actually run our jobs (fresh worker heartbeat)
        workers = await run_in_threadpool(store.live_workers, engine_key, settings.lease_s)
        body = {"ready": bool(workers), "engine_key": engine_key, "workers": [w["id"] for w in workers],
                "queue_depth": await run_in_threadpool(store.queue_depth, engine_key)}
        if not workers:
            body["hint"] = ("No live worker for this engine configuration. API and workers must share the "
                            "STT_ENGINE/MODEL/MODEL_REVISION/DEVICE/COMPUTE_TYPE/BEAM_SIZE/CHUNK_SECONDS settings.")
        return JSONResponse(body, status_code=200 if workers else 503)

    @app.get("/metrics")
    async def metrics():
        s = await run_in_threadpool(store.stats)
        lines = ["# TYPE stt_jobs gauge"]
        for status in ("queued", "running", "succeeded", "failed"):
            lines.append(f'stt_jobs{{status="{status}"}} {s["by_status"].get(status, 0)}')
        lines += [
            "# TYPE stt_audio_seconds_retained gauge", f"stt_audio_seconds_retained {s['audio_seconds_succeeded']:.1f}",
            "# TYPE stt_queue_wait_seconds_avg gauge", f"stt_queue_wait_seconds_avg {s['avg_queue_wait_s']:.3f}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app
