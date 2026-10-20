"""Job-queue consumer: claims jobs from JobStore for one engine config, keeps the lease alive with a
heartbeat thread, abandons stalled jobs, retries with backoff, and releases the in-flight job on drain.
Queue semantics and guarantees are documented in jobs.py."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid

from .errors import LONG_AUDIO_HINT, AudioTooLong, PipelineError
from .jobs import JobStore
from .pipeline import Pipeline
from .schemas import Job, JobError

log = logging.getLogger(__name__)

_MAX_RETRY_DELAY_S = 60  # cap for exponential job-retry backoff
_PROGRESS_RENEW_INTERVAL_S = 1.0  # min gap between progress writes/lease renewals from chunk callbacks
_EXIT_STALLED = 70  # sysexits EX_SOFTWARE; a literal because os.EX_SOFTWARE is Unix-only


def _retry_delay_s(attempts: int) -> int:
    """Exponential backoff before a retryable job is claimable again; checkpoints make retries cheap."""
    return min(2**attempts, _MAX_RETRY_DELAY_S)


class LeaseLost(Exception):
    """Another worker owns the job now (our lease expired); abandon without writing state."""


class Interrupted(Exception):
    """Worker is shutting down; release the job at a chunk boundary."""


class Worker:
    """Pulls jobs for one engine config and runs them through the pipeline. One per GPU/process:
    the model (and BatchedInferencePipeline, which holds mutable state) is never shared across threads."""

    def __init__(
        self,
        store: JobStore,
        pipeline: Pipeline,
        engine_key: str,
        *,
        lease_s: float = 60.0,
        poll_s: float = 0.5,
        max_attempts: int = 3,
        worker_id: str | None = None,
        stall_timeout_s: float = 1800.0,
        exit_on_stall: bool = False,
    ) -> None:
        self.store, self.pipeline, self.engine_key = store, pipeline, engine_key
        self.lease_s, self.poll_s, self.max_attempts = lease_s, poll_s, max_attempts
        self.stall_timeout_s, self.exit_on_stall = stall_timeout_s, exit_on_stall
        self.id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.stopping = threading.Event()
        self.current: Job | None = None
        self.healthy = True

    def drain(self) -> None:
        """SIGTERM: give the in-flight job back *now* (no attempt counted) instead of hoping the
        current chunk finishes within the orchestrator's grace period. Another worker resumes it
        from the last checkpoint; whatever this process still computes is discarded (lease lost)."""
        self.stopping.set()
        job = self.current
        if job is not None:
            self.store.release(job.id, self.id)

    def run(self) -> None:
        self.store.register_worker(self.id, self.engine_key, self.pipeline.engine.meta.model)
        log.info("worker %s ready (engine_key=%s)", self.id, self.engine_key)
        try:
            while not self.stopping.is_set() and self.healthy:
                self.store.worker_heartbeat(self.id)
                if not self.run_once():
                    self.stopping.wait(self.poll_s)
        finally:
            self.store.remove_worker(self.id)

    def run_once(self) -> bool:
        job = self.store.claim(self.id, self.engine_key, self.lease_s, self.max_attempts)
        if job is None:
            return False
        self._process(job)
        return True

    def _keep_lease(self, job: Job, state: dict, done: threading.Event) -> None:
        """Heartbeat thread: renew the job lease while one long chunk decodes, but only while the job
        makes progress. A hung decoder/driver call would otherwise hold the job forever (renewed lease,
        no quarantine, DELETE refused). After stall_timeout_s without a finished chunk, stop renewing:
        the lease lapses, the attempt counts, and the worker exits so the orchestrator restarts it."""
        while not done.wait(self.lease_s / 4):
            try:
                if time.monotonic() - state["last_progress"] > self.stall_timeout_s:
                    log.critical("job %s: no progress for %.0fs; abandoning lease", job.id, self.stall_timeout_s)
                    state["lost"], self.healthy = True, False
                    if self.exit_on_stall:
                        os._exit(_EXIT_STALLED)  # crash-only: a wedged native call can't be cancelled from Python
                    return
                if not self.store.renew(job.id, self.id, self.lease_s):
                    state["lost"] = True
                self.store.worker_heartbeat(self.id)
            except Exception:  # a transient DB error must not silently end lease renewal
                log.exception("heartbeat failed; retrying")

    def _process(self, job: Job) -> None:
        state = {"lost": False, "last_progress": time.monotonic(), "last_renew": 0.0}
        done = threading.Event()
        self.current = job

        def progress(done_s: float, total_s: float | None) -> None:  # called after each checkpointed chunk
            state["last_progress"] = time.monotonic()
            if state["lost"]:
                raise LeaseLost(job.id)
            if self.stopping.is_set():
                raise Interrupted(job.id)
            if time.time() - state["last_renew"] > _PROGRESS_RENEW_INTERVAL_S:
                state["last_renew"] = time.time()
                frac = min(done_s / total_s, 0.99) if total_s else None
                if not self.store.renew(job.id, self.id, self.lease_s, progress=frac, processed_s=done_s):
                    raise LeaseLost(job.id)

        hb = threading.Thread(target=self._keep_lease, args=(job, state, done), name=f"hb-{job.id[:8]}", daemon=True)
        hb.start()
        log.info("job %s: start (attempt %d)", job.id, job.attempts)
        params = self.store.run_params(job.id)
        try:
            transcript = self.pipeline.run(
                # filename stays out of the transcript: with content-hash dedup the result is shared
                # by everyone who uploads the same bytes, so one client's file name must not leak.
                self.store.input_path(job.id),
                job.options,
                filename=None,
                sha256=params.get("sha256"),
                max_duration_s=params.get("max_duration_s"),
                progress=progress,
                checkpoints=self.store.checkpoints(job.id),
            )
        except LeaseLost:
            log.warning("job %s: lease lost; abandoning (another worker owns it)", job.id)
        except Interrupted:
            log.info("job %s: released for shutdown", job.id)
            self.store.release(job.id, self.id)  # no-op if drain() already released it
        except PipelineError as e:
            message = e.message
            job_limit = params.get("max_duration_s")
            if isinstance(e, AudioTooLong) and job_limit and job_limit < self.pipeline.config.max_duration_s:
                message = f"{message} {LONG_AUDIO_HINT}"  # the job's own (sync) limit fired: point at the async API
            err = JobError(code=e.code, message=message)
            if e.retryable and job.attempts < self.max_attempts:
                delay = _retry_delay_s(job.attempts)
                log.warning("job %s: %s (attempt %d), retrying in %ds", job.id, e.code, job.attempts, delay)
                self.store.fail(job.id, self.id, err, e.status, retry_in=delay)
            else:
                self.store.fail(job.id, self.id, err, e.status)
        except Exception as e:  # pragma: no cover - defensive: one bad job must not kill the worker loop
            log.exception("job %s crashed", job.id)
            self.store.fail(job.id, self.id, JobError(code="internal_error", message=str(e)), 500)
        else:
            try:
                if self.store.complete(job.id, self.id, transcript):
                    log.info(
                        "job %s: done (%.0fs audio, RTF %.3f)", job.id, transcript.duration_s, transcript.timing.rtf
                    )
                else:
                    log.warning("job %s: finished after losing the lease; result kept, status untouched", job.id)
            except Exception as e:  # e.g. disk full while persisting: retry later, keep the worker alive
                log.exception("job %s: could not persist result", job.id)
                err = JobError(code="storage_error", message=type(e).__name__)
                retry = _retry_delay_s(job.attempts) if job.attempts < self.max_attempts else None
                self.store.fail(job.id, self.id, err, 500, retry_in=retry)
        finally:
            done.set()
            self.current = None
            hb.join(timeout=5)
