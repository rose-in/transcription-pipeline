"""Durable job queue for long audio (v1: single box).

    API process   --INSERT job row + input blob-->   data/jobs.db + data/jobs/<id>/input
    worker procs  --claim (lease) / heartbeat / checkpoint per chunk / complete-->  result.json

Job rows live in SQLite (WAL mode, safe across processes); blobs live on disk.
Production mapping is 1:1: SQLite -> Postgres (claim = SELECT ... FOR UPDATE SKIP
LOCKED), job dirs -> object storage. Same lease/heartbeat/attempt semantics.

Why a DB-backed queue instead of Celery/RQ for v1: one transactional source of
truth for job state (no Redis<->DB dual writes), and no hidden timeouts that
silently re-run or kill long jobs (Celery+Redis visibility_timeout 1 h, RQ 180 s,
Arq 300 s, Dramatiq 600 s by default).

Guarantees:
  * Idempotent submission - job id = hash(audio bytes, options, engine config), so
    a client retry returns the existing job instead of paying for it twice.
  * At-least-once execution with leases - a worker that dies stops heartbeating;
    once its lease expires another worker claims the job and resumes from the
    per-chunk checkpoints (chunk results are deterministic, so re-running is safe).
  * Poison-job quarantine - `attempts` is incremented at claim time, so a file
    that crashes workers (decoder segfault, OOM-kill) is failed after
    max_attempts instead of taking the fleet down in a loop.
  * Graceful drain - SIGTERM (deploy, spot interruption) releases the job at the
    next chunk boundary; another worker continues from the checkpoint.
  * Engine routing - workers only claim jobs whose engine_key matches their model
    config, so a "small" pool and a "large-v3-turbo" pool can share one queue.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from .errors import PipelineError
from .pipeline import DirCheckpointStore, Pipeline
from .schemas import Job, JobError, TranscribeOptions, Transcript

log = logging.getLogger(__name__)

PRIORITY_SYNC, PRIORITY_ASYNC = 0, 1
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  engine_key TEXT NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL,
  filename TEXT,
  options TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  not_before REAL NOT NULL DEFAULT 0,
  lease_until REAL,
  worker TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  progress REAL NOT NULL DEFAULT 0,
  processed_s REAL NOT NULL DEFAULT 0,
  duration_s REAL,
  max_duration_s REAL,
  sha256 TEXT,
  error_code TEXT,
  error_message TEXT,
  error_status INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (engine_key, status, priority, created_at);
CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,
  engine_key TEXT NOT NULL,
  model TEXT,
  pid INTEGER,
  started_at REAL,
  heartbeat REAL,
  stopped INTEGER NOT NULL DEFAULT 0
);
"""


class LeaseLost(Exception):
    """Another worker owns the job now (our lease expired); abandon without writing state."""


class Interrupted(Exception):
    """Worker is shutting down; release the job at a chunk boundary."""


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.blob_root = root / "jobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "jobs.db"
        with closing(self._connect()) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            # Forward-only migrations for columns added after v1.0 (Postgres: a migration tool).
            have = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
            for col, typ in (("max_duration_s", "REAL"), ("sha256", "TEXT")):
                if col not in have:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            if "stopped" not in {r["name"] for r in c.execute("PRAGMA table_info(workers)")}:
                c.execute("ALTER TABLE workers ADD COLUMN stopped INTEGER NOT NULL DEFAULT 0")

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        return c

    @contextmanager
    def _tx(self):
        """Write transaction. BEGIN IMMEDIATE takes the write lock up front, which is what makes
        claim() atomic across processes (Postgres: SELECT ... FOR UPDATE SKIP LOCKED)."""
        with closing(self._connect()) as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def _one(self, sql: str, *args) -> sqlite3.Row | None:
        with closing(self._connect()) as c:
            return c.execute(sql, args).fetchone()

    @staticmethod
    def _to_job(r: sqlite3.Row) -> Job:
        return Job(
            id=r["id"], status=r["status"], filename=r["filename"],
            # Lenient load: rows written by an older version must stay readable even if today's
            # edge validation is stricter (the pipeline re-validates before doing any work).
            options=TranscribeOptions.model_construct(**json.loads(r["options"])),
            created_at=r["created_at"], updated_at=r["updated_at"], started_at=r["started_at"],
            finished_at=r["finished_at"], attempts=r["attempts"], progress=r["progress"],
            processed_s=r["processed_s"], duration_s=r["duration_s"],
            error=JobError(code=r["error_code"], message=r["error_message"] or "", status=r["error_status"])
            if r["error_code"] else None,
        )

    # --------------------------------------------------------------------- blobs
    def dir(self, job_id: str) -> Path:
        # job_id reaches us from URL paths: never let it name anything outside jobs/.
        if not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self.blob_root / job_id

    def input_path(self, job_id: str) -> Path:
        return self.dir(job_id) / "input"

    def checkpoints(self, job_id: str) -> DirCheckpointStore:
        return DirCheckpointStore(self.dir(job_id) / "checkpoints")

    def result(self, job_id: str) -> Transcript | None:
        try:
            return Transcript.model_validate_json((self.dir(job_id) / "result.json").read_text())
        except (FileNotFoundError, KeyError):
            return None

    # ---------------------------------------------------------------- API side
    def submit(
        self, job_id: str, engine_key: str, upload: Path, filename: str | None,
        options: TranscribeOptions, *, priority: int = PRIORITY_ASYNC, duration_s: float | None = None,
        max_duration_s: float | None = None, sha256: str | None = None,
    ) -> tuple[Job, bool]:
        """Create the job, or return the existing one (idempotency).

        A job that failed *transiently* (5xx) is re-queued so the client can retry. A job that
        failed because of the input (4xx) or was quarantined as poison is returned as-is: the
        same bytes would fail the same way, and re-queueing a poison file would let any client
        retry loop crash workers indefinitely."""
        d = self.dir(job_id)
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            permanent = row is not None and row["status"] == "failed" and (
                row["error_code"] == "poison_job" or (row["error_status"] or 500) < 500)
            if row is not None and (row["status"] != "failed" or permanent):
                upload.unlink(missing_ok=True)
                if row["status"] == "queued" and priority < row["priority"]:
                    # a sync caller waiting on a deduplicated async job shouldn't wait at async priority
                    c.execute("UPDATE jobs SET priority=? WHERE id=?", (priority, job_id))
                return self._to_job(row), False
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(upload), self.input_path(job_id))
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            c.execute(
                "INSERT INTO jobs (id, engine_key, status, priority, filename, options, created_at, updated_at,"
                " duration_s, max_duration_s, sha256) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, engine_key, priority, filename, options.model_dump_json(), now, now, duration_s,
                 max_duration_s, sha256),
            )
        return self.get(job_id), True

    def run_params(self, job_id: str) -> dict:
        r = self._one("SELECT max_duration_s, sha256 FROM jobs WHERE id=?", job_id)
        return dict(r) if r else {}

    def get(self, job_id: str) -> Job | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        r = self._one("SELECT * FROM jobs WHERE id=?", job_id)
        return self._to_job(r) if r else None

    def queue_depth(self, engine_key: str | None = None) -> int:
        if engine_key is None:
            return self._one("SELECT COUNT(*) AS n FROM jobs WHERE status='queued'")["n"]
        return self._one("SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND engine_key=?", engine_key)["n"]

    def delete(self, job_id: str, *, expired_before: float | None = None) -> bool:
        """Hard-delete audio + transcript (retention policy, erasure requests). Refuses jobs a live
        worker holds; a 'running' row whose lease lapsed long ago is an orphan and can be erased.
        Blobs are removed inside the write transaction, so a concurrent resubmission of the same
        bytes can't have its fresh input deleted underneath it."""
        if not _JOB_ID.fullmatch(job_id):
            return True
        with self._tx() as c:
            row = c.execute("SELECT status, lease_until, updated_at FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None:
                if row["status"] == "running" and (row["lease_until"] or 0) > time.time():
                    return False
                if expired_before is not None and (row["status"] == "queued" or row["updated_at"] >= expired_before):
                    return False  # re-checked under the lock: it was resubmitted/updated meanwhile
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return True

    def purge_expired(self, ttl_s: float) -> int:
        cutoff = time.time() - ttl_s
        with closing(self._connect()) as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM jobs WHERE (status IN ('succeeded','failed') AND updated_at < ?)"
                " OR (status='running' AND lease_until < ?)", (cutoff, cutoff))]  # long-orphaned runs too
        return sum(self.delete(i, expired_before=cutoff) for i in ids)

    def fail_stranded(self, older_than_s: float) -> int:
        """Queued jobs for an engine config that *no worker has served for older_than_s* (a retired
        config after a redeploy) would otherwise wait forever and count against backpressure. Fail
        them with a retryable code so clients can resubmit and the rows become purgeable. Keyed on
        when the config was last served - not on whether a worker is up right now - so a routine
        worker restart or model load never fails a healthy backlog."""
        now = time.time()
        with closing(self._connect()) as c:
            cur = c.execute(
                "UPDATE jobs SET status='failed', error_code='engine_unavailable', error_status=503,"
                " error_message='No worker serves this engine configuration; resubmit.', finished_at=?, updated_at=?"
                " WHERE status='queued' AND created_at < ? AND engine_key NOT IN"
                " (SELECT engine_key FROM workers WHERE heartbeat > ?)",
                (now, now, now - older_than_s, now - older_than_s),
            )
            return cur.rowcount

    def stats(self) -> dict:
        with closing(self._connect()) as c:
            by_status = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
            audio = c.execute("SELECT COALESCE(SUM(duration_s), 0) AS s FROM jobs WHERE status='succeeded'").fetchone()["s"]
            wait = c.execute(
                "SELECT AVG(started_at - created_at) AS w FROM jobs WHERE started_at IS NOT NULL").fetchone()["w"]
        return {"by_status": by_status, "audio_seconds_succeeded": audio, "avg_queue_wait_s": wait or 0.0}

    # ------------------------------------------------------------- worker side
    def claim(self, worker_id: str, engine_key: str, lease_s: float, max_attempts: int) -> Job | None:
        now = time.time()
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET status='failed', error_code='poison_job', error_status=422, finished_at=?, updated_at=?,"
                " error_message='Job crashed or timed out on ' || attempts || ' attempts; quarantined.'"
                " WHERE engine_key=? AND status='running' AND lease_until < ? AND attempts >= ?",
                (now, now, engine_key, now, max_attempts),
            )
            row = c.execute(
                "SELECT id FROM jobs WHERE engine_key=? AND ((status='queued' AND not_before <= ?)"
                " OR (status='running' AND lease_until < ?)) ORDER BY priority, created_at LIMIT 1",
                (engine_key, now, now),
            ).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE jobs SET status='running', worker=?, lease_until=?, attempts=attempts+1,"
                " started_at=COALESCE(started_at, ?), updated_at=?, error_code=NULL, error_message=NULL WHERE id=?",
                (worker_id, now + lease_s, now, now, row["id"]),
            )
        return self.get(row["id"])

    def renew(self, job_id: str, worker_id: str, lease_s: float, *, progress: float | None = None,
              processed_s: float | None = None) -> bool:
        now = time.time()
        with closing(self._connect()) as c:
            cur = c.execute(
                "UPDATE jobs SET lease_until=?, updated_at=?, progress=COALESCE(?, progress),"
                " processed_s=COALESCE(?, processed_s) WHERE id=? AND worker=? AND status='running'",
                (now + lease_s, now, progress, processed_s, job_id, worker_id),
            )
            return cur.rowcount == 1

    def complete(self, job_id: str, worker_id: str, transcript: Transcript) -> bool:
        tmp = self.dir(job_id) / f".result.{uuid.uuid4().hex}.tmp"
        tmp.write_text(transcript.model_dump_json())
        os.replace(tmp, self.dir(job_id) / "result.json")
        now = time.time()
        with closing(self._connect()) as c:
            cur = c.execute(
                "UPDATE jobs SET status='succeeded', progress=1.0, processed_s=?, duration_s=?, finished_at=?,"
                " updated_at=?, lease_until=NULL WHERE id=? AND worker=? AND status='running'",
                (transcript.duration_s, transcript.duration_s, now, now, job_id, worker_id),
            )
            # (processed_s/duration_s are audio seconds of the file, not channel-seconds of work)
            return cur.rowcount == 1

    def fail(self, job_id: str, worker_id: str, err: JobError, status: int, *, retry_in: float | None = None) -> None:
        now = time.time()
        with closing(self._connect()) as c:
            if retry_in is not None:
                c.execute(
                    "UPDATE jobs SET status='queued', not_before=?, worker=NULL, lease_until=NULL, updated_at=?,"
                    " error_code=?, error_message=? WHERE id=? AND worker=?",
                    (now + retry_in, now, err.code, err.message, job_id, worker_id),
                )
            else:
                c.execute(
                    "UPDATE jobs SET status='failed', finished_at=?, updated_at=?, lease_until=NULL, error_code=?,"
                    " error_message=?, error_status=? WHERE id=? AND worker=?",
                    (now, now, err.code, err.message, status, job_id, worker_id),
                )

    def release(self, job_id: str, worker_id: str) -> None:
        """Graceful drain: give the job back without counting the attempt against it."""
        with closing(self._connect()) as c:
            c.execute(
                "UPDATE jobs SET status='queued', worker=NULL, lease_until=NULL, attempts=MAX(attempts-1, 0),"
                " updated_at=? WHERE id=? AND worker=? AND status='running'",
                (time.time(), job_id, worker_id),
            )

    def register_worker(self, worker_id: str, engine_key: str, model: str) -> None:
        now = time.time()
        with closing(self._connect()) as c:
            c.execute(
                "INSERT OR REPLACE INTO workers (id, engine_key, model, pid, started_at, heartbeat, stopped)"
            " VALUES (?,?,?,?,?,?,0)",
                (worker_id, engine_key, model, os.getpid(), now, now),
            )

    def worker_heartbeat(self, worker_id: str) -> None:
        with closing(self._connect()) as c:
            c.execute("UPDATE workers SET heartbeat=? WHERE id=?", (time.time(), worker_id))

    def remove_worker(self, worker_id: str) -> None:
        """Mark stopped rather than delete: the row remembers when this config was last served."""
        with closing(self._connect()) as c:
            c.execute("UPDATE workers SET stopped=1, heartbeat=? WHERE id=?", (time.time(), worker_id))
            c.execute("DELETE FROM workers WHERE stopped=1 AND heartbeat < ?", (time.time() - 7 * 86400,))

    def live_workers(self, engine_key: str, max_age_s: float) -> list[dict]:
        with closing(self._connect()) as c:
            rows = c.execute("SELECT * FROM workers WHERE engine_key=? AND stopped=0 AND heartbeat > ?",
                             (engine_key, time.time() - max_age_s)).fetchall()
        return [dict(r) for r in rows]


class Worker:
    """Pulls jobs for one engine config and runs them through the pipeline. One per GPU/process:
    the model (and BatchedInferencePipeline, which holds mutable state) is never shared across threads."""

    def __init__(
        self, store: JobStore, pipeline: Pipeline, engine_key: str, *, lease_s: float = 60.0,
        poll_s: float = 0.5, max_attempts: int = 3, worker_id: str | None = None,
        stall_timeout_s: float = 1800.0, exit_on_stall: bool = False,
    ):
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

    def _process(self, job: Job) -> None:
        state = {"lost": False, "last_progress": time.monotonic()}
        done = threading.Event()
        self.current = job

        def heartbeat() -> None:
            # Keeps the lease alive while one long chunk decodes - but only while the job is making
            # progress. A hung decoder/driver call would otherwise hold the job forever (renewed
            # lease, no quarantine, DELETE refused). After stall_timeout_s without a finished chunk
            # we stop renewing: the lease lapses, the attempt counts, and the worker exits so the
            # orchestrator restarts it.
            while not done.wait(self.lease_s / 4):
                try:
                    if time.monotonic() - state["last_progress"] > self.stall_timeout_s:
                        log.critical("job %s: no progress for %.0fs; abandoning lease", job.id, self.stall_timeout_s)
                        state["lost"], self.healthy = True, False
                        if self.exit_on_stall:
                            os._exit(70)  # crash-only: a wedged native call can't be cancelled from Python
                        return
                    if not self.store.renew(job.id, self.id, self.lease_s):
                        state["lost"] = True
                    self.store.worker_heartbeat(self.id)
                except Exception:  # a transient DB error must not silently end lease renewal
                    log.exception("heartbeat failed; retrying")

        last = [0.0]

        def progress(done_s: float, total_s: float | None) -> None:  # called after each checkpointed chunk
            state["last_progress"] = time.monotonic()
            if state["lost"]:
                raise LeaseLost(job.id)
            if self.stopping.is_set():
                raise Interrupted(job.id)
            if time.time() - last[0] > 1.0:
                last[0] = time.time()
                frac = min(done_s / total_s, 0.99) if total_s else None
                if not self.store.renew(job.id, self.id, self.lease_s, progress=frac, processed_s=done_s):
                    raise LeaseLost(job.id)

        hb = threading.Thread(target=heartbeat, name=f"hb-{job.id[:8]}", daemon=True)
        hb.start()
        log.info("job %s: start (attempt %d)", job.id, job.attempts)
        params = self.store.run_params(job.id)
        try:
            transcript = self.pipeline.run(
                # filename stays out of the transcript: with content-hash dedup the result is shared
                # by everyone who uploads the same bytes, so one client's file name must not leak.
                self.store.input_path(job.id), job.options, filename=None,
                sha256=params.get("sha256"), max_duration_s=params.get("max_duration_s"),
                progress=progress, checkpoints=self.store.checkpoints(job.id),
            )
        except LeaseLost:
            log.warning("job %s: lease lost; abandoning (another worker owns it)", job.id)
        except Interrupted:
            log.info("job %s: released for shutdown", job.id)
            self.store.release(job.id, self.id)  # no-op if drain() already released it
        except PipelineError as e:
            err = JobError(code=e.code, message=e.message)
            if e.retryable and job.attempts < self.max_attempts:
                delay = min(2 ** job.attempts, 60)  # exponential backoff; checkpoints make retries cheap
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
                    log.info("job %s: done (%.0fs audio, RTF %.3f)", job.id, transcript.duration_s, transcript.timing.rtf)
                else:
                    log.warning("job %s: finished after losing the lease; result kept, status untouched", job.id)
            except Exception as e:  # e.g. disk full while persisting: retry later, keep the worker alive
                log.exception("job %s: could not persist result", job.id)
                err = JobError(code="storage_error", message=type(e).__name__)
                retry = min(2 ** job.attempts, 60) if job.attempts < self.max_attempts else None  # bounded
                self.store.fail(job.id, self.id, err, 500, retry_in=retry)
        finally:
            done.set()
            self.current = None
            hb.join(timeout=5)


def job_id_for(sha256: str, options: TranscribeOptions, engine_key: str) -> str:
    return f"{sha256[:20]}{options.cache_key(engine_key)[:12]}"
