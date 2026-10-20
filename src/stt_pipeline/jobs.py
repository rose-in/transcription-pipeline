"""Durable job queue for long audio (v1: single box).

    API process   --INSERT job row + input blob-->   data/jobs.db + data/jobs/<id>/input
    worker procs  --claim (lease) / heartbeat / checkpoint per chunk / complete-->  result.json

This module is the queue (JobStore); worker.py is the consumer that runs claimed jobs.

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
import os
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from .pipeline import DirCheckpointStore
from .schemas import Job, JobError, TranscribeOptions, Transcript

PRIORITY_SYNC, PRIORITY_ASYNC = 0, 1
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
# Stopped worker rows are kept this long as "config last served" evidence for fail_stranded().
_STOPPED_WORKER_RETENTION_S = 7 * 86400

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


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.blob_root = root / "jobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "jobs.db"
        with closing(self._connect()) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

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
            id=r["id"],
            status=r["status"],
            filename=r["filename"],
            # Lenient load: rows written by an older version must stay readable even if today's
            # edge validation is stricter (the pipeline re-validates before doing any work).
            options=TranscribeOptions.model_construct(**json.loads(r["options"])),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
            attempts=r["attempts"],
            progress=r["progress"],
            processed_s=r["processed_s"],
            duration_s=r["duration_s"],
            error=JobError(code=r["error_code"], message=r["error_message"] or "", status=r["error_status"])
            if r["error_code"]
            else None,
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
        self,
        job_id: str,
        engine_key: str,
        upload: Path,
        filename: str | None,
        options: TranscribeOptions,
        *,
        priority: int = PRIORITY_ASYNC,
        duration_s: float | None = None,
        max_duration_s: float | None = None,
        sha256: str | None = None,
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
            permanent = (
                row is not None
                and row["status"] == "failed"
                and (row["error_code"] == "poison_job" or (row["error_status"] or 500) < 500)
            )
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
                (
                    job_id,
                    engine_key,
                    priority,
                    filename,
                    options.model_dump_json(),
                    now,
                    now,
                    duration_s,
                    max_duration_s,
                    sha256,
                ),
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

    def queue_depth(self, engine_key: str) -> int:
        """Number of jobs still queued for this engine's worker pool."""
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
            ids = [
                r["id"]
                for r in c.execute(
                    "SELECT id FROM jobs WHERE (status IN ('succeeded','failed') AND updated_at < ?)"
                    " OR (status='running' AND lease_until < ?)",
                    (cutoff, cutoff),
                )
            ]  # long-orphaned runs too
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
            by_status = {
                r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            }
            audio = c.execute("SELECT COALESCE(SUM(duration_s), 0) AS s FROM jobs WHERE status='succeeded'").fetchone()[
                "s"
            ]
            wait = c.execute(
                "SELECT AVG(started_at - created_at) AS w FROM jobs WHERE started_at IS NOT NULL"
            ).fetchone()["w"]
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

    def renew(
        self,
        job_id: str,
        worker_id: str,
        lease_s: float,
        *,
        progress: float | None = None,
        processed_s: float | None = None,
    ) -> bool:
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
            c.execute(
                "DELETE FROM workers WHERE stopped=1 AND heartbeat < ?", (time.time() - _STOPPED_WORKER_RETENTION_S,)
            )

    def live_workers(self, engine_key: str, max_age_s: float) -> list[dict]:
        with closing(self._connect()) as c:
            rows = c.execute(
                "SELECT * FROM workers WHERE engine_key=? AND stopped=0 AND heartbeat > ?",
                (engine_key, time.time() - max_age_s),
            ).fetchall()
        return [dict(r) for r in rows]


def job_id_for(sha256: str, options: TranscribeOptions, engine_key: str) -> str:
    return f"{sha256[:20]}{options.cache_key(engine_key)[:12]}"
