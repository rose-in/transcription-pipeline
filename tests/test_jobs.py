"""Queue semantics: the guarantees jobs.py documents, exercised directly."""

import threading
import time

import pytest

from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.jobs import JobStore, Worker
from stt_pipeline.pipeline import Pipeline, PipelineConfig
from stt_pipeline.schemas import TranscribeOptions

from .conftest import bursts, write_wav

KEY = "test-engine"


def submit(store, tmp_path, name, seconds=3.0):
    p = write_wav(tmp_path / f"{name}.wav", bursts([(1, 2)], seconds))
    job, created = store.submit(name, KEY, p, f"{name}.wav", TranscribeOptions())
    assert created
    return job


def worker(store, engine=None, **kw):
    return Worker(store, Pipeline(engine or MockEngine(), PipelineConfig(chunk_s=20, chunk_search_s=5)), KEY, **kw)


def test_concurrent_workers_never_double_claim(tmp_path):
    store = JobStore(tmp_path)
    for i in range(20):
        submit(store, tmp_path, f"job{i}")
    claimed, lock = [], threading.Lock()

    def grab():
        while (job := store.claim(threading.current_thread().name, KEY, lease_s=60, max_attempts=3)) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=grab, name=f"w{i}") for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(f"job{i}" for i in range(20))  # each exactly once


def test_dead_worker_lease_expires_and_job_resumes_from_checkpoint(tmp_path):
    store = JobStore(tmp_path)
    p = write_wav(tmp_path / "long.wav", bursts([(t, t + 2) for t in range(1, 79, 4)], 80.0))
    store.submit("long", KEY, p, "long.wav", TranscribeOptions())
    crashing = MockEngine(fail_on_call=3)  # dies on the 3rd chunk, after 2 checkpoints
    w1 = worker(store, crashing, lease_s=0.2, max_attempts=5)
    w1._process = lambda job: _crash_without_cleanup(w1, job)  # simulate SIGKILL: no fail() bookkeeping
    w1.run_once()
    assert store.get("long").status == "running"  # orphaned, lease ticking
    time.sleep(0.3)
    fresh = MockEngine()
    w2 = worker(store, fresh, lease_s=5)
    assert w2.run_once()
    job = store.get("long")
    assert job.status == "succeeded" and job.attempts == 2
    result = store.result("long")
    assert sum(c.cached for c in result.chunks) == 2  # resumed, not restarted
    assert fresh.calls == len(result.chunks) - 2


def _crash_without_cleanup(w, job):
    try:
        w.pipeline.run(w.store.input_path(job.id), job.options, checkpoints=w.store.checkpoints(job.id))
    except Exception:
        pass  # process "died": nobody updates the row


def test_poison_job_is_quarantined(tmp_path):
    store = JobStore(tmp_path)
    submit(store, tmp_path, "poison")
    for _ in range(3):  # three workers each claim it and die
        assert store.claim("w", KEY, lease_s=-1, max_attempts=3) is not None
    assert store.claim("w", KEY, lease_s=60, max_attempts=3) is None
    job = store.get("poison")
    assert job.status == "failed" and job.error.code == "poison_job"


def test_transient_failure_retries_with_backoff_then_succeeds(tmp_path):
    store = JobStore(tmp_path)
    submit(store, tmp_path, "flaky")
    w = worker(store, MockEngine(fail_on_call=1), max_attempts=3)
    w.run_once()
    job = store.get("flaky")
    assert job.status == "queued" and job.error.code == "engine_failure"
    assert store.claim("other", KEY, 60, 3) is None  # backoff: not claimable yet
    with store._tx() as c:
        c.execute("UPDATE jobs SET not_before=0")
    w.run_once()
    assert store.get("flaky").status == "succeeded"


def test_client_errors_fail_fast_without_retry(tmp_path):
    store = JobStore(tmp_path)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all" * 10)
    store.submit("bad", KEY, bad, "bad.wav", TranscribeOptions())
    worker(store).run_once()
    job = store.get("bad")
    assert job.status == "failed" and job.attempts == 1 and job.error.code == "unsupported_media"
    assert job.error.status == 415


def test_graceful_drain_releases_job_at_chunk_boundary(tmp_path):
    store = JobStore(tmp_path)
    p = write_wav(tmp_path / "long.wav", bursts([(t, t + 2) for t in range(1, 79, 4)], 80.0))
    store.submit("drain", KEY, p, "long.wav", TranscribeOptions())
    w = worker(store)
    w.stopping.set()  # SIGTERM arrived
    w._process(store.claim(w.id, KEY, 60, 3))
    job = store.get("drain")
    assert job.status == "queued" and job.attempts == 0  # not counted against the job
    assert len(list((store.dir("drain") / "checkpoints").glob("*.json"))) == 1  # first chunk kept
    worker(store).run_once()
    assert store.get("drain").status == "succeeded"


def test_workers_only_claim_their_engine(tmp_path):
    store = JobStore(tmp_path)
    submit(store, tmp_path, "for-turbo")
    with store._tx() as c:
        c.execute("UPDATE jobs SET engine_key='turbo-pool'")
    assert store.claim("small-worker", KEY, 60, 3) is None
    assert store.claim("turbo-worker", "turbo-pool", 60, 3).id == "for-turbo"


def test_resubmitting_a_failed_job_requeues_it(tmp_path):
    store = JobStore(tmp_path)
    job = submit(store, tmp_path, "again")
    store.claim("w", KEY, 60, 3)
    from stt_pipeline.schemas import JobError
    store.fail("again", "w", JobError(code="engine_failure", message="x"), 500)
    p = write_wav(tmp_path / "again2.wav", bursts([(1, 2)], 3.0))
    job, created = store.submit("again", KEY, p, "again.wav", TranscribeOptions())
    assert created and job.status == "queued" and job.attempts == 0


@pytest.mark.parametrize("ttl,expected", [(3600, 0), (-1, 1)])
def test_retention_purge(tmp_path, ttl, expected):
    store = JobStore(tmp_path)
    submit(store, tmp_path, "old")
    worker(store).run_once()
    assert store.purge_expired(ttl) == expected
    assert (store.get("old") is None) == bool(expected)
