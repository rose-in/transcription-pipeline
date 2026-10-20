"""JobStore queue semantics (claiming, leases, idempotent submission, retention), exercised directly."""

import threading

import pytest

from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.jobs import JobStore
from stt_pipeline.pipeline import Pipeline
from stt_pipeline.schemas import JobError, TranscribeOptions

from .conftest import ENGINE_KEY, bursts, job_worker, submit_job, write_wav


def test_concurrent_workers_never_double_claim(tmp_path):
    store = JobStore(tmp_path)
    for i in range(20):
        submit_job(store, tmp_path, f"job{i}")
    claimed, lock = [], threading.Lock()

    def grab():
        while (job := store.claim(threading.current_thread().name, ENGINE_KEY, lease_s=60, max_attempts=3)) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=grab, name=f"w{i}") for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(f"job{i}" for i in range(20))  # each exactly once


def test_poison_job_is_quarantined(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "poison")
    for _ in range(3):  # three workers each claim it and die
        assert store.claim("w", ENGINE_KEY, lease_s=-1, max_attempts=3) is not None
    assert store.claim("w", ENGINE_KEY, lease_s=60, max_attempts=3) is None
    job = store.get("poison")
    assert job.status == "failed" and job.error.code == "poison_job"


def test_poison_jobs_cannot_be_resurrected_by_resubmitting(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "p")
    for _ in range(3):
        store.claim("w", ENGINE_KEY, lease_s=-1, max_attempts=3)
    store.claim("w", ENGINE_KEY, lease_s=60, max_attempts=3)
    again = write_wav(tmp_path / "again.wav", bursts([(1, 2)], 3.0))
    job, created = store.submit("p", ENGINE_KEY, again, "x", TranscribeOptions())
    assert not created and job.status == "failed" and job.error.code == "poison_job"


def test_resubmitting_a_failed_job_requeues_it(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "again")
    store.claim("w", ENGINE_KEY, 60, 3)
    store.fail("again", "w", JobError(code="engine_failure", message="x"), 500)
    p = write_wav(tmp_path / "again2.wav", bursts([(1, 2)], 3.0))
    job, created = store.submit("again", ENGINE_KEY, p, "again.wav", TranscribeOptions())
    assert created and job.status == "queued" and job.attempts == 0


def test_sync_caller_upgrades_a_queued_async_duplicate(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "d")
    dup = write_wav(tmp_path / "d2.wav", bursts([(1, 2)], 3.0))
    store.submit("d", ENGINE_KEY, dup, "x", TranscribeOptions(), priority=0)
    assert store._one("SELECT priority FROM jobs WHERE id='d'")["priority"] == 0


def test_workers_only_claim_their_engine(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "for-turbo")
    with store._tx() as c:
        c.execute("UPDATE jobs SET engine_key='turbo-pool'")
    assert store.claim("small-worker", ENGINE_KEY, 60, 3) is None
    assert store.claim("turbo-worker", "turbo-pool", 60, 3).id == "for-turbo"


def test_complete_after_lease_takeover_is_ignored(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "taken")
    transcript = Pipeline(MockEngine()).run(store.input_path("taken"))
    store.claim("w1", ENGINE_KEY, lease_s=-1, max_attempts=3)  # w1's lease lapses...
    store.claim("w2", ENGINE_KEY, lease_s=60, max_attempts=3)  # ...and w2 takes the job over
    assert store.complete("taken", "w1", transcript) is False
    job = store.get("taken")
    assert job.status == "running" and job.attempts == 2
    # result.json is written either way: results are deterministic, so w1's copy is as good as w2's


def test_job_under_a_live_lease_cannot_be_deleted(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "busy")
    store.claim("w", ENGINE_KEY, lease_s=60, max_attempts=3)
    assert store.delete("busy") is False
    assert store.get("busy").status == "running" and store.input_path("busy").exists()


def test_orphaned_running_job_can_be_erased(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "o")
    store.claim("dead-worker", ENGINE_KEY, lease_s=-1, max_attempts=3)
    assert store.delete("o") and store.get("o") is None


@pytest.mark.parametrize("ttl,expected", [(3600, 0), (-1, 1)])
def test_retention_purge(tmp_path, ttl, expected):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "old")
    job_worker(store).run_once()
    assert store.purge_expired(ttl) == expected
    assert (store.get("old") is None) == bool(expected)


def test_retention_purge_removes_orphaned_runs_but_not_live_ones(tmp_path):
    store = JobStore(tmp_path)
    # Order matters: claim() takes the oldest claimable job, so "live" must be claimed before "orphan" exists.
    submit_job(store, tmp_path, "live")
    store.claim("alive", ENGINE_KEY, lease_s=60, max_attempts=3)
    submit_job(store, tmp_path, "orphan")
    store.claim("dead", ENGINE_KEY, lease_s=-1, max_attempts=3)
    assert store.purge_expired(-1) == 1
    assert store.get("orphan") is None and store.get("live").status == "running"


def test_stranded_jobs_fail_instead_of_blocking_backpressure(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "s")
    assert store.fail_stranded(older_than_s=-1) == 1
    assert store.get("s").error.code == "engine_unavailable"


def test_worker_restart_does_not_fail_the_backlog(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "old")
    with store._tx() as c:
        c.execute("UPDATE jobs SET created_at = created_at - 7200")  # waited 2 h: normal for long files
    store.register_worker("w1", ENGINE_KEY, "m")
    store.remove_worker("w1")  # clean shutdown during a deploy; replacement still loading
    assert store.fail_stranded(older_than_s=3600) == 0
    assert store.get("old").status == "queued"
