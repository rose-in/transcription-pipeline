"""Worker: running claimed jobs, keeping or losing the lease, retries, and drain."""

import time

from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.jobs import JobStore
from stt_pipeline.schemas import TranscribeOptions

from .conftest import ENGINE_KEY, bursts, job_worker, submit_job, write_wav


def _crash_without_cleanup(w, job):
    try:
        w.pipeline.run(w.store.input_path(job.id), job.options, checkpoints=w.store.checkpoints(job.id))
    except Exception:
        pass  # process "died": nobody updates the row


def test_dead_worker_lease_expires_and_job_resumes_from_checkpoint(tmp_path):
    store = JobStore(tmp_path)
    p = write_wav(tmp_path / "long.wav", bursts([(t, t + 2) for t in range(1, 79, 4)], 80.0))
    store.submit("long", ENGINE_KEY, p, "long.wav", TranscribeOptions())
    crashing = MockEngine(fail_on_call=3)  # dies on the 3rd chunk, after 2 checkpoints
    w1 = job_worker(store, crashing, lease_s=0.2, max_attempts=5)
    w1._process = lambda job: _crash_without_cleanup(w1, job)  # simulate SIGKILL: no fail() bookkeeping
    w1.run_once()
    assert store.get("long").status == "running"  # orphaned, lease ticking
    time.sleep(0.3)
    fresh = MockEngine()
    w2 = job_worker(store, fresh, lease_s=5)
    assert w2.run_once()
    job = store.get("long")
    assert job.status == "succeeded" and job.attempts == 2
    result = store.result("long")
    assert sum(c.cached for c in result.chunks) == 2  # resumed, not restarted
    assert fresh.calls == len(result.chunks) - 2


def test_worker_that_lost_its_lease_abandons_without_writing_state(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "stolen")
    monkeypatch.setattr(store, "renew", lambda *a, **k: False)  # another worker owns the job now
    assert job_worker(store).run_once()
    job = store.get("stolen")
    assert job.status == "running" and job.attempts == 1 and job.error is None
    assert store.result("stolen") is None


def test_stalled_job_stops_renewing_its_lease(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "slow")

    class Hang(MockEngine):
        def transcribe(self, *a, **k):
            time.sleep(1.0)
            return super().transcribe(*a, **k)

    w = job_worker(store, Hang(), lease_s=0.2, stall_timeout_s=0.3)
    w.run_once()
    assert not w.healthy  # worker marked itself wedged (the CLI worker would exit here)
    assert (
        store.claim("w2", ENGINE_KEY, lease_s=60, max_attempts=3).id == "slow"
    )  # lease lapsed: another worker can take over


def test_transient_failure_retries_with_backoff_then_succeeds(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "flaky")
    w = job_worker(store, MockEngine(fail_on_call=1), max_attempts=3)
    w.run_once()
    job = store.get("flaky")
    assert job.status == "queued" and job.error.code == "engine_failure"
    assert store.claim("other", ENGINE_KEY, 60, 3) is None  # backoff: not claimable yet
    with store._tx() as c:
        c.execute("UPDATE jobs SET not_before=0")
    w.run_once()
    assert store.get("flaky").status == "succeeded"


def test_retryable_failure_on_the_last_attempt_is_final(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "flaky")
    job_worker(store, MockEngine(fail_on_call=1), max_attempts=1).run_once()
    job = store.get("flaky")
    assert job.status == "failed" and job.attempts == 1
    assert job.error.code == "engine_failure" and job.error.status == 500


def test_client_errors_fail_fast_without_retry(tmp_path):
    store = JobStore(tmp_path)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all" * 10)
    store.submit("bad", ENGINE_KEY, bad, "bad.wav", TranscribeOptions())
    job_worker(store).run_once()
    job = store.get("bad")
    assert job.status == "failed" and job.attempts == 1 and job.error.code == "unsupported_media"
    assert job.error.status == 415


def test_rows_with_legacy_options_do_not_crash_the_worker(tmp_path):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "legacy")
    with store._tx() as c:
        c.execute("""UPDATE jobs SET options='{"language": "en-US", "word_timestamps": false, "channel_mode": "mix",
                     "redact_pii": false, "prompt": null}'""")
    job_worker(store).run_once()  # must not raise
    job = store.get("legacy")
    assert job.status == "failed" and job.error.code == "invalid_options" and job.error.status == 422


def test_result_persistence_failure_does_not_kill_the_worker(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    submit_job(store, tmp_path, "j")
    w = job_worker(store)

    def disk_full(*_):
        raise OSError("No space left on device")

    monkeypatch.setattr(store, "complete", disk_full)
    assert w.run_once()  # returned normally
    job = store.get("j")
    assert job.status == "queued" and job.error.code == "storage_error"


def test_graceful_drain_releases_job_at_chunk_boundary(tmp_path):
    store = JobStore(tmp_path)
    p = write_wav(tmp_path / "long.wav", bursts([(t, t + 2) for t in range(1, 79, 4)], 80.0))
    store.submit("drain", ENGINE_KEY, p, "long.wav", TranscribeOptions())
    w = job_worker(store)
    w.stopping.set()  # SIGTERM arrived
    w._process(store.claim(w.id, ENGINE_KEY, 60, 3))
    job = store.get("drain")
    assert job.status == "queued" and job.attempts == 0  # not counted against the job
    assert len(list((store.dir("drain") / "checkpoints").glob("*.json"))) == 1  # first chunk kept
    job_worker(store).run_once()
    assert store.get("drain").status == "succeeded"
