import sqlite3
import time
from typing import get_args

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stt_pipeline.api import OutputFormat, create_app
from stt_pipeline.config import Settings
from stt_pipeline.engines.mock import MockEngine
from stt_pipeline.formats import FORMATS

from .conftest import bursts, write_wav


def settings_for(tmp_path, **overrides) -> Settings:
    s = Settings()
    s.data_dir, s.engine = tmp_path / "data", "mock"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_client(tmp_path, worker=True, **overrides):
    s = settings_for(tmp_path, **overrides)
    return TestClient(create_app(s, worker_engine=MockEngine() if worker else None))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def post(client, path, file, url="/v1/transcriptions", **form):
    with open(file, "rb") as f:
        return client.post(url, files={"file": (path, f)}, data=form)


def wait_for(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    deadline = time.time() + 5
    while client.get("/readyz").status_code != 200 and time.time() < deadline:
        time.sleep(0.02)
    assert client.get("/readyz").json()["ready"] is True


def test_not_ready_without_workers(tmp_path):
    with make_client(tmp_path, worker=False) as c:
        assert c.get("/healthz").status_code == 200  # alive...
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["workers"] == []  # ...but can't serve


def test_every_fixture_gets_the_documented_status(client, manifest, fixtures_dir):
    for m in manifest:
        if m["file"].startswith("long_"):
            continue  # longer than the sync limit; covered below
        r = post(client, m["file"], fixtures_dir / m["file"])
        assert r.status_code == m["expect_status"], (m["file"], r.text)
        if "expect_error" in m:
            assert r.json()["error"]["code"] == m["expect_error"]


def _silent_mp3(path, seconds: float):
    with av.open(str(path), "w", format="mp3") as out:
        st = out.add_stream("libmp3lame", rate=16000)
        st.layout = "mono"
        f = av.AudioFrame.from_ndarray(
            np.zeros((1, int(16000 * seconds)), dtype=np.float32), format="fltp", layout="mono"
        )
        f.sample_rate = 16000
        for pkt in [*st.encode(f), *st.encode(None)]:
            out.mux(pkt)
    return path


def test_sync_limit_applies_in_the_worker_when_duration_is_only_an_estimate(tmp_path):
    # MP3 duration is a bitrate estimate, so intake queues the job and the worker's decode enforces the limit.
    p = _silent_mp3(tmp_path / "a.mp3", 20.0)
    with make_client(tmp_path, sync_max_duration_s=10) as c:
        r = post(c, "a.mp3", p)
        assert r.status_code == 413
        message = r.json()["error"]["message"]
        assert message.count("Use POST /v1/jobs") == 1
        (job_id,) = [row[0] for row in sqlite3.connect(tmp_path / "data" / "jobs.db").execute("SELECT id FROM jobs")]
        job = c.get(f"/v1/jobs/{job_id}").json()
        assert job["error"]["message"] == message  # a polled sync job reports the same hint
        assert post(c, "a.mp3", p, url="/v1/jobs").status_code == 202  # sync jobs have their own key


def test_sync_limit_rejects_a_trusted_header_duration_at_intake(tmp_path):
    # A WAV header duration is trusted, so the request is refused before any job is created.
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2), (8, 9)], 12.0))
    with make_client(tmp_path, sync_max_duration_s=5) as c:
        r = post(c, "a.wav", p)
        assert r.status_code == 413 and "/v1/jobs" in r.json()["error"]["message"]
        assert post(c, "a.wav", p, url="/v1/jobs").status_code == 202


def test_sync_request_that_outlives_the_timeout_becomes_a_job(tmp_path):
    with make_client(tmp_path, worker=False, sync_timeout_s=0.1) as c:  # nobody will run it
        p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
        r = post(c, "a.wav", p)
        assert r.status_code == 202 and r.headers["Location"] == f"/v1/jobs/{r.json()['id']}"
        assert r.json()["status"] == "queued"


def test_response_formats(client, tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2), (3, 4)], 5.0))
    srt = post(client, "a.wav", p, response_format="srt")
    assert srt.status_code == 200 and srt.headers["content-type"].startswith("application/x-subrip")
    assert srt.text.startswith("1\n00:00:01,000 --> ")
    vtt = post(client, "a.wav", p, response_format="vtt")
    assert vtt.text.startswith("WEBVTT")


def test_async_job_lifecycle_and_idempotency(client, tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2), (5, 6)], 8.0))
    r = post(client, "a.wav", p, url="/v1/jobs")
    assert r.status_code == 202 and r.headers["Location"] == f"/v1/jobs/{r.json()['id']}"
    job_id = r.json()["id"]
    assert wait_for(client, job_id)["status"] == "succeeded"
    assert 'stt_jobs{status="succeeded"} 1' in client.get("/metrics").text
    assert len(client.get(f"/v1/jobs/{job_id}/result").json()["segments"]) == 2
    assert client.get(f"/v1/jobs/{job_id}/result?format=txt").text.startswith("[00:00:01]")

    again = post(client, "renamed.wav", p, url="/v1/jobs")  # same bytes + options => same job
    assert again.status_code == 200 and again.json()["id"] == job_id
    assert again.headers["Idempotent-Replay"] == "true"
    different = post(client, "a.wav", p, url="/v1/jobs", word_timestamps="true")
    assert different.json()["id"] != job_id  # options are part of the key

    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_uploaded_filename_is_never_exposed(client, tmp_path):
    # Results are shared by content hash, so one uploader's file name must not reach anyone else.
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
    r = post(client, "client-secret.wav", p, url="/v1/jobs")
    job_id = r.json()["id"]
    assert wait_for(client, job_id)["status"] == "succeeded"
    for body in (r.text, client.get(f"/v1/jobs/{job_id}").text, client.get(f"/v1/jobs/{job_id}/result").text):
        assert "client-secret" not in body


def test_result_before_completion_is_409(tmp_path):
    with make_client(tmp_path, worker=False) as c:  # nobody will run it
        p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
        job_id = post(c, "a.wav", p, url="/v1/jobs").json()["id"]
        r = c.get(f"/v1/jobs/{job_id}/result")
        assert r.status_code == 409 and r.json()["error"]["code"] == "not_ready"


def test_deleting_a_running_job_is_409(tmp_path):
    with make_client(tmp_path, worker=False) as c:
        p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
        job_id = post(c, "a.wav", p, url="/v1/jobs").json()["id"]
        engine_key = c.get("/readyz").json()["engine_key"]
        claimed = c.app.state.store.claim("w", engine_key, lease_s=60, max_attempts=3)  # a live worker holds it
        assert claimed.id == job_id
        r = c.delete(f"/v1/jobs/{job_id}")
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        assert c.get(f"/v1/jobs/{job_id}").json()["status"] == "running"  # nothing was deleted


def test_queued_jobs_survive_api_restart(tmp_path):
    with make_client(tmp_path, worker=False) as c:  # API up, no worker: job just sits in the DB
        p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
        job_id = post(c, "a.wav", p, url="/v1/jobs").json()["id"]
        assert c.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"
    with make_client(tmp_path) as c:  # restart with a worker: it picks the job up
        assert wait_for(c, job_id)["status"] == "succeeded"


def test_backpressure_when_queue_is_full(tmp_path):
    with make_client(tmp_path, worker=False, max_queue=1) as c:
        a = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
        b = write_wav(tmp_path / "b.wav", bursts([(0.5, 2)], 3.0))
        assert post(c, "a.wav", a, url="/v1/jobs").status_code == 202
        r = post(c, "b.wav", b, url="/v1/jobs")
        assert r.status_code == 503 and r.headers["Retry-After"] and r.json()["error"]["code"] == "queue_full"
        assert post(c, "a.wav", a, url="/v1/jobs").status_code == 200  # idempotent replay is not new work


def test_oversized_body_is_rejected_before_parsing(tmp_path):
    with make_client(tmp_path, worker=False, max_upload_bytes=1024) as c:
        r = c.post("/v1/jobs", files={"file": ("big.wav", b"\0" * (3 << 20))})
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    uploads = tmp_path / "data" / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_oversized_chunked_body_is_rejected(tmp_path):
    # An iterator body is sent chunked (no Content-Length), so BodyLimit has to count bytes as they
    # stream. The content-type must be multipart, or the app never reads the body.
    with make_client(tmp_path, worker=False, max_upload_bytes=1024) as c:
        r = c.post(
            "/v1/jobs", content=iter([b"\0" * (3 << 20)]), headers={"content-type": "multipart/form-data; boundary=x"}
        )
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_invalid_options_are_422_and_name_the_field(client, tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
    r = post(client, "a.wav", p, channel_mode="surround")
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_options"
    assert err["message"].startswith("channel_mode: ") and "pydantic" not in err["message"]


def test_split_channel_over_http(client, tmp_path):
    st = np.stack([bursts([(1, 2)], 6.0, seed=1), bursts([(3, 4)], 6.0, seed=2)])
    p = write_wav(tmp_path / "call.wav", st)
    segs = post(client, "call.wav", p, channel_mode="split").json()["segments"]
    assert [s["speaker"] for s in segs] == ["channel_0", "channel_1"]


def test_job_ids_cannot_escape_the_jobs_directory(client):
    store = client.app.state.store
    for bad in ("..", "...", "a.b", "%2e%2e"):
        assert client.get(f"/v1/jobs/{bad}").status_code == 404
        assert client.delete(f"/v1/jobs/{bad}").status_code in (204, 404)  # nothing deleted either way
        with pytest.raises(KeyError):
            store.dir(bad.replace("%2e", "."))
    assert store.root.exists()


def test_api_accepts_exactly_the_renderable_formats():
    assert get_args(OutputFormat) == FORMATS
