"""End-to-end smoke test with real processes: API + worker(s) over HTTP.

Demonstrates, with real OS processes and signals rather than threads:
  sync transcription · precise 4xx for a hostile upload · async job with progress ·
  SIGTERM drain mid-job (job released, attempt not counted) · a second worker resuming
  from the first worker's chunk checkpoints · idempotent replay · readiness & metrics.

    uv run python scripts/smoke_e2e.py            # uses tiny.en and 2-minute chunks
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def start(*args: str, env: dict, log: Path) -> subprocess.Popen:
    """Run `stt_pipeline.cli <args>` as a child process, with stdout and stderr captured in `log`."""
    with log.open("wb") as out:  # the child keeps its own copy of the descriptor
        return subprocess.Popen(
            [sys.executable, "-m", "stt_pipeline.cli", *args], env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT
        )


def wait(pred, timeout: float, every: float = 0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(every)
    raise TimeoutError


def upload(c: httpx.Client, path: str, file: Path, **form) -> httpx.Response:
    with file.open("rb") as f:
        return c.post(path, files={"file": (file.name, f)}, data=form, timeout=180)


def main() -> int:
    data = Path(tempfile.mkdtemp(prefix="stt-smoke-"))
    env = {
        **os.environ,
        "STT_DATA_DIR": str(data),
        "STT_MODEL": "tiny.en",
        "STT_CHUNK_SECONDS": "120",
        "STT_LEASE_SECONDS": "10",
    }
    api = start("serve", "--port", str(PORT), env=env, log=data / "api.log")
    workers: list[subprocess.Popen] = []
    c = httpx.Client(base_url=BASE)
    ok = False
    try:
        wait(lambda: _ok(c, "/healthz"), 30)
        print("1. API up (liveness) - readiness without a worker:", c.get("/readyz").status_code)

        workers.append(start("worker", env=env, log=data / "worker-1.log"))
        wait(lambda: _ok(c, "/readyz"), 120)
        print("2. worker up - /readyz:", c.get("/readyz").json()["workers"])

        r = upload(c, "/v1/transcriptions", FIX / "meeting_short.mp3", response_format="srt")
        print(f"3. sync transcription -> {r.status_code}\n   " + "\n   ".join(r.text.splitlines()[:7]))

        r = upload(c, "/v1/transcriptions", FIX / "playlist_ssrf.mp3")
        print(f"4. HLS playlist disguised as .mp3 -> {r.status_code} {r.json()['error']['code']}")

        r = upload(c, "/v1/jobs", FIX / "long_meeting.mp3")
        job = r.json()
        print(f"5. async job -> {r.status_code}, Location: {r.headers['location']}")

        j = wait(lambda: (x := c.get(f"/v1/jobs/{job['id']}").json())["progress"] >= 0.25 and x, 600)
        print(f"6. progress {j['progress']:.0%}, attempts={j['attempts']} - sending SIGTERM to the worker")
        workers[0].send_signal(signal.SIGTERM)
        workers[0].wait(timeout=60)
        j = c.get(f"/v1/jobs/{job['id']}").json()
        print(f"   after drain: status={j['status']} attempts={j['attempts']} (released, attempt not counted)")
        ckpts = len(list((data / "jobs" / job["id"] / "checkpoints").glob("*.json")))
        print(f"   checkpoints on disk: {ckpts}")

        workers.append(start("worker", env=env, log=data / "worker-2.log"))
        j = wait(lambda: (x := c.get(f"/v1/jobs/{job['id']}").json())["status"] in ("succeeded", "failed") and x, 900)
        result = c.get(f"/v1/jobs/{job['id']}/result").json()
        cached = sum(ch["cached"] for ch in result["chunks"])
        print(
            f"7. second worker finished: status={j['status']}, {len(result['segments'])} segments, "
            f"{cached}/{len(result['chunks'])} chunks reused from the first worker's checkpoints"
        )

        r = upload(c, "/v1/jobs", FIX / "long_meeting.mp3")
        print(
            f"8. same file again -> {r.status_code}, Idempotent-Replay: {r.headers.get('idempotent-replay')}, "
            f"same id: {r.json()['id'] == job['id']}"
        )

        print("9. /metrics:\n   " + "\n   ".join(c.get("/metrics").text.strip().splitlines()))
        ok = j["status"] == "succeeded" and cached > 0
        return 0 if ok else 1
    finally:
        for p in [*workers, api]:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
        if ok:
            shutil.rmtree(data, ignore_errors=True)
        else:
            print(f"smoke test failed - process logs and job data kept in {data}", file=sys.stderr)


def _ok(c: httpx.Client, path: str) -> bool:
    try:
        return c.get(path, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
