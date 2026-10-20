"""FFmpeg probing of untrusted uploads in disposable, resource-limited child processes."""

from __future__ import annotations

import multiprocessing
import threading
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path

from .audio import AudioInfo, probe
from .errors import PipelineError, UndecodableAudio

# Per-child limits, so a hostile file can't pin a core or exhaust RAM.
_PROBE_CPU_LIMIT_S = 30
_PROBE_MEMORY_LIMIT_BYTES = 2 << 30  # 2 GiB address space


def _probe_child(conn: Connection, path: str, max_duration_s: float) -> None:  # pragma: no cover - runs in the child
    import resource

    for lim, val in ((resource.RLIMIT_CPU, _PROBE_CPU_LIMIT_S), (resource.RLIMIT_AS, _PROBE_MEMORY_LIMIT_BYTES)):
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
            # Preloads this module (and through it, audio), so a child imports nothing before probing.
            self._ctx.set_forkserver_preload(["stt_pipeline.sandbox"])
        self._slots = threading.BoundedSemaphore(max_concurrent)

    def probe(self, path: Path, max_duration_s: float) -> AudioInfo:
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
