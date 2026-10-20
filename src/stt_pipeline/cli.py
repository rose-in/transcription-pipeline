"""Command line entry point.

    stt transcribe meeting.mp3 --format srt -o meeting.srt
    stt transcribe call.wav --split-channels --redact-pii --format txt
    stt serve --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .errors import PipelineError
from .formats import FORMATS, render


def _transcribe(args: argparse.Namespace) -> int:
    from .engines import build_engine
    from .pipeline import DirCheckpointStore, Pipeline, PipelineConfig
    from .schemas import TranscribeOptions

    s = Settings()
    if args.model:
        s.model = args.model
    if args.engine:
        s.engine = args.engine
    options = TranscribeOptions(
        language=args.language, word_timestamps=args.words, redact_pii=args.redact_pii,
        channel_mode="split" if args.split_channels else "mix", prompt=args.prompt,
    )
    engine = build_engine(s.engine, **s.engine_kwargs())
    pipeline = Pipeline(engine, PipelineConfig(chunk_s=args.chunk_seconds or s.chunk_s, max_duration_s=s.max_duration_s,
                                               max_split_channels=s.max_split_channels))

    def progress(done: float, total: float | None) -> None:
        pct = f"{100 * done / total:5.1f}%" if total else ""
        print(f"\r  transcribed {done / 60:6.1f} min {pct}", end="", file=sys.stderr, flush=True)

    # Safe to reuse across files/models: every checkpoint carries a fingerprint of the input bytes,
    # engine, options and chunking, and is ignored unless all of them match.
    checkpoints = DirCheckpointStore(Path(args.checkpoint_dir)) if args.checkpoint_dir else None
    try:
        t = pipeline.run(Path(args.input), options, progress=progress, checkpoints=checkpoints)
    except PipelineError as e:
        print(f"\nerror [{e.code}]: {e.message}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    out = render(t, args.format)
    if args.output:
        Path(args.output).write_text(out)
    else:
        sys.stdout.write(out)
    print(
        f"  {t.duration_s / 60:.1f} min audio in {t.timing.total_s:.1f}s (RTF {t.timing.rtf:.3f}, "
        f"{1 / t.timing.rtf if t.timing.rtf else 0:.1f}x realtime), {len(t.segments)} segments, "
        f"language={t.language}, dropped={len(t.dropped)}",
        file=sys.stderr,
    )
    for w in t.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    return 0


def _worker(args: argparse.Namespace) -> int:
    import signal

    from .engines import build_engine
    from .jobs import JobStore, Worker
    from .pipeline import Pipeline, PipelineConfig

    s = Settings()
    engine = build_engine(s.engine, **s.engine_kwargs())
    if hasattr(engine, "warmup"):
        engine.warmup()  # readiness should mean "can serve at full speed", not "weights on disk"
    pipeline = Pipeline(engine, PipelineConfig(chunk_s=s.chunk_s, max_duration_s=s.max_duration_s,
                                               max_split_channels=s.max_split_channels))
    worker = Worker(JobStore(s.data_dir), pipeline, s.engine_key(), lease_s=s.lease_s, max_attempts=s.max_attempts,
                    stall_timeout_s=s.stall_timeout_s, exit_on_stall=True)

    def drain(signum, _frame):  # SIGTERM from k8s / spot notice: hand the job back immediately
        logging.getLogger(__name__).info("signal %s: releasing current job and exiting", signum)
        worker.drain()
        raise SystemExit(0)  # the in-flight native call is abandoned; another worker resumes

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    # The work runs on a separate thread and the main thread only waits: Python delivers signals
    # to the main thread *between bytecodes*, so a main thread stuck inside one long CTranslate2
    # call couldn't run drain() before the orchestrator's SIGKILL.
    import threading

    t = threading.Thread(target=worker.run, name="worker", daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.5)
    return 0


def _serve(args: argparse.Namespace) -> int:
    import signal
    import subprocess

    import uvicorn

    from .api import create_app

    proc = None
    if args.with_worker:  # dev convenience: one command, still two processes
        proc = subprocess.Popen([sys.executable, "-m", "stt_pipeline.cli", "worker"])
        # uvicorn handles SIGINT/SIGTERM itself and returns; make sure the child follows either way
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        if proc and proc.poll() is None:
            proc.terminate()  # worker drains: releases its job for the next worker
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="stt", description="Audio -> timestamped transcript")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe a file")
    t.add_argument("input")
    t.add_argument("-f", "--format", choices=FORMATS, default="txt")
    t.add_argument("-o", "--output")
    t.add_argument("--model", help="e.g. small, large-v3-turbo, distil-large-v3.5 (default: $STT_MODEL or small)")
    t.add_argument("--engine", choices=["faster-whisper", "mock"])
    t.add_argument("--language", help="ISO code; auto-detect if omitted")
    t.add_argument("--words", action="store_true", help="word-level timestamps")
    t.add_argument("--split-channels", action="store_true", help="transcribe stereo channels separately")
    t.add_argument("--redact-pii", action="store_true")
    t.add_argument("--prompt", help="domain vocabulary hint")
    t.add_argument("--chunk-seconds", type=float)
    t.add_argument("--checkpoint-dir", help="resume long runs from per-chunk checkpoints")
    t.set_defaults(fn=_transcribe)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--with-worker", action="store_true", help="also start one worker process")
    s.set_defaults(fn=_serve)

    w = sub.add_parser("worker", help="run a transcription worker (one per GPU / CPU box)")
    w.set_defaults(fn=_worker)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
