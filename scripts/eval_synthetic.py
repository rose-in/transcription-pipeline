"""Benchmarks on the synthetic TTS fixtures (scripts/make_fixtures.py); real speech is in eval_real_data.py.

    uv run python scripts/eval_synthetic.py ablation --model small
    uv run python scripts/eval_synthetic.py models --models tiny.en small large-v3-turbo
    uv run python scripts/eval_synthetic.py longform --file fixtures_60min/long_meeting.mp3
    uv run python scripts/eval_synthetic.py scaling fixtures_scaling/meeting_x{1,2,3}.mp3
    uv run python scripts/eval_synthetic.py formats --model tiny.en

Add `--out docs/eval_synthetic.md` to any subcommand to append its section to that file (it never
overwrites).

Every measurement runs in a fresh subprocess so peak RSS is attributable to that one config, and
the pipeline and the plain faster-whisper baselines run with the same number of CPU threads.
Metrics:
  WER          jiwer on Whisper's EnglishTextNormalizer output (so "12%" == "twelve percent")
  start err    |segment start - true utterance start| for the segment nearest each true turn
  words_in_silence  words placed inside a known non-speech stretch (90 s silence + 20 s tone)
  RTF          wall seconds / audio seconds (incl. decode, VAD, inference, post-processing)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from eval_common import ROOT, normalize, peak_rss_mb, percentile

FIX = ROOT / "fixtures"


# --------------------------------------------------------------------------- metrics


def wer(ref: str, hyp: str) -> float:
    import jiwer

    return round(jiwer.wer(normalize(ref), normalize(hyp)), 4)


def start_errors(segments: list[dict], truth: list[dict]) -> dict:
    if not segments:
        return {"start_err_median": None, "start_err_p90": None}
    starts = sorted(s["start"] for s in segments)
    errs = [min(abs(s - t["start"]) for s in starts) for t in truth]
    return {"start_err_median": percentile(errs, 0.5), "start_err_p90": percentile(errs, 0.9)}


def in_gaps(segments: list[dict], truth: list[dict], min_gap: float = 30.0) -> list[dict]:
    """Segments whose midpoint falls inside a known non-speech stretch. Their text is either
    invented (hallucination) or real speech whose segment wrongly spans the silence - the
    report prints the text so the two can be told apart."""
    gaps = [(a["end"] + 0.5, b["start"] - 0.5) for a, b in zip(truth, truth[1:]) if b["start"] - a["end"] > min_gap]
    gaps.append((truth[-1]["end"] + 0.5, float("inf")))
    return [s for s in segments if any(a < (s["start"] + s["end"]) / 2 < b for a, b in gaps)]


def score(stem: str, segments: list[dict], text: str, root: Path = FIX) -> dict:
    truth = json.loads((root / f"{stem}.truth.json").read_text())
    ref = (root / f"{stem}.reference.txt").read_text()
    gap_segs = in_gaps(segments, truth)
    return {
        "wer": wer(ref, text),
        "segments": len(segments),
        **start_errors(segments, truth),
        "max_seg_s": round(max((s["end"] - s["start"] for s in segments), default=0), 1),
        "words_in_silence": sum(len(s["text"].split()) for s in gap_segs),
        "_gap_text": " | ".join(f"[{s['start']:.0f}-{s['end']:.0f}] {s['text'][:120]}" for s in gap_segs),
    }


# ------------------------------------------------------------------------- single runs


def run_one(mode: str, model: str, file: str, extra: dict) -> dict:
    """Executed inside a child process."""
    from stt_pipeline.audio import load_pcm

    path = Path(file)
    if mode in ("mem_whole", "mem_stream"):
        return memory_probe(mode, path)
    from stt_pipeline.engines.devices import default_cpu_threads

    threads = default_cpu_threads()  # same CPU threads for every approach, as in eval_real_data.py
    if mode == "pipeline":
        from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
        from stt_pipeline.pipeline import Pipeline
        from stt_pipeline.schemas import TranscribeOptions

        pipe = Pipeline(FasterWhisperEngine(model, device="cpu", cpu_threads=threads))
        t0 = time.perf_counter()
        t = pipe.run(path, TranscribeOptions(**extra))
        wall = time.perf_counter() - t0
        segs = [s.model_dump() for s in t.segments]
        text, dur = t.text, t.duration_s
        reasons: dict[str, int] = {}
        for d in t.dropped:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
        info = {
            "fallback_decode": sum("fallback_decode" in s.flags for s in t.segments),
            "dropped": reasons or 0,
            "chunks": len(t.chunks),
            "continued_clips": round(t.diagnostics.get("continued_clips", 0)),
            "suspect_ranges": round(t.diagnostics.get("suspect_ranges", 0)),
            "threads": threads,
        }
    else:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        t0 = time.perf_counter()
        # faster-whisper 1.2.1's decode_audio passes av.open(metadata_errors=...), removed in PyAV 19
        audio = load_pcm(path)
        if mode == "sequential":  # reference-style long-form decoding (what openai-whisper does)
            gen, _ = m.transcribe(audio, beam_size=5, vad_filter=False, condition_on_previous_text=True)
        elif mode == "batched_default":  # faster-whisper BatchedInferencePipeline out of the box
            gen, _ = BatchedInferencePipeline(m).transcribe(audio, batch_size=8)
        else:
            raise ValueError(mode)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in gen]
        wall = time.perf_counter() - t0
        text, dur = " ".join(s["text"] for s in segs), len(audio) / 16000
        info = {"threads": threads}
    return {
        "mode": mode,
        "model": model,
        "file": path.name,
        "audio_s": round(dur, 1),
        "wall_s": round(wall, 1),
        "rtf": round(wall / dur, 4),
        "peak_rss_mb": round(peak_rss_mb()),
        **info,
        "_segments": segs,
        "_text": text,
    }


def spawn(mode: str, model: str, file: Path, **extra) -> dict:
    cmd = [sys.executable, __file__, "_child", mode, model, str(file), json.dumps(extra)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    r = json.loads(out.stdout.strip().splitlines()[-1])
    mf = file.parent / "manifest.json"
    if "_segments" in r and mf.exists():  # format variants share the ground truth of the clip they encode
        manifest = {m["file"]: m for m in json.loads(mf.read_text())}
        stem = (manifest.get(file.name) or {}).get("reference")
        if stem:
            r.update(score(stem, r["_segments"], r["_text"], root=file.parent))
    print(
        f"  {mode:16s} {model:15s} {file.name:30s} "
        + " ".join(f"{k}={v}" for k, v in r.items() if not k.startswith("_") and k not in ("mode", "model", "file")),
        file=sys.stderr,
    )
    return r


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- suites


def ablation(model: str) -> str:
    f = FIX / "long_meeting.mp3"
    rows = [spawn(m, model, f) for m in ("sequential", "batched_default", "pipeline")]
    names = {
        "sequential": "A. sequential (reference-style, no VAD)",
        "batched_default": "B. faster-whisper batched defaults",
        "pipeline": "C. this pipeline",
    }
    for r in rows:
        r["approach"] = names[r["mode"]]
    return (
        f"### Ablation: {model}, `{f.name}` ({rows[0]['audio_s'] / 60:.1f} min, 6 voices, 90 s silence + 20 s tone mid-file)\n\n"
        + table(
            rows,
            [
                "approach",
                "wer",
                "segments",
                "start_err_median",
                "start_err_p90",
                "max_seg_s",
                "words_in_silence",
                "rtf",
                "peak_rss_mb",
            ],
        )
        + "\n\nText placed inside the non-speech stretch:\n\n"
        + "\n".join(f"- **{r['approach']}**: {r.get('_gap_text') or '(none)'}" for r in rows)
    )


def models(names: list[str]) -> str:
    files = [("meeting_short.wav", {}), ("meeting_short_8k_mulaw.wav", {}), ("long_meeting.mp3", {})]
    rows = []
    for m in names:
        for fname, extra in files:
            r = spawn("pipeline", m, FIX / fname, **extra)
            r["clip"] = fname
            rows.append(r)
        s = spawn("pipeline", m, FIX / "silence_noise.wav")
        rows.append(
            {
                "model": m,
                "clip": "silence_noise.wav",
                "segments": s["segments"] if "segments" in s else len(s["_segments"]),
                "rtf": s["rtf"],
                "peak_rss_mb": s["peak_rss_mb"],
            }
        )
    return "### Model comparison (this pipeline, CPU int8, Apple M1 Pro)\n\n" + table(
        rows,
        [
            "model",
            "clip",
            "wer",
            "start_err_median",
            "words_in_silence",
            "fallback_decode",
            "segments",
            "rtf",
            "peak_rss_mb",
        ],
    )


def memory_probe(mode: str, path: Path) -> dict:
    """Memory that scales with duration, isolated from the model: decoded audio + the log-mel
    features faster-whisper's batched path builds for every VAD chunk before decoding."""
    import numpy as np
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.feature_extractor import FeatureExtractor
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    from stt_pipeline.audio import iter_pcm, load_pcm, probe
    from stt_pipeline.chunking import silence_aware_chunks

    fe, opts = FeatureExtractor(feature_size=80), VadOptions(max_speech_duration_s=30, min_silence_duration_ms=160)
    base = peak_rss_mb()
    tracemalloc.start()  # numpy reports its buffers to tracemalloc: exact, unlike RSS on macOS
    t0 = time.perf_counter()

    def features(audio):
        stamps = get_speech_timestamps(audio, opts)
        chunks, _ = collect_chunks(audio, stamps, max_duration=30)
        return np.stack([pad_or_trim(fe(c)[..., :-1]) for c in chunks]) if chunks else None

    if mode == "mem_whole":  # what BatchedInferencePipeline does with a whole file
        audio = load_pcm(path, max_duration_s=24 * 3600)
        feats = features(audio)
        dur, n = len(audio) / 16000, 0 if feats is None else len(feats)
    else:  # this pipeline: 10 min chunks, features for one chunk at a time
        dur, n = 0.0, 0
        for chunk in silence_aware_chunks(
            iter_pcm(path, probe(path, max_duration_s=24 * 3600), max_duration_s=24 * 3600)
        ):
            feats = features(chunk.audio)
            dur += chunk.duration_s
            n += 0 if feats is None else len(feats)
            del feats
    traced_peak = tracemalloc.get_traced_memory()[1]
    return {
        "mode": mode,
        "model": "-",
        "file": path.name,
        "audio_s": round(dur, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "windows": n,
        "array_peak_mb": round(traced_peak / 2**20),
        "peak_rss_mb": round(peak_rss_mb()),
        "baseline_rss_mb": round(base),
    }


def scaling(files: list[Path]) -> str:
    rows = []
    for f in files:
        for mode in ("mem_whole", "mem_stream"):
            r = spawn(mode, "-", f)
            r["approach"] = {
                "mem_whole": "whole file (faster-whisper batched)",
                "mem_stream": "this pipeline (streamed)",
            }[mode]
            r["hours"] = round(r["audio_s"] / 3600, 2)
            r["growth_mb"] = r["peak_rss_mb"] - r["baseline_rss_mb"]
            rows.append(r)
    return (
        "### Memory scaling with duration (audio + features only, model excluded)\n\n"
        "`array_peak_mb` = peak bytes held in Python/numpy buffers (tracemalloc, exact). RSS is shown for\n"
        "completeness but is unreliable on macOS (compressed pages are not counted).\n\n"
        + table(rows, ["approach", "hours", "windows", "array_peak_mb", "peak_rss_mb", "wall_s"])
    )


def long_file(file: Path, model: str) -> str:
    rows = [spawn("pipeline", model, file), spawn("batched_default", model, file)]
    for r in rows:
        r["approach"] = {
            "pipeline": "this pipeline (streamed, 10 min chunks)",
            "batched_default": "faster-whisper batched defaults (whole file)",
        }[r["mode"]]
    return f"### Long file end to end: `{file}` ({rows[0]['audio_s'] / 60:.0f} min), {model}\n\n" + table(
        rows,
        [
            "approach",
            "wer",
            "start_err_median",
            "start_err_p90",
            "max_seg_s",
            "dropped",
            "fallback_decode",
            "rtf",
            "peak_rss_mb",
        ],
    )


def formats(model: str) -> str:
    """Every valid `meeting_short` format variant in the fixture manifest (containers, codecs, sample rates)."""
    manifest = json.loads((FIX / "manifest.json").read_text())
    files = [m["file"] for m in manifest if m["kind"] == "valid" and m["file"].startswith("meeting_short")]
    rows = []
    for f in files:
        r = spawn("pipeline", model, FIX / f, language="en")
        rows.append(
            {
                "file": f,
                "WER": f"{100 * r['wer']:.1f}%",
                "clips continued": r["continued_clips"],
                "ranges flagged for re-decode": r["suspect_ranges"],
            }
        )
    return (
        f"### Format matrix: {model}\n\n"
        "`clips continued` = clips where Whisper stopped early and the engine decoded the rest. "
        "`ranges flagged for re-decode` = coverage gaps sent to the fallback.\n\n"
        + table(rows, ["file", "WER", "clips continued", "ranges flagged for re-decode"])
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        _, _, mode, model, file, extra = sys.argv
        print(json.dumps(run_one(mode, model, file, json.loads(extra))))
        return
    ap = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, help="append this section to a markdown file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ablation", parents=[common], help="sequential vs batched defaults vs this pipeline")
    a.add_argument("--model", default="small")
    m = sub.add_parser("models", parents=[common], help="this pipeline across models and clips")
    m.add_argument("--models", nargs="+", default=["tiny.en", "small", "large-v3-turbo"])
    lf = sub.add_parser("longform", parents=[common], help="one long file end to end vs batched defaults")
    lf.add_argument("--file", type=Path, required=True)
    lf.add_argument("--model", default="small")
    sc = sub.add_parser("scaling", parents=[common], help="audio + feature memory vs duration (no model)")
    sc.add_argument("files", type=Path, nargs="+")
    fm = sub.add_parser("formats", parents=[common], help="every meeting_short format variant")
    fm.add_argument("--model", default="tiny.en")
    args = ap.parse_args()
    md = {
        "ablation": lambda: ablation(args.model),
        "models": lambda: models(args.models),
        "longform": lambda: long_file(args.file, args.model),
        "scaling": lambda: scaling(args.files),
        "formats": lambda: formats(args.model),
    }[args.cmd]()
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(md + "\n\n")


if __name__ == "__main__":
    main()
