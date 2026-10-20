"""Evaluate on real, human-transcribed public speech (not the synthetic TTS fixtures).

  * LibriSpeech test-clean / test-other (CC BY 4.0): short read speech, a random sample across
    all speakers. The pipeline is compared with plain faster-whisper on the *same clips*
    (paired), which isolates what the pipeline itself changes.
  * Earnings-22 (CC BY-SA 4.0): three real ~25-min earnings calls with accented speakers.
    Long audio end to end, compared with faster-whisper's default batched mode, with segment
    timing measured against the dataset's forced-aligned word times.

All approaches run with the same number of CPU threads, each in a fresh subprocess.

    uv run python scripts/eval_real_data.py download
    uv run python scripts/eval_real_data.py run --out docs/eval_real_data.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval_common import ROOT, normalize, peak_rss_mb, percentile

DATA = ROOT / "datasets"

LS_SAMPLE = {"clean": (2620, 200), "other": (2939, 100)}  # split: (rows in test set, clips sampled)
LS_RUN = 5  # clips fetched per request from a random spot (rows are grouped by speaker)
SEED = 7
E22_FILES = {"4469075": "South Africa", "4472895": "China", "4446796": "Chile"}
E22_BASE = "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings22"
E22_MEDIA = "https://media.githubusercontent.com/media/revdotcom/speech-datasets/main/earnings22/media"
# Reference scores on the FULL test sets, from the HF Open ASR Leaderboard CSVs (fetched 2026-10-20).
# Different implementation (HF transformers, GPU, fp16) - context only, not a like-for-like target.
PUBLISHED = {
    ("large-v3-turbo", "ls-clean"): "2.10",
    ("large-v3-turbo", "ls-other"): "4.24",
    ("small", "ls-clean"): "3.05 (small.en)",
    ("small", "ls-other"): "7.25 (small.en)",
}
# Approaches: "pipeline" (this project), "no_continuation" (this project with continuation of
# early-stopped clips turned off, so the coverage fallback has to catch them: measures what
# continuation buys), "plain" (WhisperModel.transcribe, the model on its own), "batched_default"
# (faster-whisper's BatchedInferencePipeline out of the box).
PLAN = [
    ("librispeech", "pipeline"),
    ("librispeech", "plain"),
    ("earnings22", "pipeline"),
    ("earnings22", "batched_default"),
    ("earnings22", "no_continuation"),
]
E22_MODES = ("batched_default", "no_continuation", "pipeline")


def cpu_s() -> float:
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


# ----------------------------------------------------------------------------- download


def _fetch(url: str) -> bytes:
    """GET with backoff: the Hugging Face API rate-limits bursts (HTTP 429)."""
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(float(e.headers.get("retry-after") or 2**attempt))
    raise RuntimeError("unreachable")


def _get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.write_bytes(_fetch(url))


def download() -> None:
    ls = DATA / "librispeech"
    ls.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest = []
    for split, (total, n) in LS_SAMPLE.items():
        (ls / split).mkdir(exist_ok=True)
        # n/LS_RUN short runs from random, non-overlapping spots: spreads the sample over many
        # speakers while keeping API calls few.
        spots = sorted(rng.sample(range(total // LS_RUN), n // LS_RUN))
        for spot in spots:
            url = (
                f"https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr&config={split}"
                f"&split=test&offset={spot * LS_RUN}&length={LS_RUN}"
            )
            for row in json.loads(_fetch(url))["rows"]:
                x = row["row"]
                dest = ls / split / f"{x['id']}.flac"
                _get(x["audio"][0]["src"], dest)
                manifest.append(
                    {
                        "split": split,
                        "id": x["id"],
                        "speaker": x["speaker_id"],
                        "file": str(dest.relative_to(ROOT)),
                        "text": x["text"],
                    }
                )
            time.sleep(0.5)
        clips = [m for m in manifest if m["split"] == split]
        print(f"librispeech test-{split}: {len(clips)} clips, {len({m['speaker'] for m in clips})} speakers")
    (ls / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))

    e22 = DATA / "earnings22"
    e22.mkdir(parents=True, exist_ok=True)
    for fid in E22_FILES:
        _get(f"{E22_MEDIA}/{fid}.mp3", e22 / f"{fid}.mp3")
        _get(f"{E22_BASE}/transcripts/nlp_references/{fid}.nlp", e22 / f"{fid}.nlp")
        _get(f"{E22_BASE}/transcripts/force_aligned_nlp_references/{fid}.aligned.nlp", e22 / f"{fid}.aligned.nlp")
    print(f"earnings22: {len(E22_FILES)} calls")


# ----------------------------------------------------------------------------- metrics


def errors(ref: str, hyp: str) -> dict:
    """Word errors after Whisper's English normaliser (the leaderboard convention)."""
    import jiwer

    r, h = normalize(ref) or "<empty>", normalize(hyp)
    o = jiwer.process_words(r, h)
    return {
        "sub": o.substitutions,
        "del": o.deletions,
        "ins": o.insertions,
        "n": o.hits + o.substitutions + o.deletions,
    }


def pooled(errs: list[dict]) -> dict:
    n = sum(e["n"] for e in errs) or 1
    s, d, i = (sum(e[k] for e in errs) for k in ("sub", "del", "ins"))
    return {
        "wer": round(100 * (s + d + i) / n, 2),
        "sub": round(100 * s / n, 2),
        "del": round(100 * d / n, 2),
        "ins": round(100 * i / n, 2),
        "ref_words": n,
    }


def paired_delta(a: list[dict], b: list[dict], reps: int = 2000) -> tuple[float, float, float]:
    """WER(b) - WER(a) on the same clips, with a 95% bootstrap CI over clips."""
    rng = random.Random(SEED)

    def wer(errs, idx):
        n = sum(errs[i]["n"] for i in idx) or 1
        return 100 * sum(errs[i]["sub"] + errs[i]["del"] + errs[i]["ins"] for i in idx) / n

    full = list(range(len(a)))
    point = wer(b, full) - wer(a, full)
    boots = sorted(wer(b, idx) - wer(a, idx) for idx in ([rng.randrange(len(a)) for _ in full] for _ in range(reps)))
    return round(point, 2), round(boots[int(0.025 * reps)], 2), round(boots[int(0.975 * reps)], 2)


def read_nlp(path: Path) -> list[dict]:
    out = []
    for r in (line.split("|") for line in path.read_text().splitlines()[1:]):
        out.append(
            {
                "token": r[0],
                "speaker": r[1] if len(r) > 1 else "",
                "ts": float(r[2]) if len(r) > 2 and r[2].strip() else None,
                "end": float(r[3]) if len(r) > 3 and r[3].strip() else None,
                "punct": r[4] if len(r) > 4 else "",
            }
        )
    return out


def sentence_starts(aligned: list[dict], min_gap_s: float = 1.0) -> list[float]:
    """True utterance starts from the forced alignment: after sentence-ending punctuation, on a
    speaker change, or after a pause of >= min_gap_s."""
    starts, prev = [], None
    for t in aligned:
        if t["ts"] is None:
            continue
        if (
            prev is None
            or prev["punct"] in (".", "?", "!")
            or t["speaker"] != prev["speaker"]
            or (prev["end"] is not None and t["ts"] - prev["end"] >= min_gap_s)
        ):
            starts.append(t["ts"])
        prev = t
    return starts


def _tok(w: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", w.lower())


def timing(segments: list[dict], aligned: list[dict]) -> dict:
    """Two views of segment timing against forced-aligned reference word times.

    sentence-start error: for each TRUE sentence start, the distance to the nearest segment
      start. Penalises coarse segmentation (a 30 s segment leaves the sentences inside it far
      from any boundary), so it measures how well segments let you find where things were said.
    boundary precision: for each OUTPUT segment, the distance from its start to the reference
      time of its first aligned word (first 3 words tried). Measures whether the boundaries
      that do exist are placed accurately; blind to coarse segmentation, so report it second.
    """
    import jiwer

    starts = sorted(s["start"] for s in segments)
    ref_starts = sentence_starts(aligned)

    sent_err = []
    for r in ref_starts:
        i = bisect.bisect_left(starts, r)
        sent_err.append(min(abs(starts[j] - r) for j in (i - 1, i) if 0 <= j < len(starts)) if starts else 1e9)

    ref_words = [(t, r) for r in aligned if (t := _tok(r["token"]))]
    hyp_words, seg_of = [], []
    for i, s in enumerate(segments):
        for w in s["text"].split():
            if t := _tok(w):
                hyp_words.append(t)
                seg_of.append(i)
    bound_err = []
    if ref_words and hyp_words:
        o = jiwer.process_words(" ".join(t for t, _ in ref_words), " ".join(hyp_words))
        h2r = {}
        for c in o.alignments[0]:
            if c.type in ("equal", "substitute"):
                for k in range(c.hyp_end_idx - c.hyp_start_idx):
                    h2r[c.hyp_start_idx + k] = c.ref_start_idx + k
        first: dict[int, list[int]] = {}
        for h, i in enumerate(seg_of):
            first.setdefault(i, []).append(h)
        for i, hs in first.items():
            for h in hs[:3]:
                r = h2r.get(h)
                if r is not None and ref_words[r][1]["ts"] is not None:
                    bound_err.append(abs(segments[i]["start"] - ref_words[r][1]["ts"]))
                    break
    return {
        "ref_sentences": len(ref_starts),
        "sent_p50": percentile(sent_err, 0.5),
        "sent_p90": percentile(sent_err, 0.9),
        "sent_within_1s": round(100 * sum(e <= 1.0 for e in sent_err) / max(1, len(sent_err)), 1),
        "bound_p50": percentile(bound_err, 0.5),
        "bound_p90": percentile(bound_err, 0.9),
        "bound_n": len(bound_err),
    }


# ----------------------------------------------------------------------------- runs (child process)


def child(kind: str, mode: str, model: str) -> dict:
    from stt_pipeline.audio import load_pcm
    from stt_pipeline.engines.devices import default_cpu_threads

    threads = default_cpu_threads()
    diag: dict = {}
    if mode in ("pipeline", "no_continuation"):
        from stt_pipeline.engines.faster_whisper_engine import FasterWhisperEngine
        from stt_pipeline.pipeline import Pipeline
        from stt_pipeline.schemas import TranscribeOptions

        extra = {"continuation": False} if mode == "no_continuation" else {}
        pipe = Pipeline(FasterWhisperEngine(model, cpu_threads=threads, **extra))

        def transcribe(path: Path) -> tuple[list[dict], float]:
            t = pipe.run(path, TranscribeOptions(language="en"))
            for k, v in t.diagnostics.items():
                diag[k] = round(diag.get(k, 0) + v, 2)
            for d in t.dropped:
                diag[f"dropped_{d.reason}"] = diag.get(f"dropped_{d.reason}", 0) + 1
            return [s.model_dump() for s in t.segments], t.duration_s
    else:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        batched = BatchedInferencePipeline(wm)

        def transcribe(path: Path) -> tuple[list[dict], float]:
            # faster-whisper 1.2.1's decode_audio passes av.open(metadata_errors=...), removed in PyAV 19
            audio = load_pcm(path)
            if mode == "plain":
                segs, _ = wm.transcribe(audio, language="en", beam_size=5)
            else:
                segs, _ = batched.transcribe(audio, language="en", batch_size=8)
            return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs], len(audio) / 16000

    out: dict = {"kind": kind, "mode": mode, "model": model, "threads": threads}
    t0, c0, audio_s = time.perf_counter(), cpu_s(), 0.0
    if kind == "librispeech":
        clips = [json.loads(line) for line in (DATA / "librispeech" / "manifest.jsonl").read_text().splitlines()]
        per_clip = []
        for c in clips:
            segs, dur = transcribe(ROOT / c["file"])
            audio_s += dur
            per_clip.append(
                {
                    "id": c["id"],
                    "split": c["split"],
                    "speaker": c["speaker"],
                    **errors(c["text"], " ".join(s["text"] for s in segs)),
                }
            )
        out["clips"] = per_clip
    else:
        files = {}
        for fid, country in E22_FILES.items():
            segs, dur = transcribe(DATA / "earnings22" / f"{fid}.mp3")
            audio_s += dur
            ref = " ".join(r["token"] for r in read_nlp(DATA / "earnings22" / f"{fid}.nlp"))
            aligned = read_nlp(DATA / "earnings22" / f"{fid}.aligned.nlp")
            usable = sum(r["ts"] is not None for r in aligned) / max(1, len(aligned)) > 0.8
            lens = sorted(s["end"] - s["start"] for s in segs)
            files[fid] = {
                "country": country,
                "minutes": round(dur / 60, 1),
                "errors": errors(ref, " ".join(s["text"] for s in segs)),
                "segments": len(segs),
                "median_seg_s": round(lens[len(lens) // 2], 1) if lens else None,
                "max_seg_s": round(lens[-1], 1) if lens else None,
                "segs_over_30s": sum(x > 30 for x in lens),
                "timing": timing(segs, aligned) if usable else None,
            }
        out["files"] = files
    out["audio_s"] = round(audio_s, 1)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    out["cpu_s"] = round(cpu_s() - c0, 1)
    out["rtf"] = round(out["wall_s"] / audio_s, 4) if audio_s else None
    out["peak_rss_mb"] = round(peak_rss_mb())
    if diag:
        out["diagnostics"] = diag
    return out


def spawn(kind: str, mode: str, model: str) -> dict:
    p = subprocess.run(
        [sys.executable, __file__, "_child", kind, mode, model], capture_output=True, text=True, cwd=ROOT
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    print(json.dumps({k: v for k, v in r.items() if k != "clips"}), file=sys.stderr, flush=True)
    return r


# ----------------------------------------------------------------------------- report


def report(results: list[dict]) -> str:
    md = [
        "## Real-data evaluation",
        "",
        "CPU int8, Apple M1 Pro, every approach with the same number of threads. WER after Whisper's English "
        "normaliser (the leaderboard convention). `del` = reference words missing from the output (on Earnings-22 "
        "this includes verbatim repeats and false starts the references keep); `ins` = output words not in the "
        "reference (not necessarily made up).",
        "",
    ]
    by = {(r["kind"], r["mode"], r["model"]): r for r in results}
    models = sorted({r["model"] for r in results}, key=lambda m: m != "small")

    md += [
        "### LibriSpeech: does the pipeline change accuracy? (random sample, paired on the same clips)",
        "",
        "| model | split | clips / speakers | model alone (plain) | through the pipeline | difference (95% CI) | published, full set, other implementation |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        p, q = by.get(("librispeech", "plain", m)), by.get(("librispeech", "pipeline", m))
        if not (p and q):
            continue
        for split in ("clean", "other"):
            a = [c for c in p["clips"] if c["split"] == split]
            b = [c for c in q["clips"] if c["split"] == split]
            d, lo, hi = paired_delta(a, b)
            md.append(
                f"| {m} | test-{split} | {len(a)} / {len({c['speaker'] for c in a})} | {pooled(a)['wer']}% | "
                f"{pooled(b)['wer']}% | {d:+.2f} pt ({lo:+.2f} to {hi:+.2f}) | {PUBLISHED.get((m, 'ls-' + split), '-')} |"
            )
    md.append("")

    md += [
        "### Earnings-22: three real calls (South Africa 27 min, China 25 min, Chile 28 min)",
        "",
        "| model | approach | WER | sub / del / ins | per call (SA / China / Chile) | wall RTF | CPU s per audio s | peak RSS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for mode in E22_MODES:
            r = by.get(("earnings22", mode, m))
            if not r:
                continue
            errs = [f["errors"] for f in r["files"].values()]
            p = pooled(errs)
            per = " / ".join(f"{pooled([e])['wer']}" for e in errs)
            md.append(
                f"| {m} | {mode} | {p['wer']}% | {p['sub']} / {p['del']} / {p['ins']} | {per} | {r['rtf']} | "
                f"{round(r['cpu_s'] / r['audio_s'], 3)} | {r['peak_rss_mb']} MB |"
            )
    md += [
        "",
        "Segment timing against the forced-aligned reference (China and Chile; the South Africa alignment in "
        "the dataset is unusable):",
        "",
        "| model | approach | call | segments | median / max segment | segments > 30 s | sentence-start error p50 / p90 | sentence starts within 1 s | boundary precision p50 / p90 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        for mode in E22_MODES:
            r = by.get(("earnings22", mode, m))
            if not r:
                continue
            for f in r["files"].values():
                t = f["timing"]
                if not t:
                    continue
                md.append(
                    f"| {m} | {mode} | {f['country']} | {f['segments']} | {f['median_seg_s']} / {f['max_seg_s']} s | "
                    f"{f['segs_over_30s']} | {t['sent_p50']} / {t['sent_p90']} s | {t['sent_within_1s']}% | "
                    f"{t['bound_p50']} / {t['bound_p90']} s |"
                )
    rows = [
        (m, mode, by[("earnings22", mode, m)])
        for m in models
        for mode in ("no_continuation", "pipeline")
        if ("earnings22", mode, m) in by
    ]
    if rows:
        md += [
            "",
            "Safety-net activity (all three calls):",
            "",
            "| model | approach | early stops detected | ranges flagged for re-decode | sent to the slow sequential decoder | audio re-decoded | time in safety net | segments dropped by the quality gate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m, mode, r in rows:
            d = r.get("diagnostics", {})
            dropped = {k.removeprefix("dropped_"): v for k, v in d.items() if k.startswith("dropped_")}
            md.append(
                f"| {m} | {mode} | {d.get('continued_clips', 0):.0f} | {d.get('suspect_ranges', 0):.0f} | {d.get('stage2_ranges', 0):.0f} | "
                f"{d.get('redecoded_s', 0):.0f} s | {d.get('fallback_s', 0):.0f} s of {r['wall_s']:.0f} s | {dropped or 0} |"
            )
    return "\n".join(md) + "\n"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        print(json.dumps(child(*sys.argv[2:5])))
        return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    rep = sub.add_parser("report", help="rebuild the markdown from a saved --out .json")
    rep.add_argument("json", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--models", nargs="+", default=["small", "large-v3-turbo"])
    run.add_argument("--skip", nargs="*", default=[], help="model:mode pairs to skip")
    run.add_argument("--out", type=Path)
    run.add_argument("--modes", nargs="*", help="only re-run these approaches; the rest come from --reuse")
    run.add_argument("--reuse", type=Path, help="an earlier --out .json to take the other approaches' rows from")
    args = ap.parse_args()
    if args.cmd == "download":
        download()
        return
    if args.cmd == "report":
        md = report(json.loads(args.json.read_text()))
        args.json.with_suffix(".md").write_text(md)
        print(md)
        return
    old = {(r["model"], r["kind"], r["mode"]): r for r in json.loads(args.reuse.read_text())} if args.reuse else {}
    results = []
    for model in args.models:
        for kind, mode in PLAN:
            if f"{model}:{mode}" in args.skip:
                continue
            if args.modes and mode not in args.modes and (model, kind, mode) in old:
                results.append(old[(model, kind, mode)])
            else:
                results.append(spawn(kind, mode, model))
            if args.out:  # save as we go: a long run shouldn't lose everything to one failure
                args.out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    md = report(results)
    print(md)
    if args.out:
        args.out.write_text(md)


if __name__ == "__main__":
    main()
