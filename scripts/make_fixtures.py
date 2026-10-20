"""Generate mock audio fixtures with known ground truth.

Why synthetic audio: it gives us *exact* reference transcripts and *exact*
speech timestamps, so we can measure WER and timestamp error automatically,
and it lets us build a format matrix (WAV/MP3/AAC/Opus/FLAC/telephony/video)
plus hostile inputs without shipping copyrighted recordings.

Caveat (also in the README): TTS speech is far cleaner than real calls or
meetings, so absolute WER numbers here are optimistic. The fixtures are for
regression-testing the *pipeline*, not for choosing a model.

Requires macOS `say` for TTS. Everything else uses PyAV (bundled FFmpeg).

    uv run python scripts/make_fixtures.py               # ~10 min long file
    uv run python scripts/make_fixtures.py --long-minutes 60
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

SR = 22050  # `say` native-ish rate; every encoder resamples from here
# macOS `say` voices, one per English accent, for speaker/accent variety in the long meeting.
VOICES = {
    "Samantha": "en_US",
    "Daniel": "en_GB",
    "Karen": "en_AU",
    "Moira": "en_IE",
    "Tessa": "en_ZA",
    "Rishi": "en_IN",
}

# --------------------------------------------------------------------------- TTS


def tts(text: str, voice: str) -> np.ndarray:
    """Synthesize `text` with macOS `say`; return mono float32 at SR."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "u.wav"
        subprocess.run(
            ["say", "-v", voice, "-o", str(out), "--file-format=WAVE", f"--data-format=LEI16@{SR}", text],
            check=True,
        )
        with wave.open(str(out)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def tone(seconds: float, freqs=(220.0, 277.18, 329.63)) -> np.ndarray:
    """A soft major chord: stands in for 'hold music', a classic hallucination trigger."""
    t = np.arange(int(seconds * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return (0.15 * sig).astype(np.float32)


def add_noise(x: np.ndarray, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0, level, size=x.shape)).astype(np.float32)


# ---------------------------------------------------------------------- encoding


def encode(
    path: Path,
    audio: np.ndarray,
    *,
    container: str | None,
    codec: str,
    rate: int,
    layout: str = "mono",
    bit_rate: int | None = None,
    with_video: bool = False,
    with_audio: bool = True,
) -> None:
    """Encode float32 audio (shape (n,) mono or (2, n) stereo, at SR) with PyAV."""
    planar = audio[None, :] if audio.ndim == 1 else audio
    if layout == "stereo" and planar.shape[0] == 1:
        planar = np.repeat(planar, 2, axis=0)
    duration = planar.shape[1] / SR

    out = av.open(str(path), "w", format=container)
    vstream = None
    if with_video:  # video stream FIRST so audio is not stream #0
        vstream = out.add_stream("libx264", rate=10)
        vstream.width, vstream.height, vstream.pix_fmt = 160, 120, "yuv420p"
    astream = None
    if with_audio:
        astream = out.add_stream(codec, rate=rate)
        astream.layout = layout
        if bit_rate:
            astream.bit_rate = bit_rate

    if vstream is not None:
        black = np.zeros((120, 160, 3), dtype=np.uint8)
        for i in range(int(duration * 10)):
            frame = av.VideoFrame.from_ndarray(black, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 10)
            for p in vstream.encode(frame):
                out.mux(p)
        for p in vstream.encode(None):
            out.mux(p)

    if astream is not None:
        cc = astream.codec_context
        resampler = av.AudioResampler(format=cc.format.name, layout=layout, rate=rate)
        fifo = av.AudioFifo()
        src = av.AudioFrame.from_ndarray(np.ascontiguousarray(planar, dtype=np.float32), format="fltp", layout=layout)
        src.sample_rate = SR
        for f in resampler.resample(src) + resampler.resample(None):
            f.pts = None
            fifo.write(f)
        frame_size = cc.frame_size or 1024
        pts = 0
        while fifo.samples:
            f = fifo.read(min(frame_size, fifo.samples))
            f.pts = pts
            pts += f.samples
            for p in astream.encode(f):
                out.mux(p)
        for p in astream.encode(None):
            out.mux(p)
    out.close()


def write_wav(path: Path, audio: np.ndarray, rate: int = 44100, layout: str = "stereo", codec: str = "pcm_s16le"):
    encode(path, audio, container="wav", codec=codec, rate=rate, layout=layout)


# ------------------------------------------------------------------- scenarios

SHORT_SCRIPT = [
    (
        "Samantha",
        "Good morning everyone. Let's start with the quarterly review. Revenue grew twelve percent compared to last quarter, mostly driven by the new subscription tier.",
    ),
    (
        "Daniel",
        "Thanks. On the engineering side, we shipped the mobile release on Tuesday, and crash rates dropped below half a percent.",
    ),
    ("Samantha", "Great. What are the main risks for next month?"),
    (
        "Daniel",
        "The data migration is the biggest one. We need two more weeks of testing before we can switch over the production database.",
    ),
    ("Karen", "I can help with that. My team has capacity starting Monday, so we could take over the load testing."),
]

CALL_SCRIPT = [  # (channel, voice, text) -> dual-channel call-centre recording
    (0, "Samantha", "Thank you for calling Acme support, this is Sarah. How can I help you today?"),
    (1, "Daniel", "Hi Sarah. My internet has been dropping every evening for about a week."),
    (0, "Samantha", "I'm sorry to hear that. Can I get the phone number on the account?"),
    (1, "Daniel", "Sure, it's 415-555-0132."),
    (0, "Samantha", "Thanks. And the email address?"),
    (1, "Daniel", "It's john.smith@example.com."),
    (0, "Samantha", "Perfect. I can see some line errors. I'll schedule a technician for Thursday morning."),
    (1, "Daniel", "That works. Thank you so much."),
]

NAMES = ["Priya", "Marcus", "Elena", "Tom", "Aisha", "Kenji", "Laura", "Omar"]
PROJECTS = [
    "the billing service",
    "the search index",
    "the mobile app",
    "the data warehouse",
    "the onboarding flow",
    "the payments API",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEMPLATES = [
    "I think {project} is in good shape, but we still need to finish the documentation.",
    "{name} mentioned that latency on {project} went down by {n} percent after the last deploy.",
    "Can we move the review of {project} to {day}? I have a conflict this week.",
    "The customer feedback on {project} has been mostly positive so far.",
    "We found a bug in {project} that only happens when two users edit the same record.",
    "Let's make sure {name} is in the loop before we change anything in {project}.",
    "Our budget for the next quarter is about {n} thousand dollars, so we have to prioritize.",
    "I'd like to propose that we freeze new features until the end of the month.",
    "The test suite for {project} now takes {n} minutes, which is too slow for every pull request.",
    "Has anyone talked to the security team about the access logs?",
    "{name} will send a summary of the decisions after this meeting.",
    "We should write down what success looks like before we start building.",
    "There were {n} support tickets about {project} last week, mostly about login problems.",
    "I agree. Let's schedule a follow up on {day} to look at the numbers again.",
    "The new dashboard shows that most traffic comes from mobile devices in the evening.",
    "If we hire two more engineers, we could finish {project} by the end of the quarter.",
    "Honestly, I'm worried about the deadline. We underestimated the migration work.",
    "Good point. Let's ask {name} to estimate how long the rollback plan would take.",
    "The vendor contract renews on {day}, and they want to raise the price by {n} percent.",
    "Before we wrap up, does anyone have questions about {project}?",
]


def render_turn(rng: random.Random) -> str:
    k = rng.choice([1, 1, 2, 2, 3])
    return " ".join(
        rng.choice(TEMPLATES).format(
            name=rng.choice(NAMES),
            project=rng.choice(PROJECTS),
            day=rng.choice(DAYS),
            n=rng.choice([3, 5, 8, 12, 15, 20, 25, 40]),
        )
        for _ in range(k)
    )


def build_dialog(turns: list[tuple[str, str]], rng: random.Random, gap=(0.3, 2.0)) -> tuple[np.ndarray, list[dict]]:
    """Synthesize turns in parallel, then lay them out with random gaps. Returns (audio, ground-truth turns)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        clips = list(pool.map(lambda vt: tts(vt[1], vt[0]), turns))
    parts, truth, t = [silence(0.8)], [], 0.8
    for (voice, text), clip in zip(turns, clips):
        truth.append({"speaker": voice, "start": round(t, 3), "end": round(t + len(clip) / SR, 3), "text": text})
        parts.append(clip)
        t += len(clip) / SR
        g = rng.uniform(*gap)
        parts.append(silence(g))
        t += g
    return np.concatenate(parts), truth


def build_long(minutes: float, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Multi-speaker meeting with a long silent break and a 'hold music' stretch mid-way."""
    rng = random.Random(seed)
    voices = list(VOICES)
    target = minutes * 60
    pieces, truth, t = [], [], 0.0
    inserted_break = speech_after_break = False
    while t < target or not speech_after_break:
        speech_after_break = inserted_break
        batch = [(rng.choice(voices), render_turn(rng)) for _ in range(24)]
        audio, turns = build_dialog(batch, rng)
        for tr in turns:
            tr["start"] = round(tr["start"] + t, 3)
            tr["end"] = round(tr["end"] + t, 3)
        pieces.append(audio)
        truth.extend(turns)
        t += len(audio) / SR
        if not inserted_break and t > target / 2:
            # 90 s of near-silence then 20 s of tonal "music": both classic Whisper hallucination triggers.
            pieces += [silence(90.0), tone(20.0), silence(2.0)]
            t += 112.0
            inserted_break = True
    return add_noise(np.concatenate(pieces), 0.002, seed), truth


def write_reference(out: Path, stem: str, truth: list[dict]) -> None:
    (out / f"{stem}.truth.json").write_text(json.dumps(truth, indent=1))
    (out / f"{stem}.reference.txt").write_text(" ".join(tr["text"] for tr in truth) + "\n")


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "fixtures")
    ap.add_argument("--long-minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not shutil.which("say"):
        raise SystemExit("macOS `say` is required to synthesize speech fixtures.")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest: list[dict] = []

    # 1) Short multi-speaker clip, encoded in every format we claim to support.
    short, truth = build_dialog(SHORT_SCRIPT, rng)
    short = add_noise(short, 0.001, args.seed)
    write_reference(out, "meeting_short", truth)
    matrix = [
        ("meeting_short.wav", dict(container="wav", codec="pcm_s16le", rate=44100, layout="stereo")),
        ("meeting_short_f32_48k.wav", dict(container="wav", codec="pcm_f32le", rate=48000, layout="mono")),
        ("meeting_short_s24.wav", dict(container="wav", codec="pcm_s24le", rate=96000, layout="stereo")),
        ("meeting_short_8k_mulaw.wav", dict(container="wav", codec="pcm_mulaw", rate=8000, layout="mono")),
        ("meeting_short.mp3", dict(container="mp3", codec="libmp3lame", rate=44100, layout="stereo", bit_rate=128_000)),
        ("meeting_short.m4a", dict(container="ipod", codec="aac", rate=44100, layout="mono", bit_rate=96_000)),
        ("meeting_short.ogg", dict(container="ogg", codec="libopus", rate=48000, layout="mono", bit_rate=32_000)),
        ("meeting_short.webm", dict(container="webm", codec="libopus", rate=48000, layout="mono", bit_rate=24_000)),
        ("meeting_short.flac", dict(container="flac", codec="flac", rate=16000, layout="mono")),
        ("meeting_short.aiff", dict(container="aiff", codec="pcm_s16be", rate=22050, layout="mono")),
        ("meeting_short_video.mp4", dict(container="mp4", codec="aac", rate=44100, layout="stereo", with_video=True)),
    ]
    for name, kw in matrix:
        encode(out / name, short, **kw)
        manifest.append({"file": name, "kind": "valid", "reference": "meeting_short", "expect_status": 200})

    # 2) Dual-channel call: agent on L, customer on R. Contains PII for the redaction step.
    call_turns = [(v, txt) for _, v, txt in CALL_SCRIPT]
    with ThreadPoolExecutor(max_workers=8) as pool:
        clips = list(pool.map(lambda vt: tts(vt[1], vt[0]), call_turns))
    total = int(sum(len(c) for c in clips) + SR * (0.7 * len(clips) + 1))
    stereo, t, ctruth = np.zeros((2, total), dtype=np.float32), int(0.5 * SR), []
    for (ch, voice, text), clip in zip(CALL_SCRIPT, clips):
        stereo[ch, t : t + len(clip)] = clip
        ctruth.append(
            {"speaker": f"channel_{ch}", "start": round(t / SR, 3), "end": round((t + len(clip)) / SR, 3), "text": text}
        )
        t += len(clip) + int(0.7 * SR)
    stereo = add_noise(stereo, 0.001, args.seed)
    write_wav(out / "support_call_dual_channel.wav", stereo, rate=8000, layout="stereo")
    write_reference(out, "support_call_dual_channel", ctruth)
    manifest.append(
        {
            "file": "support_call_dual_channel.wav",
            "kind": "valid",
            "reference": "support_call_dual_channel",
            "expect_status": 200,
        }
    )

    # 3) Silence + noise only: the pipeline must return *no* text, not a hallucination.
    encode(out / "silence_noise.wav", add_noise(silence(30), 0.003, 1), container="wav", codec="pcm_s16le", rate=16000)
    manifest.append(
        {"file": "silence_noise.wav", "kind": "valid", "reference": None, "expect_status": 200, "expect_empty": True}
    )

    # 4) Long-form meeting (default ~10 min; use --long-minutes for hours).
    long, ltruth = build_long(args.long_minutes, args.seed)
    encode(
        out / "long_meeting.mp3", long, container="mp3", codec="libmp3lame", rate=44100, layout="mono", bit_rate=64_000
    )
    write_reference(out, "long_meeting", ltruth)
    manifest.append({"file": "long_meeting.mp3", "kind": "valid", "reference": "long_meeting", "expect_status": 200})

    # 5) Hostile / broken inputs: each must fail fast with a precise 4xx, never a 500 or a hang.
    (out / "empty.wav").write_bytes(b"")
    (out / "not_audio.mp3").write_text("this is a text file pretending to be an mp3\n" * 50)
    (out / "playlist_ssrf.mp3").write_text(  # HLS playlist disguised as MP3: SSRF / local file read probe
        "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://169.254.169.254/latest/meta-data/\n#EXTINF:10,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n"
    )
    mp3 = (out / "meeting_short.mp3").read_bytes()
    (out / "truncated.mp3").write_bytes(mp3[: len(mp3) // 3])  # decodes partially -> must warn, not crash
    (out / "corrupt.wav").write_bytes(
        (out / "meeting_short.wav").read_bytes()[:44] + bytes(random.Random(1).randbytes(20000))
    )
    encode(out / "video_only.mp4", short, container="mp4", codec="aac", rate=44100, with_video=True, with_audio=False)
    manifest += [
        {"file": "empty.wav", "kind": "invalid", "expect_status": 400, "expect_error": "empty_file"},
        {"file": "not_audio.mp3", "kind": "invalid", "expect_status": 415, "expect_error": "unsupported_media"},
        {"file": "playlist_ssrf.mp3", "kind": "invalid", "expect_status": 415, "expect_error": "unsupported_media"},
        {"file": "video_only.mp4", "kind": "invalid", "expect_status": 422, "expect_error": "no_audio_stream"},
        {"file": "truncated.mp3", "kind": "degraded", "expect_status": 200},
        {"file": "corrupt.wav", "kind": "invalid", "expect_status": 422, "expect_error": "undecodable_audio"},
    ]

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} fixtures to {out}")
    print(f"long_meeting.mp3: {len(long) / SR / 60:.1f} min, {len(ltruth)} turns")


if __name__ == "__main__":
    main()
