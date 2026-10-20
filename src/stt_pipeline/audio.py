"""Audio ingestion: sniff -> probe -> stream-decode to 16 kHz mono float32.

Design: ONE decode path for every input. Whatever arrives (WAV, MP3, AAC/M4A,
Opus/WebM from a browser, 8 kHz mu-law telephony, the audio track of an MP4...)
is canonicalised by FFmpeg's libraries (via PyAV, which bundles them, so there
is no system ffmpeg to install, patch or version-skew) into exactly what the
model consumes: 16 kHz, mono, float32 PCM. Nothing downstream ever branches on
format.

Defence in depth for untrusted uploads:
  1. Magic-byte sniff against an allowlist. File extension and client MIME type
     are ignored: both are attacker-controlled and frequently just wrong.
  2. FFmpeg itself is restricted with ``format_whitelist`` (only the demuxers
     below; never hls/concat/image2/...) and ``protocol_whitelist=file``, so a
     crafted playlist can't make the decoder fetch URLs (SSRF) or read local
     files.
  3. Decoding streams in bounded blocks, enforcing a max duration *while*
     decoding: container-reported durations are often absent or wrong (VBR MP3,
     MediaRecorder WebM), so they can't be trusted for limits.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

from .errors import AudioTooLong, EmptyFile, NoAudioStream, UndecodableAudio, UnsupportedMedia

TARGET_SR = 16_000

ALLOWED_DEMUXERS = ("wav", "w64", "mp3", "aac", "flac", "ogg", "mov", "matroska", "aiff", "amr", "asf", "caf", "au", "wv")
FFMPEG_OPTIONS = {
    "format_whitelist": ",".join(ALLOWED_DEMUXERS),
    "protocol_whitelist": "file",
    "max_streams": "16",  # crafted containers with thousands of streams are a cheap DoS
}

_ASF_GUID = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")


def sniff_container(head: bytes) -> str | None:
    """Identify the container from its first bytes. Returns None for anything not on the allowlist."""
    if len(head) < 4:
        return None
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"riff":  # Sony Wave64 GUID starts with 'riff'
        return "w64"
    if head[:3] == b"ID3":
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:5] == b"#!AMR":
        return "amr"
    if head[:16] == _ASF_GUID:
        return "asf"
    if head[:4] == b"caff":
        return "caf"
    if head[:4] == b".snd":
        return "au"
    if head[:4] == b"wvpk":
        return "wv"
    if head[0] == 0xFF and (head[1] & 0xF6) == 0xF0:
        return "aac"  # ADTS
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"  # bare MPEG audio frame sync
    return None


@dataclass(frozen=True)
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    stream_index: int
    container_duration_s: float | None
    size_bytes: int
    start_offset_s: float = 0.0
    duration_is_estimate: bool = False


# Containers whose duration FFmpeg typically *estimates from bitrate* (no reliable header):
# fine for progress bars, never for rejecting a file up front.
_ESTIMATED_DURATION_CONTAINERS = {"mp3", "aac"}
MAX_STREAMS = 16


def _open(path: Path, *, unsupported: bool = False) -> av.container.InputContainer:
    try:
        return av.open(str(path), options=FFMPEG_OPTIONS)
    except av.error.FFmpegError as e:
        if unsupported:  # neither the sniff nor any allowlisted demuxer recognises it
            raise UnsupportedMedia(
                "Unrecognised file type (content sniffing; extension and MIME type are ignored). "
                f"Supported containers: {', '.join(ALLOWED_DEMUXERS)}."
            ) from e
        raise UndecodableAudio(f"FFmpeg could not open the file: {e.strerror or e}") from e


def probe(path: Path, *, max_duration_s: float | None = None) -> AudioInfo:
    """Validate and describe the input without decoding it. Raises typed, client-facing errors."""
    size = path.stat().st_size
    if size == 0:
        raise EmptyFile("The uploaded file is empty.")
    with path.open("rb") as f:
        head = f.read(64)
    # The sniff is a cheap fast path, not the security boundary (FFmpeg's demuxer allowlist is).
    # A file it doesn't recognise (MP3 with leading junk, LATM AAC...) still gets a chance with
    # the allowlisted demuxers, and is reported as 415 if none of them can open it.
    sniffed = sniff_container(head)
    with _open(path, unsupported=sniffed is None) as container:
        if len(container.streams) >= MAX_STREAMS:
            raise UnsupportedMedia(f"Container has {len(container.streams)}+ streams; refusing (limit {MAX_STREAMS}).")
        audio = [s for s in container.streams if s.type == "audio"]
        if not audio:
            raise NoAudioStream("The file contains no audio stream.")
        # PyAV leaves codec_context=None when FFmpeg has no decoder for the codec.
        decodable = [s for s in audio if s.codec_context is not None]
        if not decodable:
            raise UndecodableAudio("No decoder is available for this file's audio codec.")
        try:
            best = container.streams.best("audio")
        except Exception:  # best() is an optimisation; any decodable stream is fine
            best = None
        stream = best if best in decodable else decodable[0]
        cc = stream.codec_context
        duration = container.duration / av.time_base if container.duration else None
        start = float(stream.start_time * stream.time_base) if stream.start_time and stream.time_base else 0.0
        info = AudioInfo(
            container=container.format.name,
            codec=cc.name,
            sample_rate=cc.sample_rate or 0,
            channels=cc.channels or 0,
            stream_index=stream.index,
            container_duration_s=duration,
            size_bytes=size,
            start_offset_s=round(max(start, 0.0), 3),
            duration_is_estimate=container.format.name in _ESTIMATED_DURATION_CONTAINERS,
        )
    # Header durations are advisory, so the real limit is enforced on decoded samples. Up front we
    # only reject what is clearly over: 5% slack for trustworthy headers, 3x for bitrate estimates
    # (a mis-estimated VBR file isn't rejected, a 10 h MP3 doesn't burn 4 h of inference first).
    slack = 3.0 if info.duration_is_estimate else 1.05
    if max_duration_s and info.container_duration_s and info.container_duration_s > max_duration_s * slack:
        raise AudioTooLong(f"Audio is {info.container_duration_s:.0f}s; the limit is {max_duration_s:.0f}s.")
    return info


@dataclass
class DecodeStats:
    samples: int = 0
    corrupt_packets: int = 0
    clipped: int = 0
    sum_sq: float = 0.0
    decode_s: float = 0.0
    stopped_early: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.samples / TARGET_SR

    def observe(self, x: np.ndarray) -> None:
        self.samples += len(x)
        self.clipped += int(np.count_nonzero(np.abs(x) >= 0.999))
        self.sum_sq += float(np.dot(x, x))

    def finalize(self, info: AudioInfo) -> None:
        # Quality signals are warnings, not rejections: the caller decides what "too bad" means.
        if self.samples and self.clipped / self.samples > 0.001:
            self.warnings.append(f"{100 * self.clipped / self.samples:.2f}% of samples are clipped; expect lower accuracy.")
        if self.samples and (self.sum_sq / self.samples) ** 0.5 < 0.003:
            self.warnings.append("Audio level is very low (< -50 dBFS RMS); the recording may be silent or mis-gained.")
        if self.corrupt_packets:
            self.warnings.append(f"Skipped {self.corrupt_packets} corrupt packet(s) while decoding.")
        if self.stopped_early:
            self.warnings.append("The stream became undecodable part-way through; transcript covers the audio before that point.")
        cd = info.container_duration_s
        if cd and self.duration_s < 0.9 * cd and cd - self.duration_s > 1.0:
            self.warnings.append(
                f"Decoded {self.duration_s:.1f}s of audio but the container reports {cd:.1f}s: "
                "the file may be truncated (or a VBR MP3 without a seek header). Transcript covers decoded audio only."
            )


def iter_pcm(
    path: Path,
    info: AudioInfo,
    *,
    channel: int | None = None,
    block_s: float = 30.0,
    max_duration_s: float | None = None,
    stats: DecodeStats | None = None,
) -> Iterator[np.ndarray]:
    """Stream-decode to float32 16 kHz blocks of ~block_s seconds.

    channel=None downmixes to mono; channel=i yields just that channel (for
    split-channel call recordings). Memory is O(block_s), independent of file length.
    Timestamps downstream are derived from *sample counts*, not container PTS, so
    broken/missing PTS (common in MediaRecorder WebM) cannot skew them.
    """
    stats = stats if stats is not None else DecodeStats()
    max_samples = int(max_duration_s * TARGET_SR) if max_duration_s else None
    block = int(block_s * TARGET_SR)
    pending: list[np.ndarray] = []
    pending_n = 0
    resampler, sig = None, None
    t0 = time.perf_counter()

    def make_resampler(frame) -> av.AudioResampler:
        # Stereo is averaged in numpy below: FFmpeg's default stereo->mono rematrix uses a
        # constant-power (-3 dB per channel) law, which makes L==R recordings 3 dB *louder* and
        # pushes peaks past full scale. For >2 channels FFmpeg's dialog-aware downmix is kept.
        many = channel is None and len(frame.layout.channels) > 2
        return av.AudioResampler(format="fltp", layout="mono" if many else frame.layout.name, rate=TARGET_SR)

    def take(frames) -> None:
        nonlocal pending_n
        for out in frames:
            arr = out.to_ndarray()
            if channel is not None:
                x = arr[min(channel, arr.shape[0] - 1)]
            else:
                x = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
            x = np.ascontiguousarray(x, dtype=np.float32)
            pending.append(x)
            pending_n += len(x)
            stats.observe(x)
            if max_samples and stats.samples > max_samples:
                raise AudioTooLong(f"Audio exceeds the {max_duration_s:.0f}s limit.")

    with _open(path) as container:
        stream = container.streams[info.stream_index]
        stream.thread_type = "AUTO"
        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except Exception:  # demuxer gave up mid-file (corrupt/truncated container)
                stats.stopped_early = True
                break
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                stats.corrupt_packets += 1
                continue
            for frame in frames:
                frame.pts = None  # we count samples instead of trusting PTS
                # Rebuild on ANY change of format/layout/rate. PyAV's resampler goes into
                # passthrough when the first frame already matches the target and then never
                # checks again, so a later 44.1 kHz frame would be counted as 16 kHz (time-stretched).
                frame_sig = (frame.format.name, frame.layout.name, frame.sample_rate)
                try:
                    if frame_sig != sig:
                        if resampler is not None:
                            _flush(resampler, take)
                        resampler, sig = make_resampler(frame), frame_sig
                    take(resampler.resample(frame))
                except AudioTooLong:
                    raise
                except (ValueError, av.error.FFmpegError):  # malformed frame (bogus rate/channels...)
                    stats.corrupt_packets += 1
                    resampler, sig = None, None  # a failed graph setup leaves the resampler unusable
            if pending_n >= block:
                stats.decode_s += time.perf_counter() - t0
                yield np.concatenate(pending)
                pending.clear()
                pending_n = 0
                t0 = time.perf_counter()
        if resampler is not None:
            _flush(resampler, take)
    if stats.samples == 0:
        raise UndecodableAudio("No decodable audio frames were found in the file.")
    stats.decode_s += time.perf_counter() - t0
    if pending:
        yield np.concatenate(pending)
    stats.finalize(info)


def _flush(resampler, take) -> None:
    try:
        take(resampler.resample(None))
    except AudioTooLong:
        raise
    except (ValueError, av.error.FFmpegError):
        pass  # nothing buffered, or the graph never got built: no samples to lose


def load_pcm(path: Path, **kw) -> np.ndarray:
    """Convenience for short files and tests: whole file as one array."""
    info = probe(path)
    return np.concatenate(list(iter_pcm(path, info, **kw)))
