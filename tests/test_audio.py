import shutil
import struct
from fractions import Fraction

import av
import numpy as np
import pytest

from stt_pipeline.audio import TARGET_SR, DecodeStats, iter_pcm, load_pcm, probe, sniff_container
from stt_pipeline.errors import AudioTooLong, PipelineError, UndecodableAudio, UnsupportedMedia

from .conftest import bursts, write_wav


def _write_adts(path, rate, seconds):
    """A 300 Hz tone as raw ADTS AAC at `rate`."""
    with av.open(str(path), "w", format="adts") as out:
        st = out.add_stream("aac", rate=rate)
        st.layout = "mono"
        cc = st.codec_context
        t = np.arange(int(rate * seconds)) / rate
        x = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)[None, :]
        fifo = av.AudioFifo()
        f = av.AudioFrame.from_ndarray(x, format="fltp", layout="mono")
        f.sample_rate = rate
        fifo.write(f)
        pts = 0
        while fifo.samples:
            fr = fifo.read(min(cc.frame_size, fifo.samples))
            fr.pts, fr.time_base = pts, Fraction(1, rate)
            pts += fr.samples
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)


@pytest.mark.parametrize(
    "head,expected",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x00" + b"\x00" * 8, "mp3"),
        (b"\xff\xf1\x50\x80" + b"\x00" * 8, "aac"),
        (b"fLaC\x00\x00\x00\x22" + b"\x00" * 4, "flac"),
        (b"OggS\x00\x02" + b"\x00" * 6, "ogg"),
        (b"\x00\x00\x00\x20ftypM4A ", "mp4"),
        (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81" + b"\x00" * 4, "matroska"),
        (b"FORM\x00\x00\x00\x00AIFF", "aiff"),
        (b"#!AMR\n" + b"\x00" * 6, "amr"),
        (b"#EXTM3U\n#EXTINF:10,", None),  # HLS playlist: not sniffed; FFmpeg's format_whitelist rejects it
        (b"ffconcat version 1.0\n", None),  # concat script (local file read): not sniffed; rejected by format_whitelist
        (b"<?xml version='1.0'?>", None),
        (b"", None),
    ],
)
def test_sniff(head, expected):
    assert sniff_container(head) == expected


@pytest.mark.parametrize(
    "payload,status,code",
    [
        (b"", 400, "empty_file"),
        (b"this is a text file pretending to be an mp3\n" * 50, 415, "unsupported_media"),
        (
            b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://169.254.169.254/latest/meta-data/\n"
            b"#EXTINF:10,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n",
            415,
            "unsupported_media",
        ),
        (b"ffconcat version 1.0\nfile '/etc/passwd'\n", 415, "unsupported_media"),
    ],
    ids=["empty", "text_as_mp3", "hls_ssrf_playlist", "ffconcat_local_read"],
)
def test_hostile_uploads_fail_with_precise_codes(tmp_path, payload, status, code):
    """Needs no generated fixtures: the security boundary (sniff + FFmpeg demuxer/protocol whitelist) is always tested."""
    p = tmp_path / "upload.mp3"  # extension is deliberately misleading
    p.write_bytes(payload)
    with pytest.raises(PipelineError) as ei:
        probe(p)
    assert (ei.value.status, ei.value.code) == (status, code)


def test_invalid_fixtures_fail_with_precise_codes(manifest, fixtures_dir):
    for m in manifest:
        if m["kind"] != "invalid":
            continue
        with pytest.raises(PipelineError) as ei:
            info = probe(fixtures_dir / m["file"])
            list(iter_pcm(fixtures_dir / m["file"], info))
        assert (ei.value.status, ei.value.code) == (m["expect_status"], m["expect_error"]), m["file"]


def test_every_supported_format_decodes_to_identical_canonical_pcm(manifest, fixtures_dir):
    ref = load_pcm(fixtures_dir / "meeting_short.wav")
    ref_rms = np.sqrt(np.mean(ref**2))
    for m in manifest:
        if m["kind"] == "valid" and m["file"].startswith("meeting_short"):
            x = load_pcm(fixtures_dir / m["file"])
            # Same speech, any container/codec/rate/bit depth -> same length at 16 kHz mono
            # (lossy codecs may add a few ms of encoder delay/padding).
            assert abs(len(x) - len(ref)) / 16000 < 0.08, m["file"]
            # Lossy decoders can overshoot full scale (inter-sample peaks); float32 keeps them
            # unclipped, unlike decoding to int16 first.
            assert x.dtype == np.float32 and np.isfinite(x).all() and np.abs(x).max() < 4.0
            # ...and the same loudness: stereo sources must not come out 3 dB hotter than mono ones.
            assert abs(np.sqrt(np.mean(x**2)) / ref_rms - 1) < 0.1, m["file"]


def test_extension_and_name_are_ignored(tmp_path, fixtures_dir):
    disguised = tmp_path / "recording.wav"
    shutil.copy(fixtures_dir / "meeting_short.mp3", disguised)
    assert probe(disguised).container == "mp3"
    fake = tmp_path / "song.mp3"
    shutil.copy(fixtures_dir / "not_audio.mp3", fake)
    with pytest.raises(UnsupportedMedia):
        probe(fake)


def test_truncated_file_warns_instead_of_failing(fixtures_dir):
    p = fixtures_dir / "truncated.mp3"
    info, stats = probe(p), DecodeStats()
    n = sum(len(b) for b in iter_pcm(p, info, stats=stats))
    assert n > 0
    assert any("truncated" in w for w in stats.warnings)


def test_unknown_wav_codec_raises_undecodable_audio(tmp_path):
    data = b"\x00\x01" * 16000
    fmt = struct.pack("<HHIIHH", 0x1234, 1, 16000, 32000, 2, 16)
    wav = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt
    (tmp_path / "x.wav").write_bytes(wav + b"data" + struct.pack("<I", len(data)) + data)
    with pytest.raises(UndecodableAudio):
        probe(tmp_path / "x.wav")


def test_estimated_duration_rejects_up_front_only_when_far_over_the_limit(tmp_path):
    p = tmp_path / "vbr.mp3"
    with av.open(str(p), "w", format="mp3") as out:  # low-bitrate start skews FFmpeg's estimate
        st = out.add_stream("libmp3lame", rate=16000)
        st.layout, st.bit_rate = "mono", 8000
        x = np.zeros((1, 16000 * 20), dtype=np.float32)
        f = av.AudioFrame.from_ndarray(x, format="fltp", layout="mono")
        f.sample_rate = 16000
        for pkt in st.encode(f):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    info = probe(p, max_duration_s=10.0)  # 20 s estimate vs 10 s limit: advisory, decode decides
    assert info.duration_is_estimate
    with pytest.raises(AudioTooLong):  # ...but an estimate far over (>3x) is rejected up front
        probe(p, max_duration_s=5.0)


def test_max_duration_enforced_while_decoding(tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 12.0))
    with pytest.raises(AudioTooLong):
        list(iter_pcm(p, probe(p), max_duration_s=5.0, block_s=1.0))


def test_streaming_decode_yields_bounded_blocks(tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 95.0))
    sizes = [len(b) for b in iter_pcm(p, probe(p), block_s=30.0)]
    assert sum(sizes) == 95 * 16000
    assert max(sizes) < 31 * 16000  # memory stays O(block), not O(file)


def test_mid_stream_sample_rate_change_is_resampled(tmp_path):
    a, b = tmp_path / "a.aac", tmp_path / "b.aac"
    _write_adts(a, 16000, 3.0)
    _write_adts(b, 44100, 3.0)
    joined = tmp_path / "joined.aac"
    joined.write_bytes(a.read_bytes() + b.read_bytes())  # 16 kHz first, so decoding starts on the no-resample path
    seconds = len(load_pcm(joined)) / TARGET_SR
    assert 5.8 < seconds < 6.3, seconds  # a time-stretched 44.1 kHz half would make this ~11.5 s


def test_channel_selection(tmp_path):
    left, right = bursts([(0.5, 1.5)], 4.0, seed=1), bursts([(2.5, 3.5)], 4.0, seed=2)
    p = write_wav(tmp_path / "st.wav", np.stack([left, right]))
    info = probe(p)
    assert info.channels == 2
    left_pcm = np.concatenate(list(iter_pcm(p, info, channel=0)))
    right_pcm = np.concatenate(list(iter_pcm(p, info, channel=1)))
    assert np.abs(left_pcm[8000:24000]).mean() > 0.05 and np.abs(left_pcm[40000:56000]).mean() < 0.01
    assert np.abs(right_pcm[40000:56000]).mean() > 0.05 and np.abs(right_pcm[8000:24000]).mean() < 0.01
