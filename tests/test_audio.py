import shutil

import numpy as np
import pytest

from stt_pipeline.audio import DecodeStats, iter_pcm, load_pcm, probe, sniff_container
from stt_pipeline.errors import AudioTooLong, PipelineError, UnsupportedMedia

from .conftest import FIXTURES, bursts, write_wav


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
        (b"#EXTM3U\n#EXTINF:10,", None),  # HLS playlist: never reaches FFmpeg
        (b"ffconcat version 1.0\n", None),  # concat demuxer script: local file read vector
        (b"<?xml version='1.0'?>", None),
        (b"", None),
    ],
)
def test_sniff(head, expected):
    assert sniff_container(head) == expected


def test_hostile_and_broken_inputs_fail_with_precise_codes(manifest):
    for m in manifest:
        if m["kind"] != "invalid":
            continue
        with pytest.raises(PipelineError) as ei:
            info = probe(FIXTURES / m["file"])
            list(iter_pcm(FIXTURES / m["file"], info))
        assert (ei.value.status, ei.value.code) == (m["expect_status"], m["expect_error"]), m["file"]


def test_every_supported_format_decodes_to_identical_canonical_pcm(manifest):
    ref = load_pcm(FIXTURES / "meeting_short.wav")
    ref_rms = np.sqrt(np.mean(ref**2))
    for m in manifest:
        if m["kind"] == "valid" and m["file"].startswith("meeting_short"):
            x = load_pcm(FIXTURES / m["file"])
            # Same speech, any container/codec/rate/bit depth -> same length at 16 kHz mono
            # (lossy codecs may add a few ms of encoder delay/padding).
            assert abs(len(x) - len(ref)) / 16000 < 0.08, m["file"]
            # Lossy decoders can overshoot full scale (inter-sample peaks); float32 keeps them
            # unclipped, unlike decoding to int16 first.
            assert x.dtype == np.float32 and np.isfinite(x).all() and np.abs(x).max() < 4.0
            # ...and the same loudness: stereo sources must not come out 3 dB hotter than mono ones.
            assert abs(np.sqrt(np.mean(x**2)) / ref_rms - 1) < 0.1, m["file"]


def test_extension_and_name_are_ignored(tmp_path, manifest):
    disguised = tmp_path / "recording.wav"
    shutil.copy(FIXTURES / "meeting_short.mp3", disguised)
    assert probe(disguised).container == "mp3"
    fake = tmp_path / "song.mp3"
    shutil.copy(FIXTURES / "not_audio.mp3", fake)
    with pytest.raises(UnsupportedMedia):
        probe(fake)


def test_truncated_file_warns_instead_of_failing(manifest):
    p = FIXTURES / "truncated.mp3"
    info, stats = probe(p), DecodeStats()
    n = sum(len(b) for b in iter_pcm(p, info, stats=stats))
    assert n > 0
    assert any("truncated" in w for w in stats.warnings)


def test_max_duration_enforced_while_decoding(tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 12.0))
    with pytest.raises(AudioTooLong):
        list(iter_pcm(p, probe(p), max_duration_s=5.0, block_s=1.0))


def test_streaming_decode_yields_bounded_blocks(tmp_path):
    p = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 95.0))
    sizes = [len(b) for b in iter_pcm(p, probe(p), block_s=30.0)]
    assert sum(sizes) == 95 * 16000
    assert max(sizes) < 31 * 16000  # memory stays O(block), not O(file)


def test_channel_selection(tmp_path):
    left, right = bursts([(0.5, 1.5)], 4.0, seed=1), bursts([(2.5, 3.5)], 4.0, seed=2)
    p = write_wav(tmp_path / "st.wav", np.stack([left, right]))
    info = probe(p)
    assert info.channels == 2
    l = np.concatenate(list(iter_pcm(p, info, channel=0)))
    r = np.concatenate(list(iter_pcm(p, info, channel=1)))
    assert np.abs(l[8000:24000]).mean() > 0.05 and np.abs(l[40000:56000]).mean() < 0.01
    assert np.abs(r[40000:56000]).mean() > 0.05 and np.abs(r[8000:24000]).mean() < 0.01
