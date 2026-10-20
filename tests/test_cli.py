from stt_pipeline.cli import main

from .conftest import bursts, write_wav


def test_transcribe_writes_the_requested_format_to_output(tmp_path):
    wav = write_wav(tmp_path / "a.wav", bursts([(1, 2)], 3.0))
    out = tmp_path / "a.srt"
    assert main(["transcribe", str(wav), "--engine", "mock", "-f", "srt", "-o", str(out)]) == 0
    assert out.read_text().startswith("1\n00:00:01,000 --> ")


def test_transcribe_reports_pipeline_errors_with_exit_code_2(tmp_path, capsys):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert main(["transcribe", str(empty), "--engine", "mock"]) == 2
    assert "error [empty_file]" in capsys.readouterr().err
