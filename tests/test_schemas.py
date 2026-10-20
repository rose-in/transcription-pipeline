import pytest

from stt_pipeline.schemas import TranscribeOptions


def test_language_must_be_a_known_code_and_is_lowercased():
    with pytest.raises(ValueError):
        TranscribeOptions(language="en-US")
    assert TranscribeOptions(language="EN").language == "en"
