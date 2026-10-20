from dataclasses import replace

import pytest

from stt_pipeline.config import Settings


@pytest.fixture
def base() -> Settings:
    # engine="mock" keeps engine_key() free of ctranslate2 / installed-version lookups
    return Settings(
        engine="mock", model="small", model_revision=None, device="cpu", compute_type="int8", beam_size=5, chunk_s=600.0
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "large-v3-turbo"),
        ("model_revision", "pinned-rev"),
        ("device", "cuda"),
        ("compute_type", "float16"),
        ("beam_size", 1),
        ("chunk_s", 300.0),
    ],
)
def test_engine_key_covers_every_output_affecting_setting(base, field, value):
    assert replace(base, **{field: value}).engine_key() != base.engine_key()


@pytest.mark.parametrize(
    "field, value",
    [
        ("batch_size", 32),
        ("cpu_threads", 4),
        ("max_queue", 5),
        ("lease_s", 5.0),
        ("max_attempts", 1),
    ],
)
def test_engine_key_ignores_throughput_only_settings(base, field, value):
    # API and workers must agree on the key; a throughput knob in it would split the pools
    assert replace(base, **{field: value}).engine_key() == base.engine_key()


def test_hosted_engine_defaults_to_a_timestamp_capable_model():
    s = Settings()
    s.engine, s.model = "openai-compatible", None
    assert s.engine_kwargs()["model"] == "whisper-1"
