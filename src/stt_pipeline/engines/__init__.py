from .base import Engine, EngineResult


def _faster_whisper(**kw) -> Engine:
    from .faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(**kw)


def _openai_compatible(**kw) -> Engine:
    from .openai_compat import OpenAICompatibleEngine

    return OpenAICompatibleEngine(**kw)


def _mock(**_kw) -> Engine:
    from .mock import MockEngine

    return MockEngine()


# Imports are deferred so the API process, which never transcribes, doesn't load model libraries.
_FACTORIES = {"faster-whisper": _faster_whisper, "openai-compatible": _openai_compatible, "mock": _mock}
ENGINES = tuple(_FACTORIES)


def build_engine(name: str, **kw) -> Engine:
    """Factory keyed by config, so swapping engines is a deployment decision, not a code change."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown engine {name!r}") from None
    return factory(**kw)


__all__ = ["ENGINES", "Engine", "EngineResult", "build_engine"]
