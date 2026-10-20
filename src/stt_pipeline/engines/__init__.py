from .base import Engine, EngineResult


def build_engine(name: str, **kw) -> Engine:
    """Factory keyed by config, so swapping engines is a deployment decision, not a code change."""
    if name == "faster-whisper":
        from .faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(**kw)
    if name == "openai-compatible":
        from .openai_compat import OpenAICompatibleEngine

        return OpenAICompatibleEngine(**kw)
    if name == "mock":
        from .mock import MockEngine

        return MockEngine()
    raise ValueError(f"unknown engine {name!r}")


__all__ = ["Engine", "EngineResult", "build_engine"]
