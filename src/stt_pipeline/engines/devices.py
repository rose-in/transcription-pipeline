"""Device, compute-type and thread defaults for local engines.

Kept free of model imports so Settings.engine_key() can resolve "auto"/"default" exactly the way a
worker will, without loading faster-whisper.
"""

from __future__ import annotations

import os
import subprocess
import sys


def resolve_device(device: str) -> str:
    """Resolve "auto" to "cuda" when CTranslate2 sees a GPU, else "cpu"."""
    if device != "auto":
        return device
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve "default" for an already-resolved device.

    int8 on CPU (no fp16 kernels; ~2x faster than fp32, near-identical WER); float16 on GPU.
    int8_float16 halves VRAM but is *slower* on many cards - benchmark before switching.
    """
    if compute_type != "default":
        return compute_type
    return "int8" if device == "cpu" else "float16"


def default_cpu_threads() -> int:
    """Performance cores on Apple Silicon (efficiency cores are slower and can hold up
    CTranslate2's parallel loops); elsewhere the core count, capped at 8."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=2
            )
            if int(out.stdout.strip()) > 0:
                return int(out.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return min(8, os.cpu_count() or 4)
