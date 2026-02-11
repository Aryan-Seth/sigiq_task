from __future__ import annotations

from typing import Dict, Type

from .base import TTSBackend
from .dummy import DummyBackend
from .piper_backend import PiperBackend

_BACKENDS: Dict[str, Type[TTSBackend]] = {
    "dummy": DummyBackend,
    "piper": PiperBackend,
}


def create_backend(name: str) -> TTSBackend:
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return _BACKENDS[key]()
