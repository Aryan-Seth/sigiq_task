from __future__ import annotations

from typing import Dict, Type

from .base import TTSBackend
from .cosyvoice_backend import CosyVoiceBackend
from .dummy import DummyBackend
from .kokoro_backend import KokoroBackend
from .piper_backend import PiperBackend
from .say_backend import SayBackend

_BACKENDS: Dict[str, Type[TTSBackend]] = {
    "dummy": DummyBackend,
    "piper": PiperBackend,
    "say": SayBackend,
    "kokoro": KokoroBackend,
    "cosyvoice": CosyVoiceBackend,
}


def create_backend(name: str) -> TTSBackend:
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    return _BACKENDS[key]()
