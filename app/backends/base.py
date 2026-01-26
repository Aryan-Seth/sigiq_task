from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


@dataclass
class SynthResult:
    audio: np.ndarray  # float32 mono
    sr: int
    char_durations_ms: Optional[Sequence[float]] = None


@dataclass
class SynthStreamChunk:
    audio: np.ndarray  # float32 mono
    sr: int


class TTSBackend(Protocol):
    def warmup(self) -> None:
        ...

    def synthesize(self, text: str) -> SynthResult:
        ...

    def synthesize_stream(self, text: str) -> Iterable[SynthStreamChunk]:
        ...
