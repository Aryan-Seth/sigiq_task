from __future__ import annotations

import random

import numpy as np

from .base import SynthResult, SynthStreamChunk


class DummyBackend:
    def __init__(self, sr: int = 22050) -> None:
        self.sr = sr

    def warmup(self) -> None:
        # Intentionally fast no-op warmup.
        return None

    def synthesize(self, text: str) -> SynthResult:
        if text.strip() == "":
            dur_ms = random.randint(80, 120)
            n_samples = max(1, int(self.sr * dur_ms / 1000.0))
            audio = np.zeros(n_samples, dtype=np.float32)
            return SynthResult(audio=audio, sr=self.sr)

        per_char_ms = 35
        total_ms = max(120, min(1200, len(text) * per_char_ms))
        n_samples = max(1, int(self.sr * total_ms / 1000.0))
        t = np.arange(n_samples, dtype=np.float32) / float(self.sr)
        freq = 220.0 + (len(text) % 7) * 30.0
        audio = 0.2 * np.sin(2.0 * np.pi * freq * t)
        return SynthResult(audio=audio.astype(np.float32), sr=self.sr)

    def synthesize_stream(self, text: str):
        result = self.synthesize(text)
        audio = result.audio
        if audio.size == 0:
            return

        chunk_samples = max(1, int(self.sr * 0.12))
        for start in range(0, audio.shape[0], chunk_samples):
            end = min(audio.shape[0], start + chunk_samples)
            yield SynthStreamChunk(audio=audio[start:end], sr=self.sr)
