from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .base import SynthResult, SynthStreamChunk


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


class SayBackend:
    """Local macOS TTS backend via `say` + `afconvert`."""

    def __init__(self) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("`say` command not found. This backend requires macOS.")
        if shutil.which("afconvert") is None:
            raise RuntimeError(
                "`afconvert` command not found. It is required to convert AIFF to WAV."
            )

        self.voice = os.environ.get("SAY_VOICE", "").strip()
        self.rate = _env_int("SAY_RATE", 190, minimum=80)
        self.sr = _env_int("SAY_SAMPLE_RATE", 44100, minimum=8000)
        self.chunk_ms = _env_int("SAY_STREAM_CHUNK_MS", 120, minimum=20)

    def warmup(self) -> None:
        if os.environ.get("SAY_SKIP_WARMUP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            return
        try:
            _ = self.synthesize("hello")
        except Exception:
            return

    def synthesize(self, text: str) -> SynthResult:
        text = str(text or "")
        if not text.strip():
            return SynthResult(audio=np.zeros(0, dtype=np.float32), sr=self.sr)

        with tempfile.TemporaryDirectory(prefix="tts_say_") as td:
            td_path = Path(td)
            aiff_path = td_path / "out.aiff"
            wav_path = td_path / "out.wav"

            say_cmd = ["say"]
            if self.voice:
                say_cmd.extend(["-v", self.voice])
            say_cmd.extend(["-r", str(self.rate), "-o", str(aiff_path), text])
            say_proc = subprocess.run(
                say_cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if say_proc.returncode != 0:
                stderr = (say_proc.stderr or "").strip()
                raise RuntimeError(f"`say` failed (code={say_proc.returncode}): {stderr}")

            conv_cmd = [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{self.sr}",
                str(aiff_path),
                str(wav_path),
            ]
            conv_proc = subprocess.run(
                conv_cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if conv_proc.returncode != 0:
                stderr = (conv_proc.stderr or "").strip()
                raise RuntimeError(
                    f"`afconvert` failed (code={conv_proc.returncode}): {stderr}"
                )

            with wave.open(str(wav_path), "rb") as wf:
                sr = int(wf.getframerate())
                channels = int(wf.getnchannels())
                sampwidth = int(wf.getsampwidth())
                frames = wf.readframes(wf.getnframes())

            if sampwidth != 2:
                raise ValueError(f"Unsupported sample width from `say`: {sampwidth}")

            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return SynthResult(audio=audio, sr=sr)

    def synthesize_stream(self, text: str):
        result = self.synthesize(text)
        audio = result.audio
        if audio.size == 0:
            return

        chunk_samples = max(1, int(result.sr * (self.chunk_ms / 1000.0)))
        for start in range(0, audio.shape[0], chunk_samples):
            end = min(audio.shape[0], start + chunk_samples)
            yield SynthStreamChunk(audio=audio[start:end], sr=result.sr)
