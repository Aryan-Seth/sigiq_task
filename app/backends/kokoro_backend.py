from __future__ import annotations

import asyncio
import os
import queue
import threading
from pathlib import Path

import numpy as np

from .base import SynthResult, SynthStreamChunk


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


class KokoroBackend:
    """Local on-device Kokoro ONNX backend."""

    def __init__(self) -> None:
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Install with `pip install kokoro-onnx`."
            ) from exc

        root = Path(__file__).resolve().parents[2]
        model_path = os.environ.get("KOKORO_MODEL", "").strip()
        voices_path = os.environ.get("KOKORO_VOICES", "").strip()

        if not model_path:
            model_path = str(root / "models" / "kokoro" / "kokoro-v1.0.onnx")
        if not voices_path:
            voices_path = str(root / "models" / "kokoro" / "voices-v1.0.bin")

        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        if not self.model_path.exists():
            raise ValueError(
                f"KOKORO_MODEL not found: {self.model_path}. "
                "Download kokoro-v1.0.onnx locally."
            )
        if not self.voices_path.exists():
            raise ValueError(
                f"KOKORO_VOICES not found: {self.voices_path}. "
                "Download voices-v1.0.bin locally."
            )

        self.voice = os.environ.get("KOKORO_VOICE", "af_sarah").strip() or "af_sarah"
        self.lang = os.environ.get("KOKORO_LANG", "en-us").strip() or "en-us"
        self.speed = min(2.0, max(0.5, _env_float("KOKORO_SPEED", 1.0)))
        self.trim = _env_flag("KOKORO_TRIM", default=True)
        self.chunk_ms = _env_int("KOKORO_STREAM_CHUNK_MS", 120, minimum=20)

        self.engine = Kokoro(str(self.model_path), str(self.voices_path))
        self._ensure_voice_exists()

    def _voice_names(self) -> list[str]:
        voices = self.engine.voices
        if hasattr(voices, "files"):
            return [str(v) for v in list(voices.files)]
        try:
            return [str(v) for v in list(voices.keys())]
        except Exception:
            return []

    def _ensure_voice_exists(self) -> None:
        names = self._voice_names()
        if not names:
            return
        if self.voice in names:
            return
        fallback = next((n for n in names if n.lower().startswith(("af_", "am_"))), names[0])
        self.voice = fallback

    def warmup(self) -> None:
        if _env_flag("KOKORO_SKIP_WARMUP", default=False):
            return
        try:
            _ = self.synthesize("hello")
        except Exception:
            return

    def synthesize(self, text: str) -> SynthResult:
        text = str(text or "")
        if not text.strip():
            return SynthResult(audio=np.zeros(0, dtype=np.float32), sr=24000)

        audio, sr = self.engine.create(
            text=text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
            trim=self.trim,
        )
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return SynthResult(audio=arr, sr=int(sr))

    def synthesize_stream(self, text: str):
        text = str(text or "")
        if not text.strip():
            return

        q: queue.Queue[tuple[np.ndarray, int] | Exception | None] = queue.Queue(maxsize=8)
        stop_event = threading.Event()

        async def _produce() -> None:
            try:
                async for audio_part, sr in self.engine.create_stream(
                    text=text,
                    voice=self.voice,
                    speed=self.speed,
                    lang=self.lang,
                    trim=self.trim,
                ):
                    if stop_event.is_set():
                        break
                    arr = np.asarray(audio_part, dtype=np.float32).reshape(-1)
                    arr = np.nan_to_num(
                        arr, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32)
                    if arr.size == 0:
                        continue
                    q.put((arr, int(sr)))
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_produce())
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                audio, sr = item
                chunk_samples = max(1, int(sr * (self.chunk_ms / 1000.0)))
                for start in range(0, audio.shape[0], chunk_samples):
                    end = min(audio.shape[0], start + chunk_samples)
                    yield SynthStreamChunk(audio=audio[start:end], sr=sr)
        finally:
            stop_event.set()
            thread.join(timeout=0.5)
