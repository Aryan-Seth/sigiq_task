from __future__ import annotations

import inspect
import os
from typing import Any, Optional

import numpy as np

from .base import SynthResult, SynthStreamChunk


def _env_flag(name: str) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str) -> Optional[float]:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    return float(val)


def _env_bool(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


class PiperBackend:
    def __init__(self) -> None:
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as exc:  # pragma: no cover - import error is user env specific
            raise RuntimeError(
                "piper-tts is not installed. Install with `pip install piper-tts`."
            ) from exc

        voice_path = os.environ.get("PIPER_VOICE", "").strip()
        if not voice_path:
            raise ValueError("PIPER_VOICE must point to a .onnx voice file.")

        config_path = os.environ.get("PIPER_CONFIG")
        use_cuda = _env_flag("PIPER_USE_CUDA")

        self.voice = self._load_voice(PiperVoice, voice_path, config_path, use_cuda)
        self.syn_config = self._build_syn_config()

    def _load_voice(
        self,
        PiperVoice: Any,
        voice_path: str,
        config_path: Optional[str],
        use_cuda: bool,
    ) -> Any:
        sig = inspect.signature(PiperVoice.load)
        kwargs: dict[str, Any] = {}
        if "config_path" in sig.parameters and config_path:
            kwargs["config_path"] = config_path
        if "use_cuda" in sig.parameters:
            kwargs["use_cuda"] = use_cuda
        return PiperVoice.load(voice_path, **kwargs)

    def _build_syn_config(self) -> Optional[Any]:
        try:
            from piper import SynthesisConfig  # type: ignore
        except Exception:
            return None

        params: dict[str, Any] = {}
        volume = _env_float("PIPER_VOLUME")
        length_scale = _env_float("PIPER_LENGTH_SCALE")
        noise_scale = _env_float("PIPER_NOISE_SCALE")
        noise_w_scale = _env_float("PIPER_NOISE_W_SCALE")
        normalize_audio = _env_bool("PIPER_NORMALIZE_AUDIO")

        if volume is not None:
            params["volume"] = volume
        if length_scale is not None:
            params["length_scale"] = length_scale
        if noise_scale is not None:
            params["noise_scale"] = noise_scale
        if noise_w_scale is not None:
            params["noise_w_scale"] = noise_w_scale
        if normalize_audio is not None:
            params["normalize_audio"] = normalize_audio

        if not params:
            return None
        return SynthesisConfig(**params)

    def warmup(self) -> None:
        return None

    def synthesize(self, text: str) -> SynthResult:
        audio_parts = []
        sr = None

        for chunk in self.synthesize_stream(text):
            if sr is None:
                sr = chunk.sr
            audio_parts.append(chunk.audio)

        if not audio_parts:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(audio_parts).astype(np.float32)
        if sr is None:
            sr = 22050

        return SynthResult(audio=audio, sr=sr)

    def synthesize_stream(self, text: str):
        if self.syn_config is not None:
            chunks = self.voice.synthesize(text, syn_config=self.syn_config)
        else:
            chunks = self.voice.synthesize(text)

        for chunk in chunks:
            sr = getattr(chunk, "sample_rate", None) or getattr(
                chunk, "sample_rate_hz", None
            )
            sample_width = getattr(chunk, "sample_width", None)
            channels = getattr(chunk, "sample_channels", None)

            if sample_width not in (None, 2):
                raise ValueError(f"Unsupported Piper sample width: {sample_width}")

            audio_bytes = getattr(chunk, "audio_int16_bytes", None)
            if audio_bytes is None:
                audio_bytes = getattr(chunk, "audio", None)
            if audio_bytes is None:
                raise ValueError("Unexpected Piper chunk format (missing audio bytes).")

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            if channels and channels > 1:
                audio_int16 = audio_int16.reshape(-1, int(channels)).mean(axis=1)
            audio = audio_int16.astype(np.float32) / 32768.0

            if sr is None:
                sr = 22050

            yield SynthStreamChunk(audio=audio, sr=sr)
