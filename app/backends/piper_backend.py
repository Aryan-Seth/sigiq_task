from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
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


def _env_int(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    return int(val)


def _env_list(name: str) -> list[str]:
    val = os.environ.get(name, "")
    return [item.strip() for item in val.split(",") if item.strip()]


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
        providers = _env_list("PIPER_PROVIDERS")
        if providers:
            return self._load_voice_with_providers(
                PiperVoice, voice_path, config_path, providers
            )

        sig = inspect.signature(PiperVoice.load)
        kwargs: dict[str, Any] = {}
        if "config_path" in sig.parameters and config_path:
            kwargs["config_path"] = config_path
        if "use_cuda" in sig.parameters:
            kwargs["use_cuda"] = use_cuda
        return PiperVoice.load(voice_path, **kwargs)

    def _load_voice_with_providers(
        self,
        PiperVoice: Any,
        voice_path: str,
        config_path: Optional[str],
        providers: list[str],
    ) -> Any:
        try:
            import onnxruntime as ort  # type: ignore
            from piper import PiperConfig  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Custom PIPER_PROVIDERS requires onnxruntime and piper config APIs."
            ) from exc

        model_path = Path(voice_path)
        cfg_path = Path(config_path) if config_path else Path(f"{voice_path}.json")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # Keep CPU as a fallback provider unless explicitly disabled.
        providers_final = list(providers)
        if _env_flag("PIPER_APPEND_CPU_PROVIDER") or "CPUExecutionProvider" not in {
            p for p in providers_final
        }:
            providers_final.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        inter = _env_int("PIPER_INTER_OP_THREADS")
        intra = _env_int("PIPER_INTRA_OP_THREADS")
        if inter is not None:
            sess_options.inter_op_num_threads = max(1, inter)
        if intra is not None:
            sess_options.intra_op_num_threads = max(1, intra)

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers_final,
        )
        return PiperVoice(
            config=PiperConfig.from_dict(cfg_dict),
            session=session,
        )

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
        # Prime runtime/session initialization so first user request avoids setup jitter.
        if _env_flag("PIPER_SKIP_WARMUP"):
            return None
        text = os.environ.get("PIPER_WARMUP_TEXT", "hello")
        try:
            stream = self.synthesize_stream(text)
            next(iter(stream), None)
        except Exception:
            # Startup warmup should never block serving if priming fails.
            return None
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
