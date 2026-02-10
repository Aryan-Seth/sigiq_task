from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .base import SynthResult, SynthStreamChunk


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is None:
        return value
    return max(minimum, value)


class CosyVoiceBackend:
    """CosyVoice backend (local model or remote model-id via ModelScope/HF)."""

    def __init__(self) -> None:
        self._inject_cosyvoice_repo_to_path()
        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
            from cosyvoice.utils.file_utils import load_wav  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "CosyVoice is not installed. Clone FunAudioLLM/CosyVoice and install its "
                "dependencies, then set COSYVOICE_REPO_DIR to that checkout."
            ) from exc

        self.model_dir = (
            os.environ.get("COSYVOICE_MODEL_DIR", "FunAudioLLM/CosyVoice2-0.5B").strip()
            or "FunAudioLLM/CosyVoice2-0.5B"
        )
        mode = (os.environ.get("COSYVOICE_MODE", "sft").strip() or "sft").lower()
        if mode not in {"sft", "zero_shot", "cross_lingual", "instruct2"}:
            raise ValueError(
                f"Unsupported COSYVOICE_MODE={mode}. "
                "Use one of: sft, zero_shot, cross_lingual, instruct2."
            )
        self.mode = mode
        self.speed = _env_float("COSYVOICE_SPEED", 1.0, minimum=0.5)
        self.text_frontend = _env_flag("COSYVOICE_TEXT_FRONTEND", default=True)
        self.chunk_ms = _env_int("COSYVOICE_STREAM_CHUNK_MS", 120, minimum=20)
        self.zero_shot_spk_id = os.environ.get("COSYVOICE_ZERO_SHOT_SPK_ID", "").strip()

        kwargs: dict[str, Any] = {"model_dir": self.model_dir}
        if _env_flag("COSYVOICE_FP16", default=False):
            kwargs["fp16"] = True
        try:
            self.model = AutoModel(**kwargs)
        except TypeError:
            # Some CosyVoice versions accept fewer kwargs.
            self.model = AutoModel(model_dir=self.model_dir)

        self.sample_rate = int(getattr(self.model, "sample_rate", 22050))
        self._load_wav = load_wav

        self.spk_id = os.environ.get("COSYVOICE_SPK_ID", "").strip()
        if self.mode == "sft":
            self.spk_id = self._resolve_sft_speaker(self.spk_id)

        self.prompt_text = os.environ.get("COSYVOICE_PROMPT_TEXT", "").strip()
        self.instruct_text = os.environ.get(
            "COSYVOICE_INSTRUCT_TEXT",
            "You are a helpful assistant. Please speak naturally.<|endofprompt|>",
        ).strip()
        self.prompt_wav = self._prepare_prompt_wav()

    def _inject_cosyvoice_repo_to_path(self) -> None:
        repo = os.environ.get("COSYVOICE_REPO_DIR", "").strip()
        if not repo:
            return
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.exists():
            raise ValueError(f"COSYVOICE_REPO_DIR not found: {repo_path}")
        matcha_path = repo_path / "third_party" / "Matcha-TTS"
        for candidate in (repo_path, matcha_path):
            if candidate.exists():
                cstr = str(candidate)
                if cstr not in sys.path:
                    sys.path.insert(0, cstr)

    def _resolve_sft_speaker(self, requested: str) -> str:
        list_fn = getattr(self.model, "list_available_spks", None)
        if not callable(list_fn):
            if requested:
                return requested
            raise ValueError(
                "COSYVOICE_MODE=sft requires COSYVOICE_SPK_ID because this model "
                "does not expose list_available_spks()."
            )

        available = [str(s) for s in list_fn()]
        if not available:
            if requested:
                return requested
            raise ValueError(
                "No built-in CosyVoice speakers found. "
                "Set COSYVOICE_MODE=zero_shot/cross_lingual with a prompt wav."
            )
        if requested and requested in available:
            return requested
        return available[0]

    def _prepare_prompt_wav(self):
        if self.mode == "sft":
            return None

        prompt_wav = os.environ.get("COSYVOICE_PROMPT_WAV", "").strip()
        if not prompt_wav:
            raise ValueError(
                f"COSYVOICE_MODE={self.mode} requires COSYVOICE_PROMPT_WAV "
                "(16k reference wav path)."
            )
        wav_path = Path(prompt_wav).expanduser()
        if not wav_path.exists():
            raise ValueError(f"COSYVOICE_PROMPT_WAV not found: {wav_path}")

        if self.mode == "zero_shot" and not self.prompt_text:
            raise ValueError(
                "COSYVOICE_MODE=zero_shot requires COSYVOICE_PROMPT_TEXT."
            )
        if self.mode == "instruct2" and not self.instruct_text:
            raise ValueError(
                "COSYVOICE_MODE=instruct2 requires COSYVOICE_INSTRUCT_TEXT."
            )
        return self._load_wav(str(wav_path), 16000)

    def _iter_model_outputs(self, text: str, stream: bool) -> Iterable[Any]:
        common = {"stream": stream, "speed": self.speed, "text_frontend": self.text_frontend}
        if self.mode == "sft":
            return self.model.inference_sft(text, self.spk_id, **common)
        if self.mode == "zero_shot":
            return self.model.inference_zero_shot(
                text,
                self.prompt_text,
                self.prompt_wav,
                zero_shot_spk_id=self.zero_shot_spk_id,
                **common,
            )
        if self.mode == "cross_lingual":
            return self.model.inference_cross_lingual(
                text,
                self.prompt_wav,
                zero_shot_spk_id=self.zero_shot_spk_id,
                **common,
            )
        return self.model.inference_instruct2(
            text,
            self.instruct_text,
            self.prompt_wav,
            zero_shot_spk_id=self.zero_shot_spk_id,
            **common,
        )

    def _to_audio(self, model_output: Any) -> np.ndarray:
        data = model_output
        if isinstance(model_output, dict):
            data = model_output.get("tts_speech")
        if data is None:
            return np.zeros(0, dtype=np.float32)

        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "float"):
            data = data.float()
        if hasattr(data, "numpy"):
            data = data.numpy()

        arr = np.asarray(data, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2:
            # Handle both [C, T] and [T, C].
            arr = arr.mean(axis=0 if arr.shape[0] <= arr.shape[1] else 1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr.reshape(-1)

    def _split_stream(self, audio: np.ndarray, sr: int) -> Iterator[np.ndarray]:
        chunk_samples = max(1, int(sr * (self.chunk_ms / 1000.0)))
        for start in range(0, audio.shape[0], chunk_samples):
            end = min(audio.shape[0], start + chunk_samples)
            yield audio[start:end]

    def warmup(self) -> None:
        if _env_flag("COSYVOICE_SKIP_WARMUP", default=False):
            return
        text = os.environ.get("COSYVOICE_WARMUP_TEXT", "hello").strip() or "hello"
        try:
            it = iter(self._iter_model_outputs(text, stream=True))
            next(it, None)
        except Exception:
            # Warmup should not block startup.
            return

    def synthesize(self, text: str) -> SynthResult:
        text = str(text or "")
        if not text.strip():
            return SynthResult(audio=np.zeros(0, dtype=np.float32), sr=self.sample_rate)

        parts: list[np.ndarray] = []
        for out in self._iter_model_outputs(text, stream=False):
            arr = self._to_audio(out)
            if arr.size > 0:
                parts.append(arr)
        if not parts:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(parts).astype(np.float32, copy=False)
        return SynthResult(audio=audio, sr=self.sample_rate)

    def synthesize_stream(self, text: str):
        text = str(text or "")
        if not text.strip():
            return
        for out in self._iter_model_outputs(text, stream=True):
            arr = self._to_audio(out)
            if arr.size == 0:
                continue
            for frame in self._split_stream(arr, self.sample_rate):
                yield SynthStreamChunk(audio=frame, sr=self.sample_rate)
