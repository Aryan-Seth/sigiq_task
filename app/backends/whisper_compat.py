from __future__ import annotations

import sys
import types


def ensure_whisper_module() -> None:
    """Install a tiny `whisper`-compatible shim if openai-whisper is absent.

    CosyVoice only needs `whisper.log_mel_spectrogram(...)` in frontend code.
    This shim keeps that symbol available so runtime doesn't hard-require
    building the `openai-whisper` package on newer Python versions.
    """
    try:
        import whisper  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    try:
        import torch
        import torchaudio
    except Exception:
        # If torch/torchaudio are missing, let the real import error surface later.
        return

    module = types.ModuleType("whisper")

    def log_mel_spectrogram(audio, n_mels: int = 80):
        # Match expected shape used by CosyVoice frontend:
        # input [T] or [1, T], output [1, n_mels, frames].
        x = audio
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim > 2:
            x = x.reshape(1, -1)

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=int(n_mels),
            center=True,
            power=2.0,
            mel_scale="htk",
            norm="slaney",
        ).to(x.device)
        spec = mel(x)
        spec = torch.clamp(spec, min=1e-10).log10()
        spec = torch.maximum(spec, spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        spec = (spec + 4.0) / 4.0
        return spec

    module.log_mel_spectrogram = log_mel_spectrogram  # type: ignore[attr-defined]
    sys.modules["whisper"] = module

