from __future__ import annotations

import base64
from typing import Tuple

import numpy as np


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int = 44100) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    duration = x.shape[0] / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    t_in = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    clipped = np.clip(x, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def pcm16_bytes_to_base64(pcm16: bytes) -> str:
    return base64.b64encode(pcm16).decode("ascii")


def audio_to_base64_pcm16(x: np.ndarray, sr_in: int, sr_out: int = 44100) -> Tuple[str, int, np.ndarray]:
    y = resample_linear(x, sr_in, sr_out=sr_out)
    pcm16 = float_to_pcm16_bytes(y)
    return pcm16_bytes_to_base64(pcm16), sr_out, y
