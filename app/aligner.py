from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .audio import resample_linear


class WhisperXAligner:
    def __init__(self) -> None:
        try:
            import whisperx  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "whisperx is not installed. Install with `pip install whisperx` (and torch)."
            ) from exc

        self.language = os.environ.get("WHISPERX_LANGUAGE", "en")
        self.device = os.environ.get("WHISPERX_DEVICE", "").strip() or None
        self.model_name = os.environ.get("WHISPERX_ALIGN_MODEL") or None
        self.model_dir = os.environ.get("WHISPERX_MODEL_DIR") or None

        # load alignment model (CTC-based)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device or "cpu",
            model_name=self.model_name,
            model_dir=self.model_dir,
        )
        self.device = self.device or "cpu"
        self.whisperx = whisperx

    def align(
        self, audio: np.ndarray, sr: int, text: str
    ) -> Optional[Dict[str, List[int]]]:
        # returns alignment dict or None on failure
        if not text:
            return None
        audio_16k = resample_linear(audio, sr, sr_out=16000)
        duration_s = audio_16k.shape[0] / 16000.0
        segments = [
            {
                "text": text,
                "start": 0.0,
                "end": duration_s,
            }
        ]

        try:
            result = self.whisperx.align(
                transcript=segments,
                model=self.align_model,
                align_model_metadata=self.align_metadata,
                audio=audio_16k,
                device=self.device,
                return_char_alignments=True,
                print_progress=False,
            )
        except Exception:
            return None

        chars: List[str] = []
        starts_ms: List[int] = []
        durs_ms: List[int] = []

        for seg in result.get("segments", []):
            for ch in seg.get("chars", []):
                ch_text = ch.get("char") or ch.get("text", "")
                start_s = float(ch.get("start", 0.0) or 0.0)
                end_s = float(ch.get("end", start_s) or start_s)
                start_ms = int(round(start_s * 1000.0))
                end_ms = int(round(end_s * 1000.0))
                dur_ms = max(0, end_ms - start_ms)
                chars.append(ch_text)
                starts_ms.append(start_ms)
                durs_ms.append(dur_ms)

        if not chars:
            return None

        return {
            "chars": chars,
            "char_start_times_ms": starts_ms,
            "char_durations_ms": durs_ms,
        }


def get_aligner_from_env():
    name = os.environ.get("ALIGNER", "").strip().lower()
    if not name or name == "none":
        return None
    if name == "whisperx":
        return WhisperXAligner()
    raise ValueError(f"Unknown aligner: {name}")
