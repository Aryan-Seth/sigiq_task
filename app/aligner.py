from __future__ import annotations

import difflib
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

        raw_chars: List[str] = []
        raw_starts_ms: List[int] = []
        raw_durs_ms: List[int] = []

        for seg in result.get("segments", []):
            for ch in seg.get("chars", []):
                ch_text = ch.get("char") or ch.get("text", "")
                start_s = float(ch.get("start", 0.0) or 0.0)
                end_s = float(ch.get("end", start_s) or start_s)
                start_ms = int(round(start_s * 1000.0))
                end_ms = int(round(end_s * 1000.0))
                dur_ms = max(0, end_ms - start_ms)
                if not ch_text:
                    continue
                raw_chars.append(str(ch_text))
                raw_starts_ms.append(start_ms)
                raw_durs_ms.append(dur_ms)

        if not raw_chars:
            return None

        return _normalize_alignment_to_text(
            text=text,
            raw_chars=raw_chars,
            raw_starts_ms=raw_starts_ms,
            raw_durs_ms=raw_durs_ms,
        )


def get_aligner_from_env():
    name = os.environ.get("ALIGNER", "whisperx").strip().lower()
    if not name or name == "none":
        return None
    if name == "whisperx":
        return WhisperXAligner()
    raise ValueError(f"Unknown aligner: {name}")


def _normalize_alignment_to_text(
    text: str,
    raw_chars: List[str],
    raw_starts_ms: List[int],
    raw_durs_ms: List[int],
) -> Optional[Dict[str, List[int]]]:
    target = list(text)
    if not target:
        return {
            "chars": [],
            "char_start_times_ms": [],
            "char_durations_ms": [],
            "char_indices": [],
        }

    # Expand multi-char tokens from WhisperX output into per-char entries.
    exp_chars: List[str] = []
    exp_starts: List[float] = []
    exp_durs: List[float] = []
    for ch, start_ms, dur_ms in zip(raw_chars, raw_starts_ms, raw_durs_ms):
        token = str(ch)
        if not token:
            continue
        token_len = len(token)
        dur_each = max(1.0, float(dur_ms) / float(token_len))
        for idx, c in enumerate(token):
            exp_chars.append(c)
            exp_starts.append(float(start_ms) + (idx * dur_each))
            exp_durs.append(dur_each)

    if not exp_chars:
        return None

    matcher = difflib.SequenceMatcher(None, exp_chars, target)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    if matched < max(1, int(0.6 * len(target))):
        # Let caller fallback to heuristic alignment if WhisperX output diverges too much.
        return None

    starts: List[Optional[float]] = [None] * len(target)
    durs: List[Optional[float]] = [None] * len(target)
    for block in matcher.get_matching_blocks():
        a0, b0, size = block
        for j in range(size):
            src_idx = a0 + j
            dst_idx = b0 + j
            starts[dst_idx] = exp_starts[src_idx]
            durs[dst_idx] = exp_durs[src_idx]

    positive = [d for d in exp_durs if d > 0.0]
    default_dur = float(np.median(positive)) if positive else 30.0
    default_dur = max(1.0, default_dur)

    out_starts: List[int] = []
    out_durs: List[int] = []
    cursor = 0.0
    for s_opt, d_opt in zip(starts, durs):
        d = float(d_opt) if d_opt is not None else default_dur
        d = max(1.0, d)
        s = float(s_opt) if s_opt is not None else cursor
        s = max(cursor, s)
        out_starts.append(int(round(s)))
        out_durs.append(int(round(d)))
        cursor = s + d

    return {
        "chars": target,
        "char_start_times_ms": out_starts,
        "char_durations_ms": out_durs,
        "char_indices": list(range(len(target))),
    }
