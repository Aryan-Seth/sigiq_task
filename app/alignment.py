from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _weights_for_text(text: str) -> np.ndarray:
    weights = []
    for ch in text:
        if ch.isspace():
            weights.append(0.5)
        elif ch in ".?!":
            weights.append(1.2)
        elif ch in ",;:":
            weights.append(0.9)
        elif ch.isalnum():
            weights.append(1.0)
        else:
            weights.append(0.8)
    return np.array(weights, dtype=np.float32)


def alignment_from_duration(
    text: str,
    total_ms: float,
    char_durations_ms: Optional[Sequence[float]] = None,
) -> Dict[str, List[int]]:
    chars = list(text)
    n = len(chars)
    if n == 0:
        return {
            "chars": [],
            "char_start_times_ms": [],
            "char_durations_ms": [],
            "char_indices": [],
        }

    total_ms_int = max(0, int(round(total_ms)))

    if char_durations_ms is not None and len(char_durations_ms) == n:
        durations = np.array(char_durations_ms, dtype=np.float32)
    else:
        weights = _weights_for_text(text)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            weights = np.ones(n, dtype=np.float32)
            total_weight = float(weights.sum())
        durations = (weights / total_weight) * float(total_ms_int)

    base = np.floor(durations).astype(np.int64)
    remainder = int(total_ms_int - base.sum())
    if remainder > 0:
        frac = durations - np.floor(durations)
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            base[idx] += 1

    base = base.astype(int)
    start_times: List[int] = []
    cur = 0
    for dur in base.tolist():
        start_times.append(cur)
        cur += int(dur)

    return {
        "chars": chars,
        "char_start_times_ms": start_times,
        "char_durations_ms": base.tolist(),
        "char_indices": list(range(n)),
    }


def split_alignment_for_window(
    alignment: Dict[str, List[int]],
    win_start_ms: float,
    win_end_ms: float,
) -> Dict[str, List[int]]:
    # Emit each character exactly once based on its start time.
    # Overlap-based slicing duplicates long characters across adjacent windows,
    # which leads to repeated subtitles like "TTThhheee".
    chars_out: List[str] = []
    starts_out: List[int] = []
    durs_out: List[int] = []

    chars = alignment.get("chars", [])
    starts = alignment.get("char_start_times_ms", [])
    durs = alignment.get("char_durations_ms", [])
    idxs = alignment.get("char_indices", [])
    if not isinstance(idxs, list) or len(idxs) != len(chars):
        idxs = list(range(len(chars)))
    idxs_out: List[int] = []

    if win_end_ms <= win_start_ms:
        return {
            "chars": [],
            "char_start_times_ms": [],
            "char_durations_ms": [],
            "char_indices": [],
        }

    for ch, start, dur, idx in zip(chars, starts, durs, idxs):
        start_f = float(start)
        dur_f = float(dur)
        if start_f < win_start_ms or start_f >= win_end_ms:
            continue
        end_f = start_f + max(0.0, dur_f)
        clipped_end_f = min(max(end_f, start_f), win_end_ms)
        rel_start = max(0, int(round(start_f - win_start_ms)))
        rel_dur = max(1, int(round(clipped_end_f - start_f)))
        chars_out.append(ch)
        starts_out.append(rel_start)
        durs_out.append(rel_dur)
        idxs_out.append(int(idx))

    return {
        "chars": chars_out,
        "char_start_times_ms": starts_out,
        "char_durations_ms": durs_out,
        "char_indices": idxs_out,
    }


def scale_alignment_to_duration(
    alignment: Dict[str, List[int]],
    target_ms: float,
) -> Dict[str, List[int]]:
    chars = alignment.get("chars", [])
    if not chars:
        return {
            "chars": [],
            "char_start_times_ms": [],
            "char_durations_ms": [],
            "char_indices": [],
        }

    durations = alignment.get("char_durations_ms", [])
    if len(durations) != len(chars):
        durations = [1 for _ in chars]

    total_ms_int = max(0, int(round(target_ms)))
    weights = np.array(durations, dtype=np.float32)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        weights = np.ones(len(chars), dtype=np.float32)
        total_weight = float(weights.sum())

    scaled = (weights / total_weight) * float(total_ms_int)
    base = np.floor(scaled).astype(np.int64)
    remainder = int(total_ms_int - base.sum())
    if remainder > 0:
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            base[idx] += 1

    base = base.astype(int)
    start_times: List[int] = []
    cur = 0
    for dur in base.tolist():
        start_times.append(cur)
        cur += int(dur)

    return {
        "chars": chars,
        "char_start_times_ms": start_times,
        "char_durations_ms": base.tolist(),
        "char_indices": alignment.get("char_indices", list(range(len(chars)))),
    }
