from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import re
import difflib
from typing import Dict, List, Tuple, Optional

import numpy as np
import websockets
import wave

from app.aligner import WhisperXAligner


def _percentile(values: List[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (pct / 100.0) * (len(vals) - 1)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


async def collect_stream(
    uri: str, text: str, chunk_size: int, delay: float
) -> Tuple[np.ndarray, int, Dict[str, List]]:
    pcm_chunks: List[bytes] = []
    align_chars: List[str] = []
    align_start_ms: List[float] = []
    align_dur_ms: List[float] = []

    samples_so_far = 0
    sr = 44100

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": " ", "flush": False}))
        for i in range(0, len(text), chunk_size):
            await ws.send(
                json.dumps({"text": text[i : i + chunk_size], "flush": False})
            )
            await asyncio.sleep(delay)

        await ws.send(json.dumps({"text": "", "flush": True}))
        await ws.send(json.dumps({"text": "", "flush": False}))

        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8")
                payload = json.loads(msg)
                audio_b64 = payload.get("audio", "")
                if not audio_b64:
                    continue
                pcm = base64.b64decode(audio_b64)
                pcm_chunks.append(pcm)

                # Compute chunk start based on samples so far
                samples = len(pcm) // 2  # int16 mono
                chunk_start_ms = (samples_so_far / float(sr)) * 1000.0
                samples_so_far += samples

                aln = payload.get("alignment", {}) or {}
                chars = aln.get("chars", [])
                starts = aln.get("char_start_times_ms", [])
                durs = aln.get("char_durations_ms", [])
                for ch, s, d in zip(chars, starts, durs):
                    if len(align_chars) >= len(text):
                        break
                    # enforce sequence consistency with source text to avoid duplicates from overlapping windows
                    expected = text[len(align_chars)]
                    if ch != expected:
                        continue
                    align_chars.append(ch)
                    align_start_ms.append(chunk_start_ms + float(s))
                    align_dur_ms.append(float(d))
        except websockets.ConnectionClosed:
            pass

    if pcm_chunks:
        pcm_bytes = b"".join(pcm_chunks)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.zeros(0, dtype=np.float32)

    return audio, sr, {
        "chars": align_chars,
        "char_start_times_ms": align_start_ms,
        "char_durations_ms": align_dur_ms,
    }


def compare_alignments(
    server_aln: Dict[str, List],
    ref_aln: Dict[str, List],
    text: str,
) -> Dict[str, float]:
    text_norm = " ".join(text.lower().split())

    def word_spans(t: str) -> List[Tuple[int, int]]:
        spans = []
        for m in re.finditer(r"\S+", t):
            spans.append((m.start(), m.end()))
        return spans

    def char_to_word_map(spans: List[Tuple[int, int]], length: int) -> List[Optional[int]]:
        mapping = [None] * length
        for idx, (s, e) in enumerate(spans):
            for i in range(s, e):
                mapping[i] = idx
        return mapping

    def build_word_alignment(aln: Dict[str, List], text: str) -> Dict[int, Tuple[float, float]]:
        chars = [c.lower() for c in aln.get("chars", [])]
        starts = aln.get("char_start_times_ms", [])
        durs = aln.get("char_durations_ms", [])

        text_chars = list(text)
        matcher = difflib.SequenceMatcher(None, chars, text_chars)
        spans = word_spans(text)
        c2w = char_to_word_map(spans, len(text_chars))

        word_times: Dict[int, Tuple[float, float]] = {}
        for block in matcher.get_matching_blocks():
            a0, b0, size = block
            for j in range(size):
                si = a0 + j
                ti = b0 + j
                widx = c2w[ti] if 0 <= ti < len(c2w) else None
                if widx is None:
                    continue
                start = float(starts[si])
                end = start + float(durs[si])
                if widx not in word_times:
                    word_times[widx] = (start, end)
                else:
                    cur_s, cur_e = word_times[widx]
                    word_times[widx] = (min(cur_s, start), max(cur_e, end))
        return word_times

    spans = word_spans(text_norm)
    server_words = build_word_alignment(server_aln, text_norm)
    ref_words = build_word_alignment(ref_aln, text_norm)

    errors = []
    matched = 0
    for idx in range(len(spans)):
        if idx in server_words and idx in ref_words:
            s_start, _ = server_words[idx]
            r_start, _ = ref_words[idx]
            errors.append(s_start - r_start)
            matched += 1

    stats = {
        "words_total": len(spans),
        "matched": matched,
        "unmatched": len(spans) - matched,
    }
    if errors:
        stats.update(
            {
                "mean_error_ms": float(np.mean(errors)),
                "median_error_ms": _percentile(errors, 50),
                "p90_error_ms": _percentile(errors, 90),
                "p99_error_ms": _percentile(errors, 99),
                "max_abs_error_ms": float(np.max(np.abs(errors))),
            }
        )
    return stats


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Alignment evaluation with WhisperX")
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--file", help="Path to text file")
    parser.add_argument("--uri", default="ws://localhost:8000/tts")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--save-wav", default="output.wav")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded text from {args.file} (chars={len(text)})")
    elif args.text:
        text = args.text
        print(f"Loaded text from --text (chars={len(text)})")
    else:
        print("Provide --text or --file")
        sys.exit(1)

    if not text.strip():
        print("Text is empty after stripping whitespace; aborting.")
        sys.exit(1)

    audio, sr, server_aln = await collect_stream(
        args.uri, text, max(1, args.chunk_size), max(0.0, args.delay)
    )
    print(f"Collected audio samples: {audio.shape[0]} at {sr} Hz")
    if args.save_wav:
        save_wav(args.save_wav, audio, sr)
        print(f"Wrote {args.save_wav}")

    print("Running WhisperX reference alignment...")
    aligner = WhisperXAligner()
    ref_aln = aligner.align(audio, sr, text) or {}
    server_len = len(server_aln.get("chars", []))
    ref_len = len(ref_aln.get("chars", [])) if ref_aln else 0
    print(f"Server chars: {server_len} | WhisperX chars: {ref_len}")

    if ref_len == 0:
        print("Ref alignment is empty; check WhisperX install/model.")
    else:
        print("Server text preview:", "".join(server_aln.get("chars", [])[:120]))
        ref_chars = ref_aln.get("chars", [])
        print("Ref text preview   :", "".join(ref_chars[:120]))
        # Debug: show raw ref char tokens to diagnose empty previews
        if ref_chars:
            print("Ref raw chars sample:", repr(ref_chars[:40]))
            print("Ref unique chars:", sorted(list({c for c in ref_chars})))
        stats = compare_alignments(server_aln, ref_aln, text)
        print("Alignment comparison stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
