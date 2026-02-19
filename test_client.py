from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
from pathlib import Path
import sys
import time
import wave

import websockets


def _load_text(args: argparse.Namespace) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Enter text, then press Ctrl-D (or Ctrl-Z on Windows):")
    return sys.stdin.read()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (pct / 100.0) * (len(vals) - 1)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


async def _run_once(
    text: str,
    chunk_size: int,
    delay: float,
    write_wav: bool,
    output_wav: str,
) -> dict[str, float | str]:
    uri = "ws://localhost:8000/tts"
    pcm_chunks = []
    t_start = None
    t_first_audio = None
    t_last_audio = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": " ", "flush": False}))

        for i in range(0, len(text), chunk_size):
            if t_start is None:
                t_start = time.perf_counter()
            await ws.send(json.dumps({"text": text[i : i + chunk_size], "flush": False}))
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
                if audio_b64:
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    t_last_audio = time.perf_counter()
                    if write_wav:
                        pcm_chunks.append(base64.b64decode(audio_b64))
        except websockets.ConnectionClosed as exc:
            if exc.code not in (1000, 1001) or exc.reason:
                print(f"WebSocket closed: code={exc.code} reason={exc.reason}")

    if t_start is None:
        t_start = time.perf_counter()
    if t_last_audio is None:
        t_last_audio = t_first_audio or time.perf_counter()

    ttft = (t_first_audio - t_start) if t_first_audio else float("nan")
    total = t_last_audio - t_start
    tokens = len(text)
    tps = tokens / total if total > 0 else float("nan")

    if write_wav:
        pcm_data = b"".join(pcm_chunks)
        if pcm_data:
            out_path = output_wav
            out_parent = Path(out_path).parent
            if str(out_parent):
                out_parent.mkdir(parents=True, exist_ok=True)
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(pcm_data)
            print(f"Wrote {out_path}")
        else:
            print("No audio received")

    return {
        "ttft_s": ttft,
        "total_s": total,
        "tokens_per_s": tps,
        "tokens": float(tokens),
    }


async def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--file", help="Path to text file")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--write-wav", action="store_true")
    parser.add_argument("--output-wav", default="artifacts/audio/output.wav")
    args = parser.parse_args()

    text = _load_text(args)
    if not text.strip():
        print("No text provided")
        return

    chunk_size = max(1, int(args.chunk_size))
    delay = max(0.0, float(args.delay))
    runs = max(1, int(args.runs))
    write_wav = args.write_wav or runs == 1

    results = []
    for idx in range(runs):
        if runs > 1:
            print(f"Run {idx + 1}/{runs}")
        result = await _run_once(
            text, chunk_size, delay, write_wav and idx == runs - 1, args.output_wav
        )
        results.append(result)
        if args.progress_every and (idx + 1) % args.progress_every == 0:
            ttfts = [r["ttft_s"] for r in results if not math.isnan(r["ttft_s"])]
            totals = [r["total_s"] for r in results if not math.isnan(r["total_s"])]
            tps_vals = [
                r["tokens_per_s"] for r in results if not math.isnan(r["tokens_per_s"])
            ]
            if ttfts:
                print(
                    f"Partial TTFT p50: {_percentile(ttfts, 50):.3f}s "
                    f"p99: {_percentile(ttfts, 99):.3f}s"
                )
            if totals:
                print(
                    f"Partial total p50: {_percentile(totals, 50):.3f}s "
                    f"p99: {_percentile(totals, 99):.3f}s"
                )
            if tps_vals:
                print(
                    f"Partial tokens/sec p50: {_percentile(tps_vals, 50):.2f} "
                    f"p99: {_percentile(tps_vals, 99):.2f}"
                )

    ttfts = [r["ttft_s"] for r in results if not math.isnan(r["ttft_s"])]
    totals = [r["total_s"] for r in results if not math.isnan(r["total_s"])]
    tps_vals = [r["tokens_per_s"] for r in results if not math.isnan(r["tokens_per_s"])]

    print(f"Tokens (chars): {int(results[0]['tokens'])}")
    if ttfts:
        print(f"TTFT p50: {_percentile(ttfts, 50):.3f}s p99: {_percentile(ttfts, 99):.3f}s")
    if totals:
        print(
            f"Total latency p50: {_percentile(totals, 50):.3f}s p99: {_percentile(totals, 99):.3f}s"
        )
    if tps_vals:
        print(
            f"Tokens/sec p50: {_percentile(tps_vals, 50):.2f} p99: {_percentile(tps_vals, 99):.2f}"
        )


if __name__ == "__main__":
    asyncio.run(run())
