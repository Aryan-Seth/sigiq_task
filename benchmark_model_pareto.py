from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import websockets
from faster_whisper import WhisperModel

from app.audio import resample_linear
from app.text_normalize import normalize_for_tts

PROFILE_PREFIX = "[tts-profile-json] "

DEFAULT_CASES = [
    (
        "plain",
        "A practical streaming text to speech benchmark should keep first audio latency low while preserving clear pronunciation.",
    ),
    (
        "math",
        r"The product of three and seven is \(3 \times 7 = 21\), and the Pythagorean theorem is \(a^2 + b^2 = c^2\).",
    ),
    (
        "math2",
        r"The derivative of \(e^x\) with respect to \(x\) is \(\frac{d}{dx} e^x = e^x\), and the quadratic formula is \(x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\).",
    ),
]


def load_cases_file(path: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name, text = line.split("\t", 1)
            name = name.strip() or f"case{i}"
            text = text.strip()
        else:
            name = f"case{i}"
            text = line
        if text:
            cases.append((name, text))
    if not cases:
        raise ValueError(f"No valid cases loaded from {path}")
    return cases


@dataclass
class ServerHandle:
    proc: asyncio.subprocess.Process
    queue: asyncio.Queue[dict[str, Any]]
    stdout_task: asyncio.Task[None]
    stderr_task: asyncio.Task[None]


@dataclass
class Variant:
    backend: str
    name: str
    env: dict[str, str]


def _parse_chunk_plan(raw: str) -> list[int]:
    if not raw.strip():
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    return out


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i]
        for j, b in enumerate(seq_b, start=1):
            cost = 0 if a == b else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def word_error_rate(ref: str, hyp: str) -> float:
    ref_w = _norm_text(ref).split()
    hyp_w = _norm_text(hyp).split()
    if not ref_w:
        return float("nan")
    return _levenshtein(ref_w, hyp_w) / float(len(ref_w))


def char_error_rate(ref: str, hyp: str) -> float:
    ref_c = list(_norm_text(ref))
    hyp_c = list(_norm_text(hyp))
    if not ref_c:
        return float("nan")
    return _levenshtein(ref_c, hyp_c) / float(len(ref_c))


def p50(vals: list[float]) -> float:
    xs = sorted(v for v in vals if not math.isnan(v))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


async def _drain(
    stream: asyncio.StreamReader | None, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            return
        txt = line.decode("utf-8", errors="replace").strip()
        if txt.startswith(PROFILE_PREFIX):
            try:
                payload = json.loads(txt[len(PROFILE_PREFIX) :].strip())
            except Exception:
                continue
            await queue.put(payload)


async def _wait_listen(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r, w = await asyncio.open_connection(host, port)
            w.close()
            await w.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"server not listening on {host}:{port}")


def resolve_server_python(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get("TTS_SERVER_PYTHON")
    if env_value:
        return env_value
    root = Path(__file__).parent
    candidates = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


async def start_server(
    host: str,
    port: int,
    server_python: str,
    backend: str,
    extra_env: dict[str, str],
    math_normalizer: str,
) -> ServerHandle:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TTS_PROFILE"] = "1"
    env["TTS_BACKEND"] = backend
    env["TTS_MATH_NORMALIZER"] = math_normalizer
    env.update(extra_env)
    proc = await asyncio.create_subprocess_exec(
        server_python,
        "-m",
        "uvicorn",
        "app.server:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
        cwd=str(Path(__file__).parent),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    out_t = asyncio.create_task(_drain(proc.stdout, q))
    err_t = asyncio.create_task(_drain(proc.stderr, q))
    try:
        await _wait_listen(host, port, timeout_s=20.0)
    except Exception:
        proc.terminate()
        await proc.wait()
        out_t.cancel()
        err_t.cancel()
        await asyncio.gather(out_t, err_t, return_exceptions=True)
        raise
    return ServerHandle(proc=proc, queue=q, stdout_task=out_t, stderr_task=err_t)


async def stop_server(h: ServerHandle) -> None:
    if h.proc.returncode is None:
        h.proc.terminate()
        try:
            await asyncio.wait_for(h.proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            h.proc.kill()
            await h.proc.wait()
    h.stdout_task.cancel()
    h.stderr_task.cancel()
    await asyncio.gather(h.stdout_task, h.stderr_task, return_exceptions=True)


async def await_profile_for_run(
    queue: asyncio.Queue[dict[str, Any]], run_id: str, timeout_s: float = 6.0
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    stash: list[dict[str, Any]] = []
    try:
        while time.monotonic() < deadline:
            rem = deadline - time.monotonic()
            item = await asyncio.wait_for(queue.get(), timeout=rem)
            if str(item.get("client_run_id", "")) == run_id:
                for s in stash:
                    queue.put_nowait(s)
                return item
            stash.append(item)
    except asyncio.TimeoutError:
        pass
    for s in stash:
        queue.put_nowait(s)
    return None


async def run_case(
    uri: str,
    text: str,
    run_id: str,
    chunk_mode: str,
    chunk_size: int,
    chunk_plan: list[int],
    delay_s: float,
) -> dict[str, Any]:
    t_first_any_send = None
    t_first_non_ws_send = None
    t_first_audio = None
    pcm_parts: list[bytes] = []

    async with websockets.connect(uri, max_size=None) as ws:
        t_first_any_send = time.perf_counter()
        await ws.send(json.dumps({"text": " ", "flush": False, "run_id": run_id}))
        if delay_s > 0:
            await asyncio.sleep(delay_s)

        i = 0
        plan_idx = 0
        while i < len(text):
            if chunk_mode == "ramp" and chunk_plan:
                size = chunk_plan[min(plan_idx, len(chunk_plan) - 1)]
                plan_idx += 1
            else:
                size = chunk_size
            part = text[i : i + size]
            if t_first_non_ws_send is None and part.strip():
                t_first_non_ws_send = time.perf_counter()
            await ws.send(json.dumps({"text": part, "flush": False, "run_id": run_id}))
            i += size
            if delay_s > 0:
                await asyncio.sleep(delay_s)

        await ws.send(json.dumps({"text": "", "flush": True, "run_id": run_id}))
        await ws.send(json.dumps({"text": "", "flush": False, "run_id": run_id}))

        try:
            while True:
                msg = await ws.recv()
                payload = json.loads(msg if isinstance(msg, str) else msg.decode("utf-8"))
                if payload.get("type") == "metrics":
                    continue
                audio_b64 = str(payload.get("audio", ""))
                if not audio_b64:
                    continue
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                import base64

                pcm_parts.append(base64.b64decode(audio_b64))
        except websockets.ConnectionClosed:
            pass

    ref = t_first_non_ws_send or t_first_any_send or time.perf_counter()
    ttft_ms = ((t_first_audio - ref) * 1000.0) if t_first_audio else float("nan")
    pcm = b"".join(pcm_parts)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0 if pcm else np.zeros(0, dtype=np.float32)
    return {"ttft_ms": ttft_ms, "audio": audio}


def transcribe(model: WhisperModel, audio: np.ndarray, sr_in: int = 44100) -> str:
    if audio.size == 0:
        return ""
    audio_16k = resample_linear(audio, sr_in, sr_out=16000)
    audio_16k = np.nan_to_num(audio_16k.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)
    if np.max(np.abs(audio_16k)) < 1e-6:
        return ""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        segments, _ = model.transcribe(
            audio=audio_16k,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
    return " ".join(seg.text.strip() for seg in segments).strip()


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Minimize both ttft_p50_ms and wer_mean.
    ordered = sorted(points, key=lambda x: (x["ttft_p50_ms"], x["wer_mean"]))
    out: list[dict[str, Any]] = []
    best_wer = float("inf")
    for p in ordered:
        w = p["wer_mean"]
        if w < best_wer:
            out.append(p)
            best_wer = w
    return out


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _slug(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(s)).strip("-")
    return out or "x"


def _build_variants(args: argparse.Namespace) -> list[Variant]:
    root = Path(__file__).parent
    variants: list[Variant] = []

    piper_voices = _split_csv(args.piper_voices)
    for voice in piper_voices:
        vpath = Path(voice)
        if not vpath.is_absolute():
            vpath = root / vpath
        if not vpath.exists():
            print(f"skip missing piper voice: {vpath}")
            continue
        cfg = Path(f"{vpath}.json")
        env = {"PIPER_VOICE": str(vpath)}
        if cfg.exists():
            env["PIPER_CONFIG"] = str(cfg)
        variants.append(Variant(backend="piper", name=vpath.name, env=env))

    say_voices = _split_csv(args.say_voices)
    if say_voices:
        for voice in say_voices:
            env = {"SAY_VOICE": voice, "SAY_RATE": str(max(80, int(args.say_rate)))}
            variants.append(Variant(backend="say", name=voice, env=env))

    kokoro_voice_list = _split_csv(args.kokoro_voice_list)
    if kokoro_voice_list:
        k_model = Path(args.kokoro_model)
        k_voices = Path(args.kokoro_voices_bin)
        if not k_model.is_absolute():
            k_model = root / k_model
        if not k_voices.is_absolute():
            k_voices = root / k_voices
        if not k_model.exists() or not k_voices.exists():
            print(
                "skip kokoro variants: model/voices files missing "
                f"(model={k_model}, voices={k_voices})"
            )
        else:
            for voice in kokoro_voice_list:
                env = {
                    "KOKORO_MODEL": str(k_model),
                    "KOKORO_VOICES": str(k_voices),
                    "KOKORO_VOICE": voice,
                    "KOKORO_LANG": str(args.kokoro_lang),
                    "KOKORO_SPEED": str(args.kokoro_speed),
                }
                variants.append(Variant(backend="kokoro", name=voice, env=env))

    return variants


async def run() -> None:
    ap = argparse.ArgumentParser(
        description="Pareto sweep: backend/voice TTFT vs ASR quality."
    )
    ap.add_argument("--uri", default="ws://127.0.0.1:8030/tts")
    ap.add_argument("--server-python")
    ap.add_argument("--math-normalizer", default="rule", choices=["rule", "off", "sre"])
    ap.add_argument("--chunk-mode", default="ramp", choices=["fixed", "ramp"])
    ap.add_argument("--chunk-size", type=int, default=22)
    ap.add_argument("--chunk-plan", default="4,8,32")
    ap.add_argument("--delay", type=float, default=0.01)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--whisper-model", default="tiny.en")
    ap.add_argument(
        "--cases-file",
        help="Optional text file with one case per line or '<name>\\t<text>' rows.",
    )
    ap.add_argument("--json-out", default="experiments/results/pareto_backends.json")
    ap.add_argument(
        "--piper-voices",
        default="en_US-lessac-low.onnx,en_US-lessac-medium.onnx,en_US-lessac-high.onnx",
        help="Comma-separated piper .onnx files.",
    )
    ap.add_argument(
        "--say-voices",
        default="Eddy (English (US)),Samantha",
        help="Comma-separated macOS `say` voice names. Empty string disables say backend.",
    )
    ap.add_argument("--say-rate", type=int, default=190)
    ap.add_argument(
        "--kokoro-voice-list",
        default="af_sarah,am_michael",
        help="Comma-separated Kokoro voice IDs. Empty string disables Kokoro backend.",
    )
    ap.add_argument("--kokoro-model", default="models/kokoro/kokoro-v1.0.onnx")
    ap.add_argument("--kokoro-voices-bin", default="models/kokoro/voices-v1.0.bin")
    ap.add_argument("--kokoro-lang", default="en-us")
    ap.add_argument("--kokoro-speed", type=float, default=1.0)
    args = ap.parse_args()

    parsed = urlparse(args.uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8030
    server_python = resolve_server_python(args.server_python)
    chunk_plan = _parse_chunk_plan(args.chunk_plan)
    cases = DEFAULT_CASES
    if args.cases_file:
        cpath = Path(args.cases_file)
        if not cpath.is_absolute():
            cpath = Path(__file__).parent / cpath
        cases = load_cases_file(cpath)

    print(f"Loading ASR model ({args.whisper_model})...")
    asr = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    variants = _build_variants(args)
    if not variants:
        raise RuntimeError("No valid variants configured to benchmark.")

    all_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for variant in variants:
        print(f"\n=== backend={variant.backend} variant={variant.name} ===")
        server = await start_server(
            host=host,
            port=port,
            server_python=server_python,
            backend=variant.backend,
            extra_env=variant.env,
            math_normalizer=args.math_normalizer,
        )
        try:
            rows_variant: list[dict[str, Any]] = []
            for case_name, case_text in cases:
                ref_spoken = (
                    normalize_for_tts(case_text).text
                    if args.math_normalizer != "off"
                    else case_text
                )
                for run_idx in range(1, max(1, int(args.runs)) + 1):
                    run_id = (
                        f"{_slug(variant.backend)}-{_slug(variant.name)}-{_slug(case_name)}-"
                        f"r{run_idx}-{uuid.uuid4().hex[:8]}"
                    )
                    out = await run_case(
                        uri=args.uri,
                        text=case_text,
                        run_id=run_id,
                        chunk_mode=args.chunk_mode,
                        chunk_size=max(1, int(args.chunk_size)),
                        chunk_plan=chunk_plan,
                        delay_s=max(0.0, float(args.delay)),
                    )
                    prof = await await_profile_for_run(server.queue, run_id, timeout_s=8.0)
                    hyp = transcribe(asr, out["audio"])
                    wer = word_error_rate(ref_spoken, hyp)
                    cer = char_error_rate(ref_spoken, hyp)
                    row = {
                        "backend": variant.backend,
                        "variant": variant.name,
                        "case": case_name,
                        "run": run_idx,
                        "ttft_ms": out["ttft_ms"],
                        "server_ttft_audio_out_ms": (
                            float(prof.get("ttft_audio_out_ms"))
                            if prof and prof.get("ttft_audio_out_ms") is not None
                            else float("nan")
                        ),
                        "wer": wer,
                        "cer": cer,
                        "hyp": hyp,
                        "ref": ref_spoken,
                    }
                    rows_variant.append(row)
                    all_rows.append(row)
                    print(
                        f"case={case_name:<5} run={run_idx} "
                        f"ttft={row['ttft_ms']:.1f}ms wer={row['wer']:.3f} cer={row['cer']:.3f}"
                    )
            ttfts = [float(r["ttft_ms"]) for r in rows_variant]
            wers = [
                float(r["wer"])
                for r in rows_variant
                if not math.isnan(float(r["wer"]))
            ]
            cers = [
                float(r["cer"])
                for r in rows_variant
                if not math.isnan(float(r["cer"]))
            ]
            point = {
                "backend": variant.backend,
                "variant": variant.name,
                "ttft_p50_ms": p50(ttfts),
                "wer_mean": (sum(wers) / len(wers)) if wers else float("nan"),
                "cer_mean": (sum(cers) / len(cers)) if cers else float("nan"),
            }
            summary.append(point)
            print(
                f"summary: ttft_p50={point['ttft_p50_ms']:.1f}ms "
                f"wer_mean={point['wer_mean']:.3f} cer_mean={point['cer_mean']:.3f}"
            )
        finally:
            await stop_server(server)

    frontier = pareto_front([p for p in summary if not math.isnan(p["wer_mean"])])

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "math_normalizer": args.math_normalizer,
                    "chunk_mode": args.chunk_mode,
                    "chunk_size": args.chunk_size,
                    "chunk_plan": chunk_plan,
                    "delay": args.delay,
                    "runs": args.runs,
                    "variants": [
                        {"backend": v.backend, "name": v.name} for v in variants
                    ],
                    "cases": [name for name, _ in cases],
                },
                "summary": summary,
                "pareto_frontier": frontier,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    print("\nPareto frontier (min TTFT, min WER):")
    for p in frontier:
        print(
            f"- {p['backend']}::{p['variant']}: ttft_p50={p['ttft_p50_ms']:.1f}ms "
            f"wer_mean={p['wer_mean']:.3f} cer_mean={p['cer_mean']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(run())
