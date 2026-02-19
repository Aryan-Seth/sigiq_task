from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import websockets

PROFILE_PREFIX = "[tts-profile-json] "


CASES = [
    r"The product of three and seven is \(3 \times 7 = 21\).",
    r"For a right triangle, the Pythagorean theorem states \(a^2 + b^2 = c^2\).",
    r"The derivative of \(e^x\) with respect to \(x\) is \(\frac{d}{dx} e^x = e^x\).",
]


@dataclass
class ServerHandle:
    proc: asyncio.subprocess.Process
    queue: asyncio.Queue[dict[str, Any]]
    stdout_task: asyncio.Task[None]
    stderr_task: asyncio.Task[None]


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


async def start_server(
    host: str,
    port: int,
    backend: str,
    aligner: str,
    math_normalizer: str,
    server_python: str,
    piper_voice: str | None,
    piper_config: str | None,
) -> ServerHandle:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TTS_PROFILE"] = "1"
    env["TTS_BACKEND"] = backend
    env["TTS_MATH_NORMALIZER"] = math_normalizer
    env.setdefault("TTS_MIN_CHARS_TO_SYNTH", "1")
    env.setdefault("TTS_IDLE_TRIGGER_MS", "5")
    env.setdefault("TTS_FIRST_SEGMENT_LIMIT", "40")
    if aligner and aligner != "none":
        env["ALIGNER"] = aligner
    else:
        env.pop("ALIGNER", None)
    if backend == "piper":
        if not piper_voice:
            raise ValueError("piper mode requires --piper-voice")
        env["PIPER_VOICE"] = piper_voice
        if piper_config:
            env["PIPER_CONFIG"] = piper_config
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
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    stdout_task = asyncio.create_task(_drain(proc.stdout, queue))
    stderr_task = asyncio.create_task(_drain(proc.stderr, queue))
    try:
        await _wait_listen(host, port, timeout_s=15.0)
    except Exception:
        proc.terminate()
        await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    return ServerHandle(proc=proc, queue=queue, stdout_task=stdout_task, stderr_task=stderr_task)


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
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


async def stop_server(server: ServerHandle) -> None:
    if server.proc.returncode is None:
        server.proc.terminate()
        try:
            await asyncio.wait_for(server.proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            server.proc.kill()
            await server.proc.wait()
    server.stdout_task.cancel()
    server.stderr_task.cancel()
    await asyncio.gather(server.stdout_task, server.stderr_task, return_exceptions=True)


async def await_profile_for_run(
    queue: asyncio.Queue[dict[str, Any]], run_id: str, timeout_s: float = 5.0
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    stash: List[dict[str, Any]] = []
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


async def run_case(uri: str, text: str, run_id: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    t_non_ws = None
    t_first_audio = None
    t_last_audio = None
    audio_bytes = 0
    samples_so_far = 0
    sr = 44100
    events: List[tuple[float, str, int]] = []

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": " ", "flush": False, "run_id": run_id}))
        await ws.send(json.dumps({"text": text, "flush": False, "run_id": run_id}))
        t_non_ws = time.perf_counter()
        await ws.send(json.dumps({"text": "", "flush": True, "run_id": run_id}))
        await ws.send(json.dumps({"text": "", "flush": False, "run_id": run_id}))
        try:
            while True:
                msg = await ws.recv()
                payload = json.loads(msg if isinstance(msg, str) else msg.decode("utf-8"))
                audio_b64 = str(payload.get("audio", ""))
                if not audio_b64:
                    continue
                audio = base64.b64decode(audio_b64)
                audio_bytes += len(audio)
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                t_last_audio = time.perf_counter()
                chunk_start_ms = (samples_so_far / float(sr)) * 1000.0
                samples_so_far += len(audio) // 2
                alignment = payload.get("alignment", {}) or {}
                chars = alignment.get("chars", []) or []
                starts = alignment.get("char_start_times_ms", []) or []
                idxs = alignment.get("char_indices", []) or []
                for i, ch in enumerate(chars):
                    start_ms = float(starts[i]) if i < len(starts) else 0.0
                    idx = -1
                    if i < len(idxs):
                        try:
                            idx = int(idxs[i])
                        except Exception:
                            idx = -1
                    events.append((chunk_start_ms + start_ms, str(ch), idx))
        except websockets.ConnectionClosed:
            pass

    events.sort(key=lambda x: (x[0], x[2]))
    spoken_chars: List[str] = []
    prev_t = -1e9
    prev_c = ""
    prev_idx = -1
    for t_ms, ch, idx in events:
        if ch == prev_c and idx == prev_idx and (t_ms - prev_t) <= 12.0:
            continue
        spoken_chars.append(ch)
        prev_t = t_ms
        prev_c = ch
        prev_idx = idx
    spoken = "".join(spoken_chars)
    ref = t_non_ws or t0
    return {
        "run_id": run_id,
        "ttft_ms": ((t_first_audio - ref) * 1000.0) if t_first_audio else float("nan"),
        "total_ms": ((t_last_audio - ref) * 1000.0) if t_last_audio else float("nan"),
        "audio_bytes": audio_bytes,
        "spoken": spoken,
    }


def pctl(values: List[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (q / 100.0) * (len(vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


async def run() -> None:
    parser = argparse.ArgumentParser(description="Benchmark math normalizers end-to-end.")
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/tts")
    parser.add_argument("--backend", default="piper", choices=["piper", "dummy"])
    parser.add_argument("--aligner", default="none")
    parser.add_argument("--math-modes", default="rule,sre")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--server-python",
        help="Python interpreter used to launch uvicorn server (defaults to TTS_SERVER_PYTHON or local .venv).",
    )
    parser.add_argument("--piper-voice")
    parser.add_argument("--piper-config")
    parser.add_argument("--json-out", default="/tmp/benchmark_math_normalizers.json")
    args = parser.parse_args()
    server_python = resolve_server_python(args.server_python)

    parsed = urlparse(args.uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    modes = [m.strip() for m in args.math_modes.split(",") if m.strip()]

    all_rows: List[dict[str, Any]] = []
    for mode in modes:
        print(f"\n=== mode={mode} ===")
        server = await start_server(
            host=host,
            port=port,
            backend=args.backend,
            aligner=args.aligner,
            math_normalizer=mode,
            server_python=server_python,
            piper_voice=args.piper_voice,
            piper_config=args.piper_config,
        )
        try:
            for case_idx, text in enumerate(CASES, start=1):
                rows: List[dict[str, Any]] = []
                for run_idx in range(1, max(1, args.runs) + 1):
                    run_id = f"{mode}-c{case_idx}-r{run_idx}-{uuid.uuid4().hex[:8]}"
                    row = await run_case(args.uri, text, run_id)
                    profile = await await_profile_for_run(server.queue, run_id, timeout_s=8.0)
                    row["server_ttft_ms"] = (
                        float(profile.get("ttft_audio_out_ms"))
                        if profile and profile.get("ttft_audio_out_ms") is not None
                        else float("nan")
                    )
                    row["mode"] = mode
                    row["case"] = case_idx
                    row["input"] = text
                    rows.append(row)
                    all_rows.append(row)
                print(
                    f"case={case_idx} ttft_client_p50={pctl([r['ttft_ms'] for r in rows], 50):.1f}ms "
                    f"ttft_server_p50={pctl([r['server_ttft_ms'] for r in rows], 50):.1f}ms"
                )
                spoken = rows[-1]["spoken"].replace("\n", " ")
                print(f"spoken(sample): {spoken[:170]}")
        finally:
            await stop_server(server)

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    print("\n=== summary ===")
    for mode in modes:
        rows = [r for r in all_rows if r["mode"] == mode]
        if not rows:
            continue
        print(
            f"{mode}: client_ttft_p50={pctl([r['ttft_ms'] for r in rows], 50):.1f}ms "
            f"client_ttft_p95={pctl([r['ttft_ms'] for r in rows], 95):.1f}ms "
            f"server_ttft_p50={pctl([r['server_ttft_ms'] for r in rows], 50):.1f}ms"
        )


if __name__ == "__main__":
    asyncio.run(run())
