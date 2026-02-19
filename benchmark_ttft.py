from __future__ import annotations

import argparse
import asyncio
import base64
import difflib
import json
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets

PROFILE_PREFIX = "[tts-profile-json] "


@dataclass
class ServerHandle:
    proc: asyncio.subprocess.Process
    profile_queue: asyncio.Queue[dict[str, Any]]
    stdout_task: asyncio.Task[None]
    stderr_task: asyncio.Task[None]


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


def _parse_lengths(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    if not out:
        raise ValueError("No valid lengths provided.")
    return out


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


def _load_source_text(path: Path | None) -> str:
    default = (
        "This is a benchmark sentence for streaming text to speech alignment and latency "
        "measurement across different payload lengths."
    )
    if path is None:
        return default
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8").strip()
    return text or default


def _load_cases(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
            out.append((name, text))
    if not out:
        raise ValueError(f"No valid cases found in {path}")
    return out


def _build_text(source: str, target_len: int) -> str:
    seed = source.strip() or "benchmark text"
    out = seed
    while len(out) < target_len:
        out += " " + seed
    return out[:target_len]


async def _drain_server_stream(
    stream: asyncio.StreamReader | None,
    label: str,
    profile_queue: asyncio.Queue[dict[str, Any]],
    verbose: bool,
) -> None:
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace").rstrip()
        if text.startswith(PROFILE_PREFIX):
            payload_txt = text[len(PROFILE_PREFIX) :].strip()
            try:
                payload = json.loads(payload_txt)
            except json.JSONDecodeError:
                if verbose:
                    print(f"[server-{label}] bad profile json: {payload_txt}")
            else:
                await profile_queue.put(payload)
        elif verbose:
            print(f"[server-{label}] {text}")


async def _wait_for_listen(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"Server did not start listening on {host}:{port} within {timeout_s}s")


async def _start_server(
    host: str,
    port: int,
    backend: str,
    aligner: str,
    math_normalizer: str,
    server_python: str,
    piper_voice: str | None,
    piper_config: str | None,
    min_chars_to_synth: int | None,
    idle_trigger_ms: int | None,
    first_segment_limit: int | None,
    est_char_ms: float | None,
    verbose: bool,
    startup_timeout_s: float,
) -> ServerHandle:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TTS_PROFILE"] = "1"
    env["TTS_BACKEND"] = backend
    env["TTS_MATH_NORMALIZER"] = math_normalizer
    if backend.lower() == "piper":
        if not piper_voice:
            raise ValueError("Piper backend requires --piper-voice /path/to/voice.onnx")
        env["PIPER_VOICE"] = piper_voice
        if piper_config:
            env["PIPER_CONFIG"] = piper_config
    if min_chars_to_synth is not None:
        env["TTS_MIN_CHARS_TO_SYNTH"] = str(max(1, int(min_chars_to_synth)))
    if idle_trigger_ms is not None:
        env["TTS_IDLE_TRIGGER_MS"] = str(max(1, int(idle_trigger_ms)))
    if first_segment_limit is not None:
        env["TTS_FIRST_SEGMENT_LIMIT"] = str(max(1, int(first_segment_limit)))
    if est_char_ms is not None:
        env["TTS_EST_CHAR_MS"] = str(max(1.0, float(est_char_ms)))
    if aligner and aligner.lower() != "none":
        env["ALIGNER"] = aligner
    else:
        env.pop("ALIGNER", None)

    cmd = [
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
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(Path(__file__).parent),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    profile_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    stdout_task = asyncio.create_task(
        _drain_server_stream(proc.stdout, "stdout", profile_queue, verbose)
    )
    stderr_task = asyncio.create_task(
        _drain_server_stream(proc.stderr, "stderr", profile_queue, verbose)
    )

    try:
        await _wait_for_listen(host, port, startup_timeout_s)
    except Exception:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        else:
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise

    return ServerHandle(
        proc=proc,
        profile_queue=profile_queue,
        stdout_task=stdout_task,
        stderr_task=stderr_task,
    )


def _resolve_server_python(cli_value: str | None) -> str:
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


async def _stop_server(handle: ServerHandle) -> None:
    if handle.proc.returncode is None:
        handle.proc.terminate()
        try:
            await asyncio.wait_for(handle.proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            handle.proc.kill()
            await handle.proc.wait()
    handle.stdout_task.cancel()
    handle.stderr_task.cancel()
    await asyncio.gather(handle.stdout_task, handle.stderr_task, return_exceptions=True)


async def _await_profile(
    queue: asyncio.Queue[dict[str, Any]], timeout_s: float
) -> dict[str, Any] | None:
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None


async def _await_profile_for_run(
    queue: asyncio.Queue[dict[str, Any]],
    run_id: str,
    timeout_s: float,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if run_id in cache:
        return cache.pop(run_id)

    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        profile = await _await_profile(queue, remaining)
        if profile is None:
            return None
        profile_run_id = str(profile.get("client_run_id", "")).strip()
        if profile_run_id == run_id:
            return profile
        if profile_run_id:
            cache[profile_run_id] = profile


async def _run_client_once(
    uri: str,
    text: str,
    chunk_size: int,
    chunk_mode: str,
    chunk_start: int,
    chunk_max: int,
    chunk_growth: float,
    chunk_plan: list[int],
    delay_s: float,
    receiver_idle_timeout_s: float,
    run_id: str,
) -> dict[str, Any]:
    t_session_start = time.perf_counter()
    t_first_any_send: float | None = None
    t_first_non_ws_send: float | None = None
    t_first_recv = None
    t_first_audio = None
    t_last_audio = None
    audio_chunks = 0
    audio_bytes = 0
    aligned_total_reported = 0
    aligned_stream_chars: list[str] = []
    aligned_unique_indices: set[int] = set()
    aligned_negative_durations = 0
    aligned_non_monotonic = 0
    aligned_prev_start_ms = -1.0
    aligned_last_end_ms = 0.0
    samples_so_far = 0
    sr = 44100
    receiver_timed_out = False
    sender_done = asyncio.Event()

    async with websockets.connect(uri, max_size=None) as ws:
        async def safe_send(payload: dict[str, Any]) -> bool:
            nonlocal t_first_any_send
            try:
                if t_first_any_send is None:
                    t_first_any_send = time.perf_counter()
                await ws.send(json.dumps(payload))
                return True
            except websockets.ConnectionClosed:
                return False

        async def sender() -> None:
            nonlocal t_first_non_ws_send
            try:
                ok = await safe_send({"text": " ", "flush": False, "run_id": run_id})
                if not ok:
                    return
                if delay_s > 0:
                    await asyncio.sleep(delay_s)

                use_ramp = chunk_mode == "ramp"
                i = 0
                ramp_size = max(1, int(chunk_start))
                ramp_cap = max(ramp_size, int(chunk_max))
                ramp_mul = max(1.0, float(chunk_growth))
                plan = [max(1, int(v)) for v in chunk_plan if int(v) > 0]
                plan_idx = 0

                while i < len(text):
                    size = int(chunk_size)
                    if use_ramp:
                        if plan:
                            size = plan[min(plan_idx, len(plan) - 1)]
                        else:
                            size = min(ramp_cap, max(1, int(round(ramp_size))))
                    part = text[i : i + size]
                    if t_first_non_ws_send is None and part.strip():
                        t_first_non_ws_send = time.perf_counter()
                    ok = await safe_send({"text": part, "flush": False, "run_id": run_id})
                    if not ok:
                        return
                    i += size
                    if use_ramp and plan:
                        plan_idx += 1
                    elif use_ramp and ramp_size < ramp_cap:
                        nxt = int(math.ceil(ramp_size * ramp_mul))
                        ramp_size = ramp_cap if nxt <= ramp_size else min(ramp_cap, nxt)
                    if delay_s > 0:
                        await asyncio.sleep(delay_s)

                ok = await safe_send({"text": "", "flush": True, "run_id": run_id})
                if not ok:
                    return
                await safe_send({"text": "", "flush": False, "run_id": run_id})
            finally:
                sender_done.set()

        async def receiver() -> None:
            nonlocal t_first_recv, t_first_audio, t_last_audio
            nonlocal audio_chunks, audio_bytes, samples_so_far
            nonlocal aligned_total_reported, aligned_stream_chars
            nonlocal aligned_unique_indices
            nonlocal aligned_negative_durations, aligned_non_monotonic, aligned_prev_start_ms
            nonlocal aligned_last_end_ms
            nonlocal receiver_timed_out
            last_msg_ts = time.perf_counter()

            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        if sender_done.is_set() and (time.perf_counter() - last_msg_ts) >= receiver_idle_timeout_s:
                            receiver_timed_out = True
                            return
                        continue
                    if t_first_recv is None:
                        t_first_recv = time.perf_counter()
                    last_msg_ts = time.perf_counter()
                    if isinstance(msg, bytes):
                        msg = msg.decode("utf-8")
                    payload = json.loads(msg)
                    audio_b64 = payload.get("audio", "")
                    alignment = payload.get("alignment", {}) or {}

                    if not audio_b64:
                        continue

                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    t_last_audio = time.perf_counter()
                    audio_chunks += 1

                    audio_chunk = base64.b64decode(audio_b64)
                    audio_bytes += len(audio_chunk)
                    chunk_start_ms = (samples_so_far / float(sr)) * 1000.0
                    samples_so_far += len(audio_chunk) // 2

                    chars = alignment.get("chars", [])
                    starts = alignment.get("char_start_times_ms", [])
                    durs = alignment.get("char_durations_ms", [])
                    idxs = alignment.get("char_indices", [])
                    for j, (ch, start_ms, dur_ms) in enumerate(zip(chars, starts, durs)):
                        aligned_total_reported += 1
                        aligned_stream_chars.append(str(ch))
                        if j < len(idxs):
                            try:
                                aligned_unique_indices.add(int(idxs[j]))
                            except Exception:
                                pass

                        abs_start_ms = chunk_start_ms + float(start_ms)
                        if abs_start_ms < aligned_prev_start_ms:
                            aligned_non_monotonic += 1
                        aligned_prev_start_ms = abs_start_ms
                        if float(dur_ms) <= 0.0:
                            aligned_negative_durations += 1

                        aligned_last_end_ms = max(
                            aligned_last_end_ms, abs_start_ms + float(dur_ms)
                        )
            except websockets.ConnectionClosed:
                return

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        await sender_task
        try:
            await asyncio.wait_for(receiver_task, timeout=max(1.0, receiver_idle_timeout_s + 1.0))
        except asyncio.TimeoutError:
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
            receiver_timed_out = True

    ref = t_first_non_ws_send or t_first_any_send or t_session_start
    target_len = len(text)
    matched_by_idx = sum(1 for idx in aligned_unique_indices if 0 <= int(idx) < target_len)
    observed_text = "".join(aligned_stream_chars)
    matcher = difflib.SequenceMatcher(None, observed_text, text)
    matched_by_seq = sum(block.size for block in matcher.get_matching_blocks())
    if aligned_unique_indices:
        aligned_matched_chars = max(matched_by_idx, matched_by_seq)
    else:
        aligned_matched_chars = matched_by_seq
    if target_len > 0:
        alignment_coverage = aligned_matched_chars / float(target_len)
    else:
        alignment_coverage = float("nan")
    aligned_skipped_chars = max(0, len(aligned_stream_chars) - aligned_matched_chars)
    aligned_missing_chars = max(0, target_len - aligned_matched_chars)
    audio_total_ms = (samples_so_far / float(sr)) * 1000.0
    alignment_tail_error_ms = aligned_last_end_ms - audio_total_ms
    out: dict[str, Any] = {
        "chars": len(text),
        "audio_chunks": audio_chunks,
        "audio_bytes": audio_bytes,
        "alignment_total_reported_chars": aligned_total_reported,
        "alignment_stream_chars": len(aligned_stream_chars),
        "alignment_unique_indices": len(aligned_unique_indices),
        "alignment_matched_chars": aligned_matched_chars,
        "alignment_skipped_chars": aligned_skipped_chars,
        "alignment_missing_chars": aligned_missing_chars,
        "alignment_coverage": alignment_coverage,
        "alignment_non_monotonic_count": aligned_non_monotonic,
        "alignment_non_positive_duration_count": aligned_negative_durations,
        "alignment_tail_error_ms": alignment_tail_error_ms,
        "alignment_abs_tail_error_ms": abs(alignment_tail_error_ms),
        "receiver_timed_out": receiver_timed_out,
        "client_ttft_recv_ms": (
            (t_first_audio - ref) * 1000.0 if t_first_audio is not None else float("nan")
        ),
        "client_ttfb_msg_ms": (
            (t_first_recv - ref) * 1000.0 if t_first_recv is not None else float("nan")
        ),
        "client_total_audio_ms": (
            (t_last_audio - ref) * 1000.0 if t_last_audio is not None else float("nan")
        ),
    }
    total_s = out["client_total_audio_ms"] / 1000.0 if not math.isnan(out["client_total_audio_ms"]) else float("nan")
    out["client_chars_per_s"] = (
        len(text) / total_s if total_s and not math.isnan(total_s) and total_s > 0 else float("nan")
    )
    return out


def _fmt_ms(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.1f}"


def _summarize(results: list[dict[str, Any]]) -> None:
    by_len: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_len[int(row["chars"])].append(row)

    print(
        "\nlength  runs  client_ttft_p50  client_ttft_p95  "
        "server_ttft_audio_p50  server_ttft_send_p50  align_cov_p50  align_tail_abs_p50"
    )
    for length in sorted(by_len):
        rows = by_len[length]
        valid_rows = [r for r in rows if not r.get("receiver_timed_out")]
        align_cov = [
            r["alignment_coverage"]
            for r in valid_rows
            if not math.isnan(r["alignment_coverage"])
        ]
        align_tail_abs = [
            r["alignment_abs_tail_error_ms"]
            for r in valid_rows
            if not math.isnan(r["alignment_abs_tail_error_ms"])
        ]
        align_cov_p50 = _percentile(align_cov, 50)
        align_tail_abs_p50 = _percentile(align_tail_abs, 50)

        print(
            f"{length:<7}{len(valid_rows):<6}"
            f"{_fmt_ms(_percentile([r['client_ttft_recv_ms'] for r in valid_rows if not math.isnan(r['client_ttft_recv_ms'])], 50)):<17}"
            f"{_fmt_ms(_percentile([r['client_ttft_recv_ms'] for r in valid_rows if not math.isnan(r['client_ttft_recv_ms'])], 95)):<17}"
            f"{_fmt_ms(_percentile([r['server_ttft_audio_out_ms'] for r in valid_rows if r.get('server_ttft_audio_out_ms') is not None], 50)):<23}"
            f"{_fmt_ms(_percentile([r['server_ttft_send_ms'] for r in valid_rows if r.get('server_ttft_send_ms') is not None], 50)):<21}"
            f"{_fmt_ms((align_cov_p50 * 100.0) if align_cov_p50 is not None else float('nan')):<15}"
            f"{_fmt_ms(align_tail_abs_p50 if align_tail_abs_p50 is not None else float('nan'))}"
        )

    print(
        "\nlength  runs  pre_model_p50  math_norm_p50  model_p50  post_total_p50  "
        "aligner_p50  packaging_p50  send_p50  client_minus_server_send_p50"
    )
    for length in sorted(by_len):
        rows = by_len[length]
        valid_rows = [r for r in rows if not r.get("receiver_timed_out")]
        pre_model = [
            r.get("server_gap_buffer_to_synth_ms")
            for r in valid_rows
            if r.get("server_gap_buffer_to_synth_ms") is not None
        ]
        math_norm = [
            r.get("server_gap_math_normalize_ms")
            for r in valid_rows
            if r.get("server_gap_math_normalize_ms") is not None
        ]
        model_ttft = [
            r.get("server_model_compute_ms")
            for r in valid_rows
            if r.get("server_model_compute_ms") is not None
        ]
        post_model = [
            r.get("server_post_model_total_incl_aligner_ms")
            for r in valid_rows
            if r.get("server_post_model_total_incl_aligner_ms") is not None
        ]
        aligner = [
            r.get("server_gap_aligner_ms")
            for r in valid_rows
            if r.get("server_gap_aligner_ms") is not None
        ]
        packaging = [
            r.get("server_packaging_only_ms")
            for r in valid_rows
            if r.get("server_packaging_only_ms") is not None
        ]
        send_ovh = [
            r.get("server_gap_audio_out_to_send_ms")
            for r in valid_rows
            if r.get("server_gap_audio_out_to_send_ms") is not None
        ]
        client_transport = [
            r["client_minus_server_send_ms"]
            for r in valid_rows
            if r.get("client_minus_server_send_ms") is not None
            and not math.isnan(r["client_minus_server_send_ms"])
        ]
        print(
            f"{length:<7}{len(valid_rows):<6}"
            f"{_fmt_ms(_percentile(pre_model, 50)):<15}"
            f"{_fmt_ms(_percentile(math_norm, 50)):<15}"
            f"{_fmt_ms(_percentile(model_ttft, 50)):<11}"
            f"{_fmt_ms(_percentile(post_model, 50)):<16}"
            f"{_fmt_ms(_percentile(aligner, 50)):<13}"
            f"{_fmt_ms(_percentile(packaging, 50)):<14}"
            f"{_fmt_ms(_percentile(send_ovh, 50)):<10}"
            f"{_fmt_ms(_percentile(client_transport, 50))}"
        )


async def run() -> None:
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for the TTS WebSocket server (client + server-side metrics)."
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/tts")
    parser.add_argument("--lengths", default="20,50,100,200,400,800")
    parser.add_argument("--runs-per-length", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument(
        "--chunk-mode",
        choices=["fixed", "ramp"],
        default="fixed",
        help="Input chunk scheduling mode for client sender.",
    )
    parser.add_argument("--chunk-start", type=int, default=4, help="Ramp mode starting chunk size.")
    parser.add_argument("--chunk-max", type=int, default=32, help="Ramp mode maximum chunk size.")
    parser.add_argument("--chunk-growth", type=float, default=2.0, help="Ramp growth multiplier.")
    parser.add_argument(
        "--chunk-plan",
        default="",
        help="Optional comma-separated ramp plan, e.g. '4,8,32'. Last size is held for remaining text.",
    )
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--source-file", default="sample_text.txt")
    parser.add_argument(
        "--cases-file",
        help="Optional file containing one prompt per line or '<name>\\t<text>' rows.",
    )
    parser.add_argument("--backend", default="dummy")
    parser.add_argument(
        "--server-python",
        help="Python interpreter used to launch uvicorn server (defaults to TTS_SERVER_PYTHON or local .venv).",
    )
    parser.add_argument("--piper-voice", help="Path to Piper .onnx voice file")
    parser.add_argument("--piper-config", help="Path to Piper voice config .json")
    parser.add_argument("--min-chars-to-synth", type=int, help="Override TTS_MIN_CHARS_TO_SYNTH")
    parser.add_argument("--idle-trigger-ms", type=int, help="Override TTS_IDLE_TRIGGER_MS")
    parser.add_argument("--first-segment-limit", type=int, help="Override TTS_FIRST_SEGMENT_LIMIT")
    parser.add_argument("--est-char-ms", type=float, help="Override TTS_EST_CHAR_MS")
    parser.add_argument("--aligner", default="none")
    parser.add_argument(
        "--math-normalizer",
        default="rule",
        choices=["rule", "sre", "off"],
        help="Math normalization mode for server-side text processing.",
    )
    parser.add_argument("--server-profile-timeout", type=float, default=5.0)
    parser.add_argument("--server-startup-timeout", type=float, default=12.0)
    parser.add_argument("--verbose-server-log", action="store_true")
    parser.add_argument(
        "--receiver-idle-timeout",
        type=float,
        default=None,
        help=(
            "Stop receiver after this idle time once sender is done. "
            "Default is 2.0s without aligner, 8.0s with aligner."
        ),
    )
    parser.add_argument(
        "--assume-synced-clocks",
        action="store_true",
        help="Enable cross-process subtraction metrics (valid only when clocks are comparable).",
    )
    parser.add_argument("--json-out", help="Optional path to write raw run data as JSON")
    parser.add_argument("--no-start-server", action="store_true")
    args = parser.parse_args()

    uri = args.uri
    parsed = urlparse(uri)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)

    lengths = _parse_lengths(args.lengths)
    source_path = Path(args.source_file) if args.source_file else None
    if source_path and not source_path.is_absolute():
        source_path = Path(__file__).parent / source_path
    source_text = _load_source_text(source_path)
    cases: list[tuple[str, str]] | None = None
    if args.cases_file:
        cpath = Path(args.cases_file)
        if not cpath.is_absolute():
            cpath = Path(__file__).parent / cpath
        cases = _load_cases(cpath)
    server_python = _resolve_server_python(args.server_python)
    default_receiver_idle_timeout = (
        8.0 if str(args.aligner).strip().lower() not in {"", "none"} else 2.0
    )
    receiver_idle_timeout_s = (
        max(0.1, float(args.receiver_idle_timeout))
        if args.receiver_idle_timeout is not None
        else default_receiver_idle_timeout
    )

    start_server = not args.no_start_server
    server: ServerHandle | None = None
    if start_server:
        print(
            f"Starting local server on {host}:{port} "
            f"(backend={args.backend}, aligner={args.aligner}, math={args.math_normalizer})..."
        )
        server = await _start_server(
            host=host,
            port=port,
            backend=args.backend,
            aligner=args.aligner,
            math_normalizer=args.math_normalizer,
            server_python=server_python,
            piper_voice=args.piper_voice,
            piper_config=args.piper_config,
            min_chars_to_synth=args.min_chars_to_synth,
            idle_trigger_ms=args.idle_trigger_ms,
            first_segment_limit=args.first_segment_limit,
            est_char_ms=args.est_char_ms,
            verbose=args.verbose_server_log,
            startup_timeout_s=args.server_startup_timeout,
        )

    results: list[dict[str, Any]] = []
    profile_cache: dict[str, dict[str, Any]] = {}
    try:
        for i in range(max(0, int(args.warmup_runs))):
            warm_seed = cases[0][1] if cases else source_text
            warmup_text = _build_text(warm_seed, lengths[0])
            warmup_run_id = f"warmup-{i + 1}"
            _ = await _run_client_once(
                uri=uri,
                text=warmup_text,
                chunk_size=max(1, int(args.chunk_size)),
                chunk_mode=str(args.chunk_mode),
                chunk_start=max(1, int(args.chunk_start)),
                chunk_max=max(1, int(args.chunk_max)),
                chunk_growth=max(1.0, float(args.chunk_growth)),
                chunk_plan=_parse_chunk_plan(str(args.chunk_plan)),
                delay_s=max(0.0, float(args.delay)),
                receiver_idle_timeout_s=receiver_idle_timeout_s,
                run_id=warmup_run_id,
            )
            if server is not None:
                _ = await _await_profile_for_run(
                    server.profile_queue,
                    warmup_run_id,
                    args.server_profile_timeout,
                    profile_cache,
                )
            print(f"Warmup {i + 1}/{args.warmup_runs} complete")

        for length in lengths:
            case_items = cases if cases else [("default", source_text)]
            for case_name, case_seed in case_items:
                for run_idx in range(max(1, int(args.runs_per_length))):
                    text = _build_text(case_seed, length)
                    run_id = (
                        f"len-{length}-case-{case_name}-run-{run_idx + 1}-{uuid.uuid4().hex[:8]}"
                    )
                    row = await _run_client_once(
                        uri=uri,
                        text=text,
                        chunk_size=max(1, int(args.chunk_size)),
                        chunk_mode=str(args.chunk_mode),
                        chunk_start=max(1, int(args.chunk_start)),
                        chunk_max=max(1, int(args.chunk_max)),
                        chunk_growth=max(1.0, float(args.chunk_growth)),
                        chunk_plan=_parse_chunk_plan(str(args.chunk_plan)),
                        delay_s=max(0.0, float(args.delay)),
                        receiver_idle_timeout_s=receiver_idle_timeout_s,
                        run_id=run_id,
                    )
                    row["run"] = run_idx + 1
                    row["run_id"] = run_id
                    row["case"] = case_name

                    if server is not None:
                        profile = await _await_profile_for_run(
                            server.profile_queue,
                            run_id,
                            args.server_profile_timeout,
                            profile_cache,
                        )
                        if profile is not None:
                            row["server_session"] = profile.get("session")
                            row["server_ttft_audio_out_ms"] = profile.get("ttft_audio_out_ms")
                            row["server_ttft_send_ms"] = profile.get("ttft_send_ms")
                            row["server_gap_buffer_to_synth_ms"] = profile.get(
                                "gap_buffer_to_synth_ms"
                            )
                            row["server_gap_synth_to_first_chunk_ms"] = profile.get(
                                "gap_synth_to_first_chunk_ms"
                            )
                            row["server_gap_synth_to_backend_done_ms"] = profile.get(
                                "gap_synth_to_backend_done_ms"
                            )
                            row["server_gap_first_chunk_to_audio_out_ms"] = profile.get(
                                "gap_first_chunk_to_audio_out_ms"
                            )
                            row["server_gap_backend_done_to_audio_out_ms"] = profile.get(
                                "gap_backend_done_to_audio_out_ms"
                            )
                            row["server_gap_aligner_ms"] = profile.get("gap_aligner_ms")
                            row["server_gap_aligner_done_to_audio_out_ms"] = profile.get(
                                "gap_aligner_done_to_audio_out_ms"
                            )
                            row["server_gap_math_normalize_ms"] = profile.get(
                                "gap_math_normalize_ms"
                            )
                            row["server_gap_audio_out_to_send_ms"] = profile.get(
                                "gap_audio_out_to_send_ms"
                            )
                            row["server_model_compute_ms"] = profile.get("model_compute_ms")
                            row["server_post_model_total_incl_aligner_ms"] = profile.get(
                                "post_model_total_incl_aligner_ms",
                                profile.get("post_model_to_audio_out_ms"),
                            )
                            row["server_packaging_only_ms"] = profile.get(
                                "packaging_only_ms",
                                profile.get("gap_aligner_done_to_audio_out_ms"),
                            )
                            row["server_post_model_to_audio_out_ms"] = profile.get(
                                "post_model_to_audio_out_ms"
                            )
                        else:
                            row["server_session"] = None
                            row["server_ttft_audio_out_ms"] = None
                            row["server_ttft_send_ms"] = None
                            row["server_gap_math_normalize_ms"] = None
                            row["server_gap_audio_out_to_send_ms"] = None
                            row["server_model_compute_ms"] = None
                            row["server_post_model_total_incl_aligner_ms"] = None
                            row["server_packaging_only_ms"] = None
                            row["server_post_model_to_audio_out_ms"] = None

                    server_send_ms = row.get("server_ttft_send_ms")
                    if (
                        args.assume_synced_clocks
                        and server_send_ms is not None
                        and not math.isnan(row["client_ttft_recv_ms"])
                    ):
                        row["client_minus_server_send_ms"] = row["client_ttft_recv_ms"] - float(
                            server_send_ms
                        )
                    else:
                        row["client_minus_server_send_ms"] = float("nan")

                    results.append(row)
                    case_label = f" case={case_name}" if cases else ""
                    print(
                        f"len={length:>4}{case_label} run={run_idx + 1:>2} "
                        f"client_ttft={_fmt_ms(row['client_ttft_recv_ms'])}ms "
                        f"client_total={_fmt_ms(row['client_total_audio_ms'])}ms "
                        f"server_ttft={_fmt_ms(row.get('server_ttft_audio_out_ms'))}ms "
                        f"align_cov={_fmt_ms(row['alignment_coverage'] * 100.0)}% "
                        f"client-server(send)={_fmt_ms(row['client_minus_server_send_ms'])}ms "
                        f"recv_timeout={row['receiver_timed_out']}"
                    )
    finally:
        if server is not None:
            await _stop_server(server)

    _summarize(results)
    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = Path(__file__).parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote raw benchmark data to {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
