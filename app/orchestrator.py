from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
import numpy as np
import os

from .alignment import (
    alignment_from_duration,
    scale_alignment_to_duration,
    split_alignment_for_window,
)
from .audio import float_to_pcm16_bytes, pcm16_bytes_to_base64, resample_linear
from .backends.base import TTSBackend
from .text_normalize import (
    has_unclosed_inline_math,
    math_normalization_enabled,
    normalize_for_tts,
)

@dataclass
class SessionState:
    buffer_text: str = ""
    synth_cursor: int = 0
    flush_requested: bool = False
    close_requested: bool = False
    last_input_ts: float = field(default_factory=time.monotonic)
    first_segment_done: bool = False
    saw_non_ws_text: bool = False


def _has_sentence_boundary(text: str) -> bool:
    for ch in text:
        if ch in ".?!\n":
            return True
    return False


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)


EST_CHAR_MS = _env_float("TTS_EST_CHAR_MS", 55.0, minimum=1.0)
MIN_CHARS_TO_SYNTH = _env_int("TTS_MIN_CHARS_TO_SYNTH", 10)
IDLE_TRIGGER_MS = _env_int("TTS_IDLE_TRIGGER_MS", 100, minimum=1)
FIRST_SEGMENT_LIMIT = _env_int("TTS_FIRST_SEGMENT_LIMIT", 100)


class Profiler:
    def __init__(self, session_id: str, enabled: bool) -> None:
        self.session_id = session_id
        self.enabled = enabled
        self.t0 = time.perf_counter()
        self.times: Dict[str, float] = {}
        self.client_run_id: Optional[str] = None
        self._final_parts: Optional[list[str]] = None
        self._final_metrics: Optional[Dict[str, float | str]] = None

    def set_client_run_id(self, run_id: str) -> None:
        if not self.enabled:
            return
        if self.client_run_id is None and run_id:
            self.client_run_id = run_id

    def mark(self, key: str) -> None:
        if not self.enabled:
            return
        if key not in self.times:
            self.times[key] = time.perf_counter()

    def set(self, key: str) -> None:
        if not self.enabled:
            return
        self.times[key] = time.perf_counter()

    def _compute_parts_and_metrics(self) -> tuple[list[str], Dict[str, float | str]]:
        parts = [f"session={self.session_id}"]
        metrics: Dict[str, float | str] = {"session": self.session_id}
        if self.client_run_id:
            parts.append(f"client_run_id={self.client_run_id}")
            metrics["client_run_id"] = self.client_run_id

        order = [
            "first_message",
            "first_text",
            "first_non_ws_text",
            "first_synth_start",
            "backend_first_chunk",
            "backend_synth_done",
            "first_audio_out",
            "first_send",
            "last_send",
            "session_end",
        ]
        for key in order:
            if key in self.times:
                delta_ms = (self.times[key] - self.t0) * 1000.0
                parts.append(f"{key}={delta_ms:.1f}ms")
                metrics[f"{key}_ms"] = round(delta_ms, 3)

        if "first_non_ws_text" in self.times and "first_audio_out" in self.times:
            ttft_ms = (self.times["first_audio_out"] - self.times["first_non_ws_text"]) * 1000.0
            parts.append(f"ttft_audio_out={ttft_ms:.1f}ms")
            metrics["ttft_audio_out_ms"] = round(ttft_ms, 3)
        if "first_non_ws_text" in self.times and "first_send" in self.times:
            ttft_send_ms = (self.times["first_send"] - self.times["first_non_ws_text"]) * 1000.0
            parts.append(f"ttft_send={ttft_send_ms:.1f}ms")
            metrics["ttft_send_ms"] = round(ttft_send_ms, 3)

        # breakdowns for latency hunting
        def gap(a: str, b: str) -> Optional[float]:
            if a in self.times and b in self.times:
                return (self.times[b] - self.times[a]) * 1000.0
            return None

        # Primary TTFT decomposition:
        #   first_non_ws_text -> first_synth_start -> backend_* -> first_audio_out -> first_send
        # Aligner can sit inside the post-model window, so we also report a packaging-only metric.
        buffer_wait = gap("first_non_ws_text", "first_synth_start")
        backend_wait = gap("first_synth_start", "backend_first_chunk")
        backend_done_wait = gap("first_synth_start", "backend_synth_done")
        post_wait = gap("backend_first_chunk", "first_audio_out")
        post_after_done_wait = gap("backend_synth_done", "first_audio_out")
        aligner_wait = gap("aligner_start", "aligner_done")
        post_after_aligner_wait = gap("aligner_done", "first_audio_out")
        math_norm_wait = gap("math_normalize_start", "math_normalize_done")
        send_wait = gap("first_audio_out", "first_send")
        if buffer_wait is not None:
            parts.append(f"gap_buffer_to_synth={buffer_wait:.1f}ms")
            metrics["gap_buffer_to_synth_ms"] = round(buffer_wait, 3)
        if backend_wait is not None:
            parts.append(f"gap_synth_to_first_chunk={backend_wait:.1f}ms")
            metrics["gap_synth_to_first_chunk_ms"] = round(backend_wait, 3)
        if backend_done_wait is not None:
            parts.append(f"gap_synth_to_backend_done={backend_done_wait:.1f}ms")
            metrics["gap_synth_to_backend_done_ms"] = round(backend_done_wait, 3)
        if post_wait is not None:
            parts.append(f"gap_first_chunk_to_audio_out={post_wait:.1f}ms")
            metrics["gap_first_chunk_to_audio_out_ms"] = round(post_wait, 3)
        if post_after_done_wait is not None:
            parts.append(f"gap_backend_done_to_audio_out={post_after_done_wait:.1f}ms")
            metrics["gap_backend_done_to_audio_out_ms"] = round(post_after_done_wait, 3)
        if aligner_wait is not None:
            parts.append(f"gap_aligner={aligner_wait:.1f}ms")
            metrics["gap_aligner_ms"] = round(aligner_wait, 3)
        if post_after_aligner_wait is not None:
            parts.append(f"gap_aligner_done_to_audio_out={post_after_aligner_wait:.1f}ms")
            metrics["gap_aligner_done_to_audio_out_ms"] = round(post_after_aligner_wait, 3)
        if math_norm_wait is not None:
            parts.append(f"gap_math_normalize={math_norm_wait:.1f}ms")
            metrics["gap_math_normalize_ms"] = round(math_norm_wait, 3)
        if send_wait is not None:
            parts.append(f"gap_audio_out_to_send={send_wait:.1f}ms")
            metrics["gap_audio_out_to_send_ms"] = round(send_wait, 3)

        model_compute_ms = backend_wait if backend_wait is not None else backend_done_wait
        if model_compute_ms is not None:
            parts.append(f"model_compute={model_compute_ms:.1f}ms")
            metrics["model_compute_ms"] = round(model_compute_ms, 3)
        post_model_total_ms = post_wait if post_wait is not None else post_after_done_wait
        if post_model_total_ms is not None:
            parts.append(f"post_model_to_audio_out={post_model_total_ms:.1f}ms")
            metrics["post_model_to_audio_out_ms"] = round(post_model_total_ms, 3)
            metrics["post_model_total_incl_aligner_ms"] = round(post_model_total_ms, 3)

        packaging_only_ms: Optional[float]
        if post_after_aligner_wait is not None:
            packaging_only_ms = post_after_aligner_wait
        elif post_model_total_ms is not None and aligner_wait is not None:
            packaging_only_ms = max(0.0, post_model_total_ms - aligner_wait)
        else:
            packaging_only_ms = post_model_total_ms
        if packaging_only_ms is not None:
            parts.append(f"packaging_only={packaging_only_ms:.1f}ms")
            metrics["packaging_only_ms"] = round(packaging_only_ms, 3)
        return parts, metrics

    def finalize_metrics(self) -> Dict[str, float | str]:
        if not self.enabled:
            return {}
        if self._final_metrics is not None:
            return dict(self._final_metrics)
        self.set("session_end")
        parts, metrics = self._compute_parts_and_metrics()
        self._final_parts = parts
        self._final_metrics = dict(metrics)
        return dict(metrics)

    def report(self) -> None:
        if not self.enabled:
            return
        metrics = self.finalize_metrics()
        parts = self._final_parts if self._final_parts is not None else []
        print("[tts-profile] " + " ".join(parts), flush=True)
        print("[tts-profile-json] " + json.dumps(metrics, sort_keys=True), flush=True)


# alignment logging toggle
LOG_ALIGNMENT = os.environ.get("LOG_ALIGNMENT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LOG_ALIGNMENT_MAX = int(os.environ.get("LOG_ALIGNMENT_MAX", "3"))  # log first N chunks per session


def _estimate_alignment(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "chars": [],
            "char_start_times_ms": [],
            "char_durations_ms": [],
            "char_indices": [],
        }
    total_ms = max(1.0, EST_CHAR_MS * len(text))
    return alignment_from_duration(text, total_ms)


async def _stream_segment(
    backend: TTSBackend,
    segment: str,
    segment_start_idx: int,
    source_indices: Optional[list[int]],
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    shutdown_event: asyncio.Event,
    target_sr: int,
    profiler: Optional[Profiler],
) -> None:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
    alignment_full = _estimate_alignment(segment)
    idxs = alignment_full.get("char_indices", [])
    if source_indices is not None and len(source_indices) == len(alignment_full.get("chars", [])):
        alignment_full["char_indices"] = [int(v) for v in source_indices]
    else:
        if not isinstance(idxs, list) or len(idxs) != len(alignment_full.get("chars", [])):
            idxs = list(range(len(alignment_full.get("chars", []))))
        alignment_full["char_indices"] = [int(segment_start_idx + int(i)) for i in idxs]
    estimated_total_ms = sum(alignment_full.get("char_durations_ms", []))
    est_cursor_ms = 0.0
    chunk_ms = 30.0
    chunk_samples = max(1, int(round(target_sr * (chunk_ms / 1000.0))))
    audio_buffer = np.zeros(0, dtype=np.float32)

    def producer() -> None:
        try:
            for chunk in backend.synthesize_stream(segment):
                if shutdown_event.is_set():
                    break
                loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(producer))

    finalizing = False
    try:
        while True:
            if shutdown_event.is_set():
                return
            item = await audio_queue.get()
            if item is None:
                finalizing = True
                break

            if profiler is not None:
                profiler.mark("backend_first_chunk")
            resampled = resample_linear(item.audio, item.sr, sr_out=target_sr)
            if resampled.size > 0:
                audio_buffer = np.concatenate([audio_buffer, resampled])

            while audio_buffer.shape[0] >= chunk_samples:
                frame = audio_buffer[:chunk_samples]
                audio_buffer = audio_buffer[chunk_samples:]
                total_ms = (frame.shape[0] / float(target_sr)) * 1000.0
                chunk_alignment = split_alignment_for_window(
                    alignment_full, est_cursor_ms, est_cursor_ms + total_ms
                )
                est_cursor_ms += total_ms
                audio_b64 = pcm16_bytes_to_base64(float_to_pcm16_bytes(frame))
                payload = {"audio": audio_b64, "alignment": chunk_alignment}
                if profiler is not None:
                    profiler.mark("first_audio_out")
                    profiler.set("last_audio_out")
                if shutdown_event.is_set():
                    return
                await out_queue.put(payload)

        if finalizing:
            frames = []
            while audio_buffer.shape[0] > 0:
                if audio_buffer.shape[0] >= chunk_samples:
                    frame = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples:]
                else:
                    frame = audio_buffer
                    audio_buffer = np.zeros(0, dtype=np.float32)
                frames.append(frame)

            for idx, frame in enumerate(frames):
                total_ms = (frame.shape[0] / float(target_sr)) * 1000.0
                is_last = idx == len(frames) - 1
                if is_last:
                    remaining = split_alignment_for_window(
                        alignment_full, est_cursor_ms, estimated_total_ms
                    )
                    chunk_alignment = scale_alignment_to_duration(remaining, total_ms)
                else:
                    chunk_alignment = split_alignment_for_window(
                        alignment_full, est_cursor_ms, est_cursor_ms + total_ms
                    )
                    est_cursor_ms += total_ms

                audio_b64 = pcm16_bytes_to_base64(float_to_pcm16_bytes(frame))
                payload = {"audio": audio_b64, "alignment": chunk_alignment}
                if profiler is not None:
                    profiler.mark("first_audio_out")
                    profiler.set("last_audio_out")
                if shutdown_event.is_set():
                    return
                await out_queue.put(payload)
    finally:
        if shutdown_event.is_set():
            producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
        else:
            await producer_task


async def recv_loop(
    ws: WebSocket,
    state: SessionState,
    state_lock: asyncio.Lock,
    wake_event: asyncio.Event,
    shutdown_event: asyncio.Event,
    profiler: Optional[Profiler],
) -> None:
    try:
        while not shutdown_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if profiler is not None:
                profiler.mark("first_message")
            text = str(msg.get("text", ""))
            flush = bool(msg.get("flush", False))
            run_id = msg.get("run_id")

            async with state_lock:
                state.last_input_ts = time.monotonic()
                if isinstance(run_id, str) and run_id.strip() and profiler is not None:
                    profiler.set_client_run_id(run_id.strip())
                if flush:
                    state.flush_requested = True
                if text == "" and not flush:
                    state.close_requested = True
                    wake_event.set()
                    return
                if text:
                    # Ignore the known protocol primer chunk (" ") before first real text.
                    if not state.saw_non_ws_text and text == " " and not flush:
                        wake_event.set()
                        continue
                    state.buffer_text += text
                    if profiler is not None:
                        profiler.mark("first_text")
                        if text.strip():
                            profiler.mark("first_non_ws_text")
                    if text.strip():
                        state.saw_non_ws_text = True
            wake_event.set()
    except WebSocketDisconnect:
        async with state_lock:
            state.close_requested = True
        shutdown_event.set()
        wake_event.set()
    except Exception:
        async with state_lock:
            state.close_requested = True
        shutdown_event.set()
        wake_event.set()
    finally:
        wake_event.set()


async def synth_loop(
    backend: TTSBackend,
    state: SessionState,
    state_lock: asyncio.Lock,
    wake_event: asyncio.Event,
    shutdown_event: asyncio.Event,
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    aligner: Optional[Any],
    math_normalizer: Optional[Any],
    profiler: Optional[Profiler],
) -> None:
    chunk_ms = 30.0
    target_sr = 44100
    idle_s = IDLE_TRIGGER_MS / 1000.0
    do_math_normalize = math_normalization_enabled()

    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(wake_event.wait(), timeout=idle_s)
        except asyncio.TimeoutError:
            pass
        wake_event.clear()

        while True:
            now = time.monotonic()
            async with state_lock:
                unsaid = state.buffer_text[state.synth_cursor :]
                flush = state.flush_requested
                close = state.close_requested
                last_input_ts = state.last_input_ts
                inline_math_open = has_unclosed_inline_math(unsaid)
                idle_ready = bool(unsaid) and (now - last_input_ts) >= idle_s
                should_synth = False

                if unsaid and (
                    flush
                    or close
                    or _has_sentence_boundary(unsaid)
                    or len(unsaid) >= MIN_CHARS_TO_SYNTH
                    or idle_ready
                ):
                    if inline_math_open and not flush and not close:
                        should_synth = False
                        segment = None
                        segment_start_idx = 0
                    else:
                        should_synth = True
                        segment_start_idx = state.synth_cursor
                        segment = unsaid
                        if (
                            not state.first_segment_done
                            and not flush
                            and not close
                            and len(unsaid) > FIRST_SEGMENT_LIMIT
                            and not has_unclosed_inline_math(unsaid[:FIRST_SEGMENT_LIMIT])
                        ):
                            segment = unsaid[:FIRST_SEGMENT_LIMIT]
                            state.synth_cursor += len(segment)
                        else:
                            state.synth_cursor = len(state.buffer_text)
                        if segment.strip():
                            state.first_segment_done = True
                        state.flush_requested = False
                elif flush and not unsaid:
                    state.flush_requested = False
                    segment = None
                    segment_start_idx = 0
                else:
                    segment = None
                    segment_start_idx = 0

                should_close = close and not unsaid and not should_synth

            if segment:
                spoken_segment = segment
                if do_math_normalize:
                    if profiler is not None:
                        profiler.mark("math_normalize_start")
                    if math_normalizer is not None:
                        normalized = await math_normalizer.normalize(segment)
                    else:
                        normalized = normalize_for_tts(segment)
                    if profiler is not None:
                        profiler.mark("math_normalize_done")
                    spoken_segment = normalized.text
                    local_source_indices = normalized.source_indices
                else:
                    local_source_indices = list(range(len(segment)))

                if not spoken_segment.strip():
                    continue
                source_indices = [int(segment_start_idx + i) for i in local_source_indices]

                stream_fn = getattr(backend, "synthesize_stream", None)
                use_stream = callable(stream_fn) and aligner is None
                if use_stream:
                    if profiler is not None:
                        profiler.mark("first_synth_start")
                        profiler.set("last_synth_start")
                    await _stream_segment(
                        backend,
                        spoken_segment,
                        segment_start_idx,
                        source_indices,
                        out_queue,
                        shutdown_event,
                        target_sr,
                        profiler,
                    )
                else:
                    if shutdown_event.is_set():
                        return
                    if profiler is not None:
                        profiler.mark("first_synth_start")
                        profiler.set("last_synth_start")
                    result = await asyncio.to_thread(backend.synthesize, spoken_segment)
                    if profiler is not None:
                        profiler.mark("backend_synth_done")
                    resampled = resample_linear(result.audio, result.sr, sr_out=target_sr)
                    total_ms = (resampled.shape[0] / float(target_sr)) * 1000.0

                    alignment = None
                    if aligner is not None:
                        if profiler is not None:
                            profiler.mark("aligner_start")
                        alignment = await asyncio.to_thread(
                            aligner.align, result.audio, result.sr, spoken_segment
                        )
                        if profiler is not None:
                            profiler.mark("aligner_done")
                    if alignment is None:
                        alignment = alignment_from_duration(
                            spoken_segment,
                            total_ms,
                            char_durations_ms=result.char_durations_ms,
                        )
                    idxs = alignment.get("char_indices", [])
                    if len(source_indices) == len(alignment.get("chars", [])):
                        alignment["char_indices"] = [int(v) for v in source_indices]
                    else:
                        if not isinstance(idxs, list) or len(idxs) != len(
                            alignment.get("chars", [])
                        ):
                            idxs = list(range(len(alignment.get("chars", []))))
                        alignment["char_indices"] = [
                            int(segment_start_idx + int(i)) for i in idxs
                        ]

                    chunk_samples = max(1, int(round(target_sr * (chunk_ms / 1000.0))))
                    total_samples = resampled.shape[0]

                    for start in range(0, total_samples, chunk_samples):
                        if shutdown_event.is_set():
                            return
                        end = min(total_samples, start + chunk_samples)
                        chunk_audio = resampled[start:end]
                        pcm16 = float_to_pcm16_bytes(chunk_audio)
                        audio_b64 = pcm16_bytes_to_base64(pcm16)
                        win_start_ms = (start / float(target_sr)) * 1000.0
                        win_end_ms = (end / float(target_sr)) * 1000.0
                        chunk_alignment = split_alignment_for_window(
                            alignment, win_start_ms, win_end_ms
                        )
                        payload = {"audio": audio_b64, "alignment": chunk_alignment}
                        if LOG_ALIGNMENT and out_queue.qsize() < LOG_ALIGNMENT_MAX:
                            print(
                                "[align-chunk]",
                                spoken_segment[:30].replace("\n", " "),
                                "chars", len(chunk_alignment.get("chars", [])),
                                "start_ms", chunk_alignment.get("char_start_times_ms", [])[:3],
                                "dur_ms", chunk_alignment.get("char_durations_ms", [])[:3],
                            )
                        if profiler is not None:
                            profiler.mark("first_audio_out")
                            profiler.set("last_audio_out")
                        await out_queue.put(payload)

                async with state_lock:
                    should_close_after = state.close_requested and (
                        state.synth_cursor >= len(state.buffer_text)
                    )
                if should_close_after:
                    await out_queue.put(None)
                    shutdown_event.set()
                    return

                continue

            if should_close:
                await out_queue.put(None)
                shutdown_event.set()
                return
            break


async def send_loop(
    ws: WebSocket,
    state: SessionState,
    state_lock: asyncio.Lock,
    wake_event: asyncio.Event,
    shutdown_event: asyncio.Event,
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    profiler: Optional[Profiler],
) -> None:
    while True:
        try:
            item = await asyncio.wait_for(out_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if shutdown_event.is_set():
                return
            continue
        if item is None:
            if profiler is not None:
                metrics = profiler.finalize_metrics()
                if metrics:
                    try:
                        await ws.send_json({"type": "metrics", "metrics": metrics})
                    except Exception:
                        pass
            shutdown_event.set()
            try:
                await ws.close()
            except Exception:
                pass
            return
        try:
            if profiler is not None:
                profiler.mark("first_send")
                profiler.set("last_send")
            await ws.send_json(item)
        except Exception:
            async with state_lock:
                state.close_requested = True
            shutdown_event.set()
            wake_event.set()
            return


async def run_session(
    ws: WebSocket,
    backend: TTSBackend,
    aligner: Optional[Any] = None,
    math_normalizer: Optional[Any] = None,
    profile_enabled: bool = False,
) -> None:
    profiler = Profiler(uuid.uuid4().hex[:8], profile_enabled)
    profiler.mark("session_start")
    state = SessionState()
    state_lock = asyncio.Lock()
    wake_event = asyncio.Event()
    shutdown_event = asyncio.Event()
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    recv_task = asyncio.create_task(
        recv_loop(ws, state, state_lock, wake_event, shutdown_event, profiler)
    )
    synth_task = asyncio.create_task(
        synth_loop(
            backend,
            state,
            state_lock,
            wake_event,
            shutdown_event,
            out_queue,
            aligner,
            math_normalizer,
            profiler,
        )
    )
    send_task = asyncio.create_task(
        send_loop(
            ws,
            state,
            state_lock,
            wake_event,
            shutdown_event,
            out_queue,
            profiler,
        )
    )

    tasks = [recv_task, synth_task, send_task]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        profiler.report()
