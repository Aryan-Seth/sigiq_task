from __future__ import annotations

import asyncio
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

# minimum information that i need to actually get something done
@dataclass
class SessionState:
    buffer_text: str = ""
    synth_cursor: int = 0
    flush_requested: bool = False
    close_requested: bool = False
    last_input_ts: float = field(default_factory=time.monotonic)
    first_segment_done: bool = False


def _has_sentence_boundary(text: str) -> bool:
    for ch in text:
        if ch in ".?!\n":
            return True
    return False


EST_CHAR_MS = 55.0
MIN_CHARS_TO_SYNTH = 10
IDLE_TRIGGER_MS = 100
FIRST_SEGMENT_LIMIT = 100


class Profiler:
    def __init__(self, session_id: str, enabled: bool) -> None:
        self.session_id = session_id
        self.enabled = enabled
        self.t0 = time.perf_counter()
        self.times: Dict[str, float] = {}

    def mark(self, key: str) -> None:
        if not self.enabled:
            return
        if key not in self.times:
            self.times[key] = time.perf_counter()

    def set(self, key: str) -> None:
        if not self.enabled:
            return
        self.times[key] = time.perf_counter()

    def report(self) -> None:
        if not self.enabled:
            return
        self.set("session_end")
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
        parts = [f"session={self.session_id}"]
        for key in order:
            if key in self.times:
                delta_ms = (self.times[key] - self.t0) * 1000.0
                parts.append(f"{key}={delta_ms:.1f}ms")
        if "first_non_ws_text" in self.times and "first_audio_out" in self.times:
            ttft_ms = (self.times["first_audio_out"] - self.times["first_non_ws_text"]) * 1000.0
            parts.append(f"ttft_audio_out={ttft_ms:.1f}ms")
        # breakdowns for latency hunting
        def gap(a: str, b: str) -> Optional[float]:
            if a in self.times and b in self.times:
                return (self.times[b] - self.times[a]) * 1000.0
            return None

        buffer_wait = gap("first_non_ws_text", "first_synth_start")
        backend_wait = gap("first_synth_start", "backend_first_chunk")
        post_wait = gap("backend_first_chunk", "first_audio_out")
        send_wait = gap("first_audio_out", "first_send")
        if buffer_wait is not None:
            parts.append(f"gap_buffer_to_synth={buffer_wait:.1f}ms")
        if backend_wait is not None:
            parts.append(f"gap_synth_to_first_chunk={backend_wait:.1f}ms")
        if post_wait is not None:
            parts.append(f"gap_first_chunk_to_audio_out={post_wait:.1f}ms")
        if send_wait is not None:
            parts.append(f"gap_audio_out_to_send={send_wait:.1f}ms")

        print("[tts-profile] " + " ".join(parts))


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
        }
    total_ms = max(1.0, EST_CHAR_MS * len(text))
    return alignment_from_duration(text, total_ms)


async def _stream_segment(
    backend: TTSBackend,
    segment: str,
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    target_sr: int,
    profiler: Optional[Profiler],
) -> None:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
    alignment_full = _estimate_alignment(segment)
    estimated_total_ms = sum(alignment_full.get("char_durations_ms", []))
    est_cursor_ms = 0.0
    chunk_ms = 30.0
    chunk_samples = max(1, int(round(target_sr * (chunk_ms / 1000.0))))
    audio_buffer = np.zeros(0, dtype=np.float32)

    def producer() -> None:
        try:
            for chunk in backend.synthesize_stream(segment):
                loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(producer))

    finalizing = False
    try:
        while True:
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
                await out_queue.put(payload)
    finally:
        await producer_task


async def recv_loop(
    ws: WebSocket,
    state: SessionState,
    state_lock: asyncio.Lock,
    wake_event: asyncio.Event,
    profiler: Optional[Profiler],
) -> None:
    try:
        while True:
            msg = await ws.receive_json()
            if profiler is not None:
                profiler.mark("first_message")
            text = str(msg.get("text", ""))
            flush = bool(msg.get("flush", False))

            async with state_lock:
                state.last_input_ts = time.monotonic() #what does this even mean, why is there something called time.monotonic()
                if flush: 
                    state.flush_requested = True 
                if text == "" and not flush:
                    state.close_requested = True
                    wake_event.set() #what case event is this
                    return
                if text:
                    state.buffer_text += text #this i assume is the first websocket received text
                    if profiler is not None:
                        profiler.mark("first_text")
                        if text.strip():
                            profiler.mark("first_non_ws_text")
            wake_event.set()
    except WebSocketDisconnect:
        async with state_lock:
            state.close_requested = True #what state.close do here since we've not used it in this loop yet
        wake_event.set()
    except Exception:
        async with state_lock:
            state.close_requested = True
        wake_event.set()


async def synth_loop(
    backend: TTSBackend,
    state: SessionState,
    state_lock: asyncio.Lock,
    wake_event: asyncio.Event,
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    aligner: Optional[Any],
    profiler: Optional[Profiler],
) -> None:
    chunk_ms = 30.0
    target_sr = 44100
    idle_s = IDLE_TRIGGER_MS / 1000.0

    while True:
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
                idle_ready = bool(unsaid) and (now - last_input_ts) >= idle_s
                should_synth = False

                if unsaid and (
                    flush
                    or close
                    or _has_sentence_boundary(unsaid)
                    or len(unsaid) >= MIN_CHARS_TO_SYNTH
                    or idle_ready
                ):
                    should_synth = True
                    segment = unsaid
                    state.synth_cursor = len(state.buffer_text)
                    state.flush_requested = False
                elif flush and not unsaid:
                    state.flush_requested = False
                    segment = None
                else:
                    segment = None

                should_close = close and not unsaid and not should_synth

            if segment:
                stream_fn = getattr(backend, "synthesize_stream", None)
                use_stream = callable(stream_fn) and aligner is None
                if use_stream:
                    if profiler is not None:
                        profiler.mark("first_synth_start")
                        profiler.set("last_synth_start")
                    await _stream_segment(
                        backend, segment, out_queue, target_sr, profiler
                    )
                else:
                    if profiler is not None:
                        profiler.mark("first_synth_start")
                        profiler.set("last_synth_start")
                    result = await asyncio.to_thread(backend.synthesize, segment)
                    if profiler is not None:
                        profiler.mark("backend_synth_done")
                    resampled = resample_linear(result.audio, result.sr, sr_out=target_sr)
                    total_ms = (resampled.shape[0] / float(target_sr)) * 1000.0

                    alignment = None
                    if aligner is not None:
                        alignment = await asyncio.to_thread(
                            aligner.align, result.audio, result.sr, segment
                        )
                    if alignment is None:
                        alignment = alignment_from_duration(
                            segment,
                            total_ms,
                            char_durations_ms=result.char_durations_ms,
                        )

                    chunk_samples = max(1, int(round(target_sr * (chunk_ms / 1000.0))))
                    total_samples = resampled.shape[0]

                    for start in range(0, total_samples, chunk_samples):
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
                            segment[:30].replace("\n", " "),
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
                    return

                continue

            if should_close:
                await out_queue.put(None)
                return
            break


async def send_loop(
    ws: WebSocket,
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    profiler: Optional[Profiler],
) -> None:
    while True:
        item = await out_queue.get()
        if item is None:
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
            return


async def run_session(
    ws: WebSocket,
    backend: TTSBackend,
    aligner: Optional[Any] = None,
    profile_enabled: bool = False,
) -> None:
    profiler = Profiler(uuid.uuid4().hex[:8], profile_enabled)
    profiler.mark("session_start")
    state = SessionState()
    state_lock = asyncio.Lock()
    wake_event = asyncio.Event()
    out_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    recv_task = asyncio.create_task(
        recv_loop(ws, state, state_lock, wake_event, profiler)
    )
    synth_task = asyncio.create_task(
        synth_loop(backend, state, state_lock, wake_event, out_queue, aligner, profiler)
    )
    send_task = asyncio.create_task(send_loop(ws, out_queue, profiler))

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
                task.cancel() #this async requests the task to close 
        await asyncio.gather(*tasks, return_exceptions=True) #now we wait for all the tasks to close and send back a cancel_true notif
        profiler.report()
