"""
One-file GPU TTS WebSocket server using NeMo FastPitch (with real durations).
Tested target: Colab / T4 GPU.

Env vars:
  MODEL_NAME   (default: "tts_en_fastpitch")  # NeMo pretrained
  HOST         (default: "0.0.0.0")
  PORT         (default: "8000")
  LOG_ALIGNMENT (1 to log first few chunk alignments)
  LOG_ALIGNMENT_MAX (default: 5)
"""

import asyncio
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import uvicorn

app = FastAPI()

# ---------------- FastPitch backend ----------------


class FastPitchBackend:
    def __init__(self, model_name: str = "tts_en_fastpitch", device: str = "cuda"):
        import nemo.collections.tts as nemo_tts

        self.device = device
        self.model = nemo_tts.models.FastPitchModel.from_pretrained(model_name).to(
            device
        )
        self.hifigan = nemo_tts.models.HifiGanModel.from_pretrained(
            "tts_hifigan"
        ).to(device)
        self.sample_rate = self.hifigan.sample_rate
        self.model.eval()
        self.hifigan.eval()

    @torch.inference_mode()
    def synthesize(self, text: str):
        parsed = self.model.parse(text, do_tokenize=True)
        # FastPitch returns spectrograms + durations
        spect, _, durations, _, _ = self.model.generate_spectrogram(
            tokens=parsed, pace=1.0, dur_tgt=None, pitch_tgt=None, speaker=None
        )
        audio = self.hifigan.convert_spectrogram_to_audio(spec=spect)
        audio = audio.squeeze().cpu().numpy().astype(np.float32)
        durations = durations.squeeze().cpu().numpy()  # per-token (chars)
        return audio, self.sample_rate, durations


# --------------- Alignment helpers ----------------


def durations_to_alignment(text: str, durations: np.ndarray, frame_hop: float, sr: int):
    # durations are in spectrogram frames; convert to ms
    hop_ms = (frame_hop / sr) * 1000.0
    chars = list(text)
    if len(durations) < len(chars):
        # pad if needed
        durations = np.pad(durations, (0, len(chars) - len(durations)), constant_values=1)
    durations = durations[: len(chars)]
    durs_ms = (durations * hop_ms).astype(int)
    starts = []
    cur = 0
    for d in durs_ms:
        starts.append(cur)
        cur += d
    return {
        "chars": chars,
        "char_start_times_ms": starts,
        "char_durations_ms": durs_ms.tolist(),
    }


def split_alignment(aln: Dict[str, Any], win_start_ms: float, win_end_ms: float):
    out_c, out_s, out_d = [], [], []
    chars = aln["chars"]
    starts = aln["char_start_times_ms"]
    durs = aln["char_durations_ms"]
    for ch, s, d in zip(chars, starts, durs):
        e = s + d
        if e <= win_start_ms or s >= win_end_ms:
            continue
        overlap_start = max(s, win_start_ms)
        overlap_end = min(e, win_end_ms)
        rel_start = int(round(overlap_start - win_start_ms))
        rel_dur = int(round(overlap_end - overlap_start))
        if rel_dur > 0:
            out_c.append(ch)
            out_s.append(rel_start)
            out_d.append(rel_dur)
    return {
        "chars": out_c,
        "char_start_times_ms": out_s,
        "char_durations_ms": out_d,
    }


# --------------- Session state ----------------


class SessionState:
    def __init__(self):
        self.buffer_text = ""
        self.synth_cursor = 0
        self.flush = False
        self.close = False


# --------------- WebSocket handlers ----------------


LOG_ALIGNMENT = os.environ.get("LOG_ALIGNMENT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LOG_ALIGNMENT_MAX = int(os.environ.get("LOG_ALIGNMENT_MAX", "5"))


async def recv_loop(ws: WebSocket, state: SessionState, event: asyncio.Event):
    try:
        while True:
            msg = await ws.receive_json()
            text = str(msg.get("text", ""))
            flush = bool(msg.get("flush", False))
            if flush:
                state.flush = True
            if text == "" and not flush:
                state.close = True
                event.set()
                return
            if text:
                state.buffer_text += text
            event.set()
    except WebSocketDisconnect:
        state.close = True
        event.set()
    except Exception:
        state.close = True
        event.set()


async def synth_loop(
    ws: WebSocket,
    backend: FastPitchBackend,
    state: SessionState,
    event: asyncio.Event,
    out_q: asyncio.Queue,
):
    chunk_ms = 30.0
    chunk_samples = int(round(backend.sample_rate * (chunk_ms / 1000.0)))
    log_count = 0

    while True:
        await event.wait()
        event.clear()

        while True:
            unsaid = state.buffer_text[state.synth_cursor :]
            if not unsaid and not state.flush and not state.close:
                break
            if not unsaid and state.flush:
                state.flush = False
                break

            # trigger: flush, close, sentence boundary, len >= 80
            trigger = (
                state.flush
                or state.close
                or any(ch in ".?!\n" for ch in unsaid)
                or len(unsaid) >= 80
            )
            if not trigger:
                break

            segment = unsaid
            state.synth_cursor = len(state.buffer_text)
            state.flush = False

            audio, sr, durations = await asyncio.to_thread(
                backend.synthesize, segment
            )
            hop = backend.model.cfg.n_mels and backend.model.cfg["mel_loss_reduction"] if hasattr(backend.model.cfg, "__getitem__") else 256  # safe fallback
            hop = getattr(backend.model, "hop_length", 256)
            alignment = durations_to_alignment(segment, durations, hop, sr)

            total_samples = audio.shape[0]
            for start in range(0, total_samples, chunk_samples):
                end = min(total_samples, start + chunk_samples)
                frame = audio[start:end]
                start_ms = (start / sr) * 1000.0
                end_ms = (end / sr) * 1000.0
                chunk_alignment = split_alignment(alignment, start_ms, end_ms)
                payload = {
                    "audio": base64_audio(frame),
                    "alignment": chunk_alignment,
                }
                if LOG_ALIGNMENT and log_count < LOG_ALIGNMENT_MAX:
                    print(
                        "[align-chunk]",
                        segment[:40].replace("\n", " "),
                        "chars", len(chunk_alignment.get("chars", [])),
                        "start_ms", chunk_alignment.get("char_start_times_ms", [])[:3],
                        "dur_ms", chunk_alignment.get("char_durations_ms", [])[:3],
                    )
                    log_count += 1
                await out_q.put(payload)

            if state.close and state.synth_cursor >= len(state.buffer_text):
                await out_q.put(None)
                return


def base64_audio(x: np.ndarray) -> str:
    import base64

    clipped = np.clip(x, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm16).decode("ascii")


async def send_loop(ws: WebSocket, out_q: asyncio.Queue):
    while True:
        item = await out_q.get()
        if item is None:
            try:
                await ws.close()
            finally:
                return
        try:
            await ws.send_json(item)
        except Exception:
            return


@app.websocket("/tts")
async def tts_ws(ws: WebSocket):
    await ws.accept()
    backend_name = os.environ.get("MODEL_NAME", "tts_en_fastpitch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = FastPitchBackend(backend_name, device=device)

    state = SessionState()
    event = asyncio.Event()
    out_q: asyncio.Queue = asyncio.Queue()

    recv_task = asyncio.create_task(recv_loop(ws, state, event))
    synth_task = asyncio.create_task(synth_loop(ws, backend, state, event, out_q))
    send_task = asyncio.create_task(send_loop(ws, out_q))

    tasks = [recv_task, synth_task, send_task]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("fast_server:app", host=host, port=port, reload=False, workers=1)
